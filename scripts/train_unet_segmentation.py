"""Train a small PyTorch U-Net for multi-label FBM defect segmentation.

This is the first real deep-learning training entrypoint. The existing
train_segmentation_smoke.py script is a NumPy wiring check; this script is the
model path to use when PyTorch is installed.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.training.segmentation import INPUT_CHANNELS, TARGET_CHANNELS, load_batch, load_manifest_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv")
    parser.add_argument("--out", default="outputs/pattern_asset_pipeline/asset_unet_segmentation.html")
    parser.add_argument("--metrics", default="outputs/pattern_asset_pipeline/asset_unet_segmentation_metrics.json")
    parser.add_argument("--model-out", default="outputs/models/asset_unet_segmentation.pt")
    parser.add_argument("--output-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--max-train-samples", type=int, default=64)
    parser.add_argument("--max-val-samples", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--min-positive-samples-per-class", type=int, default=1)
    parser.add_argument(
        "--allow-incomplete-target-coverage",
        action="store_true",
        help="Allow training even when the train split lacks positive samples for one or more target classes.",
    )
    parser.add_argument("--check-deps", action="store_true", help="Write dependency status and exit without training.")
    return parser.parse_args(argv)


def torch_status() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on local environment
        return False, f"{type(exc).__name__}: {exc}"
    return True, "available"


def require_torch() -> Any:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "PyTorch is not installed. Install torch in the training environment, "
            "then rerun this script."
        ) from exc
    return torch, nn, F, DataLoader, TensorDataset


def make_model(torch: Any, nn: Any, F: Any, input_channels: int, output_channels: int, base_channels: int) -> Any:
    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: Any) -> Any:
            return self.net(x)

    class SmallUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            c = base_channels
            self.enc1 = ConvBlock(input_channels, c)
            self.enc2 = ConvBlock(c, c * 2)
            self.bridge = ConvBlock(c * 2, c * 4)
            self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
            self.dec2 = ConvBlock(c * 4, c * 2)
            self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
            self.dec1 = ConvBlock(c * 2, c)
            self.out = nn.Conv2d(c, output_channels, kernel_size=1)

        def forward(self, x: Any) -> Any:
            e1 = self.enc1(x)
            e2 = self.enc2(F.max_pool2d(e1, 2))
            b = self.bridge(F.max_pool2d(e2, 2))
            d2 = self.up2(b)
            d2 = self.dec2(torch_cat_like(d2, e2))
            d1 = self.up1(d2)
            d1 = self.dec1(torch_cat_like(d1, e1))
            return self.out(d1)

    def torch_cat_like(left: Any, skip: Any) -> Any:
        if left.shape[-2:] != skip.shape[-2:]:
            left = F.interpolate(left, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([left, skip], dim=1)

    return SmallUNet()


def load_tensors(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    manifest = Path(args.manifest)
    train_rows = load_manifest_rows(manifest, split="train")
    val_rows = load_manifest_rows(manifest, split="val") or train_rows
    train = load_batch(train_rows, ROOT, args.output_size, args.max_train_samples)
    val = load_batch(val_rows, ROOT, args.output_size, args.max_val_samples)
    return train.inputs, train.targets, val.inputs, val.targets, train.sample_ids, val.sample_ids


def manifest_target_coverage(
    manifest: Path,
    min_positive_samples_per_class: int,
) -> dict[str, Any]:
    if not manifest.exists():
        issue = f"manifest not found: {repo_path(manifest)}"
        return {
            "status": "MISSING",
            "train_status": "CHECK",
            "validation_status": "CHECK",
            "manifest": repo_path(manifest),
            "min_positive_samples_per_class": int(min_positive_samples_per_class),
            "train_samples": 0,
            "val_samples": 0,
            "positive_train_samples": {},
            "positive_val_samples": {},
            "missing_train_classes": list(TARGET_CHANNELS),
            "missing_val_classes": list(TARGET_CHANNELS),
            "missing_classes": list(TARGET_CHANNELS),
            "blocking_issues": [issue],
            "issues": [issue],
        }
    rows = load_manifest_rows(manifest)
    train_rows = [row for row in rows if row.get("split") == "train"]
    val_rows = [row for row in rows if row.get("split") == "val"]
    minimum = max(0, int(min_positive_samples_per_class))
    train_counts = {name: sum(row_has_target(row, name) for row in train_rows) for name in TARGET_CHANNELS}
    val_counts = {name: sum(row_has_target(row, name) for row in val_rows) for name in TARGET_CHANNELS}
    missing_train = [name for name, count in train_counts.items() if count < minimum]
    missing_val = [name for name, count in val_counts.items() if count < minimum]
    issues: list[str] = []
    blocking_issues: list[str] = []
    if not train_rows:
        blocking_issues.append("manifest has no train split rows")
    if not val_rows:
        issues.append("manifest has no validation split rows")
    for name in missing_train:
        blocking_issues.append(f"{name} has {train_counts[name]} train positives, needs >= {minimum}")
    for name in missing_val:
        issues.append(f"{name} has {val_counts[name]} validation positives, metrics will be uninformative")
    issues = [*blocking_issues, *issues]
    return {
        "status": "PASS" if not issues else "CHECK",
        "train_status": "PASS" if not blocking_issues else "CHECK",
        "validation_status": "PASS" if not missing_val and val_rows else "CHECK",
        "manifest": repo_path(manifest),
        "min_positive_samples_per_class": minimum,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "positive_train_samples": train_counts,
        "positive_val_samples": val_counts,
        "missing_train_classes": missing_train,
        "missing_val_classes": missing_val,
        "missing_classes": missing_train,
        "blocking_issues": blocking_issues,
        "issues": issues,
    }


def row_has_target(row: dict[str, str], class_name: str) -> bool:
    flag = row.get(f"has_{class_name}")
    if flag not in {None, ""}:
        try:
            return float(flag) > 0.0
        except ValueError:
            return False
    ratio = row.get(f"{class_name}_mask_ratio", "")
    try:
        return float(ratio) > 0.0
    except ValueError:
        return False


def require_manifest_target_coverage(args: argparse.Namespace) -> dict[str, Any]:
    coverage = manifest_target_coverage(Path(args.manifest), args.min_positive_samples_per_class)
    if coverage["blocking_issues"] and not args.allow_incomplete_target_coverage:
        issue_text = "; ".join(str(item) for item in coverage["blocking_issues"])
        raise ValueError(
            "U-Net training requires positive train samples for every target class. "
            f"{issue_text}. Use --allow-incomplete-target-coverage only for wiring/debug runs."
        )
    return coverage


def pos_weight_from_targets(targets: np.ndarray) -> np.ndarray:
    prevalence = np.clip(targets.mean(axis=(0, 2, 3)), 1e-6, 1.0)
    return np.minimum((1.0 - prevalence) / prevalence, 50.0).astype(np.float32)


def train(args: argparse.Namespace) -> dict[str, Any]:
    coverage = require_manifest_target_coverage(args)
    torch, nn, F, DataLoader, TensorDataset = require_torch()
    torch.manual_seed(args.seed)
    x_train, y_train, x_val, y_val, train_ids, val_ids = load_tensors(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(torch, nn, F, x_train.shape[1], y_train.shape[1], args.base_channels).to(device)
    pos_weight = torch.tensor(pos_weight_from_targets(y_train), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight[:, None, None])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    history: list[float] = []
    for _ in range(args.epochs):
        model.train()
        epoch_losses = []
        for xb, yb in loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)

    val_metrics = evaluate(model, torch.tensor(x_val), torch.tensor(y_val), device, args.threshold)
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_channels": list(INPUT_CHANNELS),
            "target_channels": list(TARGET_CHANNELS),
            "output_size": args.output_size,
            "base_channels": args.base_channels,
        },
        model_path,
    )
    return {
        "status": "trained",
        "torch_available": True,
        "device": str(device),
        "manifest": repo_path(Path(args.manifest)),
        "model_out": repo_path(model_path),
        "input_channels": list(INPUT_CHANNELS),
        "target_channels": list(TARGET_CHANNELS),
        "output_size": args.output_size,
        "train_samples": len(train_ids),
        "val_samples": len(val_ids),
        "manifest_target_coverage": coverage,
        "epochs": args.epochs,
        "loss": {"history": history, "initial": history[0], "final": history[-1]},
        "validation": val_metrics,
    }


def evaluate(model: Any, x_val: Any, y_val: Any, device: Any, threshold: float) -> dict[str, Any]:
    import torch

    model.eval()
    with torch.no_grad():
        logits = model(x_val.to(device=device, dtype=torch.float32))
        probs = torch.sigmoid(logits).cpu().numpy()
    targets = y_val.numpy() > 0.5
    predictions = probs >= threshold
    rows = []
    for idx, class_name in enumerate(TARGET_CHANNELS):
        pred = predictions[:, idx]
        target = targets[:, idx]
        tp = int((pred & target).sum())
        fp = int((pred & ~target).sum())
        fn = int((~pred & target).sum())
        rows.append(
            {
                "class": class_name,
                "target_pixels": int(target.sum()),
                "predicted_pixels": int(pred.sum()),
                "precision": float(tp / max(tp + fp, 1)),
                "recall": float(tp / max(tp + fn, 1)),
                "iou": float(tp / max(tp + fp + fn, 1)),
            }
        )
    return {"per_class": rows}


def dependency_metrics(args: argparse.Namespace) -> dict[str, Any]:
    available, detail = torch_status()
    coverage = manifest_target_coverage(Path(args.manifest), args.min_positive_samples_per_class)
    return {
        "status": "dependency_check",
        "torch_available": available,
        "torch_status": detail,
        "next_action": "install_torch_and_train_unet" if not available else "run_training",
        "input_channels": list(INPUT_CHANNELS),
        "target_channels": list(TARGET_CHANNELS),
        "manifest_target_coverage": coverage,
    }


def repo_path(target: Path) -> str:
    try:
        return target.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(target)


def relpath(target: Path, base_file: Path) -> str:
    return os.path.relpath(target.resolve(), base_file.resolve().parent).replace("\\", "/")


def metric_rows(metrics: dict[str, Any]) -> str:
    rows = []
    for row in metrics.get("validation", {}).get("per_class", []):
        rows.append(
            "<tr>"
            f"<td>{html.escape(row['class'])}</td>"
            f"<td>{row['target_pixels']}</td>"
            f"<td>{row['predicted_pixels']}</td>"
            f"<td>{row['precision']:.3f}</td>"
            f"<td>{row['recall']:.3f}</td>"
            f"<td>{row['iou']:.3f}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def html_report(metrics: dict[str, Any], metrics_path: Path, out: Path) -> str:
    if not metrics.get("torch_available"):
        coverage = metrics.get("manifest_target_coverage", {})
        body = (
            "<p>PyTorch is not installed in this environment, so U-Net training was not run.</p>"
            f"<p>Status: <code>{html.escape(str(metrics.get('torch_status')))}</code></p>"
            f"<p>Manifest target coverage: <code>{html.escape(str(coverage.get('status', 'UNKNOWN')))}</code></p>"
            "<p>Install PyTorch in the training environment and rerun this script with the same manifest.</p>"
        )
    else:
        loss = metrics.get("loss", {})
        coverage = metrics.get("manifest_target_coverage", {})
        body = f"""
        <p>Device: <code>{html.escape(str(metrics.get('device')))}</code></p>
        <p>Manifest target coverage: <code>{html.escape(str(coverage.get('status', 'UNKNOWN')))}</code></p>
        <p>Loss: {float(loss.get('initial', 0.0)):.4f} -> {float(loss.get('final', 0.0)):.4f}</p>
        <table>
          <tr><th>Class</th><th>Target pixels</th><th>Predicted pixels</th><th>Precision</th><th>Recall</th><th>IoU</th></tr>
          {metric_rows(metrics)}
        </table>
        """
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>FBM U-Net Segmentation Training</title>
  <style>
    body {{ font-family: "Segoe UI", "Noto Sans KR", Arial, sans-serif; margin: 32px; color: #1f2933; line-height: 1.55; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px 10px; text-align: left; }}
    th {{ background: #eef2f7; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>FBM U-Net Segmentation Training</h1>
  {body}
  <p>Metrics JSON: <code>{html.escape(relpath(metrics_path, out))}</code></p>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    metrics = dependency_metrics(args) if args.check_deps else train(args)
    metrics_path = Path(args.metrics)
    out_path = Path(args.out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_report(metrics, metrics_path, out_path), encoding="utf-8")
    print(f"Wrote U-Net report: {out_path}")
    print(f"Wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()
