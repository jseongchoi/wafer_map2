"""Run a NumPy-only segmentation smoke training check."""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.training.segmentation import TARGET_CHANNELS, load_batch, load_manifest_rows

SMOKE_PRIORITY_CLASSES = ("scratch", "local", "ring", "stby_pattern", "edge", "shot_grid")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/reports/fbm_segmentation_manifest.csv")
    parser.add_argument("--out", default="outputs/reports/fbm_segmentation_smoke_report.html")
    parser.add_argument("--metrics", default="outputs/reports/fbm_segmentation_smoke_metrics.json")
    parser.add_argument("--figure", default="outputs/figures/fbm_segmentation_smoke_loss.png")
    parser.add_argument("--output-size", type=int, default=96)
    parser.add_argument("--max-train-samples", type=int, default=8)
    parser.add_argument("--max-val-samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.7)
    return parser.parse_args(argv)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def weighted_bce(logits: np.ndarray, targets: np.ndarray, pos_weight: np.ndarray) -> float:
    probs = sigmoid(logits)
    eps = 1e-6
    loss = -(pos_weight * targets * np.log(probs + eps) + (1.0 - targets) * np.log(1.0 - probs + eps))
    return float(loss.mean())


def train_linear_segmentation(
    x: np.ndarray,
    y: np.ndarray,
    steps: int,
    learning_rate: float,
) -> tuple[np.ndarray, np.ndarray, list[float], np.ndarray]:
    n, in_channels, height, width = x.shape
    target_channels = y.shape[1]
    features = np.moveaxis(x, 1, -1).reshape(-1, in_channels)
    targets = np.moveaxis(y, 1, -1).reshape(-1, target_channels)
    prevalence = np.clip(targets.mean(axis=0), 1e-6, 1.0)
    pos_weight = np.minimum((1.0 - prevalence) / prevalence, 50.0).astype(np.float32)
    weights = np.zeros((in_channels, target_channels), dtype=np.float32)
    bias = np.zeros(target_channels, dtype=np.float32)
    losses: list[float] = []
    for _ in range(steps):
        logits = features @ weights + bias
        losses.append(weighted_bce(logits, targets, pos_weight))
        probs = sigmoid(logits)
        grad_logits = (probs - targets) * np.where(targets > 0, pos_weight, 1.0)
        grad_logits /= max(len(features), 1)
        grad_w = features.T @ grad_logits
        grad_b = grad_logits.sum(axis=0)
        weights -= learning_rate * grad_w.astype(np.float32)
        bias -= learning_rate * grad_b.astype(np.float32)
    losses.append(weighted_bce(features @ weights + bias, targets, pos_weight))
    return weights, bias, losses, pos_weight


def evaluate_batch(x: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> dict[str, Any]:
    n, in_channels, height, width = x.shape
    features = np.moveaxis(x, 1, -1).reshape(-1, in_channels)
    targets = np.moveaxis(y, 1, -1).reshape(-1, y.shape[1])
    logits = features @ weights + bias
    probs = sigmoid(logits)
    predictions = probs >= 0.5
    target_bool = targets > 0.5
    rows = []
    for idx, class_name in enumerate(TARGET_CHANNELS):
        tp = int((predictions[:, idx] & target_bool[:, idx]).sum())
        fp = int((predictions[:, idx] & ~target_bool[:, idx]).sum())
        fn = int((~predictions[:, idx] & target_bool[:, idx]).sum())
        rows.append(
            {
                "class": class_name,
                "target_pixels": int(target_bool[:, idx].sum()),
                "predicted_pixels": int(predictions[:, idx].sum()),
                "precision": float(tp / max(tp + fp, 1)),
                "recall": float(tp / max(tp + fn, 1)),
                "iou": float(tp / max(tp + fp + fn, 1)),
            }
        )
    return {"per_class": rows}


def select_smoke_rows(rows: list[dict[str, str]], max_samples: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for class_name in SMOKE_PRIORITY_CLASSES:
        if len(selected) >= max_samples:
            break
        key = f"has_{class_name}"
        for row in rows:
            if row["sample_id"] in seen:
                continue
            if row.get(key) == "1":
                selected.append(row)
                seen.add(row["sample_id"])
                break
    for row in rows:
        if len(selected) >= max_samples:
            break
        if row["sample_id"] in seen:
            continue
        selected.append(row)
        seen.add(row["sample_id"])
    return selected


def save_loss_figure(losses: list[float], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(losses)), losses, marker="o")
    ax.set_title("Segmentation Smoke BCE Loss")
    ax.set_xlabel("step")
    ax.set_ylabel("weighted BCE")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def relpath(target: Path, base_file: Path) -> str:
    return os.path.relpath(target.resolve(), base_file.resolve().parent).replace("\\", "/")


def metric_rows(metrics: dict[str, Any]) -> str:
    rows = []
    for row in metrics["validation"]["per_class"]:
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


def html_report(metrics: dict[str, Any], figure: Path, metrics_path: Path, out: Path) -> str:
    loss_delta = metrics["loss"]["initial"] - metrics["loss"]["final"]
    verdict = "PASS" if np.isfinite(metrics["loss"]["final"]) and loss_delta >= 0 else "CHECK"
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>FBM Segmentation Smoke Training</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2933; line-height: 1.55; }}
    h1, h2 {{ color: #111827; }}
    .note {{ background: #eef6ff; border-left: 4px solid #2563eb; padding: 12px 14px; margin: 14px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f7; }}
    img {{ width: 100%; max-width: 820px; border: 1px solid #d8dee9; border-radius: 8px; background: white; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>FBM Segmentation Smoke Training</h1>
  <div class="note">이 리포트는 성능 리포트가 아니라 배관 검증 리포트다. NumPy 1x1 sigmoid baseline으로 manifest, input tensor, target mask, weighted BCE가 정상 연결되는지 확인한다.</div>
  <h2>Gate Result: {verdict}</h2>
  <ul>
    <li>Train samples: {metrics['train_samples']}, Val samples: {metrics['val_samples']}</li>
    <li>Tensor size: {metrics['output_size']} x {metrics['output_size']}</li>
    <li>Initial loss: {metrics['loss']['initial']:.4f}</li>
    <li>Final loss: {metrics['loss']['final']:.4f}</li>
    <li>Loss delta: {loss_delta:.4f}</li>
  </ul>
  <img src="{html.escape(relpath(figure, out))}" alt="smoke training loss">
  <h2>Validation Snapshot</h2>
  <table>
    <tr><th>Class</th><th>Target pixels</th><th>Predicted pixels</th><th>Precision</th><th>Recall</th><th>IoU</th></tr>
    {metric_rows(metrics)}
  </table>
  <h2>Next Gate</h2>
  <p>다음은 PyTorch/GPU 환경에서 small U-Net 또는 lightweight SegFormer로 넘어가되, 성공 기준은 전체 mIoU가 아니라 scratch recall, stby-hidden scratch recall, local small-blob recall로 둔다.</p>
  <p>Metrics JSON: <code>{html.escape(relpath(metrics_path, out))}</code></p>
</body>
</html>
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = Path(args.manifest)
    train_rows = load_manifest_rows(manifest, split="train")
    val_rows = load_manifest_rows(manifest, split="val")
    repo_root = ROOT
    selected_train_rows = select_smoke_rows(train_rows, args.max_train_samples)
    selected_val_rows = select_smoke_rows(val_rows or train_rows, args.max_val_samples)
    train = load_batch(selected_train_rows, repo_root, args.output_size, args.max_train_samples)
    val = load_batch(selected_val_rows, repo_root, args.output_size, args.max_val_samples)
    weights, bias, losses, pos_weight = train_linear_segmentation(
        train.inputs,
        train.targets,
        steps=args.steps,
        learning_rate=args.learning_rate,
    )
    metrics = {
        "manifest": str(manifest),
        "output_size": args.output_size,
        "train_samples": len(train.sample_ids),
        "val_samples": len(val.sample_ids),
        "target_channels": list(TARGET_CHANNELS),
        "pos_weight": {name: float(value) for name, value in zip(TARGET_CHANNELS, pos_weight)},
        "loss": {
            "history": [float(value) for value in losses],
            "initial": float(losses[0]),
            "final": float(losses[-1]),
        },
        "validation": evaluate_batch(val.inputs, val.targets, weights, bias),
    }
    return metrics


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    metrics = run(args)
    out_path = Path(args.out)
    metrics_path = Path(args.metrics)
    figure_path = Path(args.figure)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    save_loss_figure(metrics["loss"]["history"], figure_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_report(metrics, figure_path, metrics_path, out_path), encoding="utf-8")
    print(f"Wrote segmentation smoke report: {out_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote figure: {figure_path}")


if __name__ == "__main__":
    main()
