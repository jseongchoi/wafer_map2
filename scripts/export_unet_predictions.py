"""Export trained U-Net masks for the local segmentation tool.

The output JSON uses the fbm_prediction_masks/v1 schema consumed by
run_segmentation_tool.py --prediction-json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.assets import mask_to_rle
from wafermap.data import load_sample
from wafermap.training.segmentation import INPUT_CHANNELS, TARGET_CHANNELS, load_manifest_rows, sample_to_input_tensor

PREDICTION_SCHEMA = "fbm_prediction_masks/v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv")
    parser.add_argument("--model", default="outputs/models/asset_unet_segmentation.pt")
    parser.add_argument("--out", default="outputs/predictions/fbm_prediction_masks.json")
    parser.add_argument("--split", choices=("all", "train", "val"), default="all")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means export every matching manifest row.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device string.")
    parser.add_argument("--check-deps", action="store_true", help="Write dependency status and exit without inference.")
    return parser.parse_args(argv)


def torch_status() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on local environment
        return False, f"{type(exc).__name__}: {exc}"
    return True, "available"


def dependency_status(args: argparse.Namespace) -> dict[str, Any]:
    available, detail = torch_status()
    return {
        "status": "dependency_check",
        "torch_available": available,
        "torch_status": detail,
        "next_action": "run_export" if available else "install_torch_and_train_unet",
        "prediction_schema": PREDICTION_SCHEMA,
        "manifest": repo_path(Path(args.manifest)),
        "model": repo_path(Path(args.model)),
        "input_channels": list(INPUT_CHANNELS),
        "target_channels": list(TARGET_CHANNELS),
        "model_exists": Path(args.model).exists(),
    }


def export_predictions(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_manifest_rows(args.manifest, split=None if args.split == "all" else args.split)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError(f"manifest has no rows for split={args.split}")

    torch, model, output_size, device, checkpoint = load_checkpoint(Path(args.model), args.device)
    records = []
    for row in rows:
        sample = load_sample(sample_dir_from_manifest_row(row, ROOT))
        x = sample_to_input_tensor(sample, output_size=output_size)
        xb = torch.tensor(x[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(xb)
            probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
        valid = (sample.wafer_mask > 0) & (sample.valid_test_mask > 0)
        masks_by_family = {}
        for idx, family in enumerate(TARGET_CHANNELS):
            low_res_mask = probs[idx] >= float(args.threshold)
            masks_by_family[family] = resize_mask_nearest(low_res_mask, sample.shape) & valid
        records.append(prediction_record(sample.sample_id, masks_by_family))

    return prediction_payload(
        records,
        model_path=Path(args.model),
        threshold=args.threshold,
        manifest_path=Path(args.manifest),
        split=args.split,
        output_size=output_size,
        device=str(device),
        checkpoint=checkpoint,
    )


def load_checkpoint(model_path: Path, device_arg: str) -> tuple[Any, Any, int, Any, dict[str, Any]]:
    train_module = load_training_module()
    torch, nn, F, *_ = train_module.require_torch()
    device = choose_device(torch, device_arg)
    checkpoint = torch.load(model_path, map_location=device)
    if list(checkpoint.get("input_channels", [])) != list(INPUT_CHANNELS):
        raise ValueError("checkpoint input channels do not match current segmentation input contract")
    if list(checkpoint.get("target_channels", [])) != list(TARGET_CHANNELS):
        raise ValueError("checkpoint target channels do not match current segmentation target contract")
    output_size = int(checkpoint.get("output_size", 0))
    if output_size <= 0:
        raise ValueError("checkpoint is missing a positive output_size")
    base_channels = int(checkpoint.get("base_channels", 16))
    model = train_module.make_model(torch, nn, F, len(INPUT_CHANNELS), len(TARGET_CHANNELS), base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return torch, model, output_size, device, checkpoint


def load_training_module() -> Any:
    path = ROOT / "scripts" / "train_unet_segmentation.py"
    spec = importlib.util.spec_from_file_location("train_unet_segmentation", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"cannot load training module: {path}")
    spec.loader.exec_module(module)
    return module


def choose_device(torch: Any, device_arg: str) -> Any:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def sample_dir_from_manifest_row(row: dict[str, str], repo_root: str | Path) -> Path:
    arrays_path = Path(row["arrays_path"])
    if not arrays_path.is_absolute():
        arrays_path = Path(repo_root) / arrays_path
    return arrays_path.parent


def resize_mask_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2:
        raise ValueError("mask must be 2D")
    if source.shape[0] <= 0 or source.shape[1] <= 0:
        raise ValueError(f"invalid source mask shape: {source.shape}")
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f"invalid target shape: {shape}")
    if source.shape == shape:
        return source.copy()
    y_idx = np.minimum((np.arange(shape[0]) * source.shape[0] // shape[0]).astype(int), source.shape[0] - 1)
    x_idx = np.minimum((np.arange(shape[1]) * source.shape[1] // shape[1]).astype(int), source.shape[1] - 1)
    return source[np.ix_(y_idx, x_idx)].astype(bool)


def prediction_record(sample_id: str, masks_by_family: dict[str, np.ndarray]) -> dict[str, Any]:
    shape = first_mask_shape(masks_by_family)
    masks: dict[str, list[list[int]]] = {}
    pixel_counts: dict[str, int] = {}
    for family in TARGET_CHANNELS:
        mask = np.asarray(masks_by_family.get(family, np.zeros(shape, dtype=bool)), dtype=bool)
        if mask.shape != shape:
            raise ValueError(f"{family} mask shape {mask.shape} does not match {shape}")
        masks[family] = mask_to_rle(mask)
        pixel_counts[family] = int(mask.sum())
    return {"sample_id": str(sample_id), "masks": masks, "pixel_counts": pixel_counts}


def first_mask_shape(masks_by_family: dict[str, np.ndarray]) -> tuple[int, int]:
    for mask in masks_by_family.values():
        shape = np.asarray(mask).shape
        if len(shape) != 2:
            raise ValueError(f"mask must be 2D, got shape {shape}")
        return int(shape[0]), int(shape[1])
    raise ValueError("at least one mask is required")


def prediction_payload(
    records: list[dict[str, Any]],
    *,
    model_path: Path,
    threshold: float,
    manifest_path: Path | None = None,
    split: str = "all",
    output_size: int | None = None,
    device: str | None = None,
    checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": PREDICTION_SCHEMA,
        "source": "unet_segmentation",
        "model_path": repo_path(model_path),
        "threshold": float(threshold),
        "target_families": list(TARGET_CHANNELS),
        "samples": records,
    }
    if manifest_path is not None:
        payload["manifest"] = repo_path(manifest_path)
        payload["split"] = split
    if output_size is not None:
        payload["output_size"] = int(output_size)
    if device is not None:
        payload["device"] = device
    if checkpoint is not None:
        payload["input_channels"] = list(checkpoint.get("input_channels", INPUT_CHANNELS))
        payload["target_channels"] = list(checkpoint.get("target_channels", TARGET_CHANNELS))
    return payload


def repo_path(target: Path) -> str:
    try:
        return target.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(target)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = dependency_status(args) if args.check_deps else export_predictions(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote U-Net prediction export: {out_path}")


if __name__ == "__main__":
    main()
