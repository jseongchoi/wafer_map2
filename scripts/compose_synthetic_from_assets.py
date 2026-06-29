"""Compose labeled synthetic wafers from extracted FBM pattern assets."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.assets import DEFAULT_PROCEDURAL_FAMILIES, TARGET_FAMILIES, mask_location_summary
from wafermap.data import PATTERN_CLASSES, SyntheticSample, load_sample, save_npz, write_json
from wafermap.synth.procedural_patterns import PROCEDURAL_PROBABILITIES, add_procedural_patterns


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sample-dir", required=True, help="Sample dir containing arrays.npz and metadata.json.")
    parser.add_argument("--assets-root", default="data/pattern_assets")
    parser.add_argument("--out-dir", default="data/synthetic/asset_composed")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--assets-per-wafer", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument(
        "--placement-mode",
        choices=("source_jitter", "polar_jitter", "random_valid"),
        default="source_jitter",
        help=(
            "source_jitter preserves absolute process location; "
            "polar_jitter matches radial/angular wafer zone; random_valid stress-tests shape only."
        ),
    )
    parser.add_argument("--jitter-pixels", type=int, default=48, help="Max absolute x/y jitter for source_jitter placement.")
    parser.add_argument(
        "--procedural-families",
        default=",".join(DEFAULT_PROCEDURAL_FAMILIES),
        help="Comma-separated code-generated families. Use none to disable. Default: scratch,edge,shot_grid,random.",
    )
    return parser.parse_args(argv)


def load_assets(assets_root: Path, require_assets: bool = True) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for family in TARGET_FAMILIES:
        family_root = assets_root / family
        if not family_root.exists():
            continue
        for asset_dir in sorted(path for path in family_root.iterdir() if path.is_dir()):
            grade_path = asset_dir / "grade.png"
            mask_path = asset_dir / "mask.png"
            metadata_path = asset_dir / "metadata.json"
            if not (grade_path.exists() and mask_path.exists() and metadata_path.exists()):
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            grade = np.asarray(Image.open(grade_path).convert("L"), dtype=np.uint8)
            mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
            if grade.shape != mask.shape or not mask.any():
                continue
            assets.append({"family": family, "dir": asset_dir, "grade": grade, "mask": mask, "metadata": metadata})
    if not assets and require_assets:
        raise ValueError(f"No pattern assets found under {assets_root}")
    return assets


def parse_family_list(value: str) -> tuple[str, ...]:
    normalized = value.strip().lower()
    if normalized in {"", "none", "off", "false"}:
        return ()
    if normalized == "all":
        return DEFAULT_PROCEDURAL_FAMILIES
    families = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = sorted(set(families) - set(TARGET_FAMILIES))
    if invalid:
        raise ValueError(f"Unknown procedural families: {', '.join(invalid)}")
    return families


def compose_sample(
    base: SyntheticSample,
    assets: list[dict[str, Any]],
    rng: random.Random,
    count: int,
    sample_id: str,
    placement_mode: str = "source_jitter",
    jitter_pixels: int = 48,
    procedural_families: Sequence[str] = DEFAULT_PROCEDURAL_FAMILIES,
) -> SyntheticSample:
    severity = base.severity.copy()
    pattern_masks = np.zeros((len(PATTERN_CLASSES), *base.shape), dtype=np.uint8)
    pattern_intensity = np.zeros((len(PATTERN_CLASSES), *base.shape), dtype=np.float32)
    placed: list[dict[str, Any]] = []
    for asset in rng.sample(assets, k=min(count, len(assets))):
        placement = choose_placement(base, asset, rng, placement_mode=placement_mode, jitter_pixels=jitter_pixels)
        if placement is None:
            continue
        y, x, actual_placement_mode = placement
        grade = asset["grade"]
        mask = asset["mask"]
        h, w = grade.shape
        target = np.s_[y : y + h, x : x + w]
        valid = mask & (base.wafer_mask[target] > 0) & (base.valid_test_mask[target] > 0)
        if not valid.any():
            continue
        patch = severity[target]
        patch[valid] = np.maximum(patch[valid], grade[valid])
        severity[target] = patch
        class_idx = PATTERN_CLASSES.index(asset["family"])
        mask_patch = pattern_masks[class_idx][target]
        intensity_patch = pattern_intensity[class_idx][target]
        mask_patch[valid] = 1
        intensity_patch[valid] = np.maximum(intensity_patch[valid], grade[valid].astype(np.float32) / 7.0)
        pattern_masks[class_idx][target] = mask_patch
        pattern_intensity[class_idx][target] = intensity_patch
        placed.append(
            {
                "family": asset["family"],
                "asset_id": asset["dir"].name,
                "placed_xy": [int(x), int(y)],
                "shape_hw": [int(h), int(w)],
                "composition_rule": "max",
                "placement_mode": actual_placement_mode,
                "requested_placement_mode": placement_mode,
                "source_bbox_xywh": normalized_bbox(asset["metadata"].get("bbox_xywh")),
                "source_location_summary": asset["metadata"].get("location_summary", {}),
                "placed_location_summary": placed_location_summary(base, valid, y, x),
            }
        )
    np_rng = np.random.default_rng(rng.randrange(0, 2**32))
    procedural_patterns = add_procedural_patterns(
        base,
        severity,
        pattern_masks,
        pattern_intensity,
        np_rng,
        tuple(procedural_families),
    )

    metadata = dict(base.metadata)
    metadata.update(
        {
            "sample_id": sample_id,
            "source": "hybrid_pattern_asset_composer",
            "base_sample_id": base.sample_id,
            "pattern_classes": list(PATTERN_CLASSES),
            "composition_rule": "max",
            "placement_mode": placement_mode,
            "jitter_pixels": int(jitter_pixels),
            "procedural_families": list(procedural_families),
            "multi_label": True,
            "stby_target_excluded": True,
            "placed_assets": placed,
            "procedural_patterns": procedural_patterns,
        }
    )
    return SyntheticSample(
        sample_id=sample_id,
        severity=severity,
        wafer_mask=base.wafer_mask.copy(),
        valid_test_mask=base.valid_test_mask.copy(),
        stby_mask=base.stby_mask.copy(),
        pattern_masks=pattern_masks,
        pattern_intensity=pattern_intensity,
        chip_index=base.chip_index.copy(),
        metadata=metadata,
    )


def choose_placement(
    base: SyntheticSample,
    asset: dict[str, Any],
    rng: random.Random,
    *,
    placement_mode: str,
    jitter_pixels: int,
) -> tuple[int, int, str] | None:
    grade = asset["grade"]
    mask = asset["mask"]
    h, w = grade.shape
    if h > base.shape[0] or w > base.shape[1]:
        return None
    if placement_mode == "source_jitter":
        source = source_jitter_candidates(base, asset, rng, h, w, jitter_pixels)
        for y, x in source:
            target = np.s_[y : y + h, x : x + w]
            valid = mask & (base.wafer_mask[target] > 0) & (base.valid_test_mask[target] > 0)
            if valid.sum() >= max(1, int(mask.sum() * 0.50)):
                return y, x, "source_jitter"
    if placement_mode == "polar_jitter":
        source = polar_jitter_candidates(base, asset, rng, h, w)
        for y, x in source:
            target = np.s_[y : y + h, x : x + w]
            valid = mask & (base.wafer_mask[target] > 0) & (base.valid_test_mask[target] > 0)
            if valid.sum() >= max(1, int(mask.sum() * 0.50)):
                return y, x, "polar_jitter"
    for _ in range(200):
        y = rng.randint(0, base.shape[0] - h)
        x = rng.randint(0, base.shape[1] - w)
        target = np.s_[y : y + h, x : x + w]
        valid = mask & (base.wafer_mask[target] > 0) & (base.valid_test_mask[target] > 0)
        if valid.sum() >= max(1, int(mask.sum() * 0.70)):
            return y, x, "random_valid"
    return None


def source_jitter_candidates(
    base: SyntheticSample,
    asset: dict[str, Any],
    rng: random.Random,
    height: int,
    width: int,
    jitter_pixels: int,
) -> list[tuple[int, int]]:
    bbox = normalized_bbox(asset["metadata"].get("bbox_xywh"))
    if bbox is None:
        return []
    x, y, _, _ = bbox
    max_y = max(0, base.shape[0] - height)
    max_x = max(0, base.shape[1] - width)
    anchor = (min(max(int(y), 0), max_y), min(max(int(x), 0), max_x))
    candidates = [anchor]
    jitter = max(0, int(jitter_pixels))
    for _ in range(60):
        jy = min(max(anchor[0] + rng.randint(-jitter, jitter), 0), max_y) if jitter else anchor[0]
        jx = min(max(anchor[1] + rng.randint(-jitter, jitter), 0), max_x) if jitter else anchor[1]
        candidates.append((jy, jx))
    return candidates


def polar_jitter_candidates(
    base: SyntheticSample,
    asset: dict[str, Any],
    rng: random.Random,
    height: int,
    width: int,
) -> list[tuple[int, int]]:
    source = asset["metadata"].get("location_summary", {})
    target_radial = safe_float(source.get("radial_mean"))
    target_theta = safe_float(source.get("theta_mean_deg"))
    if target_radial is None or target_theta is None:
        return []
    max_y = max(0, base.shape[0] - height)
    max_x = max(0, base.shape[1] - width)
    local_ys, local_xs = np.nonzero(asset["mask"])
    if local_ys.size == 0:
        return []
    min_valid_pixels = max(1, int(local_ys.size * 0.50))
    wafer_radius = wafer_radius_for_shape(base.shape, base.wafer_mask > 0)
    scored: list[tuple[float, int, int]] = []
    for _ in range(240):
        y = rng.randint(0, max_y)
        x = rng.randint(0, max_x)
        abs_ys = local_ys + y
        abs_xs = local_xs + x
        valid_points = (base.wafer_mask[abs_ys, abs_xs] > 0) & (base.valid_test_mask[abs_ys, abs_xs] > 0)
        if int(valid_points.sum()) < min_valid_pixels:
            continue
        radial_mean, theta_mean = point_location_mean(
            base.shape,
            wafer_radius,
            abs_ys[valid_points],
            abs_xs[valid_points],
        )
        score = abs(radial_mean - target_radial) + circular_delta_deg(theta_mean, target_theta) / 180.0
        scored.append((float(score), int(y), int(x)))
    scored.sort(key=lambda item: item[0])
    return [(y, x) for _score, y, x in scored[:40]]


def normalized_bbox(value: Any) -> list[int] | None:
    if isinstance(value, list) and len(value) == 4:
        return [int(v) for v in value]
    if isinstance(value, str):
        parts = value.replace(",", " ").replace("[", " ").replace("]", " ").split()
        if len(parts) == 4:
            return [int(float(part)) for part in parts]
    return None


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def circular_delta_deg(left: float, right: float) -> float:
    return abs(((float(left) - float(right) + 180.0) % 360.0) - 180.0)


def wafer_radius_for_shape(shape: tuple[int, int], wafer_mask: np.ndarray) -> float:
    cx = (shape[1] - 1) / 2.0
    cy = (shape[0] - 1) / 2.0
    wy, wx = np.nonzero(wafer_mask)
    if len(wy) == 0:
        return 1.0
    distance = np.sqrt((wx.astype(np.float32) - cx) ** 2 + (wy.astype(np.float32) - cy) ** 2)
    return max(float(distance.max()), 1.0)


def point_location_mean(
    shape: tuple[int, int],
    wafer_radius: float,
    ys: np.ndarray,
    xs: np.ndarray,
) -> tuple[float, float]:
    cx = (shape[1] - 1) / 2.0
    cy = (shape[0] - 1) / 2.0
    yy = ys.astype(np.float32)
    xx = xs.astype(np.float32)
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    radius = np.clip(distance / max(wafer_radius, 1.0), 0.0, 1.0)
    theta = (np.degrees(np.arctan2(xx - cx, -(yy - cy))) + 360.0) % 360.0
    theta_rad = np.radians(theta)
    theta_mean = (np.degrees(np.arctan2(float(np.sin(theta_rad).mean()), float(np.cos(theta_rad).mean()))) + 360.0) % 360.0
    return float(radius.mean()), float(theta_mean)


def placed_location_summary(base: SyntheticSample, local_mask: np.ndarray, y: int, x: int) -> dict[str, Any]:
    full_mask = np.zeros(base.shape, dtype=bool)
    height, width = local_mask.shape
    full_mask[y : y + height, x : x + width] = local_mask
    return mask_location_summary(full_mask, base.wafer_mask > 0)


def write_sample(sample: SyntheticSample, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_npz(out_dir / "arrays.npz", sample)
    write_json(out_dir / "metadata.json", sample.metadata)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    base = load_sample(args.base_sample_dir)
    assets_root = Path(args.assets_root)
    if not assets_root.is_absolute():
        assets_root = ROOT / assets_root
    procedural_families = parse_family_list(args.procedural_families)
    assets = load_assets(assets_root, require_assets=not procedural_families)
    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    rng = random.Random(args.seed)
    for idx in range(args.count):
        sample_id = f"asset_composed_{idx:06d}"
        sample = compose_sample(
            base,
            assets,
            rng,
            args.assets_per_wafer,
            sample_id,
            placement_mode=args.placement_mode,
            jitter_pixels=args.jitter_pixels,
            procedural_families=procedural_families,
        )
        write_sample(sample, out_root / sample_id)
    print(f"Wrote {args.count} composed samples: {out_root}")
    print(f"Asset patterns per wafer: {args.assets_per_wafer}; procedural families: {', '.join(procedural_families) or 'none'}")


if __name__ == "__main__":
    main()
