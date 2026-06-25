"""Shared helpers for FBM pattern asset libraries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

TARGET_FAMILIES: tuple[str, ...] = ("local", "scratch", "ring", "edge", "shot_grid", "random")
PRIMARY_ASSET_FAMILIES: tuple[str, ...] = ("local", "scratch", "ring")
OPTIONAL_ASSET_FAMILIES: tuple[str, ...] = ("edge", "shot_grid")
PROCEDURAL_FAMILIES: tuple[str, ...] = ("scratch", "edge", "shot_grid", "random")
DEFAULT_PROCEDURAL_FAMILIES: tuple[str, ...] = PROCEDURAL_FAMILIES
FAMILY_DATA_SOURCES: dict[str, str] = {
    "local": "human_asset_primary",
    "scratch": "human_asset_primary_procedural_fallback",
    "ring": "human_asset_primary",
    "edge": "procedural_primary_asset_optional",
    "shot_grid": "procedural_primary_asset_optional",
    "random": "procedural_only",
}
FAMILY_LABELS: dict[str, str] = {
    "local": "blob/local",
    "scratch": "scratch",
    "ring": "ring",
    "edge": "edge",
    "shot_grid": "shot_grid",
    "random": "random",
}
FAMILY_COLORS: dict[str, str] = {
    "local": "#ffcc00",
    "scratch": "#af52de",
    "ring": "#34c759",
    "edge": "#ff3b30",
    "shot_grid": "#00c7be",
    "random": "#8e8e93",
}
GRADE_DISPLAY_COLORS: dict[int, tuple[int, int, int]] = {
    0: (247, 248, 248),
    1: (88, 166, 255),
    2: (35, 203, 167),
    3: (118, 219, 87),
    4: (234, 221, 72),
    5: (245, 151, 54),
    6: (220, 72, 66),
    7: (122, 24, 28),
}
REQUIRED_ASSET_FILES: tuple[str, ...] = ("grade.png", "mask.png", "preview.png", "metadata.json")


def preview_rgb(sample: Any) -> np.ndarray:
    image = np.zeros((*sample.shape, 3), dtype=np.uint8)
    image[:] = (28, 36, 33)
    wafer = sample.wafer_mask > 0
    for grade, color in GRADE_DISPLAY_COLORS.items():
        image[wafer & (sample.severity == grade)] = color
    image[sample.stby_mask > 0] = (166, 216, 240)
    return image


def rle_to_mask(runs: list[list[int]], shape: tuple[int, int]) -> np.ndarray:
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for run in runs:
        if len(run) != 2:
            raise ValueError(f"invalid RLE run: {run}")
        start, length = int(run[0]), int(run[1])
        if start < 0 or length < 0 or start + length > flat.size:
            raise ValueError(f"RLE run outside mask bounds: {run}")
        flat[start : start + length] = 1
    return flat.reshape(shape).astype(bool)


def mask_to_rle(mask: np.ndarray) -> list[list[int]]:
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    runs: list[list[int]] = []
    start = -1
    for idx, value in enumerate(flat):
        if value and start < 0:
            start = idx
        if (not value or idx == flat.size - 1) and start >= 0:
            end = idx + 1 if value and idx == flat.size - 1 else idx
            runs.append([int(start), int(end - start)])
            start = -1
    return runs


def load_prediction_masks(path: Path | None, sample_id: str, shape: tuple[int, int]) -> dict[str, list[list[int]]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("schema_version") != "fbm_prediction_masks/v1":
        raise ValueError(f"unsupported prediction schema: {payload.get('schema_version')}")
    samples = payload.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("prediction JSON samples must be a list")
    for sample in samples:
        if str(sample.get("sample_id", "")) != sample_id:
            continue
        masks = sample.get("masks", {})
        if not isinstance(masks, dict):
            raise ValueError("prediction sample masks must be an object")
        out: dict[str, list[list[int]]] = {}
        for family in TARGET_FAMILIES:
            runs = masks.get(family, [])
            mask = rle_to_mask(runs, shape)
            out[family] = mask_to_rle(mask)
        return out
    return {}


def save_pattern_assets(
    *,
    sample: Any,
    masks_by_family: dict[str, np.ndarray],
    assets_root: Path,
    margin_ratio: float,
    source_manifest: Path | None = None,
    split_components: bool = False,
) -> list[dict[str, Any]]:
    saved: list[dict[str, Any]] = []
    for family in TARGET_FAMILIES:
        mask = masks_by_family.get(family)
        if mask is None:
            continue
        mask = mask & (sample.wafer_mask > 0)
        components = connected_components(mask) if split_components else ([mask] if mask.any() else [])
        for component in components:
            bbox = bbox_with_margin(component, sample.shape, margin_ratio)
            asset_id = next_asset_id(assets_root / family, sample.sample_id, family)
            asset_dir = assets_root / family / asset_id
            write_asset(
                asset_dir=asset_dir,
                sample=sample,
                family=family,
                component=component,
                bbox=bbox,
                margin_ratio=margin_ratio,
                source_manifest=source_manifest,
            )
            saved.append({"family": family, "asset_id": asset_id, "path": str(asset_dir), "bbox": bbox})
    return saved


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[np.ndarray] = []
    height, width = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cy, cx = stack.pop()
            coords.append((cy, cx))
            for ny in range(max(0, cy - 1), min(height, cy + 2)):
                for nx in range(max(0, cx - 1), min(width, cx + 2)):
                    if not visited[ny, nx] and mask[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        component = np.zeros(mask.shape, dtype=bool)
        for cy, cx in coords:
            component[cy, cx] = True
        components.append(component)
    return components


def bbox_with_margin(component: np.ndarray, shape: tuple[int, int], margin_ratio: float) -> list[int]:
    ys, xs = np.nonzero(component)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    margin_y = max(1, int(round((y1 - y0) * margin_ratio)))
    margin_x = max(1, int(round((x1 - x0) * margin_ratio)))
    y0 = max(0, y0 - margin_y)
    x0 = max(0, x0 - margin_x)
    y1 = min(shape[0], y1 + margin_y)
    x1 = min(shape[1], x1 + margin_x)
    return [x0, y0, x1 - x0, y1 - y0]


def next_asset_id(family_root: Path, sample_id: str, family: str) -> str:
    family_root.mkdir(parents=True, exist_ok=True)
    prefix = f"{safe_name(sample_id)}_{family}_"
    existing = [path.name for path in family_root.iterdir() if path.is_dir() and path.name.startswith(prefix)]
    numbers = []
    for name in existing:
        try:
            numbers.append(int(name.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            pass
    return f"{prefix}{(max(numbers) + 1 if numbers else 1):04d}"


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value) or "sample"


def write_asset(
    *,
    asset_dir: Path,
    sample: Any,
    family: str,
    component: np.ndarray,
    bbox: list[int],
    margin_ratio: float,
    source_manifest: Path | None,
) -> None:
    x, y, width, height = bbox
    crop = np.s_[y : y + height, x : x + width]
    asset_dir.mkdir(parents=True, exist_ok=True)
    local_mask = component[crop]
    grade_patch = sample.severity[crop].astype(np.uint8)
    mask_png = local_mask.astype(np.uint8) * 255
    preview = preview_rgb(sample)[crop]
    Image.fromarray(grade_patch, mode="L").save(asset_dir / "grade.png")
    Image.fromarray(mask_png, mode="L").save(asset_dir / "mask.png")
    Image.fromarray(preview, mode="RGB").save(asset_dir / "preview.png")
    masked_values = grade_patch[local_mask]
    metadata = {
        "schema_version": "fbm_pattern_asset/v1",
        "family": family,
        "family_label": FAMILY_LABELS[family],
        "source_sample_id": sample.sample_id,
        "bbox_xywh": bbox,
        "source_image_shape": {"height": int(sample.shape[0]), "width": int(sample.shape[1])},
        "margin_ratio": float(margin_ratio),
        "composition_rule": "max",
        "mask_pixel_count": int(local_mask.sum()),
        "grade_min": int(masked_values.min()) if len(masked_values) else 0,
        "grade_max": int(masked_values.max()) if len(masked_values) else 0,
        "multi_label": True,
        "stby_target_excluded": True,
    }
    if source_manifest is not None:
        metadata["source_manifest_name"] = source_manifest.name
    (asset_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def scan_pattern_assets(assets_root: str | Path) -> list[dict[str, Any]]:
    root = Path(assets_root)
    assets: list[dict[str, Any]] = []
    for family in TARGET_FAMILIES:
        family_root = root / family
        if not family_root.exists():
            continue
        for asset_dir in sorted(path for path in family_root.iterdir() if path.is_dir()):
            files = {name: asset_dir / name for name in REQUIRED_ASSET_FILES}
            missing = [name for name, path in files.items() if not path.exists()]
            metadata = _read_metadata(files["metadata.json"]) if not missing else {}
            assets.append(
                {
                    "family": family,
                    "family_label": FAMILY_LABELS[family],
                    "asset_id": asset_dir.name,
                    "asset_dir": str(asset_dir),
                    "relative_path": f"{family}/{asset_dir.name}",
                    "grade_path": str(files["grade.png"]),
                    "mask_path": str(files["mask.png"]),
                    "preview_path": str(files["preview.png"]),
                    "metadata_path": str(files["metadata.json"]),
                    "missing_files": missing,
                    "valid": not missing and metadata.get("schema_version") == "fbm_pattern_asset/v1",
                    "mask_pixel_count": int(metadata.get("mask_pixel_count", 0) or 0),
                    "bbox_xywh": metadata.get("bbox_xywh", []),
                    "grade_min": metadata.get("grade_min", ""),
                    "grade_max": metadata.get("grade_max", ""),
                    "source_sample_id": metadata.get("source_sample_id", ""),
                    "metadata": metadata,
                }
            )
    return assets


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
