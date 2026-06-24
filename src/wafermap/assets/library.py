"""Shared helpers for FBM pattern asset libraries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
REQUIRED_ASSET_FILES: tuple[str, ...] = ("grade.png", "mask.png", "preview.png", "metadata.json")


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
