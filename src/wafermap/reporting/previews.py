"""Wafer preview rendering helpers for reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wafermap.viz import save_preview


def safe_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return safe or "wafer"


def render_wafer_previews_from_samples(samples: list[Any], out_dir: Path) -> dict[str, Path]:
    image_dir = out_dir / "wafer_images"
    image_map: dict[str, Path] = {}
    for sample in samples:
        out_path = image_dir / f"{safe_filename(sample.sample_id)}.png"
        save_preview(out_path, sample.severity, sample.wafer_mask, sample.stby_mask, sample.sample_id)
        image_map[sample.sample_id] = out_path
    return image_map
