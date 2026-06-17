"""File I/O helpers for synthetic wafer map samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from wafermap.data.schema import SyntheticSample


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_npz(path: str | Path, sample: SyntheticSample) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **sample.arrays())


def load_sample(sample_dir: str | Path) -> SyntheticSample:
    sample_dir = Path(sample_dir)
    arrays = np.load(sample_dir / "arrays.npz")
    metadata = load_metadata(sample_dir / "metadata.json")
    return SyntheticSample(
        sample_id=str(metadata["sample_id"]),
        severity=arrays["severity"],
        wafer_mask=arrays["wafer_mask"],
        valid_test_mask=arrays["valid_test_mask"],
        stby_mask=arrays["stby_mask"],
        pattern_masks=arrays["pattern_masks"],
        pattern_intensity=arrays["pattern_intensity"],
        chip_index=arrays["chip_index"],
        metadata=metadata,
    )


def load_metadata(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
