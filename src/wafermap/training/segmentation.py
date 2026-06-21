"""NumPy segmentation dataset helpers for CPU smoke tests."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from wafermap.data import PATTERN_CLASSES, load_sample
from wafermap.data.schema import SyntheticSample

INPUT_CHANNELS: tuple[str, ...] = ("severity", "wafer_mask", "valid_test_mask", "stby_mask")
TARGET_CHANNELS: tuple[str, ...] = PATTERN_CLASSES


@dataclass(frozen=True)
class SegmentationBatch:
    """Small batch container for segmentation smoke training."""

    sample_ids: list[str]
    inputs: NDArray[np.float32]
    targets: NDArray[np.float32]


def load_manifest_rows(path: str | Path, split: str | None = None) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if split is not None:
        rows = [row for row in rows if row.get("split") == split]
    return rows


def load_segmentation_tensor(
    sample_dir: str | Path,
    output_size: int = 128,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Load one sample as fixed-size input and multi-label target tensors.

    Inputs are CxHxW with channels severity, wafer mask, valid-test mask, stby mask.
    Targets are KxHxW class masks. Target masks use max pooling to preserve thin
    scratch/local defects during downsampling.
    """

    sample = load_sample(sample_dir)
    inputs = sample_to_input_tensor(sample, output_size=output_size)
    targets = sample_to_target_tensor(sample, output_size=output_size)
    return inputs, targets


def sample_to_input_tensor(
    sample: SyntheticSample,
    output_size: int = 128,
) -> NDArray[np.float32]:
    """Convert one in-memory sample to fixed-size inference input channels."""

    severity = sample.severity.astype(np.float32) / 7.0
    inputs = np.stack(
        [
            _resize_mean(severity, output_size),
            _resize_max(sample.wafer_mask > 0, output_size),
            _resize_max(sample.valid_test_mask > 0, output_size),
            _resize_max(sample.stby_mask > 0, output_size),
        ],
        axis=0,
    ).astype(np.float32)
    return inputs


def sample_to_target_tensor(
    sample: SyntheticSample,
    output_size: int = 128,
) -> NDArray[np.float32]:
    """Convert synthetic oracle masks to fixed-size multi-label targets."""

    targets = np.stack(
        [_resize_max(sample.pattern_masks[idx] > 0, output_size) for idx in range(len(TARGET_CHANNELS))],
        axis=0,
    ).astype(np.float32)
    return targets


def load_batch(
    rows: list[dict[str, str]],
    repo_root: str | Path,
    output_size: int,
    max_samples: int,
) -> SegmentationBatch:
    selected = rows[:max_samples]
    sample_ids: list[str] = []
    inputs = []
    targets = []
    for row in selected:
        sample_dir = _sample_dir_from_manifest_row(row, repo_root)
        x, y = load_segmentation_tensor(sample_dir, output_size=output_size)
        sample_ids.append(row["sample_id"])
        inputs.append(x)
        targets.append(y)
    if not inputs:
        raise ValueError("No samples available for segmentation batch")
    return SegmentationBatch(
        sample_ids=sample_ids,
        inputs=np.stack(inputs, axis=0).astype(np.float32),
        targets=np.stack(targets, axis=0).astype(np.float32),
    )


def _sample_dir_from_manifest_row(row: dict[str, str], repo_root: str | Path) -> Path:
    arrays_path = Path(row["arrays_path"])
    if not arrays_path.is_absolute():
        arrays_path = Path(repo_root) / arrays_path
    return arrays_path.parent


def _resize_mean(array: NDArray[np.float32], output_size: int) -> NDArray[np.float32]:
    return _resize_bins(array.astype(np.float32, copy=False), output_size, reduce="mean")


def _resize_max(array: NDArray[np.bool_] | NDArray[np.uint8], output_size: int) -> NDArray[np.float32]:
    return _resize_bins(array.astype(np.float32, copy=False), output_size, reduce="max")


def _resize_bins(
    array: NDArray[np.float32],
    output_size: int,
    reduce: str,
) -> NDArray[np.float32]:
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    y_edges = _bin_edges(array.shape[0], output_size)
    x_edges = _bin_edges(array.shape[1], output_size)
    out = np.zeros((output_size, output_size), dtype=np.float32)
    for y_idx in range(output_size):
        y0, y1 = int(y_edges[y_idx]), int(y_edges[y_idx + 1])
        for x_idx in range(output_size):
            x0, x1 = int(x_edges[x_idx]), int(x_edges[x_idx + 1])
            patch = array[y0:y1, x0:x1]
            if reduce == "max":
                out[y_idx, x_idx] = float(patch.max()) if patch.size else 0.0
            else:
                out[y_idx, x_idx] = float(patch.mean()) if patch.size else 0.0
    return out


def _bin_edges(length: int, bins: int) -> NDArray[np.int64]:
    if bins > length:
        raise ValueError(f"output_size {bins} cannot exceed input length {length}")
    edges = np.floor(np.linspace(0, length, bins + 1)).astype(np.int64)
    edges[-1] = length
    return np.maximum.accumulate(edges)
