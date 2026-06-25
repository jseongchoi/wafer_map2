"""Annotated wafer overlay helpers for defect review reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from wafermap.reporting.previews import safe_filename

OVERLAY_COLORS = {
    "edge": "#ff3b30",
    "ring": "#34c759",
    "scratch": "#af52de",
    "local": "#ffcc00",
    "shot_grid": "#00c7be",
    "stby_pattern": "#5ac8fa",
}
OVERLAY_ALPHAS = {
    "edge": 0.28,
    "ring": 0.42,
    "scratch": 0.42,
    "local": 0.58,
    "shot_grid": 0.42,
    "stby_pattern": 0.58,
}
OVERLAY_LABELS = {
    "local": "Local hotspot",
    "ring": "Ring / radial band",
    "edge": "Edge concentration",
    "scratch": "Scratch / line candidate",
    "shot_grid": "Shot-repeat candidate",
    "stby_pattern": "STBY / missing-test area",
}


def render_annotated_previews_from_samples(
    samples: list[Any],
    defect_rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    focus_sample_id: str | None,
) -> dict[str, Path]:
    image_dir = out_dir / "annotated_images"
    rows_by_sample = _rows_by_sample(defect_rows)
    image_map: dict[str, Path] = {}
    for sample in samples:
        if focus_sample_id and sample.sample_id != focus_sample_id:
            continue
        masks = defect_overlay_masks(sample, rows_by_sample.get(sample.sample_id, []))
        if not masks:
            continue
        out_path = image_dir / f"{safe_filename(sample.sample_id)}.png"
        save_annotated_preview(out_path, sample, masks)
        image_map[sample.sample_id] = out_path
    return image_map


def defect_overlay_masks(sample: Any, defect_rows: list[dict[str, Any]], min_score: float = 15.0) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for row in defect_rows:
        family = str(row.get("defect_family", ""))
        if family not in OVERLAY_COLORS or float(row.get("score", 0.0)) < min_score:
            continue
        mask = _family_overlay_mask(sample, family)
        if mask.any():
            masks[family] = mask
    return masks


def save_annotated_preview(path: Path, sample: Any, masks: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch

    path.parent.mkdir(parents=True, exist_ok=True)
    image = sample.severity.astype(np.float32).copy()
    image[sample.wafer_mask == 0] = np.nan
    cmap = plt.get_cmap("gray_r").copy()
    cmap.set_bad("#202624")

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.imshow(image, cmap=cmap, vmin=0, vmax=7, interpolation="nearest")
    handles: list[Patch] = []
    for family in OVERLAY_COLORS:
        mask = masks.get(family)
        if mask is None or not mask.any():
            continue
        color = OVERLAY_COLORS[family]
        rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        rgba[mask] = to_rgba(color, OVERLAY_ALPHAS.get(family, 0.46))
        ax.imshow(rgba, interpolation="nearest")
        ax.contour(mask.astype(np.float32), levels=[0.5], colors=[color], linewidths=1.2)
        handles.append(Patch(facecolor=color, edgecolor=color, label=OVERLAY_LABELS[family], alpha=0.72))

    ax.set_title(f"{sample.sample_id} - colored defect candidate regions")
    ax.axis("off")
    if handles:
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def apply_overlay_location_labels(
    defect_rows: list[dict[str, Any]],
    annotated_image_map: dict[str, Path],
    samples: list[Any],
) -> None:
    samples_by_id = {sample.sample_id: sample for sample in samples}
    rows_by_sample = _rows_by_sample(defect_rows)
    overlay_families: dict[str, set[str]] = {}
    for sample_id in annotated_image_map:
        sample = samples_by_id.get(sample_id)
        if sample is None:
            continue
        overlay_families[sample_id] = set(defect_overlay_masks(sample, rows_by_sample.get(sample_id, [])).keys())

    for row in defect_rows:
        sample_id = str(row.get("sample_id", ""))
        family = str(row.get("defect_family", ""))
        if family in overlay_families.get(sample_id, set()):
            row["location"] = "marked on annotated wafer image"


def _rows_by_sample(defect_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in defect_rows:
        grouped.setdefault(str(row.get("sample_id", "")), []).append(row)
    return grouped


def _family_overlay_mask(sample: Any, family: str) -> np.ndarray:
    valid = sample.valid_test_mask > 0
    fail = valid & (sample.severity > 0)
    if family == "stby_pattern":
        return sample.stby_mask > 0
    if not fail.any():
        return np.zeros(sample.shape, dtype=bool)
    radius, theta = _pixel_radius_theta(sample.shape, valid)
    if family == "edge":
        return fail & (radius >= 0.76)
    if family == "ring":
        return _ring_mask(sample, valid, fail, radius)
    if family == "local":
        return _local_mask(sample)
    if family == "scratch":
        return _angular_peak_mask(sample, valid, fail, theta)
    if family == "shot_grid":
        return _hot_chip_mask(sample)
    return np.zeros(sample.shape, dtype=bool)


def _ring_mask(sample: Any, valid: np.ndarray, fail: np.ndarray, radius: np.ndarray, bins: int = 24) -> np.ndarray:
    severity = sample.severity.astype(np.float32) / 7.0
    means = np.zeros(bins, dtype=np.float32)
    for idx in range(bins):
        low = idx / bins
        high = (idx + 1) / bins
        band = valid & (radius >= low) & (radius < high)
        means[idx] = float(severity[band].mean()) if band.any() else 0.0
    best = int(means.argmax())
    low = max(0.0, (best - 0.5) / bins)
    high = min(1.0, (best + 1.5) / bins)
    return fail & (radius >= low) & (radius < high)


def _local_mask(sample: Any) -> np.ndarray:
    chip_values, chip_valid = _chip_mean_grid(sample)
    if chip_values is None or chip_valid is None or not chip_valid.any():
        return _largest_component(_high_severity_mask(sample))
    values = chip_values[chip_valid]
    threshold = max(float(np.quantile(values, 0.98)), float(np.median(values) + 2.0 * values.std()))
    hot_chips = chip_valid & (chip_values >= threshold) & (chip_values > 0)
    hot_chips = _largest_component(hot_chips)
    if not hot_chips.any():
        return _largest_component(_high_severity_mask(sample))
    return _expand_chip_mask(sample, hot_chips)


def _angular_peak_mask(sample: Any, valid: np.ndarray, fail: np.ndarray, theta: np.ndarray, bins: int = 24) -> np.ndarray:
    severity = sample.severity.astype(np.float32) / 7.0
    scaled = theta / (2.0 * np.pi)
    means = np.zeros(bins, dtype=np.float32)
    for idx in range(bins):
        low = idx / bins
        high = (idx + 1) / bins
        sector = valid & (scaled >= low) & (scaled < high)
        means[idx] = float(severity[sector].mean()) if sector.any() else 0.0
    best = int(means.argmax())
    low = max(0.0, (best - 0.7) / bins)
    high = min(1.0, (best + 1.7) / bins)
    sector = fail & (scaled >= low) & (scaled < high)
    focused = sector & _high_severity_mask(sample)
    return focused if focused.any() else sector


def _hot_chip_mask(sample: Any) -> np.ndarray:
    chip_values, chip_valid = _chip_mean_grid(sample)
    if chip_values is None or chip_valid is None or not chip_valid.any():
        return _high_severity_mask(sample)
    values = chip_values[chip_valid]
    threshold = max(float(np.quantile(values, 0.88)), float(values.mean() + values.std()))
    return _expand_chip_mask(sample, chip_valid & (chip_values >= threshold) & (chip_values > 0))


def _high_severity_mask(sample: Any) -> np.ndarray:
    valid = sample.valid_test_mask > 0
    severity = sample.severity.astype(np.float32) / 7.0
    values = severity[valid & (sample.severity > 0)]
    if len(values) == 0:
        return np.zeros(sample.shape, dtype=bool)
    threshold = max(float(np.quantile(values, 0.90)), float(values.mean() + values.std()))
    return valid & (severity >= threshold) & (sample.severity > 0)


def _pixel_radius_theta(shape: tuple[int, int], valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.indices(shape, dtype=np.float32)
    cx = (shape[1] - 1) / 2.0
    cy = (shape[0] - 1) / 2.0
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_distance = float(distance[valid].max()) if valid.any() else 1.0
    radius = distance / max(max_distance, 1.0)
    theta = (np.arctan2(x - cx, -(y - cy)) + 2.0 * np.pi) % (2.0 * np.pi)
    return radius.astype(np.float32), theta.astype(np.float32)


def _chip_mean_grid(sample: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        rows = int(sample.metadata["grid"]["rows"])
        cols = int(sample.metadata["grid"]["cols"])
        chip_width = int(sample.metadata["chip_blocks"]["width"])
        chip_height = int(sample.metadata["chip_blocks"]["height"])
    except (KeyError, TypeError, ValueError):
        return None, None

    severity = sample.severity.astype(np.float32) / 7.0
    valid = sample.valid_test_mask > 0
    values = np.zeros((rows, cols), dtype=np.float32)
    chip_valid = np.zeros((rows, cols), dtype=bool)
    for row in range(rows):
        y0 = row * chip_height
        y1 = min(y0 + chip_height, sample.shape[0])
        for col in range(cols):
            x0 = col * chip_width
            x1 = min(x0 + chip_width, sample.shape[1])
            block_valid = valid[y0:y1, x0:x1]
            if block_valid.any():
                values[row, col] = float(severity[y0:y1, x0:x1][block_valid].mean())
                chip_valid[row, col] = True
    return values, chip_valid


def _expand_chip_mask(sample: Any, chip_mask: np.ndarray) -> np.ndarray:
    try:
        chip_width = int(sample.metadata["chip_blocks"]["width"])
        chip_height = int(sample.metadata["chip_blocks"]["height"])
    except (KeyError, TypeError, ValueError):
        return np.zeros(sample.shape, dtype=bool)
    mask = np.zeros(sample.shape, dtype=bool)
    rows, cols = chip_mask.shape
    for row in range(rows):
        y0 = row * chip_height
        y1 = min(y0 + chip_height, sample.shape[0])
        for col in range(cols):
            if not chip_mask[row, col]:
                continue
            x0 = col * chip_width
            x1 = min(x0 + chip_width, sample.shape[1])
            mask[y0:y1, x0:x1] = sample.valid_test_mask[y0:y1, x0:x1] > 0
    return mask & (sample.severity > 0)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    visited = np.zeros(mask.shape, dtype=bool)
    best: list[tuple[int, int]] = []
    height, width = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        component: list[tuple[int, int]] = []
        while stack:
            cy, cx = stack.pop()
            component.append((cy, cx))
            for ny in range(max(0, cy - 1), min(height, cy + 2)):
                for nx in range(max(0, cx - 1), min(width, cx + 2)):
                    if not visited[ny, nx] and mask[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        if len(component) > len(best):
            best = component
    out = np.zeros(mask.shape, dtype=bool)
    for y, x in best:
        out[y, x] = True
    return out
