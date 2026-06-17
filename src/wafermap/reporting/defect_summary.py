"""Convert segmentation masks into structured wafer defect features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wafermap.data import PATTERN_CLASSES, SyntheticSample
from wafermap.reporting.clock_position import clock_position_from_xy

REPORT_CLASSES: tuple[str, ...] = ("scratch", "ring", "edge", "local", "shot_grid", "stby_pattern")


@dataclass(frozen=True)
class DefectRegionSummary:
    """Structured feature row for one predicted/synthetic defect mask."""

    sample_id: str
    class_name: str
    feature_key: str
    pixel_ratio: float
    centroid_clock: str
    location_label: str
    radial_zone: str
    top_clock_positions: tuple[str, ...]
    top_sector_share: float
    stby_overlap_ratio: float


def summarize_sample_defects(
    sample: SyntheticSample,
    min_pixel_ratio: float = 1e-5,
    class_names: tuple[str, ...] = REPORT_CLASSES,
) -> list[DefectRegionSummary]:
    """Summarize class masks into location-aware structured features.

    This works with synthetic oracle masks today and model-predicted masks later.
    """

    summaries = []
    wafer = sample.wafer_mask > 0
    stby = sample.stby_mask > 0
    denominator = max(int(wafer.sum()), 1)
    height, width = sample.shape
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    wafer_y, wafer_x = np.nonzero(wafer)
    max_radius = float(np.hypot(wafer_x.astype(np.float32) - cx, wafer_y.astype(np.float32) - cy).max())
    for class_name in class_names:
        class_idx = PATTERN_CLASSES.index(class_name)
        mask = (sample.pattern_masks[class_idx] > 0) & wafer
        ratio = float(mask.sum() / denominator)
        if ratio < min_pixel_ratio:
            continue
        location = _location_stats(mask, class_name, cx, cy, max_radius)
        stby_overlap = float((mask & stby).sum() / max(int(mask.sum()), 1)) if class_name != "stby_pattern" else 1.0
        location_label = str(location["location_label"])
        radial_zone = str(location["radial_zone"])
        summaries.append(
            DefectRegionSummary(
                sample_id=sample.sample_id,
                class_name=class_name,
                feature_key=_feature_key(class_name, location_label, radial_zone),
                pixel_ratio=ratio,
                centroid_clock=str(location["centroid_clock"]),
                location_label=location_label,
                radial_zone=radial_zone,
                top_clock_positions=tuple(location["top_clock_positions"]),
                top_sector_share=float(location["top_sector_share"]),
                stby_overlap_ratio=stby_overlap,
            )
        )
    return sorted(summaries, key=_summary_sort_key)


def _feature_key(class_name: str, location_label: str, radial_zone: str) -> str:
    location = location_label.replace(":", "").replace(" ", "_").lower()
    radial = radial_zone.replace("-", "_").lower()
    return f"{class_name}__{location}__{radial}"


def _summary_sort_key(summary: DefectRegionSummary) -> tuple[int, float]:
    priority = {
        "scratch": 0,
        "local": 1,
        "edge": 2,
        "ring": 3,
        "shot_grid": 4,
        "stby_pattern": 5,
    }.get(summary.class_name, 9)
    hidden_bonus = 1.0 + summary.stby_overlap_ratio if summary.class_name == "scratch" else 1.0
    return priority, -summary.pixel_ratio * hidden_bonus


def _location_stats(
    mask: NDArray[np.bool_],
    class_name: str,
    cx: float,
    cy: float,
    max_radius: float,
) -> dict[str, object]:
    ys, xs = np.nonzero(mask)
    centroid_x = float(xs.mean()) if len(xs) else cx
    centroid_y = float(ys.mean()) if len(ys) else cy
    centroid_clock = clock_position_from_xy(centroid_x, centroid_y, cx, cy)
    top_clocks, top_share = _top_clock_positions(xs, ys, cx, cy)
    radial_mean = _mean_radius(xs, ys, cx, cy, max_radius)
    radial_zone = _radial_zone(radial_mean)
    location_label = _location_label(class_name, centroid_clock, top_clocks, top_share, radial_mean)
    return {
        "centroid_clock": centroid_clock,
        "location_label": location_label,
        "radial_zone": radial_zone,
        "top_clock_positions": top_clocks,
        "top_sector_share": top_share,
    }


def _top_clock_positions(
    xs: NDArray[np.int64],
    ys: NDArray[np.int64],
    cx: float,
    cy: float,
) -> tuple[list[str], float]:
    if len(xs) == 0:
        return ["center"], 0.0
    theta = (np.arctan2(xs.astype(np.float32) - cx, -(ys.astype(np.float32) - cy)) + 2 * np.pi) % (2 * np.pi)
    sector = np.rint(theta / (2 * np.pi) * 12.0).astype(np.int32) % 12
    counts = np.bincount(sector, minlength=12)
    order = np.argsort(counts)[::-1]
    labels = [_sector_to_clock(int(idx)) for idx in order[:3] if counts[idx] > 0]
    return labels, float(counts[order[0]] / max(int(counts.sum()), 1))


def _sector_to_clock(sector: int) -> str:
    if sector == 0:
        return "12:00"
    return f"{sector:02d}:00"


def _mean_radius(
    xs: NDArray[np.int64],
    ys: NDArray[np.int64],
    cx: float,
    cy: float,
    max_distance: float,
) -> float:
    if len(xs) == 0:
        return 0.0
    distance = np.hypot(xs.astype(np.float32) - cx, ys.astype(np.float32) - cy)
    return float(distance.mean() / max(max_distance, 1.0))


def _radial_zone(radius: float) -> str:
    if radius < 0.33:
        return "center"
    if radius < 0.72:
        return "middle"
    return "edge-side"


def _location_label(
    class_name: str,
    centroid_clock: str,
    top_clocks: list[str],
    top_share: float,
    radial_mean: float,
) -> str:
    if class_name == "ring" and top_share < 0.20:
        return "ring_global"
    if class_name == "edge" and radial_mean >= 0.72 and top_share < 0.24:
        return "edge_global"
    if class_name == "shot_grid" and top_share < 0.22:
        return "wafer_global"
    if top_share < 0.16:
        return "wafer_global"
    if top_clocks:
        return top_clocks[0]
    return centroid_clock
