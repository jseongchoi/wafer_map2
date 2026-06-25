"""Code-generated defect patterns for hybrid synthetic wafer composition."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from wafermap.data import PATTERN_CLASSES, SyntheticSample

PROCEDURAL_PROBABILITIES: dict[str, float] = {
    "scratch": 0.65,
    "edge": 0.70,
    "shot_grid": 0.45,
    "random": 1.00,
}


def add_procedural_patterns(
    base: SyntheticSample,
    severity: np.ndarray,
    pattern_masks: np.ndarray,
    pattern_intensity: np.ndarray,
    rng: np.random.Generator,
    families: Sequence[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    valid = (base.wafer_mask > 0) & (base.valid_test_mask > 0)
    if not valid.any():
        return records
    radius, theta = polar_coordinates(base.shape, base.wafer_mask)
    for family in families:
        probability = PROCEDURAL_PROBABILITIES.get(family, 0.0)
        if rng.random() > probability:
            continue
        if family == "scratch":
            strength, params = procedural_scratch(radius, theta, valid, rng)
        elif family == "edge":
            strength, params = procedural_edge(radius, theta, valid, rng)
        elif family == "shot_grid":
            strength, params = procedural_shot_grid(base, valid, rng)
        elif family == "random":
            strength, params = procedural_random(valid, rng)
        else:
            continue
        record = paint_procedural_family(
            family,
            strength,
            valid,
            severity,
            pattern_masks,
            pattern_intensity,
            params,
        )
        if record is not None:
            records.append(record)
    return records


def polar_coordinates(shape: tuple[int, int], wafer_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    yy, xx = np.indices(shape, dtype=np.float32)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    dx = xx - cx
    dy = yy - cy
    distance = np.sqrt(dx * dx + dy * dy)
    max_distance = float(distance[wafer_mask > 0].max()) if (wafer_mask > 0).any() else 1.0
    radius = np.clip(distance / max(max_distance, 1.0), 0.0, 1.0).astype(np.float32)
    theta = np.arctan2(dx, -dy).astype(np.float32)
    return radius, theta


def angle_delta(theta: np.ndarray, center: float) -> np.ndarray:
    return np.angle(np.exp(1j * (theta - center))).astype(np.float32)


def procedural_edge(
    radius: np.ndarray,
    theta: np.ndarray,
    valid: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    center_angle = float(rng.uniform(0.0, 2.0 * np.pi))
    angular_span = float(rng.uniform(0.45, 1.45))
    radial_center = float(rng.uniform(0.80, 0.96))
    radial_width = float(rng.uniform(0.025, 0.075))
    angular = np.abs(angle_delta(theta, center_angle))
    radial = np.exp(-0.5 * ((radius - radial_center) / radial_width) ** 2)
    sparse = rng.random(radius.shape) < float(rng.uniform(0.18, 0.46))
    strength = radial * (angular <= angular_span / 2.0) * sparse * float(rng.uniform(0.48, 0.92))
    strength *= valid
    return strength.astype(np.float32), {
        "mode": "edge_sector",
        "angle_rad": round(center_angle, 4),
        "angular_span": round(angular_span, 4),
        "radial_center": round(radial_center, 4),
        "radial_width": round(radial_width, 4),
    }


def procedural_scratch(
    radius: np.ndarray,
    theta: np.ndarray,
    valid: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    mode = str(rng.choice(["spin_arc", "radial"], p=[0.58, 0.42]))
    center_angle = float(rng.uniform(0.0, 2.0 * np.pi))
    if mode == "spin_arc":
        radial_center = float(rng.uniform(0.28, 0.92))
        radial_width = float(rng.uniform(0.006, 0.018))
        angular_span = float(rng.uniform(0.55, 2.8))
        angular = np.abs(angle_delta(theta, center_angle))
        radial = np.exp(-0.5 * ((radius - radial_center) / radial_width) ** 2)
        envelope = np.exp(-0.5 * (angular / (angular_span / 2.8)) ** 2)
        sparse = rng.random(radius.shape) < float(rng.uniform(0.16, 0.34))
        strength = radial * envelope * sparse * float(rng.uniform(0.55, 1.00))
        params = {
            "mode": mode,
            "angle_rad": round(center_angle, 4),
            "radius": round(radial_center, 4),
            "radial_width": round(radial_width, 4),
            "angular_span": round(angular_span, 4),
        }
    else:
        radial_start = float(rng.uniform(0.03, 0.22))
        radial_end = float(rng.uniform(0.58, 0.98))
        scratch_width = float(rng.uniform(0.008, 0.022))
        cross = np.abs(angle_delta(theta, center_angle)) * np.maximum(radius, 0.08)
        segment = (radius >= radial_start) & (radius <= radial_end)
        taper = np.clip((radius - radial_start) / 0.12, 0.0, 1.0) * np.clip((radial_end - radius) / 0.12, 0.0, 1.0)
        sparse = rng.random(radius.shape) < float(rng.uniform(0.14, 0.32))
        strength = np.exp(-0.5 * (cross / scratch_width) ** 2) * segment * taper * sparse * float(rng.uniform(0.55, 1.00))
        params = {
            "mode": mode,
            "angle_rad": round(center_angle, 4),
            "radial_start": round(radial_start, 4),
            "radial_end": round(radial_end, 4),
            "scratch_width": round(scratch_width, 4),
        }
    strength *= valid
    return strength.astype(np.float32), params


def procedural_random(valid: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    probability = float(rng.uniform(0.0035, 0.0120))
    impulses = (rng.random(valid.shape) < probability) & valid
    values = rng.uniform(0.18, 0.58, size=valid.shape).astype(np.float32)
    return impulses.astype(np.float32) * values, {
        "mode": "random_sparse_impulse",
        "pixel_probability": round(probability, 5),
    }


def procedural_shot_grid(
    base: SyntheticSample,
    valid: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = base.shape
    chip_blocks = base.metadata.get("chip_blocks", {}) if isinstance(base.metadata, dict) else {}
    chip_width = max(1, int(chip_blocks.get("width", max(1, width // 16))))
    chip_height = max(1, int(chip_blocks.get("height", max(1, height // 16))))
    chip_rows = max(1, height // chip_height)
    chip_cols = max(1, width // chip_width)
    shot_rows, shot_cols = tuple(int(v) for v in str(rng.choice(["3x3", "3x2", "2x3"])).split("x"))
    mode = str(rng.choice(["lower_left_region", "shot_edge_band"], p=[0.70, 0.30]))
    anchor = "lower_left" if mode == "lower_left_region" else str(rng.choice(["left_edge", "bottom_edge"]))
    row_offset = int(rng.integers(0, shot_rows))
    col_offset = int(rng.integers(0, shot_cols))
    affected_slot = (
        (shot_rows - 1, 0)
        if anchor == "lower_left"
        else (int(rng.integers(0, shot_rows)), 0 if anchor == "left_edge" else int(rng.integers(0, shot_cols)))
    )
    chip_y, chip_x = np.indices((chip_height, chip_width), dtype=np.float32)
    intra_x = (chip_x + 0.5) / chip_width
    intra_y = (chip_y + 0.5) / chip_height
    template = shot_template(intra_x, intra_y, anchor)
    strength = np.zeros(base.shape, dtype=np.float32)
    touched_chips = 0
    for row in range(chip_rows):
        for col in range(chip_cols):
            if ((row - row_offset) % shot_rows, (col - col_offset) % shot_cols) != affected_slot:
                continue
            if rng.random() < 0.26:
                continue
            y0 = row * chip_height
            x0 = col * chip_width
            y1 = min(y0 + chip_height, height)
            x1 = min(x0 + chip_width, width)
            patch_template = template[: y1 - y0, : x1 - x0]
            grain = rng.random(patch_template.shape) < np.clip(0.10 + patch_template * 0.42, 0.0, 0.58)
            amplitude = float(rng.uniform(0.32, 0.76))
            patch = patch_template * grain * amplitude * valid[y0:y1, x0:x1]
            if patch.any():
                touched_chips += 1
                strength[y0:y1, x0:x1] = np.maximum(strength[y0:y1, x0:x1], patch)
    return strength, {
        "mode": mode,
        "anchor_region": anchor,
        "shot_rows": shot_rows,
        "shot_cols": shot_cols,
        "shot_row_offset": row_offset,
        "shot_col_offset": col_offset,
        "affected_slot": list(affected_slot),
        "touched_chip_count": touched_chips,
    }


def shot_template(intra_x: np.ndarray, intra_y: np.ndarray, anchor: str) -> np.ndarray:
    if anchor == "left_edge":
        template = np.exp(-0.5 * ((intra_x - 0.08) / 0.12) ** 2)
        template *= 0.55 + 0.45 * np.exp(-0.5 * ((intra_y - 0.58) / 0.34) ** 2)
    elif anchor == "bottom_edge":
        template = np.exp(-0.5 * ((intra_y - 0.92) / 0.12) ** 2)
        template *= 0.55 + 0.45 * np.exp(-0.5 * ((intra_x - 0.42) / 0.36) ** 2)
    else:
        template = np.exp(-0.5 * (((intra_x - 0.18) / 0.16) ** 2 + ((intra_y - 0.82) / 0.18) ** 2))
    return template.astype(np.float32)


def paint_procedural_family(
    family: str,
    strength: np.ndarray,
    valid: np.ndarray,
    severity: np.ndarray,
    pattern_masks: np.ndarray,
    pattern_intensity: np.ndarray,
    params: dict[str, Any],
) -> dict[str, Any] | None:
    mask = (strength > 0.06) & valid
    if not mask.any():
        return None
    class_idx = PATTERN_CLASSES.index(family)
    grade = np.clip(np.rint(1.0 + strength * 7.0), 1, 7).astype(np.uint8)
    severity[mask] = np.maximum(severity[mask], grade[mask])
    pattern_masks[class_idx][mask] = 1
    pattern_intensity[class_idx][mask] = np.maximum(pattern_intensity[class_idx][mask], strength[mask])
    ys, xs = np.nonzero(mask)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    return {
        "family": family,
        "source": "procedural",
        "composition_rule": "max",
        "bbox_xywh": bbox,
        "mask_pixel_count": int(mask.sum()),
        "grade_min": int(grade[mask].min()),
        "grade_max": int(grade[mask].max()),
        "parameters": params,
    }
