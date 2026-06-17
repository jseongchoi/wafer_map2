"""Synthetic Fail Bit Map generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

from wafermap.data import PATTERN_CLASSES, SyntheticSample, save_npz, write_json
from wafermap.reporting import clock_position_from_xy

PATTERN_TO_INDEX = {name: idx for idx, name in enumerate(PATTERN_CLASSES)}
StbySeed = tuple[float, float, str]


@dataclass(frozen=True)
class SyntheticConfig:
    """Config for synthetic wafer map generation."""

    seed: int = 7
    count: int = 20
    target_net_die: int = 600
    chip_width: int = 100
    chip_height: int = 50
    stby_min_chips: int = 4
    stby_max_chips: int = 10
    grade_thresholds: tuple[float, ...] = (0.04, 0.10, 0.18, 0.30, 0.45, 0.63, 0.82)
    pattern_probabilities: dict[str, float] = field(
        default_factory=lambda: {
            "scratch": 0.75,
            "ring": 0.35,
            "edge": 0.70,
            "local": 0.80,
            "random": 1.00,
            "shot_grid": 0.35,
            "stby_pattern": 0.65,
        }
    )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SyntheticConfig":
        chip_blocks = payload.get("chip_blocks", {})
        stby_chips = payload.get("stby_chips", {})
        return cls(
            seed=int(payload.get("seed", 7)),
            count=int(payload.get("count", 20)),
            target_net_die=int(payload.get("target_net_die", 600)),
            chip_width=int(chip_blocks.get("width", payload.get("chip_width", 100))),
            chip_height=int(chip_blocks.get("height", payload.get("chip_height", 50))),
            stby_min_chips=int(stby_chips.get("min", payload.get("stby_min_chips", 4))),
            stby_max_chips=int(stby_chips.get("max", payload.get("stby_max_chips", 10))),
            grade_thresholds=tuple(
                float(v) for v in payload.get("grade_thresholds", cls().grade_thresholds)
            ),
            pattern_probabilities=dict(
                payload.get("pattern_probabilities", cls().pattern_probabilities)
            ),
        )


@dataclass(frozen=True)
class Geometry:
    rows: int
    cols: int
    chip_width: int
    chip_height: int
    chip_index: NDArray[np.int32]
    wafer_mask: NDArray[np.uint8]
    chip_centers: NDArray[np.float32]

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows * self.chip_height, self.cols * self.chip_width

    @property
    def net_die(self) -> int:
        return int(self.chip_centers.shape[0])


def generate_dataset(config: SyntheticConfig) -> Iterable[SyntheticSample]:
    """Yield synthetic samples one at a time."""

    for idx in range(config.count):
        yield generate_sample(config, idx)


def generate_sample(config: SyntheticConfig, sample_index: int = 0) -> SyntheticSample:
    """Generate one synthetic wafer map sample."""

    rng = np.random.default_rng(config.seed + sample_index)
    geometry = make_geometry(config.target_net_die, config.chip_width, config.chip_height)
    height, width = geometry.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    x_norm = (xx - cx) / (width / 2.0)
    y_norm = (yy - cy) / (height / 2.0)
    radius, theta = _polar_coordinates(height, width, geometry.wafer_mask)

    n_classes = len(PATTERN_CLASSES)
    intensity = np.zeros((n_classes, height, width), dtype=np.float32)
    instances: list[dict[str, Any]] = []
    stby_seed_points: list[StbySeed] = []

    _maybe_add(
        "scratch",
        config,
        rng,
        lambda: _add_scratch(
            intensity, rng, radius, theta, width, height, instances, stby_seed_points
        ),
    )
    _maybe_add("ring", config, rng, lambda: _add_ring(intensity, rng, radius, theta, width, height, instances))
    _maybe_add("edge", config, rng, lambda: _add_edge(intensity, rng, radius, theta, width, height, instances))
    _maybe_add(
        "local",
        config,
        rng,
        lambda: _add_local(
            intensity, rng, x_norm, y_norm, cx, cy, instances, sample_index, stby_seed_points
        ),
    )
    _maybe_add("random", config, rng, lambda: _add_random(intensity, rng, geometry.wafer_mask))
    _maybe_add("shot_grid", config, rng, lambda: _add_shot_grid(intensity, rng, geometry, instances))

    background = _make_background(rng, geometry, radius)
    signal_indices = [idx for name, idx in PATTERN_TO_INDEX.items() if name != "stby_pattern"]
    latent = np.clip(background + intensity[signal_indices].sum(axis=0), 0.0, 1.5)
    latent *= geometry.wafer_mask

    stby_mask = np.zeros((height, width), dtype=np.uint8)
    if rng.random() < config.pattern_probabilities.get("stby_pattern", 0.65):
        stby_mask = _make_stby_mask(
            rng,
            geometry,
            latent,
            config.stby_min_chips,
            config.stby_max_chips,
            stby_seed_points,
        )
        intensity[PATTERN_TO_INDEX["stby_pattern"]] = stby_mask.astype(np.float32)
        if stby_mask.any():
            seeded_stby_count = _count_seeded_stby_chips(geometry, stby_mask, stby_seed_points)
            stby_chip_count = int(stby_mask.sum() // (config.chip_width * config.chip_height))
            yx = np.argwhere(stby_mask > 0).mean(axis=0)
            instances.append(
                {
                    "type": "stby_pattern",
                    "instance_id": len(instances) + 1,
                    "clock_position": clock_position_from_xy(yx[1], yx[0], cx, cy),
                    "severity": 1.0,
                    "parameters": {
                        "mode": (
                            "origin_coupled_or_random_chip_missing"
                            if seeded_stby_count > 0
                            else "latent_weighted_random_chip_missing"
                        ),
                        "origin_seed_count": len(stby_seed_points),
                        "seeded_stby_chip_count": seeded_stby_count,
                        "latent_weighted_stby_chip_count_est": max(0, stby_chip_count - seeded_stby_count),
                        "stby_chip_count_est": stby_chip_count,
                    },
                }
            )

    severity = quantize_severity(latent, config.grade_thresholds)
    severity[geometry.wafer_mask == 0] = 0
    severity[stby_mask > 0] = 0
    valid_test_mask = ((geometry.wafer_mask > 0) & (stby_mask == 0)).astype(np.uint8)
    pattern_masks = (intensity > 0.08).astype(np.uint8)
    pattern_masks[PATTERN_TO_INDEX["stby_pattern"]] = stby_mask

    sample_id = f"synth_{sample_index:06d}"
    metadata = {
        "sample_id": sample_id,
        "target_net_die": config.target_net_die,
        "actual_net_die": geometry.net_die,
        "chip_blocks": {"width": config.chip_width, "height": config.chip_height},
        "stby_chips": {"min": config.stby_min_chips, "max": config.stby_max_chips},
        "grid": {"rows": geometry.rows, "cols": geometry.cols},
        "image_shape": {"height": height, "width": width},
        "pattern_classes": list(PATTERN_CLASSES),
        "patterns": instances,
        "grade_thresholds": list(config.grade_thresholds),
    }
    return SyntheticSample(
        sample_id=sample_id,
        severity=severity,
        wafer_mask=geometry.wafer_mask,
        valid_test_mask=valid_test_mask,
        stby_mask=stby_mask,
        pattern_masks=pattern_masks,
        pattern_intensity=intensity.astype(np.float32),
        chip_index=geometry.chip_index,
        metadata=metadata,
    )


def make_geometry(target_net_die: int, chip_width: int, chip_height: int) -> Geometry:
    """Create a rectangular chip grid clipped by an elliptical wafer boundary."""

    best: tuple[float, int, int, int] | None = None
    for cols in range(8, 80):
        for rows in range(8, 100):
            count = _count_ellipse_chips(rows, cols)
            aspect = abs((cols * chip_width) / (rows * chip_height) - 1.0)
            score = abs(count - target_net_die) + aspect * 12.0
            if best is None or score < best[0]:
                best = (score, rows, cols, count)

    if best is None:
        raise RuntimeError("Unable to create wafer geometry")
    _, rows, cols, _ = best

    chip_grid = np.full((rows, cols), -1, dtype=np.int32)
    centers: list[tuple[float, float, int, int]] = []
    next_idx = 0
    for r in range(rows):
        for c in range(cols):
            x = ((c + 0.5) / cols - 0.5) * 2.0
            y = ((r + 0.5) / rows - 0.5) * 2.0
            if x * x + y * y <= 1.0:
                chip_grid[r, c] = next_idx
                centers.append((x, y, r, c))
                next_idx += 1

    chip_index = np.repeat(np.repeat(chip_grid, chip_height, axis=0), chip_width, axis=1)
    wafer_mask = (chip_index >= 0).astype(np.uint8)
    chip_centers = np.array(centers, dtype=np.float32)
    return Geometry(rows, cols, chip_width, chip_height, chip_index, wafer_mask, chip_centers)


def quantize_severity(
    latent: NDArray[np.float32], thresholds: tuple[float, ...]
) -> NDArray[np.uint8]:
    """Quantize continuous latent severity into grades 0..7."""

    grade = np.zeros(latent.shape, dtype=np.uint8)
    for idx, threshold in enumerate(thresholds, start=1):
        grade[latent >= threshold] = idx
    return grade


def save_sample(sample: SyntheticSample, sample_dir: str | Path) -> None:
    """Persist one synthetic sample as arrays.npz and metadata.json."""

    sample_dir = Path(sample_dir)
    save_npz(sample_dir / "arrays.npz", sample)
    write_json(sample_dir / "metadata.json", sample.metadata)


def _count_ellipse_chips(rows: int, cols: int) -> int:
    count = 0
    for r in range(rows):
        for c in range(cols):
            x = ((c + 0.5) / cols - 0.5) * 2.0
            y = ((r + 0.5) / rows - 0.5) * 2.0
            if x * x + y * y <= 1.0:
                count += 1
    return count


def _maybe_add(name: str, config: SyntheticConfig, rng: np.random.Generator, fn: Any) -> None:
    if rng.random() < config.pattern_probabilities.get(name, 0.0):
        fn()


def _angle_delta(theta: NDArray[np.float32], center: float) -> NDArray[np.float32]:
    return np.angle(np.exp(1j * (theta - center))).astype(np.float32)


def _polar_coordinates(
    height: int,
    width: int,
    wafer_mask: NDArray[np.uint8],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return center-distance polar coordinates normalized to the wafer edge."""

    yy, xx = np.indices((height, width), dtype=np.float32)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    dx = xx - cx
    dy = yy - cy
    distance = np.sqrt(dx * dx + dy * dy)
    max_distance = float(distance[wafer_mask > 0].max()) if (wafer_mask > 0).any() else 1.0
    radius = np.clip(distance / max(max_distance, 1.0), 0.0, 1.0).astype(np.float32)
    theta = np.arctan2(dx, -dy).astype(np.float32)
    return radius, theta


def _add_scratch(
    intensity: NDArray[np.float32],
    rng: np.random.Generator,
    radius: NDArray[np.float32],
    theta: NDArray[np.float32],
    width: int,
    height: int,
    instances: list[dict[str, Any]],
    stby_seed_points: list[StbySeed],
) -> None:
    idx = PATTERN_TO_INDEX["scratch"]
    mode = str(rng.choice(["spin_arc", "radial"], p=[0.62, 0.38]))

    if mode == "spin_arc":
        center_angle = rng.uniform(0, 2 * np.pi)
        angular_span = rng.uniform(0.55, 2.6)
        radial_center = rng.uniform(0.28, 0.92)
        radial_width = rng.uniform(0.0045, 0.014)
        angular = np.abs(_angle_delta(theta, center_angle))
        radial = np.exp(-0.5 * ((radius - radial_center) / radial_width) ** 2)
        envelope = np.exp(-0.5 * (angular / (angular_span / 2.8)) ** 2)
        sparse = rng.random(radius.shape) < rng.uniform(0.12, 0.34)
        field = radial * envelope * sparse
        field *= rng.uniform(0.30, 0.70)
        report_x = float(np.sin(center_angle) * radial_center)
        report_y = float(-np.cos(center_angle) * radial_center)
        stby_seed_points.append((report_x, report_y, "scratch_arc_contact"))
        params = {
            "mode": mode,
            "radius": round(float(radial_center), 4),
            "angular_span": round(float(angular_span), 4),
        }
    else:
        center_angle = rng.uniform(0, 2 * np.pi)
        radial_start = rng.uniform(0.02, 0.22)
        radial_end = rng.uniform(0.58, 0.98)
        scratch_width = rng.uniform(0.006, 0.018)
        cross = np.abs(_angle_delta(theta, center_angle)) * np.maximum(radius, 0.08)
        along = radius
        segment = (along >= radial_start) & (along <= radial_end)
        taper = np.clip((along - radial_start) / 0.12, 0, 1) * np.clip((radial_end - along) / 0.12, 0, 1)
        sparse = rng.random(radius.shape) < rng.uniform(0.10, 0.30)
        field = np.exp(-0.5 * (cross / scratch_width) ** 2) * segment * taper * sparse
        field *= rng.uniform(0.28, 0.68)
        report_radius = (radial_start + radial_end) / 2.0
        report_x = float(np.sin(center_angle) * report_radius)
        report_y = float(-np.cos(center_angle) * report_radius)
        origin_x = float(np.sin(center_angle) * radial_start)
        origin_y = float(-np.cos(center_angle) * radial_start)
        stby_seed_points.append((origin_x, origin_y, "scratch_radial_origin"))
        params = {
            "mode": mode,
            "radial_start": round(float(radial_start), 4),
            "radial_end": round(float(radial_end), 4),
        }

    intensity[idx] = np.maximum(intensity[idx], field.astype(np.float32))
    instance = _instance("scratch", len(instances) + 1, report_x, report_y, (width - 1) / 2, (height - 1) / 2, (height, width))
    instance["parameters"].update(params)
    instances.append(instance)


def _add_ring(
    intensity: NDArray[np.float32],
    rng: np.random.Generator,
    radius: NDArray[np.float32],
    theta: NDArray[np.float32],
    width: int,
    height: int,
    instances: list[dict[str, Any]],
) -> None:
    idx = PATTERN_TO_INDEX["ring"]
    mode = str(rng.choice(["donut", "partial_ring"], p=[0.45, 0.55]))
    radial_center = rng.uniform(0.35, 0.82)
    radial_width = rng.uniform(0.010, 0.030)
    radial = np.exp(-0.5 * ((radius - radial_center) / radial_width) ** 2)
    if mode == "partial_ring":
        center_angle = rng.uniform(0, 2 * np.pi)
        angular_span = rng.uniform(1.1, 4.8)
        angular = np.exp(-0.5 * (_angle_delta(theta, center_angle) / (angular_span / 2.8)) ** 2)
        field = radial * angular
        report_x = float(np.sin(center_angle) * radial_center)
        report_y = float(-np.cos(center_angle) * radial_center)
    else:
        field = radial
        report_x = 0.0
        report_y = 0.0
    sparse = rng.random(radius.shape) < rng.uniform(0.16, 0.40)
    field *= sparse
    field *= rng.uniform(0.20, 0.55)
    intensity[idx] = np.maximum(intensity[idx], field.astype(np.float32))
    instance = _instance("ring", len(instances) + 1, report_x, report_y, (width - 1) / 2, (height - 1) / 2, (height, width))
    instance["parameters"].update(
        {
            "mode": mode,
            "radius": round(float(radial_center), 4),
        }
    )
    instances.append(instance)


def _add_edge(
    intensity: NDArray[np.float32],
    rng: np.random.Generator,
    radius: NDArray[np.float32],
    theta: NDArray[np.float32],
    width: int,
    height: int,
    instances: list[dict[str, Any]],
) -> None:
    idx = PATTERN_TO_INDEX["edge"]
    center_angle = rng.uniform(0, 2 * np.pi)
    angular_span = rng.uniform(0.45, 1.35)
    radial_center = rng.uniform(0.80, 0.94)
    radial_width = rng.uniform(0.025, 0.070)
    angular = np.abs(_angle_delta(theta, center_angle))
    radial = np.exp(-0.5 * ((radius - radial_center) / radial_width) ** 2)
    sparse = rng.random(radius.shape) < rng.uniform(0.18, 0.45)
    field = radial * (angular <= angular_span / 2.0) * sparse
    field *= rng.uniform(0.25, 0.60)
    intensity[idx] = np.maximum(intensity[idx], field.astype(np.float32))
    x = np.sin(center_angle) * radial_center
    y = -np.cos(center_angle) * radial_center
    instances.append(_instance("edge", len(instances) + 1, x, y, (width - 1) / 2, (height - 1) / 2, (height, width)))


def _add_local(
    intensity: NDArray[np.float32],
    rng: np.random.Generator,
    x_norm: NDArray[np.float32],
    y_norm: NDArray[np.float32],
    cx: float,
    cy: float,
    instances: list[dict[str, Any]],
    sample_index: int,
    stby_seed_points: list[StbySeed],
) -> None:
    idx = PATTERN_TO_INDEX["local"]
    blob_modes = ("single_blob", "double_blob", "triple_triangle")
    mode = blob_modes[sample_index % len(blob_modes)]
    radius0 = np.sqrt(rng.uniform(0.02, 0.76))
    angle0 = rng.uniform(0, 2 * np.pi)
    anchor_x = float(np.sin(angle0) * radius0)
    anchor_y = float(-np.cos(angle0) * radius0)
    spacing = rng.uniform(0.045, 0.115)
    rotation = rng.uniform(0, 2 * np.pi)
    if mode == "single_blob":
        offsets = [(0.0, 0.0)]
    elif mode == "double_blob":
        offsets = [
            (-0.5 * spacing * np.cos(rotation), -0.5 * spacing * np.sin(rotation)),
            (0.5 * spacing * np.cos(rotation), 0.5 * spacing * np.sin(rotation)),
        ]
    else:
        offsets = [
            (
                spacing * np.cos(rotation + 2 * np.pi * k / 3),
                spacing * np.sin(rotation + 2 * np.pi * k / 3),
            )
            for k in range(3)
        ]

    centers: list[tuple[float, float]] = []
    for dx, dy in offsets:
        center_x = anchor_x + float(dx) + rng.normal(0, 0.008)
        center_y = anchor_y + float(dy) + rng.normal(0, 0.008)
        sigma = rng.uniform(0.010, 0.035)
        field = np.exp(-0.5 * (((x_norm - center_x) ** 2 + (y_norm - center_y) ** 2) / sigma**2))
        field *= rng.uniform(0.45, 0.95)
        intensity[idx] = np.maximum(intensity[idx], field.astype(np.float32))
        centers.append((center_x, center_y))
    mean_x = float(np.mean([p[0] for p in centers]))
    mean_y = float(np.mean([p[1] for p in centers]))
    stby_seed_points.append((mean_x, mean_y, "local_impact_origin"))
    instance = _instance("local", len(instances) + 1, mean_x, mean_y, cx, cy, x_norm.shape)
    instance["parameters"].update(
        {
            "mode": mode,
            "blob_count": len(offsets),
            "spacing": round(float(spacing), 4),
        }
    )
    instances.append(instance)


def _add_random(
    intensity: NDArray[np.float32],
    rng: np.random.Generator,
    wafer_mask: NDArray[np.uint8],
) -> None:
    idx = PATTERN_TO_INDEX["random"]
    probability = rng.uniform(0.0035, 0.0120)
    impulses = (rng.random(wafer_mask.shape) < probability) & (wafer_mask > 0)
    random_values = rng.uniform(0.08, 0.35, size=wafer_mask.shape).astype(np.float32)
    intensity[idx] = np.maximum(intensity[idx], impulses.astype(np.float32) * random_values)


def _add_shot_grid(
    intensity: NDArray[np.float32],
    rng: np.random.Generator,
    geometry: Geometry,
    instances: list[dict[str, Any]],
) -> None:
    idx = PATTERN_TO_INDEX["shot_grid"]
    layout = tuple(rng.choice(["3x3", "3x2", "2x3"], p=[0.70, 0.15, 0.15]).split("x"))
    shot_rows, shot_cols = int(layout[0]), int(layout[1])
    mode = str(
        rng.choice(
            ["lower_left_region", "shot_edge_band", "corner_gradient"],
            p=[0.65, 0.25, 0.10],
        )
    )
    if mode == "lower_left_region":
        anchor = "lower_left"
    elif mode == "shot_edge_band":
        anchor = str(rng.choice(["left_edge", "bottom_edge"]))
    else:
        anchor = "lower_left"

    row_offset = int(rng.integers(0, shot_rows))
    col_offset = int(rng.integers(0, shot_cols))

    field = np.zeros(geometry.shape, dtype=np.float32)
    chip_y, chip_x = np.indices((geometry.chip_height, geometry.chip_width), dtype=np.float32)
    intra_x = (chip_x + 0.5) / geometry.chip_width
    intra_y = (chip_y + 0.5) / geometry.chip_height

    slot_scores: dict[tuple[int, int], float] = {}
    for slot_r in range(shot_rows):
        for slot_c in range(shot_cols):
            shot_x = (slot_c + intra_x) / shot_cols
            shot_y = (slot_r + intra_y) / shot_rows
            slot_scores[(slot_r, slot_c)] = float(_shot_relative_template(shot_x, shot_y, anchor).mean())
    max_slot_score = max(slot_scores.values())
    affected_slots = {
        slot for slot, score in slot_scores.items() if score >= max(max_slot_score * 0.38, 0.06)
    }

    touched_centers: list[tuple[float, float]] = []
    touched_shots: set[tuple[int, int]] = set()
    shot_amplitudes: dict[tuple[int, int], float] = {}
    dropout_probability = rng.uniform(0.20, 0.35)
    for center in geometry.chip_centers:
        x_norm, y_norm, r, c = center
        rr = int(r)
        cc = int(c)
        slot_r = (rr - row_offset) % shot_rows
        slot_c = (cc - col_offset) % shot_cols
        if (slot_r, slot_c) not in affected_slots:
            continue
        shot_key = ((rr - row_offset) // shot_rows, (cc - col_offset) // shot_cols)
        if shot_key not in shot_amplitudes:
            high = 0.075 if mode == "shot_edge_band" else 0.095
            low = 0.025 if mode == "shot_edge_band" else 0.035
            shot_amplitudes[shot_key] = 0.0 if rng.random() < dropout_probability else rng.uniform(low, high)
        amplitude = shot_amplitudes[shot_key]
        if amplitude <= 0:
            continue

        y0 = rr * geometry.chip_height
        y1 = y0 + geometry.chip_height
        x0 = cc * geometry.chip_width
        x1 = x0 + geometry.chip_width
        chip_mask = geometry.wafer_mask[y0:y1, x0:x1] > 0
        shot_x = (slot_c + intra_x) / shot_cols
        shot_y = (slot_r + intra_y) / shot_rows
        template = _shot_relative_template(shot_x, shot_y, anchor)
        noise = rng.uniform(0.88, 1.12, size=template.shape)
        grain_probability = np.clip(0.10 + template * 0.40, 0.0, 0.55)
        grain = rng.random(template.shape) < grain_probability
        field[y0:y1, x0:x1] = np.maximum(
            field[y0:y1, x0:x1],
            (template * noise * grain * amplitude * chip_mask).astype(np.float32),
        )
        touched_centers.append((float(x_norm), float(y_norm)))
        touched_shots.add(shot_key)

    intensity[idx] = np.maximum(intensity[idx], field)
    if not touched_centers:
        return

    mean_x = float(np.mean([p[0] for p in touched_centers]))
    mean_y = float(np.mean([p[1] for p in touched_centers]))
    instance = _instance(
        "shot_grid",
        len(instances) + 1,
        mean_x,
        mean_y,
        (geometry.shape[1] - 1) / 2,
        (geometry.shape[0] - 1) / 2,
        geometry.shape,
    )
    instance["parameters"].update(
        {
            "mode": mode,
            "anchor_region": anchor,
            "shot_rows": shot_rows,
            "shot_cols": shot_cols,
            "shot_row_offset": row_offset,
            "shot_col_offset": col_offset,
            "affected_slots": sorted([list(slot) for slot in affected_slots]),
            "selected_slots": sorted([list(slot) for slot in affected_slots]),
            "touched_chip_count": len(touched_centers),
            "touched_shot_count": len(touched_shots),
        }
    )
    instances.append(instance)


def _shot_relative_template(
    shot_x: NDArray[np.float32],
    shot_y: NDArray[np.float32],
    anchor: str,
) -> NDArray[np.float32]:
    if anchor == "lower_left":
        template = np.exp(-0.5 * (((shot_x - 0.18) / 0.16) ** 2 + ((shot_y - 0.82) / 0.18) ** 2))
    elif anchor == "upper_left":
        template = np.exp(-0.5 * (((shot_x - 0.18) / 0.16) ** 2 + ((shot_y - 0.18) / 0.18) ** 2))
    elif anchor == "lower_right":
        template = np.exp(-0.5 * (((shot_x - 0.82) / 0.16) ** 2 + ((shot_y - 0.82) / 0.18) ** 2))
    elif anchor == "left_edge":
        template = np.exp(-0.5 * ((shot_x - 0.08) / 0.11) ** 2)
        template *= 0.55 + 0.45 * np.exp(-0.5 * ((shot_y - 0.58) / 0.34) ** 2)
    elif anchor == "bottom_edge":
        template = np.exp(-0.5 * ((shot_y - 0.92) / 0.11) ** 2)
        template *= 0.55 + 0.45 * np.exp(-0.5 * ((shot_x - 0.42) / 0.36) ** 2)
    elif anchor == "top_edge":
        template = np.exp(-0.5 * ((shot_y - 0.08) / 0.11) ** 2)
        template *= 0.55 + 0.45 * np.exp(-0.5 * ((shot_x - 0.42) / 0.36) ** 2)
    else:
        template = np.exp(-0.5 * ((shot_x - 0.92) / 0.11) ** 2)
        template *= 0.55 + 0.45 * np.exp(-0.5 * ((shot_y - 0.58) / 0.34) ** 2)
    return template.astype(np.float32)


def _make_stby_mask(
    rng: np.random.Generator,
    geometry: Geometry,
    latent: NDArray[np.float32],
    min_chips: int,
    max_chips: int,
    seed_points: list[StbySeed],
) -> NDArray[np.uint8]:
    chip_scores: list[tuple[float, int, int]] = []
    for center in geometry.chip_centers:
        _, _, r, c = center
        rr = int(r)
        cc = int(c)
        y0 = rr * geometry.chip_height
        y1 = y0 + geometry.chip_height
        x0 = cc * geometry.chip_width
        x1 = x0 + geometry.chip_width
        chip_scores.append((float(latent[y0:y1, x0:x1].mean()), rr, cc))

    mask = np.zeros_like(geometry.wafer_mask, dtype=np.uint8)
    if not chip_scores:
        return mask

    scores = np.array([v[0] for v in chip_scores], dtype=np.float32)
    weights = scores - scores.min()
    weights = weights + float(np.quantile(weights, 0.75) + 1e-4)
    weights = weights / weights.sum()
    lower = max(1, min_chips)
    upper = max(lower, max_chips)
    target_count = int(rng.integers(lower, upper + 1))
    selected_chips: set[tuple[int, int]] = set()

    for x_norm, y_norm, _ in seed_points:
        chip = _chip_for_normalized_point(geometry, x_norm, y_norm)
        if chip is not None:
            selected_chips.add(chip)
            if len(selected_chips) >= target_count:
                break

    remaining = max(0, min(target_count, len(chip_scores)) - len(selected_chips))
    if remaining:
        candidate_indices = [
            idx
            for idx, (_, rr, cc) in enumerate(chip_scores)
            if (rr, cc) not in selected_chips
        ]
        candidate_weights = weights[candidate_indices]
        candidate_weights = candidate_weights / candidate_weights.sum()
        chosen = rng.choice(
            candidate_indices,
            size=min(remaining, len(candidate_indices)),
            replace=False,
            p=candidate_weights,
        )
        for chip_idx in chosen:
            _, rr, cc = chip_scores[int(chip_idx)]
            selected_chips.add((rr, cc))

    for rr, cc in selected_chips:
        _paint_chip(mask, geometry, rr, cc)
    return mask


def _chip_for_normalized_point(
    geometry: Geometry,
    x_norm: float,
    y_norm: float,
) -> tuple[int, int] | None:
    if geometry.chip_centers.size == 0:
        return None

    height, width = geometry.shape
    x = int(round((x_norm * (width / 2.0)) + (width - 1) / 2.0))
    y = int(round((y_norm * (height / 2.0)) + (height - 1) / 2.0))
    x = int(np.clip(x, 0, width - 1))
    y = int(np.clip(y, 0, height - 1))
    if geometry.chip_index[y, x] >= 0:
        return y // geometry.chip_height, x // geometry.chip_width

    centers = geometry.chip_centers
    distances = (centers[:, 0] - x_norm) ** 2 + (centers[:, 1] - y_norm) ** 2
    nearest = centers[int(np.argmin(distances))]
    return int(nearest[2]), int(nearest[3])


def _count_seeded_stby_chips(
    geometry: Geometry,
    stby_mask: NDArray[np.uint8],
    seed_points: list[StbySeed],
) -> int:
    seeded_chips: set[tuple[int, int]] = set()
    for x_norm, y_norm, _ in seed_points:
        chip = _chip_for_normalized_point(geometry, x_norm, y_norm)
        if chip is None:
            continue
        row, col = chip
        y0 = row * geometry.chip_height
        y1 = y0 + geometry.chip_height
        x0 = col * geometry.chip_width
        x1 = x0 + geometry.chip_width
        if stby_mask[y0:y1, x0:x1].any():
            seeded_chips.add(chip)
    return len(seeded_chips)


def _paint_chip(
    mask: NDArray[np.uint8],
    geometry: Geometry,
    row: int,
    col: int,
) -> None:
    y0 = row * geometry.chip_height
    y1 = y0 + geometry.chip_height
    x0 = col * geometry.chip_width
    x1 = x0 + geometry.chip_width
    mask[y0:y1, x0:x1] = 1


def _make_background(
    rng: np.random.Generator,
    geometry: Geometry,
    radius: NDArray[np.float32],
) -> NDArray[np.float32]:
    wafer_mask = geometry.wafer_mask
    base = np.zeros(wafer_mask.shape, dtype=np.float32)
    edge_lift = np.clip((radius - 0.72) / 0.28, 0, 1) ** 1.45
    edge_face_gradient = _edge_chip_face_gradient(geometry, radius)
    interior_event = rng.random(wafer_mask.shape) < rng.uniform(0.020, 0.055)
    edge_event_probability = np.clip(
        0.04
        + edge_lift * rng.uniform(0.28, 0.58)
        + edge_face_gradient * rng.uniform(0.24, 0.48),
        0,
        0.82,
    )
    edge_event = rng.random(wafer_mask.shape) < edge_event_probability
    base += interior_event * rng.exponential(scale=0.035, size=wafer_mask.shape)
    base += edge_event * rng.gamma(shape=1.4, scale=0.060, size=wafer_mask.shape) * (
        0.55 + edge_lift * 1.15 + edge_face_gradient * 1.20
    )
    speckle = rng.random(wafer_mask.shape)
    base += (speckle < 0.010) * rng.uniform(0.05, 0.18, size=wafer_mask.shape)
    base += (speckle < 0.0015) * rng.uniform(0.18, 0.45, size=wafer_mask.shape)
    base *= wafer_mask
    return base.astype(np.float32)


def _edge_chip_face_gradient(
    geometry: Geometry,
    radius: NDArray[np.float32],
) -> NDArray[np.float32]:
    gradient = np.zeros(geometry.shape, dtype=np.float32)
    for center in geometry.chip_centers:
        _, _, r, c = center
        rr = int(r)
        cc = int(c)
        y0 = rr * geometry.chip_height
        y1 = y0 + geometry.chip_height
        x0 = cc * geometry.chip_width
        x1 = x0 + geometry.chip_width
        chip_mask = geometry.wafer_mask[y0:y1, x0:x1] > 0
        chip_radius = radius[y0:y1, x0:x1]
        if not chip_mask.any():
            continue
        r_min = float(chip_radius[chip_mask].min())
        r_max = float(chip_radius[chip_mask].max())
        edge_strength = np.clip((r_max - 0.78) / 0.22, 0, 1)
        if edge_strength <= 0 or r_max <= r_min:
            continue
        local_rank = np.clip((chip_radius - r_min) / (r_max - r_min), 0, 1) ** 1.35
        gradient[y0:y1, x0:x1] = (local_rank * edge_strength * chip_mask).astype(np.float32)
    return gradient


def _instance(
    pattern_type: str,
    instance_id: int,
    x_norm: float,
    y_norm: float,
    cx: float,
    cy: float,
    shape: tuple[int, int],
) -> dict[str, Any]:
    height, width = shape
    x = (x_norm * (width / 2.0)) + cx
    y = (y_norm * (height / 2.0)) + cy
    return {
        "type": pattern_type,
        "instance_id": instance_id,
        "clock_position": clock_position_from_xy(x, y, cx, cy),
        "severity": 1.0,
        "parameters": {"x_norm": round(float(x_norm), 4), "y_norm": round(float(y_norm), 4)},
    }
