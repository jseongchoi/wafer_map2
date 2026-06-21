"""Wafer-level feature extraction for EDA and similarity search."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from wafermap.data import PATTERN_CLASSES, SyntheticSample


def extract_feature_vector(sample: SyntheticSample, radial_bins: int = 5, angular_bins: int = 12) -> dict[str, float]:
    """Extract observable wafer-level features usable on real unlabeled FBM."""

    return extract_observable_feature_vector(sample, radial_bins=radial_bins, angular_bins=angular_bins)


def extract_observable_feature_vector(
    sample: SyntheticSample,
    radial_bins: int = 5,
    angular_bins: int = 12,
) -> dict[str, float]:
    """Extract a compact wafer-level feature vector without synthetic oracle fields."""

    wafer = sample.wafer_mask > 0
    valid = sample.valid_test_mask > 0
    severity = sample.severity.astype(np.float32)
    denominator = max(int(wafer.sum()), 1)
    valid_denominator = max(int(valid.sum()), 1)
    features: dict[str, float] = {
        "total_fail_density": float(((severity > 0) & valid).sum() / valid_denominator),
        "grade_weighted_severity": float(severity[valid].sum() / (valid_denominator * 7.0)),
        "stby_ratio": float((sample.stby_mask > 0).sum() / denominator),
    }

    normalized_severity = severity / 7.0
    radius, theta = _pixel_radius_theta(sample.shape, wafer)
    for idx, value in enumerate(_binned_mean(normalized_severity, valid, radius, radial_bins)):
        features[f"radial_zone_{idx:02d}_severity"] = float(value)
    for idx, value in enumerate(_binned_mean(normalized_severity, valid, theta, angular_bins)):
        features[f"angular_sector_{idx:02d}_severity"] = float(value)

    density_profile = _binned_mean((severity > 0).astype(np.float32), valid, radius, 5)
    features["edge_density"] = float(np.mean(density_profile[-1:]))
    features["center_density"] = float(density_profile[0])
    features["edge_minus_center_density"] = float(features["edge_density"] - features["center_density"])
    features["edge_chip_outer_minus_inner_density"] = edge_chip_outer_minus_inner_density(sample, radius)
    chip_values, chip_valid = _chip_mean_grid(sample)
    features.update(polar_spatial_features(sample, chip_values, chip_valid))
    features.update(radial_angular_morphology_features(normalized_severity, valid, radius, theta))
    features.update(local_hotspot_features(sample, chip_values, chip_valid))
    features.update(component_morphology_features(sample, chip_values, chip_valid))
    features.update(edge_localized_features(sample, chip_values, chip_valid))
    features.update(shot_relative_features(sample, chip_values, chip_valid))
    return features


def extract_validation_feature_vector(sample: SyntheticSample) -> dict[str, float]:
    """Extract synthetic-only oracle fields for generator validation reports."""

    denominator = max(int((sample.wafer_mask > 0).sum()), 1)
    return {
        f"{name}_mask_ratio": float((sample.pattern_masks[idx] > 0).sum() / denominator)
        for idx, name in enumerate(PATTERN_CLASSES)
    }


def radial_profile(values: NDArray[np.float32], mask: NDArray[np.bool_], bins: int) -> NDArray[np.float32]:
    radius, _ = _pixel_radius_theta(values.shape, mask)
    return _binned_mean(values, mask, radius, bins)


def angular_profile(values: NDArray[np.float32], mask: NDArray[np.bool_], bins: int) -> NDArray[np.float32]:
    _, theta = _pixel_radius_theta(values.shape, mask)
    return _binned_mean(values, mask, theta, bins)


def polar_spatial_features(
    sample: SyntheticSample,
    chip_values: NDArray[np.float32],
    chip_valid: NDArray[np.bool_],
    radial_bins: int = 3,
    angular_bins: int = 12,
) -> dict[str, float]:
    """Capture location-aware observed signal on a coarse chip-level polar grid."""

    wafer_grid, stby_grid_values = _chip_wafer_stby_grid(sample)
    radius, theta = _chip_center_polar_grid(sample, wafer_grid)
    severity_grid = _binned_2d_mean(chip_values, chip_valid, radius, theta / (2 * np.pi), radial_bins, angular_bins)
    fail_grid = _binned_2d_mean(
        (chip_values > 0).astype(np.float32),
        chip_valid,
        radius,
        theta / (2 * np.pi),
        radial_bins,
        angular_bins,
    )
    stby_grid = _binned_2d_mean(stby_grid_values, wafer_grid, radius, theta / (2 * np.pi), radial_bins, angular_bins)
    features: dict[str, float] = {}
    for radial_idx in range(radial_bins):
        for angular_idx in range(angular_bins):
            prefix = f"polar_r{radial_idx:02d}_a{angular_idx:02d}"
            features[f"{prefix}_severity"] = float(severity_grid[radial_idx, angular_idx])
            features[f"{prefix}_fail_density"] = float(fail_grid[radial_idx, angular_idx])
            features[f"stby_{prefix}_ratio"] = float(stby_grid[radial_idx, angular_idx])
    return features


def _pixel_radius_theta(
    shape: tuple[int, int],
    mask: NDArray[np.bool_],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    cx = (shape[1] - 1) / 2.0
    cy = (shape[0] - 1) / 2.0
    yy, xx = np.indices(shape, dtype=np.float32)
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_distance = float(distance[mask].max()) if mask.any() else 1.0
    radius = np.clip(distance / max(max_distance, 1.0), 0.0, 0.999)
    theta = (np.arctan2(xx - cx, -(yy - cy)) + 2 * np.pi) % (2 * np.pi)
    return radius.astype(np.float32), (theta / (2 * np.pi)).astype(np.float32)


def edge_chip_outer_minus_inner_density(
    sample: SyntheticSample,
    radius: NDArray[np.float32] | None = None,
) -> float:
    """Compare fail density on edge-facing and center-facing sides of edge chips."""

    severity = sample.severity.astype(np.float32)
    chip_width = int(sample.metadata["chip_blocks"]["width"])
    chip_height = int(sample.metadata["chip_blocks"]["height"])
    rows = int(sample.metadata["grid"]["rows"])
    cols = int(sample.metadata["grid"]["cols"])
    wafer = sample.wafer_mask > 0
    if radius is None:
        radius, _ = _pixel_radius_theta(sample.shape, wafer)
    valid = sample.valid_test_mask > 0
    inner_face = np.zeros(sample.shape, dtype=bool)
    outer_face = np.zeros(sample.shape, dtype=bool)

    for row in range(rows):
        y0 = row * chip_height
        y1 = y0 + chip_height
        for col in range(cols):
            x0 = col * chip_width
            x1 = x0 + chip_width
            chip_valid = valid[y0:y1, x0:x1]
            if not chip_valid.any():
                continue
            chip_radius = radius[y0:y1, x0:x1]
            valid_radius = chip_radius[chip_valid]
            if valid_radius.max() < 0.78:
                continue
            radius_min = valid_radius.min()
            radius_max = valid_radius.max()
            if radius_max <= radius_min:
                continue
            local_rank = (chip_radius - radius_min) / (radius_max - radius_min)
            inner_face[y0:y1, x0:x1] |= chip_valid & (local_rank < 0.34)
            outer_face[y0:y1, x0:x1] |= chip_valid & (local_rank > 0.66)

    inner_density = float(((severity > 0) & inner_face).sum() / max(int(inner_face.sum()), 1))
    outer_density = float(((severity > 0) & outer_face).sum() / max(int(outer_face.sum()), 1))
    return outer_density - inner_density


def radial_angular_morphology_features(
    normalized_severity: NDArray[np.float32],
    valid: NDArray[np.bool_],
    radius: NDArray[np.float32] | None = None,
    theta: NDArray[np.float32] | None = None,
) -> dict[str, float]:
    """Capture ring-like radial peaks and scratch-like angular concentration."""

    if radius is None or theta is None:
        radius, theta = _pixel_radius_theta(normalized_severity.shape, valid)
    radial = _binned_mean(normalized_severity, valid, radius, 24)
    angular = _binned_mean(normalized_severity, valid, theta, 24)
    return {
        "ring_radial_peak_contrast": _profile_peak_contrast(radial),
        "ring_radial_peak_width_ratio": _profile_peak_width_ratio(radial),
        "scratch_angular_peak_contrast": _profile_peak_contrast(angular),
        "scratch_angular_peak_width_ratio": _profile_peak_width_ratio(angular),
    }


def local_hotspot_features(
    sample: SyntheticSample,
    chip_values: NDArray[np.float32] | None = None,
    chip_valid: NDArray[np.bool_] | None = None,
) -> dict[str, float]:
    """Summarize compact high-severity regions on the chip grid."""

    if chip_values is None or chip_valid is None:
        chip_values, chip_valid = _chip_mean_grid(sample)
    values = chip_values[chip_valid]
    if len(values) == 0:
        return {
            "local_hotspot_peak_contrast": 0.0,
            "local_hotspot_top3_mean_contrast": 0.0,
            "local_hotspot_top3_spread": 0.0,
            "local_hotspot_count_ratio": 0.0,
        }

    median = float(np.median(values))
    std = float(values.std())
    sorted_values = np.sort(values)[::-1]
    top_n = min(3, len(sorted_values))
    threshold = median + 2.0 * std
    top_positions = np.argwhere(chip_valid)
    top_order = np.argsort(chip_values[chip_valid])[::-1][:top_n]
    selected_positions = top_positions[top_order]
    spread = _mean_normalized_pair_distance(selected_positions, chip_values.shape)
    return {
        "local_hotspot_peak_contrast": float(sorted_values[0] - median),
        "local_hotspot_top3_mean_contrast": float(sorted_values[:top_n].mean() - median),
        "local_hotspot_top3_spread": spread,
        "local_hotspot_count_ratio": float((values > threshold).sum() / max(len(values), 1)),
    }


def component_morphology_features(
    sample: SyntheticSample,
    chip_values: NDArray[np.float32] | None = None,
    chip_valid: NDArray[np.bool_] | None = None,
) -> dict[str, float]:
    """Capture connected local blobs and elongated scratch-like structures on the chip grid."""

    if chip_values is None or chip_valid is None:
        chip_values, chip_valid = _chip_mean_grid(sample)
    values = chip_values[chip_valid]
    empty = {
        "morph_hot_chip_ratio": 0.0,
        "morph_component_count_ratio": 0.0,
        "local_component_largest_ratio": 0.0,
        "local_component_compactness": 0.0,
        "local_component_top3_spread": 0.0,
        "local_component_triangle_score": 0.0,
        "scratch_component_elongation": 0.0,
        "scratch_component_linear_score": 0.0,
        "scratch_component_radial_span": 0.0,
        "scratch_component_angular_span": 0.0,
    }
    if len(values) == 0:
        return empty

    median = float(np.median(values))
    std = float(values.std())
    threshold = max(median + 0.5 * std, float(np.quantile(values, 0.84)))
    hot = chip_valid & (chip_values >= threshold)
    components = _connected_components(hot)
    valid_count = max(int(chip_valid.sum()), 1)
    hot_count = int(hot.sum())
    if not components:
        return {**empty, "morph_hot_chip_ratio": float(hot_count / valid_count)}

    radius, theta = _chip_center_polar_grid(sample, chip_valid)
    stats = [_component_stats(component, chip_values, radius, theta) for component in components]
    sorted_stats = sorted(stats, key=lambda item: item["mean_value"], reverse=True)
    top_centroids = np.array([item["centroid"] for item in sorted_stats[:3]], dtype=np.float32)
    linear_item = max(stats, key=lambda item: item["linear_score"])
    compact_values = [item["compactness"] for item in sorted_stats[: min(3, len(sorted_stats))]]
    return {
        "morph_hot_chip_ratio": float(hot_count / valid_count),
        "morph_component_count_ratio": float(len(components) / valid_count),
        "local_component_largest_ratio": float(max(item["size"] for item in stats) / valid_count),
        "local_component_compactness": float(np.mean(compact_values)) if compact_values else 0.0,
        "local_component_top3_spread": _mean_normalized_pair_distance(top_centroids.astype(np.int64), chip_values.shape),
        "local_component_triangle_score": _triangle_score(top_centroids, chip_values.shape),
        "scratch_component_elongation": float(linear_item["elongation"]),
        "scratch_component_linear_score": float(linear_item["linear_score"]),
        "scratch_component_radial_span": float(linear_item["radial_span"]),
        "scratch_component_angular_span": float(linear_item["angular_span"]),
    }


def edge_localized_features(
    sample: SyntheticSample,
    chip_values: NDArray[np.float32] | None = None,
    chip_valid: NDArray[np.bool_] | None = None,
    angular_bins: int = 12,
) -> dict[str, float]:
    """Measure whether edge fail is globally uniform or localized by angle."""

    if chip_values is None or chip_valid is None:
        chip_values, chip_valid = _chip_mean_grid(sample)
    radius, theta = _chip_center_polar_grid(sample, chip_valid)
    edge = chip_valid & (radius >= 0.78)
    center = chip_valid & (radius < 0.55)
    if not edge.any():
        return {
            "edge_chip_peak_contrast": 0.0,
            "edge_sector_peak_contrast": 0.0,
            "edge_sector_concentration": 0.0,
            "edge_localized_sector_ratio": 0.0,
        }

    baseline = float(np.median(chip_values[chip_valid])) if chip_valid.any() else 0.0
    center_mean = float(chip_values[center].mean()) if center.any() else baseline
    edge_values = chip_values[edge]
    sector_values = np.zeros(angular_bins, dtype=np.float32)
    sector_present = np.zeros(angular_bins, dtype=bool)
    scaled_theta = theta / (2 * np.pi)
    for idx in range(angular_bins):
        low = idx / angular_bins
        high = (idx + 1) / angular_bins
        sector = edge & (scaled_theta >= low) & (scaled_theta < high)
        if sector.any():
            sector_values[idx] = float(chip_values[sector].mean())
            sector_present[idx] = True

    present_values = sector_values[sector_present]
    sector_median = float(np.median(present_values)) if len(present_values) else 0.0
    sector_peak = float(present_values.max()) if len(present_values) else 0.0
    sector_total = float(present_values.sum())
    sector_threshold = sector_median + float(present_values.std()) if len(present_values) else 0.0
    return {
        "edge_chip_peak_contrast": float(edge_values.max() - center_mean),
        "edge_sector_peak_contrast": float(sector_peak - sector_median),
        "edge_sector_concentration": float(sector_peak / max(sector_total, 1e-6)),
        "edge_localized_sector_ratio": float((present_values > sector_threshold).sum() / max(len(present_values), 1)),
    }


def shot_relative_features(
    sample: SyntheticSample,
    chip_values: NDArray[np.float32] | None = None,
    chip_valid: NDArray[np.bool_] | None = None,
) -> dict[str, float]:
    """Measure repeated shot-relative lower-left and edge-band responses from observed severity."""

    if chip_values is None or chip_valid is None:
        chip_values, chip_valid = _chip_mean_grid(sample)
    anchors = ("lower_left", "bottom_edge", "left_edge")
    features: dict[str, float] = {}
    best = 0.0
    for anchor in anchors:
        score = _best_shot_anchor_contrast(chip_values, chip_valid, anchor)
        features[f"shot_{anchor}_contrast"] = score
        best = max(best, score)
    features["shot_best_contrast"] = best
    return features


def _chip_mean_grid(sample: SyntheticSample) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    rows = int(sample.metadata["grid"]["rows"])
    cols = int(sample.metadata["grid"]["cols"])
    chip_width = int(sample.metadata["chip_blocks"]["width"])
    chip_height = int(sample.metadata["chip_blocks"]["height"])
    severity = sample.severity.astype(np.float32) / 7.0
    valid = sample.valid_test_mask > 0
    values = np.zeros((rows, cols), dtype=np.float32)
    valid_grid = np.zeros((rows, cols), dtype=bool)

    for row in range(rows):
        y0 = row * chip_height
        y1 = y0 + chip_height
        for col in range(cols):
            x0 = col * chip_width
            x1 = x0 + chip_width
            chip_valid = valid[y0:y1, x0:x1]
            if chip_valid.any():
                values[row, col] = float(severity[y0:y1, x0:x1][chip_valid].mean())
                valid_grid[row, col] = True
    return values, valid_grid


def _chip_wafer_stby_grid(sample: SyntheticSample) -> tuple[NDArray[np.bool_], NDArray[np.float32]]:
    rows = int(sample.metadata["grid"]["rows"])
    cols = int(sample.metadata["grid"]["cols"])
    chip_width = int(sample.metadata["chip_blocks"]["width"])
    chip_height = int(sample.metadata["chip_blocks"]["height"])
    wafer = sample.wafer_mask > 0
    stby = sample.stby_mask > 0
    wafer_grid = np.zeros((rows, cols), dtype=bool)
    stby_values = np.zeros((rows, cols), dtype=np.float32)

    for row in range(rows):
        y0 = row * chip_height
        y1 = y0 + chip_height
        for col in range(cols):
            x0 = col * chip_width
            x1 = x0 + chip_width
            chip_wafer = wafer[y0:y1, x0:x1]
            if chip_wafer.any():
                wafer_grid[row, col] = True
                stby_values[row, col] = float(stby[y0:y1, x0:x1][chip_wafer].mean())
    return wafer_grid, stby_values


def _chip_center_polar_grid(
    sample: SyntheticSample,
    chip_valid: NDArray[np.bool_],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    rows, cols = chip_valid.shape
    chip_width = int(sample.metadata["chip_blocks"]["width"])
    chip_height = int(sample.metadata["chip_blocks"]["height"])
    y = (np.arange(rows, dtype=np.float32) + 0.5) * chip_height
    x = (np.arange(cols, dtype=np.float32) + 0.5) * chip_width
    yy, xx = np.meshgrid(y, x, indexing="ij")
    cx = (sample.shape[1] - 1) / 2.0
    cy = (sample.shape[0] - 1) / 2.0
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_distance = float(distance[chip_valid].max()) if chip_valid.any() else 1.0
    radius = distance / max(max_distance, 1.0)
    theta = (np.arctan2(xx - cx, -(yy - cy)) + 2 * np.pi) % (2 * np.pi)
    return radius.astype(np.float32), theta.astype(np.float32)


def _best_shot_anchor_contrast(
    chip_values: NDArray[np.float32],
    chip_valid: NDArray[np.bool_],
    anchor: str,
) -> float:
    best = 0.0
    layouts = ((3, 3), (3, 2), (2, 3))
    for shot_rows, shot_cols in layouts:
        for row_offset in range(shot_rows):
            for col_offset in range(shot_cols):
                weights = np.zeros(chip_values.shape, dtype=np.float32)
                for row in range(chip_values.shape[0]):
                    slot_r = (row - row_offset) % shot_rows
                    for col in range(chip_values.shape[1]):
                        slot_c = (col - col_offset) % shot_cols
                        shot_x = (slot_c + 0.5) / shot_cols
                        shot_y = (slot_r + 0.5) / shot_rows
                        weights[row, col] = _shot_anchor_weight(shot_x, shot_y, anchor)
                high = chip_valid & (weights >= max(float(weights.max()) * 0.38, 0.06))
                low = chip_valid & (weights < max(float(weights.max()) * 0.22, 0.03))
                if not high.any() or not low.any():
                    continue
                contrast = float(chip_values[high].mean() - chip_values[low].mean())
                best = max(best, contrast)
    return best


def _shot_anchor_weight(shot_x: float, shot_y: float, anchor: str) -> float:
    if anchor == "lower_left":
        return float(np.exp(-0.5 * (((shot_x - 0.18) / 0.16) ** 2 + ((shot_y - 0.82) / 0.18) ** 2)))
    if anchor == "bottom_edge":
        value = np.exp(-0.5 * ((shot_y - 0.92) / 0.11) ** 2)
        return float(value * (0.55 + 0.45 * np.exp(-0.5 * ((shot_x - 0.42) / 0.36) ** 2)))
    value = np.exp(-0.5 * ((shot_x - 0.08) / 0.11) ** 2)
    return float(value * (0.55 + 0.45 * np.exp(-0.5 * ((shot_y - 0.58) / 0.34) ** 2)))


def _profile_peak_contrast(profile: NDArray[np.float32]) -> float:
    if len(profile) == 0:
        return 0.0
    return float(profile.max() - np.median(profile))


def _profile_peak_width_ratio(profile: NDArray[np.float32]) -> float:
    if len(profile) == 0:
        return 0.0
    baseline = float(np.median(profile))
    contrast = float(profile.max() - baseline)
    if contrast <= 1e-6:
        return 0.0
    return float((profile >= baseline + 0.5 * contrast).sum() / len(profile))


def _mean_normalized_pair_distance(
    positions: NDArray[np.int64],
    shape: tuple[int, int],
) -> float:
    if len(positions) < 2:
        return 0.0
    distances = []
    denom = max(float(np.hypot(max(shape[0] - 1, 1), max(shape[1] - 1, 1))), 1.0)
    for left_idx in range(len(positions)):
        for right_idx in range(left_idx + 1, len(positions)):
            distances.append(float(np.linalg.norm(positions[left_idx] - positions[right_idx]) / denom))
    return float(np.mean(distances)) if distances else 0.0


def _connected_components(mask: NDArray[np.bool_]) -> list[NDArray[np.int64]]:
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[NDArray[np.int64]] = []
    rows, cols = mask.shape
    for row in range(rows):
        for col in range(cols):
            if visited[row, col] or not mask[row, col]:
                continue
            stack = [(row, col)]
            visited[row, col] = True
            coords: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                coords.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny = y + dy
                        nx = x + dx
                        if ny < 0 or ny >= rows or nx < 0 or nx >= cols:
                            continue
                        if visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            components.append(np.array(coords, dtype=np.int64))
    return components


def _component_stats(
    coords: NDArray[np.int64],
    chip_values: NDArray[np.float32],
    radius: NDArray[np.float32],
    theta: NDArray[np.float32],
) -> dict[str, Any]:
    size = int(len(coords))
    ys = coords[:, 0]
    xs = coords[:, 1]
    height = int(ys.max() - ys.min() + 1)
    width = int(xs.max() - xs.min() + 1)
    bbox_area = max(height * width, 1)
    compactness = float(size / bbox_area)
    centered = coords.astype(np.float32) - coords.astype(np.float32).mean(axis=0, keepdims=True)
    if size >= 2:
        cov = centered.T @ centered / max(size - 1, 1)
        eigvals = np.linalg.eigvalsh(cov)
        major = float(max(eigvals[-1], 0.0))
        minor = float(max(eigvals[0], 0.0))
        elongation = float((major - minor) / max(major + minor, 1e-6))
    else:
        elongation = 0.0
    size_weight = min(1.0, size / 6.0)
    linear_score = float(elongation * size_weight * (1.0 - 0.45 * compactness))
    component_theta = theta[ys, xs]
    component_radius = radius[ys, xs]
    return {
        "size": size,
        "centroid": coords.astype(np.float32).mean(axis=0),
        "compactness": compactness,
        "elongation": elongation,
        "linear_score": max(0.0, linear_score),
        "radial_span": float(component_radius.max() - component_radius.min()) if size else 0.0,
        "angular_span": _circular_span(component_theta),
        "mean_value": float(chip_values[ys, xs].mean()) if size else 0.0,
    }


def _circular_span(theta: NDArray[np.float32]) -> float:
    if len(theta) <= 1:
        return 0.0
    values = np.sort((theta.astype(np.float64) % (2 * np.pi)))
    gaps = np.diff(np.concatenate([values, values[:1] + 2 * np.pi]))
    span = 2 * np.pi - float(gaps.max())
    return float(span / (2 * np.pi))


def _triangle_score(positions: NDArray[np.float32], shape: tuple[int, int]) -> float:
    if len(positions) < 3:
        return 0.0
    points = positions[:3].astype(np.float32)
    distances = []
    for left_idx in range(3):
        for right_idx in range(left_idx + 1, 3):
            distances.append(float(np.linalg.norm(points[left_idx] - points[right_idx])))
    mean_distance = float(np.mean(distances))
    if mean_distance <= 1e-6:
        return 0.0
    balance = max(0.0, 1.0 - float(np.std(distances)) / mean_distance)
    left = points[1] - points[0]
    right = points[2] - points[0]
    area = abs(float(left[0] * right[1] - left[1] * right[0])) / 2.0
    diag = max(float(np.hypot(max(shape[0] - 1, 1), max(shape[1] - 1, 1))), 1.0)
    area_score = min(1.0, area / max((mean_distance * diag) / 8.0, 1e-6))
    return float(balance * area_score)


def _binned_mean(
    values: NDArray[np.float32],
    mask: NDArray[np.bool_],
    scaled_position: NDArray[np.float32],
    bins: int,
) -> NDArray[np.float32]:
    out = np.zeros(bins, dtype=np.float32)
    if not mask.any():
        return out
    positions = np.clip(scaled_position[mask], 0.0, 0.999999)
    indices = np.minimum((positions * bins).astype(np.int32), bins - 1)
    selected = values[mask].astype(np.float64, copy=False)
    sums = np.bincount(indices, weights=selected, minlength=bins)
    counts = np.bincount(indices, minlength=bins)
    present = counts > 0
    out[present] = (sums[present] / counts[present]).astype(np.float32)
    return out


def _binned_2d_mean(
    values: NDArray[np.float32],
    mask: NDArray[np.bool_],
    scaled_radius: NDArray[np.float32],
    scaled_theta: NDArray[np.float32],
    radial_bins: int,
    angular_bins: int,
) -> NDArray[np.float32]:
    out = np.zeros((radial_bins, angular_bins), dtype=np.float32)
    if not mask.any():
        return out
    r_idx = np.minimum((np.clip(scaled_radius[mask], 0.0, 0.999999) * radial_bins).astype(np.int32), radial_bins - 1)
    a_idx = np.minimum((np.clip(scaled_theta[mask], 0.0, 0.999999) * angular_bins).astype(np.int32), angular_bins - 1)
    flat_idx = r_idx * angular_bins + a_idx
    selected = values[mask].astype(np.float64, copy=False)
    size = radial_bins * angular_bins
    sums = np.bincount(flat_idx, weights=selected, minlength=size)
    counts = np.bincount(flat_idx, minlength=size)
    present = counts > 0
    flat = out.ravel()
    flat[present] = (sums[present] / counts[present]).astype(np.float32)
    return out
