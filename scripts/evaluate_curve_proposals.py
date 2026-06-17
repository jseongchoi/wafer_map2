"""Evaluate polar curve proposals for scratch and ring-like FBM defects."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.data import PATTERN_CLASSES
from wafermap.features.spatial_pool import grid_edges, pooled_mean

CURVE_CLASSES = ("scratch", "ring")


@dataclass
class CurveSample:
    sample_id: str
    sample_dir: Path
    severity: NDArray[np.uint8]
    wafer_mask: NDArray[np.uint8]
    valid_test_mask: NDArray[np.uint8]
    stby_mask: NDArray[np.uint8]
    pattern_masks: NDArray[np.uint8]


@dataclass(frozen=True)
class CurveCandidate:
    kind: str
    r0: int
    r1: int
    a0: int
    a1: int
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/synthetic/fbm_grouping_scale_pilot")
    parser.add_argument("--out", default="outputs/reports/fbm_curve_proposal_scale_report.html")
    parser.add_argument("--metrics", default="outputs/reports/fbm_curve_proposal_scale_metrics.json")
    parser.add_argument("--details", default="outputs/reports/fbm_curve_proposal_scale_details.csv")
    parser.add_argument("--gallery", default="outputs/figures/fbm_curve_proposal_scale_gallery.png")
    parser.add_argument("--grid-size", type=int, default=72)
    parser.add_argument("--radial-bins", type=int, default=32)
    parser.add_argument("--angular-bins", type=int, default=48)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=29)
    return parser.parse_args()


def sample_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("synth_*") if (path / "arrays.npz").exists())


def load_curve_sample(sample_dir: Path) -> CurveSample:
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    arrays = np.load(sample_dir / "arrays.npz")
    return CurveSample(
        sample_id=str(metadata["sample_id"]),
        sample_dir=sample_dir,
        severity=arrays["severity"],
        wafer_mask=arrays["wafer_mask"],
        valid_test_mask=arrays["valid_test_mask"],
        stby_mask=arrays["stby_mask"],
        pattern_masks=arrays["pattern_masks"],
    )


def semantic_maps(sample: CurveSample, grid_size: int) -> dict[str, NDArray[np.float32]]:
    y_edges, x_edges = grid_edges(sample.severity.shape, grid_size)
    severity = np.clip(sample.severity.astype(np.float32), 0, 7) / 7.0
    wafer = sample.wafer_mask > 0
    valid = sample.valid_test_mask > 0
    stby = sample.stby_mask > 0
    valid_w = valid.astype(np.float32)
    wafer_w = wafer.astype(np.float32)
    return {
        "severity": pooled_mean(severity, valid_w, y_edges, x_edges),
        "severity_peak": pooled_max(severity, valid, y_edges, x_edges),
        "fail": pooled_mean((sample.severity > 0).astype(np.float32), valid_w, y_edges, x_edges),
        "high": pooled_mean((sample.severity >= 6).astype(np.float32), valid_w, y_edges, x_edges),
        "stby": pooled_mean(stby.astype(np.float32), wafer_w, y_edges, x_edges),
        "wafer": pooled_mean(wafer.astype(np.float32), np.ones_like(wafer_w), y_edges, x_edges),
    }


def pooled_max(
    values: NDArray[np.float32],
    mask: NDArray[np.bool_],
    y_edges: NDArray[np.int32],
    x_edges: NDArray[np.int32],
) -> NDArray[np.float32]:
    out = np.zeros((len(y_edges) - 1, len(x_edges) - 1), dtype=np.float32)
    for y_idx, (y0, y1) in enumerate(zip(y_edges[:-1], y_edges[1:])):
        for x_idx, (x0, x1) in enumerate(zip(x_edges[:-1], x_edges[1:])):
            cell_mask = mask[y0:y1, x0:x1]
            if cell_mask.any():
                out[y_idx, x_idx] = float(values[y0:y1, x0:x1][cell_mask].max())
    return out


def polar_grid(grid_size: int, wafer: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    yy, xx = np.mgrid[0:grid_size, 0:grid_size].astype(np.float32)
    cy = (grid_size - 1) / 2.0
    cx = (grid_size - 1) / 2.0
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = wafer > 0.2
    max_distance = float(distance[mask].max()) if mask.any() else max(cx, cy, 1.0)
    radius = np.clip(distance / max(max_distance, 1.0), 0.0, 0.999)
    theta = (np.arctan2(xx - cx, -(yy - cy)) + 2 * np.pi) % (2 * np.pi)
    return radius.astype(np.float32), theta.astype(np.float32)


def polar_bin_means(
    score: NDArray[np.float32],
    wafer: NDArray[np.float32],
    radius: NDArray[np.float32],
    theta: NDArray[np.float32],
    radial_bins: int,
    angular_bins: int,
) -> NDArray[np.float32]:
    valid = wafer > 0.2
    r_idx = np.minimum((radius[valid] * radial_bins).astype(np.int32), radial_bins - 1)
    a_idx = np.minimum(((theta[valid] / (2 * np.pi)) * angular_bins).astype(np.int32), angular_bins - 1)
    flat = r_idx * angular_bins + a_idx
    sums = np.bincount(flat, weights=score[valid].astype(np.float64), minlength=radial_bins * angular_bins)
    counts = np.bincount(flat, minlength=radial_bins * angular_bins)
    means = np.divide(sums, np.maximum(counts, 1), out=np.zeros_like(sums), where=counts > 0)
    return means.reshape(radial_bins, angular_bins).astype(np.float32)


def pixel_radius_theta(
    shape: tuple[int, int],
    wafer: NDArray[np.bool_],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    yy, xx = np.indices(shape, dtype=np.float32)
    cy = (shape[0] - 1) / 2.0
    cx = (shape[1] - 1) / 2.0
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_distance = float(distance[wafer].max()) if wafer.any() else max(cx, cy, 1.0)
    radius = np.clip(distance / max(max_distance, 1.0), 0.0, 0.999)
    theta = (np.arctan2(xx - cx, -(yy - cy)) + 2 * np.pi) % (2 * np.pi)
    return radius.astype(np.float32), theta.astype(np.float32)


def pixel_curve_score(sample: CurveSample, class_name: str) -> NDArray[np.float32]:
    severity = np.clip(sample.severity.astype(np.float32), 0, 7) / 7.0
    fail = (sample.severity > 0).astype(np.float32)
    high = (sample.severity >= 6).astype(np.float32)
    stby = (sample.stby_mask > 0).astype(np.float32)
    if class_name == "ring":
        score = 0.50 * severity + 0.35 * fail + 0.15 * high
    else:
        score = 0.45 * severity + 0.30 * fail + 0.15 * high + 0.25 * stby
    return (score * (sample.wafer_mask > 0)).astype(np.float32)


def pixel_polar_mean(
    values: NDArray[np.float32],
    mask: NDArray[np.bool_],
    radius: NDArray[np.float32],
    theta: NDArray[np.float32],
    radial_bins: int,
    angular_bins: int,
) -> NDArray[np.float32]:
    r_idx = np.minimum((radius[mask] * radial_bins).astype(np.int32), radial_bins - 1)
    a_idx = np.minimum(((theta[mask] / (2 * np.pi)) * angular_bins).astype(np.int32), angular_bins - 1)
    flat = r_idx * angular_bins + a_idx
    sums = np.bincount(flat, weights=values[mask].astype(np.float64), minlength=radial_bins * angular_bins)
    counts = np.bincount(flat, minlength=radial_bins * angular_bins)
    means = np.divide(sums, np.maximum(counts, 1), out=np.zeros_like(sums), where=counts > 0)
    return means.reshape(radial_bins, angular_bins).astype(np.float32)


def pixel_target_bins(
    sample: CurveSample,
    class_name: str,
    radius: NDArray[np.float32],
    theta: NDArray[np.float32],
    radial_bins: int,
    angular_bins: int,
) -> NDArray[np.float32]:
    class_idx = PATTERN_CLASSES.index(class_name)
    target = sample.pattern_masks[class_idx] > 0
    if not target.any():
        return np.zeros((radial_bins, angular_bins), dtype=np.float32)
    r_idx = np.minimum((radius[target] * radial_bins).astype(np.int32), radial_bins - 1)
    a_idx = np.minimum(((theta[target] / (2 * np.pi)) * angular_bins).astype(np.int32), angular_bins - 1)
    flat = r_idx * angular_bins + a_idx
    counts = np.bincount(flat, minlength=radial_bins * angular_bins)
    return counts.reshape(radial_bins, angular_bins).astype(np.float32)


def candidate_mask(
    candidate: CurveCandidate,
    radius: NDArray[np.float32],
    theta: NDArray[np.float32],
    wafer: NDArray[np.float32],
    radial_bins: int,
    angular_bins: int,
) -> NDArray[np.bool_]:
    r_idx = np.minimum((radius * radial_bins).astype(np.int32), radial_bins - 1)
    a_idx = np.minimum(((theta / (2 * np.pi)) * angular_bins).astype(np.int32), angular_bins - 1)
    radial = (r_idx >= candidate.r0) & (r_idx < candidate.r1)
    angular = circular_bin_mask(a_idx, candidate.a0, candidate.a1, angular_bins)
    return radial & angular & (wafer > 0.2)


def circular_bin_mask(indices: NDArray[np.int32], start: int, end: int, bins: int) -> NDArray[np.bool_]:
    start = start % bins
    width = max(1, end - start)
    if width >= bins:
        return np.ones(indices.shape, dtype=bool)
    return ((indices - start) % bins) < width


def polar_rect_mean(polar_score: NDArray[np.float32], r0: int, r1: int, a0: int, a1: int) -> float:
    angular_bins = polar_score.shape[1]
    a_width = max(1, a1 - a0)
    cols = (np.arange(a_width, dtype=np.int32) + a0) % angular_bins
    values = polar_score[r0:r1, :][:, cols]
    return float(values.mean()) if values.size else 0.0


def build_candidates(
    class_name: str,
    polar_score: NDArray[np.float32],
    top_k: int,
) -> list[CurveCandidate]:
    radial_bins, angular_bins = polar_score.shape
    baseline = float(np.median(polar_score[polar_score > 0])) if (polar_score > 0).any() else 0.0
    spread = float(polar_score[polar_score > 0].std()) if (polar_score > 0).any() else 0.0
    candidates: list[CurveCandidate] = []
    if class_name == "ring":
        max_ring_r = max(2, int(radial_bins * 0.90))
        for r_width in (1, 2, 3):
            for r0 in range(1, min(radial_bins - r_width, max_ring_r)):
                r1 = r0 + r_width
                score = curve_candidate_score("annulus", polar_score, r0, r1, 0, angular_bins, baseline, spread)
                candidates.append(CurveCandidate("annulus", r0, r1, 0, angular_bins, score))
        for r_width in (1, 2, 3):
            for a_width in (8, 12, 16, 24):
                for r0 in range(1, min(radial_bins - r_width, max_ring_r)):
                    for a0 in range(0, angular_bins, 2):
                        r1 = r0 + r_width
                        score = curve_candidate_score(
                            "partial_ring",
                            polar_score,
                            r0,
                            r1,
                            a0,
                            a0 + a_width,
                            baseline,
                            spread,
                        )
                        candidates.append(CurveCandidate("partial_ring", r0, r1, a0, a0 + a_width, score))
    else:
        for r_width in (1, 2, 3):
            for a_width in (4, 8, 12, 16):
                for r0 in range(1, radial_bins - r_width):
                    for a0 in range(0, angular_bins, 2):
                        r1 = r0 + r_width
                        score = curve_candidate_score("spin_arc", polar_score, r0, r1, a0, a0 + a_width, baseline, spread)
                        candidates.append(CurveCandidate("spin_arc", r0, r1, a0, a0 + a_width, score))
        for r0 in range(1, max(2, radial_bins // 2)):
            for r1 in range(max(r0 + radial_bins // 3, radial_bins // 2), radial_bins):
                for a_width in (1, 2, 3):
                    for a0 in range(0, angular_bins):
                        score = curve_candidate_score(
                            "radial_scratch",
                            polar_score,
                            r0,
                            r1,
                            a0,
                            a0 + a_width,
                            baseline,
                            spread,
                        )
                        candidates.append(CurveCandidate("radial_scratch", r0, r1, a0, a0 + a_width, score))
    candidates.sort(key=lambda item: item.score, reverse=True)
    return nms_candidates(candidates, top_k, angular_bins)


def curve_candidate_score(
    kind: str,
    polar_score: NDArray[np.float32],
    r0: int,
    r1: int,
    a0: int,
    a1: int,
    baseline: float,
    spread: float,
) -> float:
    angular_bins = polar_score.shape[1]
    a_width = max(1, a1 - a0)
    cols = (np.arange(a_width, dtype=np.int32) + a0) % angular_bins
    values = polar_score[r0:r1, :][:, cols]
    if values.size == 0:
        return 0.0
    residual = values - baseline
    threshold = max(0.01, 0.20 * spread)
    positive = residual > threshold
    if not positive.any():
        return float(residual.mean())
    strength = float(residual[positive].mean())
    if kind == "radial_scratch":
        continuity = float(positive.any(axis=1).mean())
    else:
        continuity = float(positive.any(axis=0).mean())
    support = float(positive.mean())
    return strength * (0.15 + continuity) * np.sqrt(max(support, 1.0 / values.size))


def nms_candidates(
    candidates: list[CurveCandidate],
    top_k: int,
    angular_bins: int,
    iou_threshold: float = 0.35,
) -> list[CurveCandidate]:
    selected: list[CurveCandidate] = []
    for candidate in candidates:
        if all(candidate_iou(candidate, item, angular_bins) <= iou_threshold for item in selected):
            selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected


def candidate_iou(left: CurveCandidate, right: CurveCandidate, angular_bins: int) -> float:
    r_inter = max(0, min(left.r1, right.r1) - max(left.r0, right.r0))
    a_inter = circular_interval_intersection(left.a0, left.a1, right.a0, right.a1, angular_bins)
    inter = r_inter * a_inter
    if inter == 0:
        return 0.0
    left_area = (left.r1 - left.r0) * (left.a1 - left.a0)
    right_area = (right.r1 - right.r0) * (right.a1 - right.a0)
    return float(inter / max(left_area + right_area - inter, 1))


def circular_interval_intersection(left0: int, left1: int, right0: int, right1: int, bins_hint: int) -> int:
    bins = max(bins_hint, left1 - left0, right1 - right0)
    left = set(((np.arange(left1 - left0) + left0) % bins).tolist())
    right = set(((np.arange(right1 - right0) + right0) % bins).tolist())
    return len(left & right)


def random_candidates(
    class_name: str,
    top_k: int,
    radial_bins: int,
    angular_bins: int,
    rng: np.random.Generator,
) -> list[CurveCandidate]:
    candidates = []
    for _ in range(top_k):
        if class_name == "ring":
            kind = str(rng.choice(["annulus", "partial_ring"], p=[0.35, 0.65]))
            r_width = int(rng.choice([1, 2, 3]))
            r0 = int(rng.integers(1, radial_bins - r_width))
            if kind == "annulus":
                a0, a1 = 0, angular_bins
            else:
                a_width = int(rng.choice([8, 12, 16, 24]))
                a0, a1 = int(rng.integers(0, angular_bins)), int(rng.integers(0, angular_bins)) + a_width
                a1 = a0 + a_width
        else:
            kind = str(rng.choice(["spin_arc", "radial_scratch"], p=[0.65, 0.35]))
            if kind == "spin_arc":
                r_width = int(rng.choice([1, 2, 3]))
                r0 = int(rng.integers(1, radial_bins - r_width))
                a_width = int(rng.choice([4, 8, 12, 16]))
                a0 = int(rng.integers(0, angular_bins))
                a1 = a0 + a_width
            else:
                r0 = int(rng.integers(1, max(2, radial_bins // 2)))
                r1 = int(rng.integers(max(r0 + radial_bins // 3, radial_bins // 2), radial_bins))
                a_width = int(rng.choice([1, 2, 3]))
                a0 = int(rng.integers(0, angular_bins))
                candidates.append(CurveCandidate(kind, r0, r1, a0, a0 + a_width, 0.0))
                continue
        candidates.append(CurveCandidate(kind, r0, r0 + r_width, a0, a1, 0.0))
    return candidates


def target_occupancy(sample: CurveSample, class_name: str, grid_size: int) -> NDArray[np.float32]:
    class_idx = PATTERN_CLASSES.index(class_name)
    target = sample.pattern_masks[class_idx].astype(np.float32)
    y_edges, x_edges = grid_edges(sample.severity.shape, grid_size)
    return pooled_mean(target, np.ones(target.shape, dtype=np.float32), y_edges, x_edges)


def coverage_recall(
    target: NDArray[np.float32],
    candidates: list[CurveCandidate],
    radius: NDArray[np.float32],
    theta: NDArray[np.float32],
    wafer: NDArray[np.float32],
    radial_bins: int,
    angular_bins: int,
) -> float:
    target_mass = float(target.sum())
    if target_mass <= 0:
        return 0.0
    covered = np.zeros(target.shape, dtype=bool)
    for candidate in candidates:
        covered |= candidate_mask(candidate, radius, theta, wafer, radial_bins, angular_bins)
    return float(target[covered].sum() / target_mass)


def candidate_bin_mask(candidate: CurveCandidate, radial_bins: int, angular_bins: int) -> NDArray[np.bool_]:
    r_idx = np.arange(radial_bins, dtype=np.int32)[:, None]
    a_idx = np.arange(angular_bins, dtype=np.int32)[None, :]
    radial = (r_idx >= candidate.r0) & (r_idx < candidate.r1)
    angular = circular_bin_mask(a_idx, candidate.a0, candidate.a1, angular_bins)
    return radial & angular


def polar_recall(target_bins: NDArray[np.float32], candidates: list[CurveCandidate]) -> float:
    target_mass = float(target_bins.sum())
    if target_mass <= 0:
        return 0.0
    covered = np.zeros(target_bins.shape, dtype=bool)
    radial_bins, angular_bins = target_bins.shape
    for candidate in candidates:
        covered |= candidate_bin_mask(candidate, radial_bins, angular_bins)
    return float(target_bins[covered].sum() / target_mass)


def curve_score_map(class_name: str, maps: dict[str, NDArray[np.float32]]) -> NDArray[np.float32]:
    if class_name == "ring":
        score = 0.35 * maps["severity"] + 0.30 * maps["severity_peak"] + 0.25 * maps["fail"] + 0.15 * maps["high"]
    else:
        score = (
            0.25 * maps["severity"]
            + 0.45 * maps["severity_peak"]
            + 0.25 * maps["fail"]
            + 0.15 * maps["high"]
            + 0.10 * maps["stby"]
        )
    return (score * maps["wafer"]).astype(np.float32)


def evaluate_sample(
    sample: CurveSample,
    grid_size: int,
    radial_bins: int,
    angular_bins: int,
    top_k: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    wafer = sample.wafer_mask > 0
    radius, theta = pixel_radius_theta(sample.severity.shape, wafer)
    rows: list[dict[str, Any]] = []
    for class_name in CURVE_CLASSES:
        target = pixel_target_bins(sample, class_name, radius, theta, radial_bins, angular_bins)
        if float(target.sum()) <= 0:
            continue
        score = pixel_curve_score(sample, class_name)
        polar_score = pixel_polar_mean(score, wafer, radius, theta, radial_bins, angular_bins)
        proposals = build_candidates(class_name, polar_score, top_k)
        random = random_candidates(class_name, top_k, radial_bins, angular_bins, rng)
        proposal_cov = polar_recall(target, proposals)
        random_cov = polar_recall(target, random)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "class_name": class_name,
                "target_grid_mass": float(target.sum()),
                "proposal_recall": proposal_cov,
                "random_recall": random_cov,
                "hit_at_30": int(proposal_cov >= 0.30),
                "hit_at_50": int(proposal_cov >= 0.50),
                "random_hit_at_30": int(random_cov >= 0.30),
                "random_hit_at_50": int(random_cov >= 0.50),
                "proposal_kinds": ",".join(candidate.kind for candidate in proposals),
                "proposal_specs": serialize_candidates(proposals),
                "random_specs": serialize_candidates(random),
            }
        )
    return rows


def serialize_candidates(candidates: list[CurveCandidate]) -> str:
    return ";".join(f"{c.kind}:{c.r0}:{c.r1}:{c.a0}:{c.a1}:{c.score:.4f}" for c in candidates)


def parse_candidates(value: str) -> list[CurveCandidate]:
    candidates = []
    if not value:
        return candidates
    for item in value.split(";"):
        kind, r0, r1, a0, a1, score = item.split(":")
        candidates.append(CurveCandidate(kind, int(r0), int(r1), int(a0), int(a1), float(score)))
    return candidates


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for class_name in CURVE_CLASSES:
        items = [row for row in rows if row["class_name"] == class_name]
        if not items:
            out[class_name] = {
                "positive_count": 0,
                "mean_proposal_recall": 0.0,
                "mean_random_recall": 0.0,
                "recall_lift": 0.0,
                "hit_at_30": 0.0,
                "hit_at_50": 0.0,
            }
            continue
        proposal = np.array([float(row["proposal_recall"]) for row in items], dtype=np.float32)
        random = np.array([float(row["random_recall"]) for row in items], dtype=np.float32)
        out[class_name] = {
            "positive_count": len(items),
            "mean_proposal_recall": float(proposal.mean()),
            "mean_random_recall": float(random.mean()),
            "recall_lift": float(proposal.mean() / max(float(random.mean()), 1e-9)),
            "hit_at_30": float(np.mean([int(row["hit_at_30"]) for row in items])),
            "hit_at_50": float(np.mean([int(row["hit_at_50"]) for row in items])),
            "random_hit_at_30": float(np.mean([int(row["random_hit_at_30"]) for row in items])),
            "random_hit_at_50": float(np.mean([int(row["random_hit_at_50"]) for row in items])),
        }
    return out


def render_sample(sample: CurveSample) -> NDArray[np.float32]:
    values = np.clip(sample.severity.astype(np.float32), 0, 7) / 7.0
    image = plt.get_cmap("turbo")(values)
    image[(sample.wafer_mask == 0) | ((sample.severity == 0) & (sample.stby_mask == 0))] = (0.0, 0.0, 0.0, 1.0)
    image[sample.stby_mask > 0] = (1.0, 1.0, 1.0, 1.0)
    return image


def proposal_grid_mask(
    candidates: list[CurveCandidate],
    grid_size: int,
    radial_bins: int,
    angular_bins: int,
    wafer: NDArray[np.float32],
) -> NDArray[np.bool_]:
    radius, theta = polar_grid(grid_size, wafer)
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    for candidate in candidates:
        mask |= candidate_mask(candidate, radius, theta, wafer, radial_bins, angular_bins)
    return mask


def save_gallery(
    rows: list[dict[str, Any]],
    sample_by_id: dict[str, Path],
    grid_size: int,
    radial_bins: int,
    angular_bins: int,
    out: Path,
) -> None:
    examples = []
    for class_name in CURVE_CLASSES:
        candidates = [row for row in rows if row["class_name"] == class_name]
        if candidates:
            examples.append(max(candidates, key=lambda row: float(row["proposal_recall"])))
    if not examples:
        return
    fig, axes = plt.subplots(len(examples), 2, figsize=(8.2, 4.0 * len(examples)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row_idx, row in enumerate(examples):
        sample = load_curve_sample(sample_by_id[row["sample_id"]])
        maps = semantic_maps(sample, grid_size)
        proposals = parse_candidates(row["proposal_specs"])
        proposal_mask = proposal_grid_mask(proposals, grid_size, radial_bins, angular_bins, maps["wafer"])
        class_idx = PATTERN_CLASSES.index(row["class_name"])
        target = sample.pattern_masks[class_idx] > 0
        target_overlay = np.zeros((*target.shape, 4), dtype=np.float32)
        target_overlay[target] = (1.0, 0.0, 0.0, 0.36)
        proposal_overlay = np.zeros((grid_size, grid_size, 4), dtype=np.float32)
        proposal_overlay[proposal_mask] = (0.0, 1.0, 1.0, 0.34)
        for col, show_target in enumerate((False, True)):
            ax = axes[row_idx, col]
            ax.imshow(render_sample(sample), interpolation="nearest")
            ax.imshow(
                proposal_overlay,
                interpolation="nearest",
                extent=(0, sample.severity.shape[1], sample.severity.shape[0], 0),
            )
            if show_target:
                ax.imshow(target_overlay, interpolation="nearest")
            title = f"{row['class_name']} {row['sample_id']} curve recall={float(row['proposal_recall']):.2f}"
            if show_target:
                title += " / oracle overlay"
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def write_details(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def relpath(target: Path, base_file: Path) -> str:
    return os.path.relpath(target.resolve(), base_file.resolve().parent).replace("\\", "/")


def summary_rows(metrics: dict[str, Any]) -> str:
    rows = []
    for class_name, item in metrics["summary_by_class"].items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(class_name)}</td>"
            f"<td>{item['positive_count']}</td>"
            f"<td>{item['mean_proposal_recall']:.3f}</td>"
            f"<td>{item['mean_random_recall']:.3f}</td>"
            f"<td>{item['recall_lift']:.2f}x</td>"
            f"<td>{item['hit_at_30']:.3f}</td>"
            f"<td>{item['hit_at_50']:.3f}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def html_report(metrics: dict[str, Any], gallery: Path, details: Path, metrics_path: Path, out: Path) -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>FBM Curve Proposal Evaluation</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2933; line-height: 1.55; }}
    h1, h2 {{ color: #111827; }}
    .note {{ background: #eef6ff; border-left: 4px solid #2563eb; padding: 12px 14px; margin: 14px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f7; }}
    img {{ width: 100%; max-width: 1400px; border: 1px solid #d8dee9; border-radius: 8px; background: white; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>FBM Curve Proposal Evaluation</h1>
  <p>네모 patch proposal이 놓치기 쉬운 scratch/ring 계열을 polar curve 후보로 평가한다. Cyan 영역은 관측 FBM에서 계산한 curve proposal이고, red overlay는 synthetic oracle mask다.</p>
  <div class="note">Synthetic oracle은 recall 채점과 gallery overlay에만 사용한다. Proposal score는 severity/fail/high/stby/wafer channel에서 계산한다.</div>

  <h2>Curve Proposal Recall</h2>
  <table>
    <tr><th>Class</th><th>Positive</th><th>Proposal Recall</th><th>Random Recall</th><th>Lift</th><th>Hit@30%</th><th>Hit@50%</th></tr>
    {summary_rows(metrics)}
  </table>

  <h2>Gallery</h2>
  <img src="{html.escape(relpath(gallery, out))}" alt="curve proposal gallery">

  <h2>설정</h2>
  <table>
    <tr><td>Samples</td><td>{metrics['sample_count']}</td></tr>
    <tr><td>Grid size</td><td>{metrics['grid_size']}</td></tr>
    <tr><td>Radial bins</td><td>{metrics['radial_bins']}</td></tr>
    <tr><td>Angular bins</td><td>{metrics['angular_bins']}</td></tr>
    <tr><td>Top-K proposals</td><td>{metrics['top_k']}</td></tr>
  </table>

  <h2>Outputs</h2>
  <ul>
    <li>Metrics JSON: <code>{html.escape(relpath(metrics_path, out))}</code></li>
    <li>Details CSV: <code>{html.escape(relpath(details, out))}</code></li>
    <li>Gallery: <code>{html.escape(relpath(gallery, out))}</code></li>
  </ul>
</body>
</html>
"""


def evaluate(
    dirs: list[Path],
    grid_size: int,
    radial_bins: int,
    angular_bins: int,
    top_k: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Path]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    sample_by_id: dict[str, Path] = {}
    for sample_dir in dirs:
        sample = load_curve_sample(sample_dir)
        sample_by_id[sample.sample_id] = sample_dir
        rows.extend(evaluate_sample(sample, grid_size, radial_bins, angular_bins, top_k, rng))
    metrics = {
        "sample_count": len(dirs),
        "grid_size": grid_size,
        "radial_bins": radial_bins,
        "angular_bins": angular_bins,
        "top_k": top_k,
        "curve_classes": list(CURVE_CLASSES),
        "summary_by_class": summarize(rows),
    }
    return metrics, rows, sample_by_id


def main() -> None:
    args = parse_args()
    dirs = sample_dirs(Path(args.data))
    if not dirs:
        raise SystemExit(f"No samples found under {args.data}")
    metrics, rows, sample_by_id = evaluate(
        dirs,
        args.grid_size,
        args.radial_bins,
        args.angular_bins,
        args.top_k,
        args.seed,
    )
    metrics_path = Path(args.metrics)
    details_path = Path(args.details)
    gallery_path = Path(args.gallery)
    out_path = Path(args.out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_details(details_path, rows)
    save_gallery(rows, sample_by_id, args.grid_size, args.radial_bins, args.angular_bins, gallery_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_report(metrics, gallery_path, details_path, metrics_path, out_path), encoding="utf-8")
    print(f"Wrote curve proposal report: {out_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote details: {details_path}")
    print(f"Wrote gallery: {gallery_path}")


if __name__ == "__main__":
    main()
