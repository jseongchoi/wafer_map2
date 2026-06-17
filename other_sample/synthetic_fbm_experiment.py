from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import binom
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


PATTERNS = ("scratch", "ring", "blob", "local_cluster", "grid_signature", "repeat_coord")
PLOT_CMAP = "turbo"
SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SampleMeta:
    wafer_id: str
    lot_id: str
    tool_id: str
    chamber_id: str
    primary_pattern: str
    active_patterns: tuple[str, ...]
    injected_repeat_xy: tuple[int, int] | None


@dataclass(frozen=True)
class Dataset:
    maps: np.ndarray
    metas: list[SampleMeta]
    label_scores: np.ndarray
    pattern_names: tuple[str, ...]
    valid_mask: np.ndarray


def make_coordinate_cache(size: int) -> dict[str, np.ndarray]:
    yy, xx = np.mgrid[0:size, 0:size]
    cx = (size - 1) / 2
    cy = (size - 1) / 2
    xn = (xx - cx) / (size * 0.48)
    yn = (yy - cy) / (size * 0.48)
    rr = np.sqrt(xn * xn + yn * yn)
    theta = np.arctan2(yn, xn)
    valid = rr <= 1.0
    return {
        "xx": xx,
        "yy": yy,
        "xn": xn,
        "yn": yn,
        "rr": rr,
        "theta": theta,
        "valid": valid,
    }


def add_gaussian(field: np.ndarray, cache: dict[str, np.ndarray], x0: float, y0: float, sigma: float, amp: float) -> None:
    dist2 = (cache["xn"] - x0) ** 2 + (cache["yn"] - y0) ** 2
    field += amp * np.exp(-dist2 / (2 * sigma * sigma))


def add_scratch(
    field: np.ndarray,
    cache: dict[str, np.ndarray],
    rng: np.random.Generator,
    curved: bool = False,
) -> None:
    angle = rng.uniform(0, math.pi)
    length = rng.uniform(1.0, 1.9)
    width = rng.uniform(0.012, 0.027)
    offset = rng.uniform(-0.45, 0.45)
    c, s = math.cos(angle), math.sin(angle)
    xrot = cache["xn"] * c + cache["yn"] * s
    yrot = -cache["xn"] * s + cache["yn"] * c
    target = offset
    if curved:
        target = offset + rng.uniform(0.025, 0.075) * np.sin(rng.uniform(2.0, 4.0) * xrot + rng.uniform(0, math.tau))
    mask = (np.abs(yrot - target) < width) & (np.abs(xrot) < length / 2) & cache["valid"]
    field[mask] += rng.uniform(0.85, 1.45)


def add_ring(field: np.ndarray, cache: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    r0 = rng.uniform(0.58, 0.93)
    width = rng.uniform(0.025, 0.065)
    partial = rng.random() < 0.45
    ring = np.exp(-((cache["rr"] - r0) ** 2) / (2 * width * width))
    if partial:
        center = rng.uniform(-math.pi, math.pi)
        span = rng.uniform(math.pi / 2, math.tau * 0.82)
        delta = np.angle(np.exp(1j * (cache["theta"] - center)))
        angular = np.exp(-(delta * delta) / (2 * (span / 3) ** 2))
        ring *= angular
    field += rng.uniform(0.65, 1.25) * ring * cache["valid"]


def add_blob(field: np.ndarray, cache: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    for _ in range(rng.integers(1, 4)):
        radius = math.sqrt(rng.uniform(0.0, 0.72))
        angle = rng.uniform(-math.pi, math.pi)
        x0, y0 = radius * math.cos(angle), radius * math.sin(angle)
        add_gaussian(field, cache, x0, y0, rng.uniform(0.035, 0.11), rng.uniform(0.75, 1.55))


def add_local_cluster(field: np.ndarray, cache: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    radius = math.sqrt(rng.uniform(0.0, 0.85))
    angle = rng.uniform(-math.pi, math.pi)
    cx, cy = radius * math.cos(angle), radius * math.sin(angle)
    for _ in range(rng.integers(5, 13)):
        add_gaussian(
            field,
            cache,
            cx + rng.normal(0, 0.055),
            cy + rng.normal(0, 0.055),
            rng.uniform(0.008, 0.022),
            rng.uniform(0.55, 1.35),
        )


def add_grid_signature(field: np.ndarray, cache: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    size = field.shape[0]
    pitch = int(rng.choice([24, 32, 40]))
    width = int(rng.choice([1, 2]))
    phase_x = int(rng.integers(0, pitch))
    phase_y = int(rng.integers(0, pitch))
    grid = np.zeros_like(field, dtype=bool)
    grid[:, phase_x::pitch] = True
    grid[phase_y::pitch, :] = True
    if width > 1:
        grid = ndi.binary_dilation(grid, iterations=width - 1)
    sparse = rng.random((size, size)) < rng.uniform(0.25, 0.55)
    field[grid & sparse & cache["valid"]] += rng.uniform(0.45, 0.85)


def add_repeat_coord(
    field: np.ndarray,
    cache: dict[str, np.ndarray],
    rng: np.random.Generator,
    xy: tuple[int, int],
) -> None:
    size = field.shape[0]
    x0 = (xy[0] - (size - 1) / 2) / (size * 0.48)
    y0 = (xy[1] - (size - 1) / 2) / (size * 0.48)
    add_gaussian(field, cache, x0, y0, rng.uniform(0.010, 0.025), rng.uniform(1.15, 1.85))


def quantize_to_grade(field: np.ndarray, valid_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = rng.exponential(scale=0.045, size=field.shape)
    speckle = rng.random(field.shape)
    score = field + noise
    score += (speckle < 0.013) * rng.uniform(0.12, 0.36, size=field.shape)
    score += (speckle < 0.0025) * rng.uniform(0.32, 0.72, size=field.shape)
    thresholds = np.array([0.14, 0.28, 0.44, 0.66, 0.94, 1.24])
    grades = 1 + np.digitize(score, thresholds).astype(np.uint8)
    grades[~valid_mask] = 0
    return grades


def make_synthetic_dataset(
    size: int = 256,
    lots: int = 8,
    wafers_per_lot: int = 20,
    seed: int = 7,
) -> Dataset:
    rng = np.random.default_rng(seed)
    cache = make_coordinate_cache(size)
    repeat_lot_coords = {
        "LOT02": (int(size * 0.65), int(size * 0.35)),
        "LOT05": (int(size * 0.30), int(size * 0.72)),
    }
    maps: list[np.ndarray] = []
    metas: list[SampleMeta] = []
    labels: list[np.ndarray] = []
    pattern_to_idx = {name: idx for idx, name in enumerate(PATTERNS)}

    for lot_idx in range(lots):
        lot_id = f"LOT{lot_idx:02d}"
        for wafer_idx in range(wafers_per_lot):
            wafer_id = f"{lot_id}_W{wafer_idx:02d}"
            tool_id = f"T{lot_idx % 4 + 1}"
            chamber_id = f"C{lot_idx % 3 + 1}"
            field = np.zeros((size, size), dtype=np.float32)
            label_score = np.zeros(len(PATTERNS), dtype=np.float32)

            choices: list[str] = []
            if rng.random() < 0.20:
                choices.append("scratch")
            if rng.random() < 0.18:
                choices.append("ring")
            if rng.random() < 0.18:
                choices.append("blob")
            if rng.random() < 0.16:
                choices.append("local_cluster")
            if rng.random() < 0.10:
                choices.append("grid_signature")
            if not choices:
                choices.append("normal")
            if len(choices) == 1 and choices[0] != "normal" and rng.random() < 0.22:
                extra = rng.choice([p for p in PATTERNS[:-1] if p not in choices])
                choices.append(str(extra))

            if "scratch" in choices:
                add_scratch(field, cache, rng, curved=rng.random() < 0.25)
                label_score[pattern_to_idx["scratch"]] = rng.uniform(0.65, 1.0)
            if "ring" in choices:
                add_ring(field, cache, rng)
                label_score[pattern_to_idx["ring"]] = rng.uniform(0.65, 1.0)
            if "blob" in choices:
                add_blob(field, cache, rng)
                label_score[pattern_to_idx["blob"]] = rng.uniform(0.65, 1.0)
            if "local_cluster" in choices:
                add_local_cluster(field, cache, rng)
                label_score[pattern_to_idx["local_cluster"]] = rng.uniform(0.65, 1.0)
            if "grid_signature" in choices:
                add_grid_signature(field, cache, rng)
                label_score[pattern_to_idx["grid_signature"]] = rng.uniform(0.65, 1.0)

            repeat_xy = None
            if lot_id in repeat_lot_coords and rng.random() < 0.72:
                repeat_xy = repeat_lot_coords[lot_id]
                add_repeat_coord(field, cache, rng, repeat_xy)
                label_score[pattern_to_idx["repeat_coord"]] = rng.uniform(0.75, 1.0)
                if "repeat_coord" not in choices:
                    choices.append("repeat_coord")

            grades = quantize_to_grade(field, cache["valid"], rng)
            maps.append(grades)
            primary = choices[0]
            active = tuple(p for p in choices if p != "normal")
            metas.append(SampleMeta(wafer_id, lot_id, tool_id, chamber_id, primary, active, repeat_xy))
            labels.append(label_score)

    return Dataset(
        maps=np.stack(maps),
        metas=metas,
        label_scores=np.stack(labels),
        pattern_names=PATTERNS,
        valid_mask=cache["valid"],
    )


def _profile(values: np.ndarray, bin_idx: np.ndarray, valid: np.ndarray, bins: int) -> np.ndarray:
    idx = bin_idx[valid].ravel()
    weights = values[valid].ravel()
    sums = np.bincount(idx, weights=weights, minlength=bins)
    counts = np.bincount(idx, minlength=bins).clip(min=1)
    return sums / counts


def component_stats(mask: np.ndarray) -> tuple[float, float, float, float]:
    labeled, ncomp = ndi.label(mask)
    if ncomp == 0:
        return 0.0, 0.0, 0.0, 0.0
    max_scratch = 0.0
    max_blob = 0.0
    max_area = 0.0
    count_large = 0
    for comp_id in range(1, ncomp + 1):
        coords = np.argwhere(labeled == comp_id)
        area = float(len(coords))
        if area < 12:
            continue
        count_large += 1
        max_area = max(max_area, area)
        centered = coords - coords.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(area - 1.0, 1.0)
        eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
        elongation = math.sqrt((eig[0] + 1e-6) / (eig[1] + 1e-6))
        edge = np.logical_xor(labeled == comp_id, ndi.binary_erosion(labeled == comp_id))
        perimeter = float(edge.sum())
        compactness = 4.0 * math.pi * area / max(perimeter * perimeter, 1.0)
        scratch = min(1.0, max(0.0, (elongation - 2.5) / 12.0)) * min(1.0, area / 600.0)
        blob = min(1.0, compactness) * min(1.0, area / 500.0) * max(0.0, 1.0 - (elongation - 1.0) / 6.0)
        max_scratch = max(max_scratch, scratch)
        max_blob = max(max_blob, blob)
    return max_scratch, max_blob, min(1.0, max_area / 2500.0), min(1.0, count_large / 30.0)


def rule_scores(grades: np.ndarray, valid: np.ndarray, rr: np.ndarray) -> dict[str, float]:
    severity = np.where(valid, (grades.astype(np.float32) - 1.0) / 6.0, 0.0)
    fail = (grades >= 3) & valid
    edge = (rr > 0.78) & valid
    center = (rr < 0.55) & valid
    edge_lift = float(severity[edge].mean() / (severity[center].mean() + 1e-5))

    r_bins = np.clip((rr * 36).astype(int), 0, 35)
    radial = _profile(severity, r_bins, valid, 36)
    ring_score = float(np.clip((radial.max() - np.median(radial)) / (radial.std() + 1e-5) / 5.0, 0, 1))
    scratch_score, blob_score, area_score, comp_count_score = component_stats(fail)
    local_density = ndi.uniform_filter(fail.astype(np.float32), size=17)
    local_score = float(np.clip(local_density.max() / (fail.mean() + 1e-5) / 28.0, 0, 1))
    row = fail.mean(axis=1)
    col = fail.mean(axis=0)
    grid_score = float(np.clip(max(row.std(), col.std()) / (fail.mean() + 1e-5) / 8.0, 0, 1))
    return {
        "fail_rate_g3": float(fail[valid].mean()),
        "mean_severity": float(severity[valid].mean()),
        "edge_lift": min(edge_lift / 5.0, 1.0),
        "ring_score": ring_score,
        "scratch_score": scratch_score,
        "blob_score": blob_score,
        "local_score": local_score,
        "grid_score": grid_score,
        "area_score": area_score,
        "component_count_score": comp_count_score,
    }


def extract_features(maps: np.ndarray, valid_mask: np.ndarray, mode: str) -> np.ndarray:
    size = maps.shape[-1]
    cache = make_coordinate_cache(size)
    valid = valid_mask
    r_bins = np.clip((cache["rr"] * 32).astype(int), 0, 31)
    theta_bins = np.clip(((cache["theta"] + math.pi) / math.tau * 36).astype(int), 0, 35)
    x_bins = np.clip((cache["xx"] / size * 32).astype(int), 0, 31)
    y_bins = np.clip((cache["yy"] / size * 32).astype(int), 0, 31)
    feats: list[np.ndarray] = []

    for grades in maps:
        severity = np.where(valid, (grades.astype(np.float32) - 1.0) / 6.0, 0.0)
        hist = np.array([(grades[valid] == g).mean() for g in range(1, 8)], dtype=np.float32)
        radial = _profile(severity, r_bins, valid, 32)
        angular = _profile(severity, theta_bins, valid, 36)
        scores = rule_scores(grades, valid, cache["rr"])
        score_values = np.array(list(scores.values()), dtype=np.float32)

        if mode == "cartesian":
            xprof = _profile(severity, x_bins, valid, 32)
            yprof = _profile(severity, y_bins, valid, 32)
            feat = np.concatenate([hist, xprof, yprof, radial, angular, score_values])
        elif mode == "polar_invariant":
            spectrum = np.abs(np.fft.rfft(angular))
            spectrum = spectrum[1:13] / (spectrum[0:1].mean() + 1e-6)
            feat = np.concatenate([hist, radial, spectrum, score_values])
        else:
            raise ValueError(f"Unknown feature mode: {mode}")
        feats.append(feat.astype(np.float32))

    return np.stack(feats)


def active_label_set(meta: SampleMeta) -> set[str]:
    if meta.active_patterns:
        return set(meta.active_patterns)
    return {"normal"}


def evaluate_similarity(features: np.ndarray, metas: list[SampleMeta], k: int = 5) -> dict[str, float]:
    scaled = StandardScaler().fit_transform(features)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(scaled)
    distances, indices = nn.kneighbors(scaled)
    hits = []
    reciprocal_ranks = []
    for i, neighbors in enumerate(indices):
        query_labels = active_label_set(metas[i])
        rank_hit = 0
        top_hits = 0
        for rank, j in enumerate(neighbors[1:], start=1):
            if query_labels & active_label_set(metas[j]):
                top_hits += 1
                if rank_hit == 0:
                    rank_hit = rank
        hits.append(top_hits / k)
        reciprocal_ranks.append(0.0 if rank_hit == 0 else 1.0 / rank_hit)
    return {
        "top5_label_precision": float(np.mean(hits)),
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
    }


def train_defect_score_model(features: np.ndarray, labels: np.ndarray, pattern_names: Iterable[str]) -> dict[str, dict[str, float]]:
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.28, random_state=11)
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    model = RandomForestRegressor(n_estimators=260, max_depth=10, random_state=11)
    model.fit(x_train_s, y_train)
    pred = np.clip(model.predict(x_test_s), 0.0, 1.0)

    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(pattern_names):
        y_true_binary = y_test[:, i] > 0.5
        auc = float("nan")
        if len(np.unique(y_true_binary)) == 2:
            auc = float(roc_auc_score(y_true_binary, pred[:, i]))
        out[name] = {
            "mae": float(mean_absolute_error(y_test[:, i], pred[:, i])),
            "auc": auc,
        }
    return out


def downsample_hotspots(maps: np.ndarray, block: int = 4) -> np.ndarray:
    n, h, w = maps.shape
    hot = (maps >= 4).astype(np.uint8)
    return hot.reshape(n, h // block, block, w // block, block).max(axis=(2, 4))


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    flat = pvals.ravel()
    n = flat.size
    order = np.argsort(flat)
    ranked = flat[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(flat)
    out[order] = np.clip(q, 0, 1)
    return out.reshape(pvals.shape)


def detect_repeating_hotspots(dataset: Dataset, block: int = 4, top_n: int = 20) -> list[dict[str, float | int | str]]:
    hot = downsample_hotspots(dataset.maps, block=block)
    lots = sorted({m.lot_id for m in dataset.metas})
    lot_arr = np.array([m.lot_id for m in dataset.metas])
    results: list[dict[str, float | int | str]] = []
    eps = 1e-4

    for lot_id in lots:
        in_lot = lot_arr == lot_id
        obs = hot[in_lot].sum(axis=0)
        n_lot = int(in_lot.sum())
        baseline = hot[~in_lot].mean(axis=0)
        baseline = np.clip(baseline, eps, 0.99)
        obs_rate = obs / max(n_lot, 1)
        lift = obs_rate / baseline
        z = (obs - n_lot * baseline) / np.sqrt(n_lot * baseline * (1 - baseline) + eps)
        pvals = binom.sf(obs - 1, n_lot, baseline)
        qvals = benjamini_hochberg(pvals)
        candidate = (obs >= max(5, int(0.25 * n_lot))) & (lift >= 3.0) & (qvals <= 0.05)
        ys, xs = np.where(candidate)
        for y, x in zip(ys, xs, strict=False):
            px = int(x * block + block / 2)
            py = int(y * block + block / 2)
            rr = math.sqrt((px - dataset.maps.shape[-1] / 2) ** 2 + (py - dataset.maps.shape[-2] / 2) ** 2)
            theta = math.degrees(math.atan2(py - dataset.maps.shape[-2] / 2, px - dataset.maps.shape[-1] / 2))
            results.append(
                {
                    "lot_id": lot_id,
                    "x_bin": int(x),
                    "y_bin": int(y),
                    "x_pixel": px,
                    "y_pixel": py,
                    "observed_count": int(obs[y, x]),
                    "observed_rate": float(obs_rate[y, x]),
                    "baseline_rate": float(baseline[y, x]),
                    "lift": float(lift[y, x]),
                    "z_score": float(z[y, x]),
                    "q_value": float(qvals[y, x]),
                    "radius_pixel": float(rr),
                    "theta_deg": float(theta),
                }
            )

    results.sort(key=lambda row: (float(row["q_value"]), -float(row["lift"]), -float(row["observed_rate"])))
    return results[:top_n]


def nearest_examples(features: np.ndarray, metas: list[SampleMeta], query_idx: int, k: int = 5) -> list[int]:
    scaled = StandardScaler().fit_transform(features)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(scaled)
    return [int(i) for i in nn.kneighbors(scaled[query_idx : query_idx + 1], return_distance=False)[0][1:]]


def save_gallery(dataset: Dataset, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = ["normal", "scratch", "ring", "blob", "local_cluster", "grid_signature", "repeat_coord"]
    indices = []
    for target in wanted:
        for i, meta in enumerate(dataset.metas):
            labels = active_label_set(meta)
            if (target == "normal" and labels == {"normal"}) or target in labels:
                indices.append(i)
                break
    indices = indices[:12]
    fig, axes = plt.subplots(2, math.ceil(len(indices) / 2), figsize=(14, 6), constrained_layout=True)
    axes = np.array(axes).ravel()
    for ax, idx in zip(axes, indices, strict=False):
        im = ax.imshow(dataset.maps[idx], vmin=0, vmax=7, cmap=PLOT_CMAP, interpolation="nearest")
        meta = dataset.metas[idx]
        ax.set_title(f"{meta.wafer_id}\n{','.join(sorted(active_label_set(meta)))}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(indices) :]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.75, label="Grade")
    fig.savefig(out_dir / "sample_gallery.png", dpi=180)
    plt.close(fig)


def save_query_plot(dataset: Dataset, cart_features: np.ndarray, polar_features: np.ndarray, out_dir: Path) -> dict[str, object]:
    query_idx = next(
        i for i, meta in enumerate(dataset.metas) if "ring" in active_label_set(meta) and "repeat_coord" not in active_label_set(meta)
    )
    cart_neighbors = nearest_examples(cart_features, dataset.metas, query_idx)
    polar_neighbors = nearest_examples(polar_features, dataset.metas, query_idx)
    plot_indices = [query_idx] + cart_neighbors + polar_neighbors
    titles = ["Query"] + [f"Cartesian {i + 1}" for i in range(len(cart_neighbors))] + [f"Polar {i + 1}" for i in range(len(polar_neighbors))]

    fig, axes = plt.subplots(2, 6, figsize=(15, 5.2), constrained_layout=True)
    axes = axes.ravel()
    for ax, idx, title in zip(axes, plot_indices, titles, strict=False):
        ax.imshow(dataset.maps[idx], vmin=0, vmax=7, cmap=PLOT_CMAP, interpolation="nearest")
        meta = dataset.metas[idx]
        ax.set_title(f"{title}\n{meta.wafer_id}\n{','.join(sorted(active_label_set(meta)))}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(plot_indices) :]:
        ax.axis("off")
    fig.savefig(out_dir / "query_results.png", dpi=180)
    plt.close(fig)

    return {
        "query": dataset.metas[query_idx].wafer_id,
        "query_labels": sorted(active_label_set(dataset.metas[query_idx])),
        "cartesian_neighbors": [
            {"wafer_id": dataset.metas[i].wafer_id, "labels": sorted(active_label_set(dataset.metas[i]))} for i in cart_neighbors
        ],
        "polar_neighbors": [
            {"wafer_id": dataset.metas[i].wafer_id, "labels": sorted(active_label_set(dataset.metas[i]))} for i in polar_neighbors
        ],
    }


def write_hotspot_csv(rows: list[dict[str, float | int | str]], out_dir: Path) -> None:
    if not rows:
        return
    with (out_dir / "repeating_hotspots.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic wafer fail bitmap benchmark")
    parser.add_argument("--out", default=str(SCRIPT_DIR / "outputs"), help="Output directory")
    parser.add_argument("--size", type=int, default=256, help="Synthetic map size")
    parser.add_argument("--lots", type=int, default=8, help="Number of synthetic lots")
    parser.add_argument("--wafers-per-lot", type=int, default=20, help="Wafers per lot")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = make_synthetic_dataset(args.size, args.lots, args.wafers_per_lot, args.seed)
    cart_features = extract_features(dataset.maps, dataset.valid_mask, mode="cartesian")
    polar_features = extract_features(dataset.maps, dataset.valid_mask, mode="polar_invariant")

    metrics = {
        "dataset": {
            "samples": int(dataset.maps.shape[0]),
            "map_size": int(dataset.maps.shape[-1]),
            "patterns": list(dataset.pattern_names),
        },
        "similarity_cartesian": evaluate_similarity(cart_features, dataset.metas),
        "similarity_polar_invariant": evaluate_similarity(polar_features, dataset.metas),
        "defect_score_model": train_defect_score_model(cart_features, dataset.label_scores, dataset.pattern_names),
    }
    hotspots = detect_repeating_hotspots(dataset)
    metrics["repeating_hotspot_detection"] = {
        "top_count": len(hotspots),
        "top_rows": hotspots[:5],
        "injected_repeat_lots": {
            meta.lot_id: meta.injected_repeat_xy
            for meta in dataset.metas
            if meta.injected_repeat_xy is not None
        },
    }

    query_example = save_query_plot(dataset, cart_features, polar_features, out_dir)
    metrics["query_example"] = query_example
    save_gallery(dataset, out_dir)
    write_hotspot_csv(hotspots, out_dir)

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nArtifacts written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
