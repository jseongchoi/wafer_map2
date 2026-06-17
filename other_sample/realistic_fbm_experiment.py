from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.detectors.stby import detect_stby_fail_chip_mask
from wafermap.features.summary import FeatureConfig, extract_summary_features
from wafermap.hotspots.repeating import detect_repeating_hotspots as detect_repeating_hotspot_rows
from wafermap.hotspots.repeating import hotspots_to_dicts
from wafermap.reports.artifacts import dumps_json, write_csv_artifact, write_json_artifact
from wafermap.search.nearest import label_precision_at_k, top_k_neighbors

PATTERNS = ("scratch", "ring", "edge_local", "blob", "local_cluster", "repeat_coord")
CMAP = "turbo"


@dataclass(frozen=True)
class Meta:
    wafer_id: str
    lot_id: str
    active_patterns: tuple[str, ...]
    has_no_test: bool
    repeat_xy: tuple[int, int] | None


@dataclass(frozen=True)
class RealisticDataset:
    maps: np.ndarray
    valid_mask: np.ndarray
    no_test_masks: np.ndarray
    labels: np.ndarray
    metas: list[Meta]


def coordinate_cache(size: int) -> dict[str, np.ndarray]:
    yy, xx = np.mgrid[0:size, 0:size]
    cx = (size - 1) / 2
    cy = (size - 1) / 2
    xn = (xx - cx) / (size * 0.48)
    yn = (yy - cy) / (size * 0.48)
    rr = np.sqrt(xn * xn + yn * yn)
    theta = np.arctan2(yn, xn)
    return {"xx": xx, "yy": yy, "xn": xn, "yn": yn, "rr": rr, "theta": theta, "valid": rr <= 1.0}


def add_gaussian(field: np.ndarray, cache: dict[str, np.ndarray], x0: float, y0: float, sigma: float, amp: float) -> None:
    dist2 = (cache["xn"] - x0) ** 2 + (cache["yn"] - y0) ** 2
    field += amp * np.exp(-dist2 / (2 * sigma * sigma))


def add_subtle_scratch(field: np.ndarray, cache: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    angle = rng.uniform(0, math.pi)
    offset = rng.uniform(-0.42, 0.42)
    width = rng.uniform(0.006, 0.016)
    length = rng.uniform(1.0, 1.8)
    c, s = math.cos(angle), math.sin(angle)
    xrot = cache["xn"] * c + cache["yn"] * s
    yrot = -cache["xn"] * s + cache["yn"] * c
    line = np.exp(-((yrot - offset) ** 2) / (2 * width * width)) * (np.abs(xrot) < length / 2)
    sparse = rng.random(field.shape) < rng.uniform(0.18, 0.38)
    field += line * sparse * rng.uniform(0.32, 0.62) * cache["valid"]


def add_subtle_ring(field: np.ndarray, cache: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    r0 = rng.uniform(0.62, 0.94)
    width = rng.uniform(0.018, 0.045)
    ring = np.exp(-((cache["rr"] - r0) ** 2) / (2 * width * width))
    if rng.random() < 0.65:
        center = rng.uniform(-math.pi, math.pi)
        span = rng.uniform(math.pi / 2, math.tau * 0.75)
        delta = np.angle(np.exp(1j * (cache["theta"] - center)))
        ring *= np.exp(-(delta * delta) / (2 * (span / 3) ** 2))
    sparse = rng.random(field.shape) < rng.uniform(0.25, 0.55)
    field += ring * sparse * rng.uniform(0.28, 0.58) * cache["valid"]


def add_edge_local(field: np.ndarray, cache: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    center = rng.uniform(-math.pi, math.pi)
    span = rng.uniform(math.pi / 10, math.pi / 3)
    delta = np.angle(np.exp(1j * (cache["theta"] - center)))
    angular = np.exp(-(delta * delta) / (2 * span * span))
    edge_band = np.clip((cache["rr"] - rng.uniform(0.78, 0.88)) / 0.06, 0, 1)
    pattern = angular * edge_band * cache["valid"]
    sparse = rng.random(field.shape) < rng.uniform(0.28, 0.52)
    field += pattern * sparse * rng.uniform(0.32, 0.68)


def add_subtle_blob(field: np.ndarray, cache: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    radius = math.sqrt(rng.uniform(0.0, 0.75))
    angle = rng.uniform(-math.pi, math.pi)
    add_gaussian(field, cache, radius * math.cos(angle), radius * math.sin(angle), rng.uniform(0.025, 0.075), rng.uniform(0.36, 0.72))


def add_local_cluster(field: np.ndarray, cache: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    radius = math.sqrt(rng.uniform(0.0, 0.80))
    angle = rng.uniform(-math.pi, math.pi)
    cx, cy = radius * math.cos(angle), radius * math.sin(angle)
    for _ in range(rng.integers(4, 11)):
        add_gaussian(
            field,
            cache,
            cx + rng.normal(0, 0.045),
            cy + rng.normal(0, 0.045),
            rng.uniform(0.006, 0.018),
            rng.uniform(0.32, 0.70),
        )


def add_repeat_coord(field: np.ndarray, cache: dict[str, np.ndarray], xy: tuple[int, int], rng: np.random.Generator) -> None:
    size = field.shape[0]
    x0 = (xy[0] - (size - 1) / 2) / (size * 0.48)
    y0 = (xy[1] - (size - 1) / 2) / (size * 0.48)
    add_gaussian(field, cache, x0, y0, rng.uniform(0.006, 0.014), rng.uniform(0.68, 1.05))


def quantize_realistic(risk: np.ndarray, cache: dict[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    valid = cache["valid"]
    rr = cache["rr"]
    base = rng.exponential(scale=0.010, size=risk.shape)

    edge_lift = np.clip((rr - 0.78) / 0.22, 0, 1) ** 1.8
    base += edge_lift * rng.gamma(shape=1.4, scale=0.055, size=risk.shape)
    base += (rng.random(risk.shape) < 0.010) * rng.uniform(0.05, 0.18, size=risk.shape)
    base += (rng.random(risk.shape) < 0.0015) * rng.uniform(0.18, 0.45, size=risk.shape)

    score = risk + base
    thresholds = np.array([0.055, 0.115, 0.205, 0.340, 0.530, 0.760])
    grade = 1 + np.digitize(score, thresholds).astype(np.uint8)
    grade[~valid] = 0
    return grade


def add_no_test_chip(
    grade: np.ndarray,
    cache: dict[str, np.ndarray],
    rng: np.random.Generator,
    chip_pitch: int,
) -> np.ndarray:
    mask = np.zeros_like(grade, dtype=bool)
    size = grade.shape[0]
    chips = rng.integers(1, 4)
    for _ in range(chips):
        w = chip_pitch
        h = chip_pitch
        for _attempt in range(100):
            grid_x = int(rng.integers(0, max(1, size // chip_pitch)))
            grid_y = int(rng.integers(0, max(1, size // chip_pitch)))
            x0 = grid_x * chip_pitch
            y0 = grid_y * chip_pitch
            if x0 + w > size or y0 + h > size:
                continue
            candidate = np.zeros_like(mask)
            candidate[y0 : y0 + h, x0 : x0 + w] = True
            if (candidate & cache["valid"]).sum() >= 0.98 * candidate.sum():
                mask |= candidate & cache["valid"]
                break
    grade[mask] = 7
    return mask


def make_dataset(size: int, lots: int, wafers_per_lot: int, seed: int) -> RealisticDataset:
    rng = np.random.default_rng(seed)
    cache = coordinate_cache(size)
    chip_pitch = max(18, int(round(size / 18)))
    repeat_coords = {
        "LOT03": (int(size * 0.66), int(size * 0.33)),
        "LOT08": (int(size * 0.31), int(size * 0.70)),
    }

    maps: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    metas: list[Meta] = []
    pattern_to_idx = {p: i for i, p in enumerate(PATTERNS)}

    for lot_idx in range(lots):
        lot_id = f"LOT{lot_idx:02d}"
        for wafer_idx in range(wafers_per_lot):
            wafer_id = f"{lot_id}_W{wafer_idx:02d}"
            risk = np.zeros((size, size), dtype=np.float32)
            active: list[str] = []
            y = np.zeros(len(PATTERNS), dtype=np.float32)

            if rng.random() < 0.18:
                add_subtle_scratch(risk, cache, rng)
                active.append("scratch")
                y[pattern_to_idx["scratch"]] = rng.uniform(0.55, 1.0)
            if rng.random() < 0.20:
                add_subtle_ring(risk, cache, rng)
                active.append("ring")
                y[pattern_to_idx["ring"]] = rng.uniform(0.55, 1.0)
            if rng.random() < 0.18:
                add_edge_local(risk, cache, rng)
                active.append("edge_local")
                y[pattern_to_idx["edge_local"]] = rng.uniform(0.55, 1.0)
            if rng.random() < 0.18:
                add_subtle_blob(risk, cache, rng)
                active.append("blob")
                y[pattern_to_idx["blob"]] = rng.uniform(0.55, 1.0)
            if rng.random() < 0.16:
                add_local_cluster(risk, cache, rng)
                active.append("local_cluster")
                y[pattern_to_idx["local_cluster"]] = rng.uniform(0.55, 1.0)

            repeat_xy = None
            if lot_id in repeat_coords and rng.random() < 0.70:
                repeat_xy = repeat_coords[lot_id]
                add_repeat_coord(risk, cache, repeat_xy, rng)
                active.append("repeat_coord")
                y[pattern_to_idx["repeat_coord"]] = rng.uniform(0.70, 1.0)

            grade = quantize_realistic(risk, cache, rng)
            has_no_test = rng.random() < 0.28
            no_test_mask = np.zeros_like(grade, dtype=bool)
            if has_no_test:
                no_test_mask = add_no_test_chip(grade, cache, rng, chip_pitch)
                has_no_test = bool(no_test_mask.any())

            maps.append(grade)
            masks.append(no_test_mask)
            labels.append(y)
            metas.append(Meta(wafer_id, lot_id, tuple(active), has_no_test, repeat_xy))

    return RealisticDataset(
        maps=np.stack(maps),
        valid_mask=cache["valid"],
        no_test_masks=np.stack(masks),
        labels=np.stack(labels),
        metas=metas,
    )


def detect_no_test_mask(grade: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return detect_stby_fail_chip_mask(grade, valid_mask=valid)


def no_test_detection_metrics(dataset: RealisticDataset) -> dict[str, float]:
    ious = []
    recalls = []
    precisions = []
    for grade, truth in zip(dataset.maps, dataset.no_test_masks, strict=True):
        pred = detect_no_test_mask(grade, dataset.valid_mask)
        inter = (pred & truth).sum()
        union = (pred | truth).sum()
        if union > 0:
            ious.append(inter / union)
        if truth.sum() > 0:
            recalls.append(inter / truth.sum())
        if pred.sum() > 0:
            precisions.append(inter / pred.sum())
    return {
        "mean_iou_on_present_or_predicted": float(np.mean(ious)) if ious else 1.0,
        "mean_recall_when_present": float(np.mean(recalls)) if recalls else 1.0,
        "mean_precision_when_predicted": float(np.mean(precisions)) if precisions else 1.0,
    }


def extract_features(maps: np.ndarray, valid_mask: np.ndarray, clean_no_test: bool) -> np.ndarray:
    return extract_summary_features(
        maps,
        valid_mask=valid_mask,
        config=FeatureConfig(mode="combined", clean_stby=clean_no_test),
    )


def label_set(meta: Meta) -> set[str]:
    return set(meta.active_patterns) if meta.active_patterns else {"normal"}


def similarity_metrics(features: np.ndarray, metas: list[Meta], k: int = 5) -> dict[str, float]:
    neighbors = top_k_neighbors(features, k=k, metric="cosine", standardize=True)
    return label_precision_at_k(neighbors, [label_set(meta) for meta in metas], k=k)


def train_score_model(features: np.ndarray, labels: np.ndarray) -> dict[str, dict[str, float]]:
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=17)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = RandomForestRegressor(n_estimators=220, max_depth=10, random_state=17)
    model.fit(x_train, y_train)
    pred = np.clip(model.predict(x_test), 0, 1)
    out = {}
    for i, name in enumerate(PATTERNS):
        binary = y_test[:, i] > 0.5
        auc = float("nan")
        if len(np.unique(binary)) == 2:
            auc = float(roc_auc_score(binary, pred[:, i]))
        out[name] = {"mae": float(mean_absolute_error(y_test[:, i], pred[:, i])), "auc": auc}
    return out


def detect_repeating_hotspots(dataset: RealisticDataset, block: int = 4, top_n: int = 12) -> list[dict[str, float | int | str]]:
    group_ids = np.array([m.lot_id for m in dataset.metas])
    exclude_masks = np.stack([detect_no_test_mask(grade, dataset.valid_mask) for grade in dataset.maps])
    rows = detect_repeating_hotspot_rows(
        dataset.maps,
        group_ids,
        valid_mask=dataset.valid_mask,
        exclude_masks=exclude_masks,
        block=block,
        top_n=top_n,
    )
    return [
        {"lot_id": row.group_id, **{k: v for k, v in data.items() if k != "group_id"}}
        for row, data in zip(rows, hotspots_to_dicts(rows), strict=True)
    ]


def save_gallery(dataset: RealisticDataset, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = []
    targets = ["normal", "scratch", "ring", "blob", "local_cluster", "repeat_coord"]
    for target in targets:
        for i, meta in enumerate(dataset.metas):
            labels = label_set(meta)
            if (target == "normal" and labels == {"normal"}) or target in labels:
                indices.append(i)
                break
    no_test_idx = next((i for i, meta in enumerate(dataset.metas) if meta.has_no_test), None)
    if no_test_idx is not None and no_test_idx not in indices:
        indices.append(no_test_idx)

    fig, axes = plt.subplots(2, math.ceil(len(indices) / 2), figsize=(14, 6), constrained_layout=True)
    axes = np.array(axes).ravel()
    for ax, idx in zip(axes, indices, strict=False):
        im = ax.imshow(dataset.maps[idx], vmin=0, vmax=7, cmap=CMAP, interpolation="nearest")
        meta = dataset.metas[idx]
        title = ",".join(sorted(label_set(meta)))
        if meta.has_no_test:
            title += "+stby_fail_chip"
        ax.set_title(f"{meta.wafer_id}\n{title}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(indices) :]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.75, label="Grade")
    fig.savefig(out_dir / "realistic_sample_gallery.png", dpi=180)
    plt.close(fig)


def save_hotspots(rows: list[dict[str, float | int | str]], out_dir: Path) -> None:
    write_csv_artifact(rows, out_dir / "realistic_repeating_hotspots.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Realistic synthetic wafer FBM benchmark")
    parser.add_argument("--out", default=str(SCRIPT_DIR / "outputs" / "realistic_v2"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--lots", type=int, default=12)
    parser.add_argument("--wafers-per-lot", type=int, default=18)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = make_dataset(args.size, args.lots, args.wafers_per_lot, args.seed)
    naive_features = extract_features(dataset.maps, dataset.valid_mask, clean_no_test=False)
    clean_features = extract_features(dataset.maps, dataset.valid_mask, clean_no_test=True)
    hotspots = detect_repeating_hotspots(dataset)

    metrics = {
        "dataset": {
            "samples": int(dataset.maps.shape[0]),
            "map_size": int(dataset.maps.shape[-1]),
            "lots": args.lots,
            "wafers_per_lot": args.wafers_per_lot,
            "grade_histogram": {
                str(g): int((dataset.maps == g).sum()) for g in range(8)
            },
            "no_test_wafer_count": int(sum(m.has_no_test for m in dataset.metas)),
            "stby_fail_chip_wafer_count": int(sum(m.has_no_test for m in dataset.metas)),
        },
        "no_test_region_detection": no_test_detection_metrics(dataset),
        "similarity_naive_grade7_as_defect": similarity_metrics(naive_features, dataset.metas),
        "similarity_clean_no_test_separated": similarity_metrics(clean_features, dataset.metas),
        "defect_score_model_clean_features": train_score_model(clean_features, dataset.labels),
        "repeating_hotspot_detection": {
            "top_rows": hotspots[:8],
            "injected_repeat_lots": {
                meta.lot_id: meta.repeat_xy for meta in dataset.metas if meta.repeat_xy is not None
            },
        },
    }

    save_gallery(dataset, out_dir)
    save_hotspots(hotspots, out_dir)
    write_json_artifact(metrics, out_dir / "realistic_metrics.json")

    print(dumps_json(metrics))
    print(f"\nArtifacts written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
