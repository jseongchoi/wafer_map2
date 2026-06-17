"""Benchmark resize and semantic pooling representations for FBM retrieval."""

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
from wafermap.evaluation import nearest_neighbor_indices, standardize as standardize_matrix
from wafermap.features import compact_observable_feature_names
from wafermap.features.spatial_pool import grid_edges, pooled_mean, pooled_occupancy

LABEL_CLASSES = ("scratch", "ring", "edge", "local", "shot_grid", "stby_pattern")
EXCLUDED_FEATURE_COLUMNS = {
    "sample_id",
    "actual_net_die",
    "cluster_id",
    "pca_0",
    "pca_1",
    *(f"label_{name}" for name in LABEL_CLASSES),
}


@dataclass
class LightSample:
    sample_id: str
    sample_dir: Path
    severity: NDArray[np.uint8]
    wafer_mask: NDArray[np.uint8]
    valid_test_mask: NDArray[np.uint8]
    stby_mask: NDArray[np.uint8]
    labels: NDArray[np.int32]


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    sample_dir: Path
    labels: NDArray[np.int32]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/synthetic/fbm_grouping_scale_pilot")
    parser.add_argument("--features", default="outputs/reports/fbm_grouping_scale_features.csv")
    parser.add_argument("--out", default="outputs/reports/fbm_resize_benchmark_scale_report.html")
    parser.add_argument("--metrics", default="outputs/reports/fbm_resize_benchmark_scale_metrics.json")
    parser.add_argument("--gallery", default="outputs/figures/fbm_resize_benchmark_scale_gallery.png")
    parser.add_argument("--neighbors-out", default="outputs/reports/fbm_resize_benchmark_scale_neighbors.csv")
    parser.add_argument("--grid-sizes", default="32,64")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def sample_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("synth_*") if (path / "arrays.npz").exists())


def load_light_sample(sample_dir: Path) -> LightSample:
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    arrays = np.load(sample_dir / "arrays.npz")
    pattern_masks = arrays["pattern_masks"]
    labels = np.array(
        [int(pattern_masks[PATTERN_CLASSES.index(name)].sum() > 0) for name in LABEL_CLASSES],
        dtype=np.int32,
    )
    return LightSample(
        sample_id=str(metadata["sample_id"]),
        sample_dir=sample_dir,
        severity=arrays["severity"],
        wafer_mask=arrays["wafer_mask"],
        valid_test_mask=arrays["valid_test_mask"],
        stby_mask=arrays["stby_mask"],
        labels=labels,
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compact_feature_names(row: dict[str, str]) -> list[str]:
    return compact_observable_feature_names(row, extra_excluded=EXCLUDED_FEATURE_COLUMNS)


def compact_feature_matrix(feature_csv: Path, sample_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    rows = read_csv_rows(feature_csv)
    if not rows:
        raise ValueError(f"No feature rows found in {feature_csv}")
    row_by_id = {row["sample_id"]: row for row in rows}
    names = compact_feature_names(rows[0])
    missing = [sample_id for sample_id in sample_ids if sample_id not in row_by_id]
    if missing:
        raise ValueError(f"Feature CSV is missing samples: {missing[:5]}")
    x = np.array([[float(row_by_id[sample_id][name]) for name in names] for sample_id in sample_ids], dtype=np.float32)
    return x, names


def naive_grayscale_vector(sample: LightSample, grid_size: int) -> NDArray[np.float32]:
    y_edges, x_edges = grid_edges(sample.severity.shape, grid_size)
    image = np.clip(sample.severity.astype(np.float32), 0, 7) / 7.0
    image[sample.wafer_mask == 0] = 0.0
    image[sample.stby_mask > 0] = 1.0
    return pooled_occupancy(image > -1, y_edges, x_edges) * pooled_mean(
        image,
        np.ones(sample.severity.shape, dtype=np.float32),
        y_edges,
        x_edges,
    )


def semantic_pool_vector(sample: LightSample, grid_size: int) -> NDArray[np.float32]:
    y_edges, x_edges = grid_edges(sample.severity.shape, grid_size)
    severity = np.clip(sample.severity.astype(np.float32), 0, 7) / 7.0
    wafer = sample.wafer_mask > 0
    valid = sample.valid_test_mask > 0
    stby = sample.stby_mask > 0
    valid_w = valid.astype(np.float32)
    wafer_w = wafer.astype(np.float32)
    channels = [
        pooled_mean(severity, valid_w, y_edges, x_edges),
        pooled_mean((sample.severity > 0).astype(np.float32), valid_w, y_edges, x_edges),
        pooled_mean((sample.severity >= 6).astype(np.float32), valid_w, y_edges, x_edges),
        pooled_mean(stby.astype(np.float32), wafer_w, y_edges, x_edges),
        pooled_mean(valid.astype(np.float32), wafer_w, y_edges, x_edges),
        pooled_occupancy(wafer, y_edges, x_edges),
    ]
    return np.stack(channels, axis=0).astype(np.float32)


def standardize(x: NDArray[np.float32]) -> NDArray[np.float32]:
    return standardize_matrix(x)


def nearest_neighbors(x: NDArray[np.float32], top_k: int) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    return nearest_neighbor_indices(x.astype(np.float32, copy=False), top_k)


def label_jaccard(left: NDArray[np.bool_], right: NDArray[np.bool_]) -> float:
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(left, right).sum() / union)


def retrieval_metrics(labels: NDArray[np.int32], neighbors: NDArray[np.int32]) -> dict[str, Any]:
    neighbor_scores = []
    for idx in range(len(labels)):
        query = labels[idx].astype(bool)
        for nn in neighbors[idx]:
            neighbor_scores.append(label_jaccard(query, labels[int(nn)].astype(bool)))
    random_scores = []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            random_scores.append(label_jaccard(labels[left].astype(bool), labels[right].astype(bool)))
    neighbor_mean = float(np.mean(neighbor_scores)) if neighbor_scores else 0.0
    random_mean = float(np.mean(random_scores)) if random_scores else 0.0
    return {
        "mean_neighbor_label_jaccard": neighbor_mean,
        "random_pair_label_jaccard": random_mean,
        "jaccard_lift": float(neighbor_mean / max(random_mean, 1e-9)),
        "class_metrics": class_metrics(labels, neighbors),
    }


def class_metrics(labels: NDArray[np.int32], neighbors: NDArray[np.int32]) -> dict[str, dict[str, float]]:
    out = {}
    for class_idx, class_name in enumerate(LABEL_CLASSES):
        target = labels[:, class_idx].astype(bool)
        positives = np.where(target)[0]
        base_rate = float(target.mean())
        if len(positives) == 0:
            out[class_name] = {"precision_at_k": 0.0, "lift": 0.0, "hit_rate_at_k": 0.0, "base_rate": base_rate}
            continue
        precisions = []
        hits = []
        for idx in positives:
            nn_labels = target[neighbors[idx]]
            precisions.append(float(nn_labels.mean()))
            hits.append(float(nn_labels.any()))
        precision = float(np.mean(precisions))
        out[class_name] = {
            "precision_at_k": precision,
            "lift": float(precision / max(base_rate, 1e-9)),
            "hit_rate_at_k": float(np.mean(hits)),
            "base_rate": base_rate,
        }
    return out


def render_sample(sample: LightSample) -> NDArray[np.float32]:
    values = np.clip(sample.severity.astype(np.float32), 0, 7) / 7.0
    image = plt.get_cmap("turbo")(values)
    image[(sample.wafer_mask == 0) | ((sample.severity == 0) & (sample.stby_mask == 0))] = (0.0, 0.0, 0.0, 1.0)
    image[sample.stby_mask > 0] = (1.0, 1.0, 1.0, 1.0)
    return image


def compact_label(labels: NDArray[np.int32]) -> str:
    active = [class_name for class_name, value in zip(LABEL_CLASSES, labels.tolist()) if value]
    return ", ".join(active) if active else "background"


def save_gallery(
    records: list[SampleRecord],
    representations: dict[str, tuple[NDArray[np.int32], NDArray[np.float32]]],
    labels: NDArray[np.int32],
    out: Path,
    top_k: int,
) -> None:
    show_reps = [name for name in ("compact_features", "naive_gray_64", "semantic_pool_64") if name in representations]
    if not show_reps:
        return
    query_idx = int(np.argmax(labels.sum(axis=1)))
    cols = min(top_k, 3) + 1
    fig, axes = plt.subplots(len(show_reps), cols, figsize=(3.25 * cols, 3.25 * len(show_reps)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row_idx, rep_name in enumerate(show_reps):
        neighbors, distances = representations[rep_name]
        indices = [query_idx, *[int(v) for v in neighbors[query_idx, : cols - 1]]]
        for col_idx, sample_idx in enumerate(indices):
            ax = axes[row_idx, col_idx]
            sample = load_light_sample(records[sample_idx].sample_dir)
            ax.imshow(render_sample(sample), interpolation="nearest")
            if col_idx == 0:
                title = f"{rep_name}\nQUERY {records[sample_idx].sample_id}\n{compact_label(labels[sample_idx])}"
            else:
                jac = label_jaccard(labels[query_idx].astype(bool), labels[sample_idx].astype(bool))
                title = f"NN {col_idx} {records[sample_idx].sample_id}\nJ={jac:.2f}, d={distances[query_idx, sample_idx]:.2f}"
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def write_neighbor_csv(
    path: Path,
    records: list[SampleRecord],
    labels: NDArray[np.int32],
    representations: dict[str, tuple[NDArray[np.int32], NDArray[np.float32]]],
    top_k: int,
) -> None:
    rows = []
    for rep_name, (neighbors, distances) in representations.items():
        for query_idx, sample in enumerate(records):
            for rank, nn in enumerate(neighbors[query_idx, :top_k], start=1):
                nn_idx = int(nn)
                rows.append(
                    {
                        "representation": rep_name,
                        "query_sample_id": sample.sample_id,
                        "rank": rank,
                        "neighbor_sample_id": records[nn_idx].sample_id,
                        "distance": float(distances[query_idx, nn_idx]),
                        "label_jaccard_validation_only": label_jaccard(
                            labels[query_idx].astype(bool),
                            labels[nn_idx].astype(bool),
                        ),
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def metric_rows(metrics: dict[str, Any]) -> str:
    rows = []
    for name, item in metrics["representations"].items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{item['dimension']}</td>"
            f"<td>{item['mean_neighbor_label_jaccard']:.4f}</td>"
            f"<td>{item['random_pair_label_jaccard']:.4f}</td>"
            f"<td>{item['jaccard_lift']:.2f}x</td>"
            f"<td>{item['class_metrics']['scratch']['lift']:.2f}x</td>"
            f"<td>{item['class_metrics']['local']['lift']:.2f}x</td>"
            f"<td>{item['class_metrics']['stby_pattern']['lift']:.2f}x</td>"
            "</tr>"
        )
    return "\n".join(rows)


def relpath(target: Path, base_file: Path) -> str:
    return os.path.relpath(target.resolve(), base_file.resolve().parent).replace("\\", "/")


def html_report(metrics: dict[str, Any], gallery: Path, neighbors: Path, metrics_path: Path, out: Path) -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>FBM Resize / Aggregation Benchmark</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2933; line-height: 1.55; }}
    h1, h2 {{ color: #111827; }}
    .note {{ background: #fff7ed; border-left: 4px solid #f97316; padding: 12px 14px; margin: 14px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f7; }}
    img {{ width: 100%; max-width: 1400px; border: 1px solid #d8dee9; border-radius: 8px; background: white; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>FBM Resize / Aggregation Benchmark</h1>
  <p>이 리포트는 리사이즈 표현이 유사 wafer 검색에 쓸 수 있는지 비교한다. naive grayscale은 stby와 Grade7을 섞는 의도적 비교군이고, semantic pooling은 severity/fail/stby/valid/wafer channel을 분리한다.</p>
  <div class="note">Synthetic label은 평가에만 사용한다. 실제 inference에서는 label이나 oracle mask를 쓰지 않는다.</div>

  <h2>Representation별 Retrieval</h2>
  <table>
    <tr><th>Representation</th><th>Dim</th><th>Neighbor Jaccard</th><th>Random Jaccard</th><th>Lift</th><th>Scratch Lift</th><th>Local Lift</th><th>Stby Lift</th></tr>
    {metric_rows(metrics)}
  </table>

  <h2>Neighbor Gallery</h2>
  <img src="{html.escape(relpath(gallery, out))}" alt="resize benchmark neighbor gallery">

  <h2>Outputs</h2>
  <ul>
    <li>Metrics JSON: <code>{html.escape(relpath(metrics_path, out))}</code></li>
    <li>Neighbor CSV: <code>{html.escape(relpath(neighbors, out))}</code></li>
    <li>Gallery: <code>{html.escape(relpath(gallery, out))}</code></li>
  </ul>
</body>
</html>
"""


def build_resize_vectors(sample_paths: list[Path], grid_sizes: list[int]) -> tuple[list[SampleRecord], NDArray[np.int32], dict[str, NDArray[np.float32]]]:
    records: list[SampleRecord] = []
    labels = []
    vector_lists: dict[str, list[NDArray[np.float32]]] = {}
    for grid_size in grid_sizes:
        vector_lists[f"naive_gray_{grid_size}"] = []
        vector_lists[f"semantic_pool_{grid_size}"] = []
    for sample_path in sample_paths:
        sample = load_light_sample(sample_path)
        records.append(SampleRecord(sample.sample_id, sample.sample_dir, sample.labels))
        labels.append(sample.labels)
        for grid_size in grid_sizes:
            vector_lists[f"naive_gray_{grid_size}"].append(naive_grayscale_vector(sample, grid_size).ravel())
            vector_lists[f"semantic_pool_{grid_size}"].append(semantic_pool_vector(sample, grid_size).ravel())
    vectors = {name: np.stack(items, axis=0).astype(np.float32) for name, items in vector_lists.items()}
    return records, np.stack(labels, axis=0).astype(np.int32), vectors


def evaluate(
    sample_paths: list[Path],
    feature_csv: Path,
    grid_sizes: list[int],
    top_k: int,
) -> tuple[dict[str, Any], dict[str, tuple[NDArray[np.int32], NDArray[np.float32]]], list[SampleRecord], NDArray[np.int32]]:
    records, labels, vectors = build_resize_vectors(sample_paths, grid_sizes)
    sample_ids = [record.sample_id for record in records]
    compact_x, compact_names = compact_feature_matrix(feature_csv, sample_ids)
    vectors = {"compact_features": compact_x, **vectors}

    metrics = {
        "sample_count": len(records),
        "top_k": top_k,
        "label_classes": list(LABEL_CLASSES),
        "compact_feature_count": len(compact_names),
        "grid_sizes": grid_sizes,
        "representations": {},
    }
    neighbor_payload: dict[str, tuple[NDArray[np.int32], NDArray[np.float32]]] = {}
    for name, x in vectors.items():
        neighbors, distances = nearest_neighbors(x, top_k)
        item = retrieval_metrics(labels, neighbors)
        item["dimension"] = int(x.shape[1])
        metrics["representations"][name] = item
        neighbor_payload[name] = (neighbors, distances)
    return metrics, neighbor_payload, records, labels


def main() -> None:
    args = parse_args()
    data_root = Path(args.data)
    dirs = sample_dirs(data_root)
    if not dirs:
        raise SystemExit(f"No samples found under {data_root}")
    grid_sizes = [int(value.strip()) for value in args.grid_sizes.split(",") if value.strip()]
    metrics, representations, records, labels = evaluate(dirs, Path(args.features), grid_sizes, args.top_k)
    metrics_path = Path(args.metrics)
    neighbors_path = Path(args.neighbors_out)
    gallery_path = Path(args.gallery)
    out_path = Path(args.out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_neighbor_csv(neighbors_path, records, labels, representations, args.top_k)
    save_gallery(records, representations, labels, gallery_path, args.top_k)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_report(metrics, gallery_path, neighbors_path, metrics_path, out_path), encoding="utf-8")
    print(f"Wrote resize benchmark report: {out_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote neighbors: {neighbors_path}")
    print(f"Wrote gallery: {gallery_path}")


if __name__ == "__main__":
    main()
