"""Evaluate defect-interest-specific FBM nearest-neighbor retrieval."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.data import load_sample
from wafermap.evaluation import nearest_neighbor_indices, standardize as standardize_matrix

EXCLUDED_COLUMNS = {"sample_id", "actual_net_die", "cluster_id", "pca_0", "pca_1"}
LABEL_COLUMNS = (
    "label_scratch",
    "label_ring",
    "label_edge",
    "label_local",
    "label_shot_grid",
    "label_stby_pattern",
)
CRITERIA = {
    "overall": {
        "target": "multi_label_jaccard",
        "label": "전체 불량 조합",
        "description": "모든 observable feature를 사용해 전체적인 불량 조합이 비슷한 wafer를 찾는다.",
    },
    "edge_focus": {
        "target": "label_edge",
        "label": "Edge 관심",
        "description": "edge density, edge-chip outer-face gradient, edge sector feature 중심 검색이다.",
    },
    "shot_focus": {
        "target": "label_shot_grid",
        "label": "Shot/reticle 관심",
        "description": "photo shot 내부 상대 위치 반복성 feature 중심 검색이다.",
    },
    "stby_focus": {
        "target": "label_stby_pattern",
        "label": "Stby 관심",
        "description": "현재는 stby_ratio 중심 검색이다. stby 공간 배열 feature는 다음 보강 후보이다.",
    },
    "ring_focus": {
        "target": "label_ring",
        "label": "Ring 관심",
        "description": "radial zone과 ring morphology feature 중심 검색이다.",
    },
    "scratch_focus": {
        "target": "label_scratch",
        "label": "Scratch 관심",
        "description": "scratch morphology와 angular sector feature 중심 검색이다.",
    },
    "local_focus": {
        "target": "label_local",
        "label": "Local blob 관심",
        "description": "local hotspot peak/top3/spread/count feature 중심 검색이다.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="outputs/reports/fbm_grouping_scale_features.csv")
    parser.add_argument("--data", default="data/synthetic/fbm_grouping_scale_pilot")
    parser.add_argument("--out", default="outputs/reports/fbm_interest_retrieval_scale_report.html")
    parser.add_argument("--metrics", default="outputs/reports/fbm_interest_retrieval_scale_metrics.json")
    parser.add_argument("--neighbors-out", default="outputs/reports/fbm_interest_retrieval_scale_neighbors.csv")
    parser.add_argument("--gallery", default="outputs/figures/fbm_interest_neighbor_gallery_scale.png")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--gallery-top-k", type=int, default=3)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def observable_feature_names(row: dict[str, str]) -> list[str]:
    return [
        key
        for key in row
        if key not in EXCLUDED_COLUMNS and not key.startswith("label_") and not key.endswith("_mask_ratio")
    ]


def criterion_feature_names(all_features: list[str], criterion: str) -> list[str]:
    compact_features = [
        name for name in all_features if not name.startswith("polar_") and not name.startswith("stby_polar_")
    ]
    if criterion == "overall":
        return compact_features
    if criterion == "edge_focus":
        return [
            name
            for name in compact_features
            if name.startswith("edge_")
            or name == "center_density"
            or name == "radial_zone_04_severity"
        ]
    if criterion == "shot_focus":
        return [name for name in compact_features if name.startswith("shot_")]
    if criterion == "stby_focus":
        return [name for name in compact_features if name == "stby_ratio"]
    if criterion == "ring_focus":
        return [name for name in compact_features if name.startswith("ring_") or name.startswith("radial_zone_")]
    if criterion == "scratch_focus":
        return [
            name
            for name in compact_features
            if name.startswith("scratch_") or name.startswith("angular_sector_")
        ]
    if criterion == "local_focus":
        return [name for name in compact_features if name.startswith("local_") or name.startswith("morph_")]
    raise ValueError(f"Unknown criterion: {criterion}")


def standardize(x: np.ndarray) -> np.ndarray:
    return standardize_matrix(x)


def neighbor_indices(x: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    return nearest_neighbor_indices(x, top_k)


def label_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(left, right).sum() / union)


def build_arrays(rows: list[dict[str, str]], features: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    sample_ids = [row["sample_id"] for row in rows]
    x = np.array([[float(row[name]) for name in features] for row in rows], dtype=np.float32)
    y = np.array([[int(row[label]) for label in LABEL_COLUMNS] for row in rows], dtype=np.int32)
    return sample_ids, x, y


def display_target(target: str) -> str:
    if target.startswith("label_"):
        return target.replace("label_", "", 1)
    return target


def binary_metrics(
    labels: np.ndarray,
    neighbors: np.ndarray,
    target_index: int,
) -> dict[str, Any]:
    target = labels[:, target_index].astype(bool)
    positives = np.where(target)[0]
    base_rate = float(target.mean())
    if len(positives) == 0:
        return {
            "positive_queries": 0,
            "base_rate": base_rate,
            "precision_at_k": 0.0,
            "lift": 0.0,
            "hit_rate_at_k": 0.0,
            "mrr": 0.0,
        }
    precisions = []
    hits = []
    reciprocal_ranks = []
    for idx in positives:
        nn_labels = target[neighbors[idx]]
        precisions.append(float(nn_labels.mean()))
        hits.append(float(nn_labels.any()))
        hit_positions = np.where(nn_labels)[0]
        reciprocal_ranks.append(float(1.0 / (int(hit_positions[0]) + 1)) if len(hit_positions) else 0.0)
    precision = float(np.mean(precisions))
    return {
        "positive_queries": int(len(positives)),
        "base_rate": base_rate,
        "precision_at_k": precision,
        "lift": float(precision / max(base_rate, 1e-9)),
        "hit_rate_at_k": float(np.mean(hits)),
        "mrr": float(np.mean(reciprocal_ranks)),
    }


def overall_metrics(labels: np.ndarray, neighbors: np.ndarray) -> dict[str, Any]:
    neighbor_scores = []
    for idx in range(len(labels)):
        query = labels[idx].astype(bool)
        neighbor_scores.extend(label_jaccard(query, labels[nn].astype(bool)) for nn in neighbors[idx])
    random_scores = []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            random_scores.append(label_jaccard(labels[left].astype(bool), labels[right].astype(bool)))
    neighbor_mean = float(np.mean(neighbor_scores)) if neighbor_scores else 0.0
    random_mean = float(np.mean(random_scores)) if random_scores else 0.0
    return {
        "mean_neighbor_label_jaccard": neighbor_mean,
        "random_pair_label_jaccard": random_mean,
        "lift": float(neighbor_mean / max(random_mean, 1e-9)),
    }


def compact_labels(row: dict[str, str]) -> str:
    active = [name.replace("label_", "") for name in LABEL_COLUMNS if int(row[name]) == 1]
    return ", ".join(active) if active else "background"


def make_neighbor_rows(
    rows: list[dict[str, str]],
    sample_ids: list[str],
    labels: np.ndarray,
    neighbors_by_criterion: dict[str, np.ndarray],
    distances_by_criterion: dict[str, np.ndarray],
    top_k: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for criterion, neighbors in neighbors_by_criterion.items():
        target = CRITERIA[criterion]["target"]
        target_index = LABEL_COLUMNS.index(target) if target in LABEL_COLUMNS else -1
        for query_idx, query_id in enumerate(sample_ids):
            for rank, neighbor_idx in enumerate(neighbors[query_idx, :top_k], start=1):
                nn = int(neighbor_idx)
                row: dict[str, Any] = {
                    "criterion": criterion,
                    "criterion_label": CRITERIA[criterion]["label"],
                    "query_sample_id": query_id,
                    "rank": rank,
                    "neighbor_sample_id": sample_ids[nn],
                    "distance": float(distances_by_criterion[criterion][query_idx, nn]),
                }
                if target_index >= 0:
                    row["target_label"] = target.replace("label_", "")
                    row["query_has_target"] = int(labels[query_idx, target_index])
                    row["neighbor_has_target"] = int(labels[nn, target_index])
                    row["synthetic_target_match"] = int(labels[query_idx, target_index] == labels[nn, target_index])
                else:
                    row["target_label"] = "multi_label_jaccard"
                    row["query_has_target"] = ""
                    row["neighbor_has_target"] = ""
                    row["synthetic_target_match"] = label_jaccard(labels[query_idx].astype(bool), labels[nn].astype(bool))
                out.append(row)
    return out


def selected_query_for_criterion(rows: list[dict[str, str]], criterion: str) -> int:
    target = CRITERIA[criterion]["target"]
    features = criterion_feature_names(observable_feature_names(rows[0]), criterion)
    x = np.array([[float(row[name]) for name in features] for row in rows], dtype=np.float32)
    scores = standardize(x).mean(axis=1)
    if target in LABEL_COLUMNS:
        positives = [idx for idx, row in enumerate(rows) if int(row[target]) == 1]
        if positives:
            return max(positives, key=lambda idx: float(scores[idx]))
    return int(np.argmax(scores))


def render_sample_image(sample_dir: Path) -> np.ndarray:
    sample = load_sample(sample_dir)
    values = np.clip(sample.severity.astype(np.float32), 0, 7) / 7.0
    image = plt.get_cmap("turbo")(values)
    image[(sample.wafer_mask == 0) | ((sample.severity == 0) & (sample.stby_mask == 0))] = (0.0, 0.0, 0.0, 1.0)
    image[sample.stby_mask > 0] = (1.0, 1.0, 1.0, 1.0)
    return image


def save_interest_gallery(
    rows: list[dict[str, str]],
    sample_ids: list[str],
    labels: np.ndarray,
    neighbors_by_criterion: dict[str, np.ndarray],
    distances_by_criterion: dict[str, np.ndarray],
    data_root: Path,
    top_k: int,
    out: Path,
) -> None:
    criteria = [name for name in CRITERIA if name != "overall"]
    cols = top_k + 1
    fig, axes = plt.subplots(len(criteria), cols, figsize=(3.2 * cols, 3.2 * len(criteria)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row_pos, criterion in enumerate(criteria):
        query_idx = selected_query_for_criterion(rows, criterion)
        indices = [query_idx, *[int(v) for v in neighbors_by_criterion[criterion][query_idx, :top_k]]]
        target = CRITERIA[criterion]["target"]
        target_index = LABEL_COLUMNS.index(target)
        for col_pos, sample_idx in enumerate(indices):
            ax = axes[row_pos, col_pos]
            ax.imshow(render_sample_image(data_root / sample_ids[sample_idx]), interpolation="nearest")
            if col_pos == 0:
                title = f"{criterion}\nQUERY {sample_ids[sample_idx]}\n{compact_labels(rows[sample_idx])}"
            else:
                match = "Y" if labels[sample_idx, target_index] else "N"
                dist = distances_by_criterion[criterion][query_idx, sample_idx]
                title = f"NN {col_pos} {sample_ids[sample_idx]}\n{target.replace('label_', '')}={match}, d={dist:.2f}"
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def evaluate(rows: list[dict[str, str]], top_k: int) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray], list[str], np.ndarray]:
    all_features = observable_feature_names(rows[0])
    sample_ids = [row["sample_id"] for row in rows]
    all_labels = np.array([[int(row[label]) for label in LABEL_COLUMNS] for row in rows], dtype=np.int32)
    metrics: dict[str, Any] = {
        "sample_count": len(rows),
        "top_k": top_k,
        "label_columns": list(LABEL_COLUMNS),
        "criteria": {},
    }
    neighbors_by_criterion: dict[str, np.ndarray] = {}
    distances_by_criterion: dict[str, np.ndarray] = {}
    for criterion in CRITERIA:
        features = criterion_feature_names(all_features, criterion)
        if not features:
            raise ValueError(f"No observable features selected for {criterion}")
        _, x, labels = build_arrays(rows, features)
        neighbors, distances = neighbor_indices(x, top_k)
        neighbors_by_criterion[criterion] = neighbors
        distances_by_criterion[criterion] = distances
        target = CRITERIA[criterion]["target"]
        criterion_metrics: dict[str, Any] = {
            "label": CRITERIA[criterion]["label"],
            "description": CRITERIA[criterion]["description"],
            "target": display_target(target),
            "feature_count": len(features),
            "features": features,
        }
        if target == "multi_label_jaccard":
            criterion_metrics.update(overall_metrics(labels, neighbors))
        else:
            criterion_metrics.update(binary_metrics(labels, neighbors, LABEL_COLUMNS.index(target)))
        metrics["criteria"][criterion] = criterion_metrics
    neighbor_rows = make_neighbor_rows(rows, sample_ids, all_labels, neighbors_by_criterion, distances_by_criterion, top_k)
    return metrics, neighbor_rows, neighbors_by_criterion, distances_by_criterion, sample_ids, all_labels


def relpath(target: Path, base_file: Path) -> str:
    return Path(os.path.relpath(Path(target).resolve(), base_file.resolve().parent)).as_posix()


def criteria_rows(metrics: dict[str, Any]) -> str:
    rows = []
    for criterion, item in metrics["criteria"].items():
        if criterion == "overall":
            metric_text = f"Jaccard lift {item['lift']:.2f}x"
            precision = f"neighbor {item['mean_neighbor_label_jaccard']:.3f} / random {item['random_pair_label_jaccard']:.3f}"
        else:
            metric_text = f"lift {item['lift']:.2f}x"
            precision = f"P@K {item['precision_at_k']:.3f}, hit@K {item['hit_rate_at_k']:.3f}, MRR {item['mrr']:.3f}"
        rows.append(
            "<tr>"
            f"<td>{html.escape(criterion)}</td>"
            f"<td>{html.escape(item['label'])}</td>"
            f"<td>{html.escape(item['target'])}</td>"
            f"<td>{item['feature_count']}</td>"
            f"<td>{html.escape(metric_text)}</td>"
            f"<td>{html.escape(precision)}</td>"
            f"<td>{html.escape(item['description'])}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def feature_rows(metrics: dict[str, Any]) -> str:
    rows = []
    for criterion, item in metrics["criteria"].items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(criterion)}</td>"
            f"<td>{html.escape(', '.join(item['features']))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def html_report(metrics: dict[str, Any], gallery: Path, neighbors: Path, metrics_path: Path, out: Path) -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>FBM Interest-Based Retrieval Report</title>
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
  <h1>FBM Interest-Based Retrieval Report</h1>
  <p>같은 wafer라도 관심 defect에 따라 유사도의 기준은 달라진다. 이 리포트는 검색 기준을 feature subset으로 나누고, synthetic label을 검증용 정답지로만 사용해 각 기준이 얼마나 잘 맞는지 평가한다.</p>
  <div class="note">중요: synthetic label은 채점용이다. 실제 real wafer inference feature에는 <code>label_*</code>나 <code>*_mask_ratio</code>가 들어가지 않는다.</div>

  <h2>관심 기준별 성능</h2>
  <table>
    <tr><th>Criterion</th><th>관심 기준</th><th>채점 label</th><th>Feature 수</th><th>Metric</th><th>Top-k detail</th><th>해석</th></tr>
    {criteria_rows(metrics)}
  </table>

  <h2>관심 기준별 Neighbor 예시</h2>
  <p>각 행은 하나의 관심 기준이다. 왼쪽이 query wafer이고 오른쪽은 해당 기준 feature subset으로 찾은 nearest neighbor다.</p>
  <img src="{html.escape(relpath(gallery, out))}" alt="interest based nearest-neighbor gallery">

  <h2>검색에 사용한 Feature 기준</h2>
  <table>
    <tr><th>Criterion</th><th>Feature subset</th></tr>
    {feature_rows(metrics)}
  </table>

  <h2>Outputs</h2>
  <ul>
    <li>Metrics JSON: <code>{html.escape(relpath(metrics_path, out))}</code></li>
    <li>Neighbor CSV: <code>{html.escape(relpath(neighbors, out))}</code></li>
    <li>Gallery: <code>{html.escape(relpath(gallery, out))}</code></li>
  </ul>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    features_path = Path(args.features)
    out_path = Path(args.out)
    metrics_path = Path(args.metrics)
    neighbors_path = Path(args.neighbors_out)
    gallery_path = Path(args.gallery)
    rows = read_rows(features_path)
    if not rows:
        raise SystemExit(f"No rows found in {features_path}")
    metrics, neighbor_rows, neighbors_by_criterion, distances_by_criterion, sample_ids, labels = evaluate(rows, args.top_k)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(neighbors_path, neighbor_rows)
    save_interest_gallery(
        rows,
        sample_ids,
        labels,
        neighbors_by_criterion,
        distances_by_criterion,
        Path(args.data),
        args.gallery_top_k,
        gallery_path,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_report(metrics, gallery_path, neighbors_path, metrics_path, out_path), encoding="utf-8")
    print(f"Wrote interest retrieval report: {out_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote neighbors: {neighbors_path}")
    print(f"Wrote gallery: {gallery_path}")


if __name__ == "__main__":
    main()
