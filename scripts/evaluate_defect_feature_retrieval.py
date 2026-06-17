"""Evaluate retrieval against structured defect feature targets."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.data import load_sample
from wafermap.evaluation import nearest_neighbor_indices, standardize as standardize_matrix
from wafermap.features import feature_matrix as build_feature_matrix

EXCLUDED_COLUMNS = {"sample_id", "actual_net_die", "cluster_id", "pca_0", "pca_1"}
TARGET_KIND_ORDER = ("class", "class_radial", "class_location", "feature_key")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="outputs/reports/fbm_grouping_scale_features.csv")
    parser.add_argument("--defect-features", default="outputs/reports/fbm_defect_location_summary.csv")
    parser.add_argument("--data", default="data/synthetic/fbm_grouping_scale_pilot")
    parser.add_argument("--out", default="outputs/reports/fbm_defect_feature_retrieval_scale_report.html")
    parser.add_argument("--metrics", default="outputs/reports/fbm_defect_feature_retrieval_scale_metrics.json")
    parser.add_argument("--neighbors-out", default="outputs/reports/fbm_defect_feature_retrieval_scale_neighbors.csv")
    parser.add_argument("--gallery", default="outputs/figures/fbm_defect_feature_retrieval_scale_gallery.png")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-support", type=int, default=6)
    parser.add_argument("--max-gallery-targets", type=int, default=6)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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


def class_feature_names(all_features: list[str], class_name: str, target_kind: str = "class") -> list[str]:
    use_spatial = target_kind in {"class_location", "feature_key"}
    polar = [name for name in all_features if use_spatial and name.startswith("polar_")]
    compact_features = [
        name for name in all_features if not name.startswith("polar_") and not name.startswith("stby_polar_")
    ]
    if class_name == "edge":
        return [
            name
            for name in compact_features
            if name.startswith("edge_") or name == "center_density" or name == "radial_zone_04_severity"
        ] + polar
    if class_name == "shot_grid":
        return [name for name in compact_features if name.startswith("shot_")]
    if class_name == "stby_pattern":
        features = [name for name in compact_features if name == "stby_ratio"]
        if use_spatial:
            features += [name for name in all_features if name.startswith("stby_polar_")]
        return features
    if class_name == "ring":
        return [name for name in compact_features if name.startswith("ring_") or name.startswith("radial_zone_")]
    if class_name == "scratch":
        return [
            name
            for name in compact_features
            if name.startswith("scratch_") or name.startswith("angular_sector_")
        ] + polar
    if class_name == "local":
        return [name for name in compact_features if name.startswith("local_") or name.startswith("morph_")] + polar
    return compact_features


def standardize(x: np.ndarray) -> np.ndarray:
    return standardize_matrix(x)


def neighbor_indices(x: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    return nearest_neighbor_indices(x, top_k)


def target_keys(row: dict[str, str]) -> dict[str, str]:
    class_name = row["class_name"]
    return {
        "class": class_name,
        "class_radial": f"{class_name}__{row['radial_zone']}",
        "class_location": f"{class_name}__{row['location_label']}",
        "feature_key": row["feature_key"],
    }


def build_targets(
    defect_rows: list[dict[str, str]],
    sample_ids: list[str],
    min_support: int,
) -> list[dict[str, Any]]:
    sample_set = set(sample_ids)
    target_samples: dict[tuple[str, str], set[str]] = defaultdict(set)
    target_classes: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in defect_rows:
        sample_id = row["sample_id"]
        if sample_id not in sample_set:
            continue
        for kind, target_id in target_keys(row).items():
            key = (kind, target_id)
            target_samples[key].add(sample_id)
            target_classes[key][row["class_name"]] += 1

    targets = []
    for (kind, target_id), samples in target_samples.items():
        support = len(samples)
        if support < min_support or support >= len(sample_ids):
            continue
        class_name = target_classes[(kind, target_id)].most_common(1)[0][0]
        targets.append(
            {
                "target_kind": kind,
                "target_id": target_id,
                "class_name": class_name,
                "sample_ids": samples,
                "support": support,
            }
        )
    order = {name: idx for idx, name in enumerate(TARGET_KIND_ORDER)}
    return sorted(targets, key=lambda item: (order[item["target_kind"]], item["class_name"], -item["support"], item["target_id"]))


def evaluate_target(
    sample_ids: list[str],
    positive_ids: set[str],
    neighbors: np.ndarray,
    top_k: int,
) -> dict[str, float | int]:
    positives = [idx for idx, sample_id in enumerate(sample_ids) if sample_id in positive_ids]
    support = len(positives)
    if support == 0:
        return {
            "support": 0,
            "random_precision": 0.0,
            "precision_at_k": 0.0,
            "lift": 0.0,
            "hit_rate_at_k": 0.0,
            "mrr": 0.0,
        }
    precisions = []
    hits = []
    reciprocal_ranks = []
    for idx in positives:
        nn_ids = [sample_ids[int(nn)] for nn in neighbors[idx, :top_k]]
        matches = np.array([item in positive_ids for item in nn_ids], dtype=bool)
        precisions.append(float(matches.mean()))
        hits.append(float(matches.any()))
        hit_positions = np.where(matches)[0]
        reciprocal_ranks.append(float(1.0 / (int(hit_positions[0]) + 1)) if len(hit_positions) else 0.0)
    random_precision = float((support - 1) / max(len(sample_ids) - 1, 1))
    precision = float(np.mean(precisions))
    return {
        "support": support,
        "random_precision": random_precision,
        "precision_at_k": precision,
        "lift": float(precision / max(random_precision, 1e-9)),
        "hit_rate_at_k": float(np.mean(hits)),
        "mrr": float(np.mean(reciprocal_ranks)),
    }


def feature_matrix(rows: list[dict[str, str]], features: list[str]) -> np.ndarray:
    return build_feature_matrix(rows, features)


def summarize_by_kind(target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for kind in TARGET_KIND_ORDER:
        rows = [row for row in target_rows if row["target_kind"] == kind]
        if not rows:
            summary[kind] = {"target_count": 0, "mean_lift": 0.0, "mean_precision_at_k": 0.0, "mean_hit_rate_at_k": 0.0}
            continue
        weights = np.array([float(row["support"]) for row in rows], dtype=np.float32)
        summary[kind] = {
            "target_count": len(rows),
            "mean_lift": float(np.average([float(row["lift"]) for row in rows], weights=weights)),
            "mean_precision_at_k": float(np.average([float(row["precision_at_k"]) for row in rows], weights=weights)),
            "mean_hit_rate_at_k": float(np.average([float(row["hit_rate_at_k"]) for row in rows], weights=weights)),
        }
    return summary


def make_neighbor_rows(
    sample_ids: list[str],
    targets: list[dict[str, Any]],
    neighbors_by_target: dict[str, np.ndarray],
    distances_by_target: dict[str, np.ndarray],
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in targets:
        target_key = target["target_key"]
        positive_ids = target["sample_ids"]
        neighbors = neighbors_by_target[target_key]
        distances = distances_by_target[target_key]
        for query_idx, query_id in enumerate(sample_ids):
            if query_id not in positive_ids:
                continue
            for rank, neighbor_idx in enumerate(neighbors[query_idx, :top_k], start=1):
                nn = int(neighbor_idx)
                rows.append(
                    {
                        "target_kind": target["target_kind"],
                        "target_id": target["target_id"],
                        "class_name": target["class_name"],
                        "query_sample_id": query_id,
                        "rank": rank,
                        "neighbor_sample_id": sample_ids[nn],
                        "neighbor_has_target": int(sample_ids[nn] in positive_ids),
                        "distance": float(distances[query_idx, nn]),
                    }
                )
    return rows


def render_sample_image(sample_dir: Path) -> np.ndarray:
    sample = load_sample(sample_dir)
    values = np.clip(sample.severity.astype(np.float32), 0, 7) / 7.0
    image = plt.get_cmap("turbo")(values)
    image[(sample.wafer_mask == 0) | ((sample.severity == 0) & (sample.stby_mask == 0))] = (0.0, 0.0, 0.0, 1.0)
    image[sample.stby_mask > 0] = (1.0, 1.0, 1.0, 1.0)
    return image


def select_gallery_targets(target_rows: list[dict[str, Any]], max_targets: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_classes: set[str] = set()
    candidates = sorted(target_rows, key=lambda row: (float(row["lift"]), int(row["support"])), reverse=True)
    for row in candidates:
        if len(selected) >= max_targets:
            break
        if row["target_kind"] == "feature_key" and row["class_name"] not in seen_classes:
            selected.append(row)
            seen_classes.add(row["class_name"])
    for row in candidates:
        if len(selected) >= max_targets:
            break
        if row not in selected:
            selected.append(row)
    return selected


def save_gallery(
    target_rows: list[dict[str, Any]],
    sample_ids: list[str],
    neighbors_by_target: dict[str, np.ndarray],
    distances_by_target: dict[str, np.ndarray],
    data_root: Path,
    top_k: int,
    max_targets: int,
    out: Path,
) -> None:
    selected = select_gallery_targets(target_rows, max_targets)
    if not selected:
        return
    cols = min(top_k, 3) + 1
    fig, axes = plt.subplots(len(selected), cols, figsize=(3.2 * cols, 3.1 * len(selected)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row_idx, target in enumerate(selected):
        target_key = target["target_key"]
        positive_ids = target["sample_ids"]
        neighbors = neighbors_by_target[target_key]
        distances = distances_by_target[target_key]
        query_indices = [idx for idx, sample_id in enumerate(sample_ids) if sample_id in positive_ids]
        query_idx = max(
            query_indices,
            key=lambda idx: float(np.mean([sample_ids[int(nn)] in positive_ids for nn in neighbors[idx, :top_k]])),
        )
        indices = [query_idx, *[int(v) for v in neighbors[query_idx, : cols - 1]]]
        for col_idx, sample_idx in enumerate(indices):
            ax = axes[row_idx, col_idx]
            sample_id = sample_ids[sample_idx]
            ax.imshow(render_sample_image(data_root / sample_id), interpolation="nearest")
            if col_idx == 0:
                title = f"{target['target_kind']}\n{target['target_id']}\nQUERY {sample_id}"
            else:
                match = "Y" if sample_id in positive_ids else "N"
                title = f"NN {col_idx} {sample_id}\nmatch={match}, d={distances[query_idx, sample_idx]:.2f}"
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def evaluate(
    feature_rows: list[dict[str, str]],
    defect_rows: list[dict[str, str]],
    top_k: int,
    min_support: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    if not feature_rows:
        raise ValueError("feature_rows is empty")
    sample_ids = [row["sample_id"] for row in feature_rows]
    all_features = observable_feature_names(feature_rows[0])
    targets = build_targets(defect_rows, sample_ids, min_support)
    if not targets:
        raise ValueError("No supported defect feature targets found")

    target_rows = []
    neighbors_by_target: dict[str, np.ndarray] = {}
    distances_by_target: dict[str, np.ndarray] = {}
    feature_cache: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    for idx, target in enumerate(targets):
        class_name = target["class_name"]
        features = class_feature_names(all_features, class_name, target["target_kind"])
        if not features:
            features = all_features
        cache_key = "|".join(features)
        if cache_key not in feature_cache:
            x = feature_matrix(feature_rows, features)
            feature_cache[cache_key] = (*neighbor_indices(x, top_k), features)
        neighbors, distances, used_features = feature_cache[cache_key]
        target_key = f"{target['target_kind']}::{target['target_id']}"
        neighbors_by_target[target_key] = neighbors
        distances_by_target[target_key] = distances
        metrics = evaluate_target(sample_ids, target["sample_ids"], neighbors, top_k)
        target_rows.append(
            {
                "target_key": target_key,
                "target_kind": target["target_kind"],
                "target_id": target["target_id"],
                "class_name": class_name,
                "sample_ids": target["sample_ids"],
                "feature_count": len(used_features),
                "feature_names": used_features,
                **metrics,
            }
        )

    target_rows = sorted(
        target_rows,
        key=lambda row: (TARGET_KIND_ORDER.index(row["target_kind"]), row["class_name"], -int(row["support"]), row["target_id"]),
    )
    neighbor_rows = make_neighbor_rows(sample_ids, target_rows, neighbors_by_target, distances_by_target, top_k)
    metrics = {
        "sample_count": len(sample_ids),
        "top_k": top_k,
        "min_support": min_support,
        "target_count": len(target_rows),
        "summary_by_target_kind": summarize_by_kind(target_rows),
        "targets": [
            {key: value for key, value in row.items() if key not in {"sample_ids", "feature_names"}}
            for row in target_rows
        ],
    }
    return metrics, target_rows, neighbor_rows, neighbors_by_target, distances_by_target, sample_ids


def relpath(target: Path, base_file: Path) -> str:
    return os.path.relpath(target.resolve(), base_file.resolve().parent).replace("\\", "/")


def summary_rows(metrics: dict[str, Any]) -> str:
    rows = []
    for kind, item in metrics["summary_by_target_kind"].items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(kind)}</td>"
            f"<td>{item['target_count']}</td>"
            f"<td>{item['mean_precision_at_k']:.3f}</td>"
            f"<td>{item['mean_hit_rate_at_k']:.3f}</td>"
            f"<td>{item['mean_lift']:.2f}x</td>"
            "</tr>"
        )
    return "\n".join(rows)


def target_table_rows(target_rows: list[dict[str, Any]], limit: int = 60) -> str:
    rows = []
    for row in target_rows[:limit]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(row['target_kind'])}</td>"
            f"<td>{html.escape(row['target_id'])}</td>"
            f"<td>{html.escape(row['class_name'])}</td>"
            f"<td>{int(row['support'])}</td>"
            f"<td>{int(row['feature_count'])}</td>"
            f"<td>{float(row['precision_at_k']):.3f}</td>"
            f"<td>{float(row['random_precision']):.3f}</td>"
            f"<td>{float(row['lift']):.2f}x</td>"
            f"<td>{float(row['hit_rate_at_k']):.3f}</td>"
            f"<td>{float(row['mrr']):.3f}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def html_report(
    metrics: dict[str, Any],
    target_rows: list[dict[str, Any]],
    gallery: Path,
    neighbors: Path,
    metrics_path: Path,
    out: Path,
) -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>FBM Defect Feature Retrieval</title>
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
  <h1>FBM Defect Feature Retrieval</h1>
  <p>이 리포트는 구조화 defect feature target을 기준으로, observable FBM feature만 사용한 nearest-neighbor 검색이 얼마나 잘 맞는지 평가한다.</p>
  <div class="note">Synthetic defect feature row는 채점용 target이다. 검색 feature에는 <code>label_*</code>, <code>*_mask_ratio</code>, oracle mask field를 넣지 않는다.</div>

  <h2>Target Kind별 요약</h2>
  <table>
    <tr><th>Target kind</th><th>Target count</th><th>Mean P@K</th><th>Mean Hit@K</th><th>Mean lift</th></tr>
    {summary_rows(metrics)}
  </table>

  <h2>Defect Feature Target별 검색 성능</h2>
  <table>
    <tr><th>Kind</th><th>Target</th><th>Class</th><th>Support</th><th>Feature 수</th><th>P@K</th><th>Random</th><th>Lift</th><th>Hit@K</th><th>MRR</th></tr>
    {target_table_rows(target_rows)}
  </table>

  <h2>대표 검색 Gallery</h2>
  <img src="{html.escape(relpath(gallery, out))}" alt="defect feature retrieval gallery">

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
    feature_rows = read_csv(Path(args.features))
    defect_rows = read_csv(Path(args.defect_features))
    metrics, target_rows, neighbor_rows, neighbors_by_target, distances_by_target, sample_ids = evaluate(
        feature_rows,
        defect_rows,
        args.top_k,
        args.min_support,
    )

    metrics_path = Path(args.metrics)
    neighbors_path = Path(args.neighbors_out)
    gallery_path = Path(args.gallery)
    out_path = Path(args.out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(neighbors_path, neighbor_rows)
    save_gallery(
        target_rows,
        sample_ids,
        neighbors_by_target,
        distances_by_target,
        Path(args.data),
        args.top_k,
        args.max_gallery_targets,
        gallery_path,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_report(metrics, target_rows, gallery_path, neighbors_path, metrics_path, out_path), encoding="utf-8")
    print(f"Wrote defect feature retrieval report: {out_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote neighbors: {neighbors_path}")
    print(f"Wrote gallery: {gallery_path}")


if __name__ == "__main__":
    main()
