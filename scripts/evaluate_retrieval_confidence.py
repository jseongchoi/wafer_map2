"""Estimate confidence for FBM nearest-neighbor retrieval metrics."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.evaluation import nearest_neighbor_indices, standardize as standardize_matrix
from wafermap.features import compact_observable_feature_names

LABEL_COLUMNS = (
    "label_scratch",
    "label_ring",
    "label_edge",
    "label_local",
    "label_shot_grid",
    "label_stby_pattern",
)
EXCLUDED_COLUMNS = {"sample_id", "actual_net_die", "cluster_id", "pca_0", "pca_1", *LABEL_COLUMNS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="outputs/reports/fbm_grouping_scale_features.csv")
    parser.add_argument("--out", default="outputs/reports/fbm_retrieval_confidence_report.html")
    parser.add_argument("--metrics", default="outputs/reports/fbm_retrieval_confidence_metrics.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bootstrap-runs", type=int, default=1000)
    parser.add_argument("--permutation-runs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=29)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def feature_names(rows: list[dict[str, str]]) -> list[str]:
    return compact_observable_feature_names(rows, extra_excluded=EXCLUDED_COLUMNS)


def standardize(x: np.ndarray) -> np.ndarray:
    return standardize_matrix(x)


def nearest_neighbors(x: np.ndarray, top_k: int) -> np.ndarray:
    neighbors, _ = nearest_neighbor_indices(x, top_k)
    return neighbors


def label_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(left, right).sum() / union)


def query_neighbor_jaccards(y: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    values = np.zeros(len(y), dtype=np.float32)
    for idx in range(len(y)):
        query = y[idx].astype(bool)
        scores = [label_jaccard(query, y[nn].astype(bool)) for nn in neighbors[idx]]
        values[idx] = float(np.mean(scores)) if scores else 0.0
    return values


def random_pair_jaccards(y: np.ndarray) -> np.ndarray:
    values = []
    for left in range(len(y)):
        for right in range(left + 1, len(y)):
            values.append(label_jaccard(y[left].astype(bool), y[right].astype(bool)))
    return np.array(values, dtype=np.float32)


def bootstrap_lift_ci(
    neighbor_query_scores: np.ndarray,
    random_pair_scores: np.ndarray,
    rng: np.random.Generator,
    runs: int,
) -> dict[str, float]:
    lifts = np.zeros(runs, dtype=np.float32)
    neighbor_means = np.zeros(runs, dtype=np.float32)
    random_means = np.zeros(runs, dtype=np.float32)
    for idx in range(runs):
        neighbor_sample = rng.choice(neighbor_query_scores, size=len(neighbor_query_scores), replace=True)
        random_sample = rng.choice(random_pair_scores, size=len(random_pair_scores), replace=True)
        neighbor_mean = float(neighbor_sample.mean())
        random_mean = float(random_sample.mean())
        neighbor_means[idx] = neighbor_mean
        random_means[idx] = random_mean
        lifts[idx] = neighbor_mean / max(random_mean, 1e-9)
    return {
        "neighbor_mean_ci_low": float(np.quantile(neighbor_means, 0.025)),
        "neighbor_mean_ci_high": float(np.quantile(neighbor_means, 0.975)),
        "random_mean_ci_low": float(np.quantile(random_means, 0.025)),
        "random_mean_ci_high": float(np.quantile(random_means, 0.975)),
        "lift_ci_low": float(np.quantile(lifts, 0.025)),
        "lift_ci_high": float(np.quantile(lifts, 0.975)),
    }


def permutation_p_value(
    y: np.ndarray,
    neighbors: np.ndarray,
    observed_neighbor_mean: float,
    rng: np.random.Generator,
    runs: int,
) -> float:
    exceed = 0
    for _ in range(runs):
        permuted = y[rng.permutation(len(y))]
        permuted_mean = float(query_neighbor_jaccards(permuted, neighbors).mean())
        if permuted_mean >= observed_neighbor_mean:
            exceed += 1
    return float((exceed + 1) / (runs + 1))


def class_neighbor_metrics(y: np.ndarray, neighbors: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for class_idx, label in enumerate(LABEL_COLUMNS):
        positives = y[:, class_idx].astype(bool)
        base_rate = float(positives.mean())
        positive_indices = np.where(positives)[0]
        if len(positive_indices) == 0:
            rows.append(
                {
                    "class": label.replace("label_", ""),
                    "positive_count": 0,
                    "base_rate": base_rate,
                    "precision_at_k": 0.0,
                    "lift": 0.0,
                    "hit_rate_at_k": 0.0,
                }
            )
            continue
        precision_values = []
        hit_values = []
        for idx in positive_indices:
            neighbor_labels = positives[neighbors[idx]]
            precision_values.append(float(neighbor_labels.mean()))
            hit_values.append(float(neighbor_labels.any()))
        precision = float(np.mean(precision_values))
        rows.append(
            {
                "class": label.replace("label_", ""),
                "positive_count": int(positives.sum()),
                "base_rate": base_rate,
                "precision_at_k": precision,
                "lift": float(precision / max(base_rate, 1e-9)),
                "hit_rate_at_k": float(np.mean(hit_values)),
            }
        )
    return rows


def relpath(target: Path, base_file: Path) -> str:
    return Path(os.path.relpath(Path(target).resolve(), base_file.resolve().parent)).as_posix()


def html_report(metrics: dict[str, Any], features: Path, metrics_path: Path) -> str:
    class_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(item['class'])}</td>"
        f"<td>{item['positive_count']}</td>"
        f"<td>{item['base_rate']:.3f}</td>"
        f"<td>{item['precision_at_k']:.3f}</td>"
        f"<td>{item['lift']:.2f}x</td>"
        f"<td>{item['hit_rate_at_k']:.3f}</td>"
        "</tr>"
        for item in metrics["class_metrics"]
    )
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>FBM Retrieval Confidence Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ color: #111827; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; margin: 18px 0; }}
    .card {{ border: 1px solid #d8dee9; border-radius: 8px; padding: 14px; background: #f8fafc; }}
    .metric {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px 10px; text-align: left; }}
    th {{ background: #eef2f7; }}
    .note {{ background: #fff7ed; border-left: 4px solid #f97316; padding: 12px 14px; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>FBM Retrieval Confidence Report</h1>
  <p>이 리포트는 기존 feature CSV에서 nearest-neighbor retrieval의 bootstrap confidence interval과 permutation p-value를 계산한다. Synthetic label은 검증용으로만 사용한다.</p>
  <div class="note">이 수치는 synthetic scale pilot 내부 검증이다. Real wafer 성능 보장은 아니며, class별 prevalence가 높은 항목은 lift 해석을 조심해야 한다.</div>

  <div class="summary">
    <div class="card"><div>Samples</div><div class="metric">{metrics['sample_count']}</div></div>
    <div class="card"><div>Top K</div><div class="metric">{metrics['top_k']}</div></div>
    <div class="card"><div>Lift</div><div class="metric">{metrics['jaccard_lift']:.2f}x</div></div>
    <div class="card"><div>Permutation p</div><div class="metric">{metrics['permutation_p_value']:.3f}</div></div>
  </div>

  <h2>Overall Retrieval</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Neighbor label-Jaccard</td><td>{metrics['mean_neighbor_label_jaccard']:.4f}</td></tr>
    <tr><td>Random pair label-Jaccard</td><td>{metrics['random_pair_label_jaccard']:.4f}</td></tr>
    <tr><td>Lift 95% bootstrap CI</td><td>{metrics['bootstrap']['lift_ci_low']:.2f}x - {metrics['bootstrap']['lift_ci_high']:.2f}x</td></tr>
    <tr><td>Neighbor mean 95% CI</td><td>{metrics['bootstrap']['neighbor_mean_ci_low']:.4f} - {metrics['bootstrap']['neighbor_mean_ci_high']:.4f}</td></tr>
    <tr><td>Random mean 95% CI</td><td>{metrics['bootstrap']['random_mean_ci_low']:.4f} - {metrics['bootstrap']['random_mean_ci_high']:.4f}</td></tr>
  </table>

  <h2>Class Retrieval</h2>
  <table>
    <tr><th>Class</th><th>Positive</th><th>Base Rate</th><th>Precision@K</th><th>Lift</th><th>Hit Rate@K</th></tr>
    {class_rows}
  </table>

  <h2>Inputs</h2>
  <ul>
    <li>Feature CSV: <code>{html.escape(relpath(features, metrics_path))}</code></li>
    <li>Metrics JSON: <code>{html.escape(metrics_path.name)}</code></li>
  </ul>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    features_path = Path(args.features)
    rows = read_rows(features_path)
    if not rows:
        raise SystemExit(f"No rows found in {features_path}")

    names = feature_names(rows)
    x = np.array([[float(row[name]) for name in names] for row in rows], dtype=np.float32)
    y = np.array([[int(row[label]) for label in LABEL_COLUMNS] for row in rows], dtype=np.int32)
    neighbors = nearest_neighbors(x, args.top_k)
    neighbor_scores = query_neighbor_jaccards(y, neighbors)
    random_scores = random_pair_jaccards(y)
    observed_neighbor_mean = float(neighbor_scores.mean())
    random_mean = float(random_scores.mean())
    rng = np.random.default_rng(args.seed)
    metrics = {
        "sample_count": len(rows),
        "feature_count_observable": len(names),
        "top_k": args.top_k,
        "mean_neighbor_label_jaccard": observed_neighbor_mean,
        "random_pair_label_jaccard": random_mean,
        "jaccard_lift": float(observed_neighbor_mean / max(random_mean, 1e-9)),
        "bootstrap_runs": args.bootstrap_runs,
        "permutation_runs": args.permutation_runs,
        "bootstrap": bootstrap_lift_ci(neighbor_scores, random_scores, rng, args.bootstrap_runs),
        "permutation_p_value": permutation_p_value(y, neighbors, observed_neighbor_mean, rng, args.permutation_runs),
        "class_metrics": class_neighbor_metrics(y, neighbors),
        "feature_names": names,
    }

    metrics_path = Path(args.metrics)
    out = Path(args.out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_report(metrics, features_path, metrics_path), encoding="utf-8")
    print(f"Wrote confidence report: {out}")
    print(f"Wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()
