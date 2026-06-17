"""Sweep CPU-friendly FBM grouping parameters."""

from __future__ import annotations

import argparse
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

from wafermap.evaluation import nearest_neighbor_indices
from evaluate_grouping_stability import load_rows, observable_feature_names, relpath, run_stability, standardize


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="outputs/reports/fbm_grouping_features.csv")
    parser.add_argument("--out", default="outputs/reports/fbm_grouping_parameter_sweep_report.html")
    parser.add_argument("--metrics", default="outputs/reports/fbm_grouping_parameter_sweep_metrics.json")
    parser.add_argument("--figure", default="outputs/figures/fbm_grouping_parameter_sweep.png")
    parser.add_argument("--cluster-values", default="3,4,5,6,7,8")
    parser.add_argument("--top-k-values", default="3,5,10")
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--feature-rate", type=float, default=0.85)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def label_names(row: dict[str, str]) -> list[str]:
    return [key for key in row if key.startswith("label_")]


def label_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.logical_or(left, right).sum())
    inter = int(np.logical_and(left, right).sum())
    return float(inter / union) if union else 1.0


def retrieval_metrics(rows: list[dict[str, str]], top_k: int) -> dict[str, Any]:
    feature_names = observable_feature_names(rows[0])
    label_cols = label_names(rows[0])
    x = np.array([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float32)
    y = np.array([[int(row[name]) for name in label_cols] for row in rows], dtype=np.int32)
    neighbors, _ = nearest_neighbor_indices(x, top_k)
    k = neighbors.shape[1]

    neighbor_scores = []
    for idx in range(len(rows)):
        query = y[idx].astype(bool)
        for nn in neighbors[idx]:
            neighbor_scores.append(label_jaccard(query, y[int(nn)].astype(bool)))

    random_scores = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            random_scores.append(label_jaccard(y[i].astype(bool), y[j].astype(bool)))

    neighbor_mean = float(np.mean(neighbor_scores)) if neighbor_scores else 0.0
    random_mean = float(np.mean(random_scores)) if random_scores else 0.0
    return {
        "top_k": k,
        "mean_neighbor_label_jaccard": neighbor_mean,
        "random_pair_label_jaccard": random_mean,
        "jaccard_lift": neighbor_mean / max(random_mean, 1e-9),
    }


def rows_without_cluster_id(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    cleaned = []
    for row in rows:
        copied = dict(row)
        copied.pop("cluster_id", None)
        copied.pop("pca_0", None)
        copied.pop("pca_1", None)
        cleaned.append(copied)
    return cleaned


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_rows(Path(args.features))
    if not rows:
        raise SystemExit(f"No rows found in {args.features}")
    cluster_values = parse_int_list(args.cluster_values)
    top_k_values = parse_int_list(args.top_k_values)
    clean_rows = rows_without_cluster_id(rows)

    cluster_results = []
    for cluster_count in cluster_values:
        stability = run_stability(
            rows=clean_rows,
            runs=args.runs,
            clusters=cluster_count,
            top_k=min(top_k_values) if top_k_values else 5,
            feature_rate=args.feature_rate,
            noise_std=args.noise_std,
            seed=args.seed + cluster_count * 17,
        )
        cluster_results.append(
            {
                "clusters": cluster_count,
                "same_cluster_mean_coassociation": stability["same_cluster_mean_coassociation"],
                "different_cluster_mean_coassociation": stability["different_cluster_mean_coassociation"],
                "coassociation_separation": stability["coassociation_separation"],
                "mean_nearest_neighbor_overlap": stability["mean_nearest_neighbor_overlap"],
                "run_unique_cluster_count_mean": stability["run_unique_cluster_count_mean"],
                "all_stability_acceptance": all(bool(v) for v in stability["acceptance"].values()),
            }
        )

    top_k_results = [retrieval_metrics(clean_rows, top_k) for top_k in top_k_values]
    best_cluster = max(cluster_results, key=lambda item: (item["all_stability_acceptance"], item["coassociation_separation"]))
    acceptance = {
        "cpu_only_validation": True,
        "any_cluster_setting_passes_stability": any(item["all_stability_acceptance"] for item in cluster_results),
        "all_top_k_lift_above_1_10": all(item["jaccard_lift"] >= 1.10 for item in top_k_results),
        "mean_top_k_lift_above_1_25": float(np.mean([item["jaccard_lift"] for item in top_k_results])) >= 1.25,
    }
    return {
        "sample_count": len(rows),
        "feature_count_observable": len(observable_feature_names(clean_rows[0])),
        "cluster_values": cluster_values,
        "top_k_values": top_k_values,
        "runs_per_cluster": args.runs,
        "feature_subsample_rate": args.feature_rate,
        "noise_std": args.noise_std,
        "cluster_stability_sweep": cluster_results,
        "retrieval_top_k_sweep": top_k_results,
        "recommended_cluster_count_cpu_pilot": best_cluster["clusters"],
        "acceptance": acceptance,
    }


def save_figure(metrics: dict[str, Any], path: Path) -> None:
    cluster_results = metrics["cluster_stability_sweep"]
    top_k_results = metrics["retrieval_top_k_sweep"]
    ks = [item["clusters"] for item in cluster_results]
    same = [item["same_cluster_mean_coassociation"] for item in cluster_results]
    diff = [item["different_cluster_mean_coassociation"] for item in cluster_results]
    sep = [item["coassociation_separation"] for item in cluster_results]
    topks = [item["top_k"] for item in top_k_results]
    lifts = [item["jaccard_lift"] for item in top_k_results]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    ax = axes[0]
    ax.plot(ks, same, marker="o", label="same cluster")
    ax.plot(ks, diff, marker="o", label="different cluster")
    ax.plot(ks, sep, marker="o", label="separation")
    ax.axhline(0.20, color="#9ca3af", linestyle="--", linewidth=1)
    ax.set_title("Cluster Stability Sweep")
    ax.set_xlabel("Cluster count")
    ax.set_ylabel("Co-association")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1]
    ax.plot(topks, lifts, marker="o", color="#2563eb")
    ax.axhline(1.10, color="#9ca3af", linestyle="--", linewidth=1)
    ax.set_title("Top-K Retrieval Lift")
    ax.set_xlabel("Top K")
    ax.set_ylabel("Label-Jaccard lift")
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def table_rows(items: list[dict[str, Any]], columns: list[str]) -> str:
    rows = []
    for item in items:
        cells = []
        for col in columns:
            value = item[col]
            if isinstance(value, float):
                text = f"{value:.4f}"
            else:
                text = str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(rows)


def acceptance_rows(metrics: dict[str, Any]) -> str:
    labels = {
        "cpu_only_validation": "GPU 없이 CPU-only 검증으로 수행",
        "any_cluster_setting_passes_stability": "하나 이상의 cluster 설정이 안정성 기준 통과",
        "all_top_k_lift_above_1_10": "모든 top-k 설정에서 유사 검색 lift 1.10 이상",
        "mean_top_k_lift_above_1_25": "top-k lift 평균 1.25 이상",
    }
    rows = []
    for key, value in metrics["acceptance"].items():
        cls = "pass" if value else "fail"
        text = "PASS" if value else "CHECK"
        rows.append(
            "<tr>"
            f"<td>{html.escape(labels.get(key, key))}</td>"
            f"<td class=\"{cls}\">{text}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def html_report(metrics: dict[str, Any], figure: Path, metrics_path: Path, features: Path, out: Path) -> str:
    all_acceptance = all(bool(v) for v in metrics["acceptance"].values())
    verdict = "PASS" if all_acceptance else "CHECK"
    cluster_cols = [
        "clusters",
        "same_cluster_mean_coassociation",
        "different_cluster_mean_coassociation",
        "coassociation_separation",
        "mean_nearest_neighbor_overlap",
        "run_unique_cluster_count_mean",
        "all_stability_acceptance",
    ]
    topk_cols = ["top_k", "mean_neighbor_label_jaccard", "random_pair_label_jaccard", "jaccard_lift"]
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>FBM 그룹핑 파라미터 스윕 리포트</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ color: #111827; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(170px, 1fr)); gap: 12px; margin: 18px 0; }}
    .card {{ border: 1px solid #d8dee9; border-radius: 8px; padding: 14px; background: #f8fafc; }}
    .metric {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f7; }}
    .pass {{ color: #096b3b; font-weight: 700; }}
    .fail {{ color: #b42318; font-weight: 700; }}
    .note {{ background: #eef6ff; border-left: 4px solid #2563eb; padding: 12px 14px; }}
    img {{ max-width: 100%; border: 1px solid #d8dee9; border-radius: 8px; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>FBM 그룹핑 파라미터 스윕 리포트</h1>
  <p>이 리포트는 노트북 CPU에서 cluster count와 top-k 검색 조건을 바꿔 보며, 현재 observable feature 기반 방법론이 어느 정도 granularity에서 안정적인지 확인한다.</p>

  <div class="note">
    현재 결론은 “최근접 검색은 안정적이고, cluster count는 너무 세밀하게 잡지 않는 편이 좋다”에 가깝다. 따라서 실무 적용 초기는 hard cluster label보다 nearest-neighbor review와 multi-view score ranking을 먼저 쓰는 방향이 안전하다.
  </div>

  <div class="summary">
    <div class="card"><div>Sample</div><div class="metric">{metrics['sample_count']}</div></div>
    <div class="card"><div>Feature</div><div class="metric">{metrics['feature_count_observable']}</div></div>
    <div class="card"><div>권장 K</div><div class="metric">{metrics['recommended_cluster_count_cpu_pilot']}</div></div>
    <div class="card"><div>검증 상태</div><div class="metric {'pass' if all_acceptance else 'fail'}">{verdict}</div></div>
  </div>

  <h2>스윕 요약 그림</h2>
  <img src="{html.escape(relpath(figure, out))}" alt="FBM grouping parameter sweep">

  <h2>Cluster Count Sweep</h2>
  <table>
    <tr><th>K</th><th>내부 동시 배정</th><th>외부 동시 배정</th><th>차이</th><th>NN overlap</th><th>반복 cluster 수 평균</th><th>통과</th></tr>
    {table_rows(metrics['cluster_stability_sweep'], cluster_cols)}
  </table>

  <h2>Top-K Retrieval Sweep</h2>
  <table>
    <tr><th>Top K</th><th>Neighbor label Jaccard</th><th>Random label Jaccard</th><th>Lift</th></tr>
    {table_rows(metrics['retrieval_top_k_sweep'], topk_cols)}
  </table>

  <h2>Acceptance</h2>
  <table>
    <tr><th>항목</th><th>상태</th></tr>
    {acceptance_rows(metrics)}
  </table>

  <h2>산출물</h2>
  <ul>
    <li>Sweep Metrics JSON: <code>{html.escape(relpath(metrics_path, out))}</code></li>
    <li>Sweep Figure: <code>{html.escape(relpath(figure, out))}</code></li>
    <li>Input Feature CSV: <code>{html.escape(relpath(features, out))}</code></li>
  </ul>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    metrics_path = Path(args.metrics)
    figure = Path(args.figure)
    features = Path(args.features)
    metrics = run_sweep(args)
    save_figure(metrics, figure)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_report(metrics, figure, metrics_path, features, out), encoding="utf-8")
    print(f"Wrote parameter sweep report: {out}")
    print(f"Wrote parameter sweep metrics: {metrics_path}")
    print(f"Wrote parameter sweep figure: {figure}")


if __name__ == "__main__":
    main()
