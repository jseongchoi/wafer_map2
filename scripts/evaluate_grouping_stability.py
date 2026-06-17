"""Evaluate CPU-only stability of FBM feature grouping."""

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

from wafermap.evaluation import fit_standardizer, nearest_neighbor_indices
from wafermap.features import compact_observable_feature_names

EXCLUDED_COLUMNS = {"sample_id", "actual_net_die", "cluster_id", "pca_0", "pca_1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="outputs/reports/fbm_grouping_features.csv")
    parser.add_argument("--out", default="outputs/reports/fbm_grouping_stability_report.html")
    parser.add_argument("--metrics", default="outputs/reports/fbm_grouping_stability_metrics.json")
    parser.add_argument("--figure", default="outputs/figures/fbm_grouping_coassociation.png")
    parser.add_argument("--runs", type=int, default=64)
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--feature-rate", type=float, default=0.85)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=29)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def observable_feature_names(row: dict[str, str]) -> list[str]:
    return compact_observable_feature_names(row, extra_excluded=EXCLUDED_COLUMNS)


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, sigma = fit_standardizer(x)
    z = (x - mu) / sigma
    return z.astype(np.float32), mu.ravel(), sigma.ravel()


def kmeans(z: np.ndarray, clusters: int, seed: int, iterations: int = 80) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = min(max(1, clusters), len(z))
    centers = z[rng.choice(len(z), size=k, replace=False)].copy()
    labels = np.zeros(len(z), dtype=np.int32)
    for _ in range(iterations):
        distances = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(distances, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for idx in range(k):
            members = z[labels == idx]
            if len(members):
                centers[idx] = members.mean(axis=0)
            else:
                centers[idx] = z[rng.integers(0, len(z))]
    return labels


def nearest_neighbor_sets(z: np.ndarray, top_k: int) -> list[set[int]]:
    order, _ = nearest_neighbor_indices(z, top_k, standardize_input=False)
    return [set(int(v) for v in row) for row in order]


def mean_neighbor_overlap(left: list[set[int]], right: list[set[int]]) -> float:
    values = []
    for a, b in zip(left, right, strict=True):
        denom = max(len(a), 1)
        values.append(len(a & b) / denom)
    return float(np.mean(values)) if values else 0.0


def pair_mean(matrix: np.ndarray, mask: np.ndarray) -> float:
    values = matrix[mask]
    return float(values.mean()) if values.size else 0.0


def cluster_stability_rows(
    sample_ids: list[str],
    baseline_labels: np.ndarray,
    coassociation: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for cluster_id in sorted(int(v) for v in set(baseline_labels.tolist())):
        members = np.where(baseline_labels == cluster_id)[0]
        if len(members) == 1:
            internal = 1.0
            fragile_pairs: list[str] = []
        else:
            block = coassociation[np.ix_(members, members)]
            internal_mask = ~np.eye(len(members), dtype=bool)
            internal = float(block[internal_mask].mean()) if internal_mask.any() else 1.0
            pair_scores = []
            for a_pos, a in enumerate(members):
                for b in members[a_pos + 1 :]:
                    pair_scores.append((float(coassociation[a, b]), sample_ids[int(a)], sample_ids[int(b)]))
            pair_scores.sort(key=lambda item: item[0])
            fragile_pairs = [f"{left}-{right}: {score:.2f}" for score, left, right in pair_scores[:3]]
        rows.append(
            {
                "cluster_id": cluster_id,
                "size": int(len(members)),
                "internal_coassociation": internal,
                "fragile_pairs": fragile_pairs,
            }
        )
    return rows


def run_stability(
    rows: list[dict[str, str]],
    runs: int,
    clusters: int,
    top_k: int,
    feature_rate: float,
    noise_std: float,
    seed: int,
) -> dict[str, Any]:
    feature_names = observable_feature_names(rows[0])
    sample_ids = [row["sample_id"] for row in rows]
    x = np.array([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float32)
    z, _, _ = standardize(x)
    if "cluster_id" in rows[0] and rows[0]["cluster_id"] != "":
        baseline_labels = np.array([int(row["cluster_id"]) for row in rows], dtype=np.int32)
    else:
        baseline_labels = kmeans(z, clusters, seed)

    rng = np.random.default_rng(seed)
    n_samples, n_features = z.shape
    selected_count = max(2, min(n_features, int(round(n_features * feature_rate))))
    coassociation = np.zeros((n_samples, n_samples), dtype=np.float32)
    neighbor_overlaps = []
    unique_cluster_counts = []
    baseline_neighbors = nearest_neighbor_sets(z, top_k)

    for run_idx in range(runs):
        selected = np.sort(rng.choice(n_features, size=selected_count, replace=False))
        z_run = z[:, selected].copy()
        if noise_std > 0:
            z_run += rng.normal(0.0, noise_std, size=z_run.shape).astype(np.float32)
        run_labels = kmeans(z_run, clusters, seed + 1009 + run_idx)
        same = run_labels[:, None] == run_labels[None, :]
        coassociation += same.astype(np.float32)
        neighbor_overlaps.append(mean_neighbor_overlap(baseline_neighbors, nearest_neighbor_sets(z_run, top_k)))
        unique_cluster_counts.append(int(len(set(run_labels.tolist()))))

    coassociation /= max(runs, 1)
    off_diag = ~np.eye(n_samples, dtype=bool)
    same_baseline = (baseline_labels[:, None] == baseline_labels[None, :]) & off_diag
    diff_baseline = (baseline_labels[:, None] != baseline_labels[None, :]) & off_diag
    same_mean = pair_mean(coassociation, same_baseline)
    diff_mean = pair_mean(coassociation, diff_baseline)
    separation = same_mean - diff_mean
    neighbor_overlap = float(np.mean(neighbor_overlaps)) if neighbor_overlaps else 0.0
    cluster_rows = cluster_stability_rows(sample_ids, baseline_labels, coassociation)

    acceptance = {
        "cpu_only_validation": True,
        "sample_count_at_least_30": n_samples >= 30,
        "same_cluster_coassociation_above_0_55": same_mean >= 0.55,
        "coassociation_separation_above_0_20": separation >= 0.20,
        "neighbor_overlap_above_0_30": neighbor_overlap >= 0.30,
    }

    return {
        "sample_count": n_samples,
        "feature_count_observable": n_features,
        "baseline_cluster_count": int(len(set(baseline_labels.tolist()))),
        "runs": runs,
        "clusters_requested": clusters,
        "top_k": top_k,
        "feature_subsample_rate": feature_rate,
        "selected_feature_count_per_run": selected_count,
        "noise_std": noise_std,
        "same_cluster_mean_coassociation": same_mean,
        "different_cluster_mean_coassociation": diff_mean,
        "coassociation_separation": separation,
        "mean_nearest_neighbor_overlap": neighbor_overlap,
        "run_unique_cluster_count_mean": float(np.mean(unique_cluster_counts)) if unique_cluster_counts else 0.0,
        "run_unique_cluster_count_min": int(min(unique_cluster_counts)) if unique_cluster_counts else 0,
        "run_unique_cluster_count_max": int(max(unique_cluster_counts)) if unique_cluster_counts else 0,
        "cluster_stability": cluster_rows,
        "baseline_labels": baseline_labels.tolist(),
        "sample_ids": sample_ids,
        "feature_names": feature_names,
        "acceptance": acceptance,
        "_coassociation": coassociation.tolist(),
    }


def relpath(target: Path, base_file: Path) -> str:
    return Path(os.path.relpath(Path(target).resolve(), base_file.resolve().parent)).as_posix()


def save_heatmap(metrics: dict[str, Any], path: Path) -> None:
    coassociation = np.array(metrics["_coassociation"], dtype=np.float32)
    labels = np.array(metrics["baseline_labels"], dtype=np.int32)
    sample_ids = metrics["sample_ids"]
    order = np.lexsort((np.arange(len(labels)), labels))
    sorted_matrix = coassociation[np.ix_(order, order)]
    sorted_ids = [sample_ids[int(idx)] for idx in order]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(sorted_matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title("FBM Grouping Co-association Stability")
    ax.set_xlabel("Samples sorted by baseline cluster")
    ax.set_ylabel("Samples sorted by baseline cluster")
    if len(sorted_ids) <= 48:
        ax.set_xticks(np.arange(len(sorted_ids)))
        ax.set_yticks(np.arange(len(sorted_ids)))
        ax.set_xticklabels(sorted_ids, rotation=90, fontsize=6)
        ax.set_yticklabels(sorted_ids, fontsize=6)
    fig.colorbar(im, ax=ax, label="Same-cluster frequency")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def acceptance_rows(metrics: dict[str, Any]) -> str:
    labels = {
        "cpu_only_validation": "GPU 없이 CPU-only 검증으로 수행",
        "sample_count_at_least_30": "pilot sample 30개 이상",
        "same_cluster_coassociation_above_0_55": "기준 cluster 내부 동시 배정 평균 0.55 이상",
        "coassociation_separation_above_0_20": "내부/외부 동시 배정 차이 0.20 이상",
        "neighbor_overlap_above_0_30": "최근접 wafer overlap 평균 0.30 이상",
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


def cluster_rows(metrics: dict[str, Any]) -> str:
    rows = []
    for item in metrics["cluster_stability"]:
        fragile = ", ".join(item["fragile_pairs"]) if item["fragile_pairs"] else "-"
        rows.append(
            "<tr>"
            f"<td>{item['cluster_id']}</td>"
            f"<td>{item['size']}</td>"
            f"<td>{item['internal_coassociation']:.3f}</td>"
            f"<td>{html.escape(fragile)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def html_report(metrics: dict[str, Any], figure: Path, metrics_path: Path, features: Path, out: Path) -> str:
    all_acceptance = all(bool(v) for v in metrics["acceptance"].values())
    verdict = "PASS" if all_acceptance else "CHECK"
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>FBM 그룹핑 안정성 검증 리포트</title>
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
  <h1>FBM 그룹핑 안정성 검증 리포트</h1>
  <p>이 리포트는 기존 <code>fbm_grouping_features.csv</code>의 observable feature만 사용한다. Synthetic mask나 label은 feature에 넣지 않았고, GPU 없이 CPU에서 반복 실험으로 안정성을 확인했다.</p>

  <div class="note">
    검증 목적은 단순 정확도 하나가 아니라, feature 일부를 빼거나 작은 노이즈를 줘도 비슷한 wafer들이 계속 비슷하게 묶이는지 보는 것이다. 이 기준을 통과하면 다음 단계에서 real unlabeled wafer에도 feature extraction과 nearest-neighbor review를 먼저 적용해볼 근거가 생긴다.
  </div>

  <div class="summary">
    <div class="card"><div>Sample</div><div class="metric">{metrics['sample_count']}</div></div>
    <div class="card"><div>반복 실험</div><div class="metric">{metrics['runs']}</div></div>
    <div class="card"><div>동시 배정 차이</div><div class="metric">{metrics['coassociation_separation']:.2f}</div></div>
    <div class="card"><div>검증 상태</div><div class="metric {'pass' if all_acceptance else 'fail'}">{verdict}</div></div>
  </div>

  <h2>핵심 지표</h2>
  <table>
    <tr><th>지표</th><th>값</th><th>해석</th></tr>
    <tr><td>기준 cluster 내부 동시 배정 평균</td><td>{metrics['same_cluster_mean_coassociation']:.4f}</td><td>기존 같은 cluster wafer들이 반복 실험에서도 같이 묶이는 빈도</td></tr>
    <tr><td>기준 cluster 외부 동시 배정 평균</td><td>{metrics['different_cluster_mean_coassociation']:.4f}</td><td>기존 다른 cluster wafer들이 우연히 같이 묶이는 빈도</td></tr>
    <tr><td>내부/외부 동시 배정 차이</td><td>{metrics['coassociation_separation']:.4f}</td><td>클수록 그룹 경계가 안정적임</td></tr>
    <tr><td>최근접 wafer overlap 평균</td><td>{metrics['mean_nearest_neighbor_overlap']:.4f}</td><td>feature subset/noise 이후에도 가까운 wafer 목록이 유지되는 정도</td></tr>
    <tr><td>반복별 cluster 수 평균</td><td>{metrics['run_unique_cluster_count_mean']:.2f}</td><td>k-means가 비어 있는 cluster 없이 구조를 유지하는지 확인</td></tr>
  </table>

  <h2>Co-association Heatmap</h2>
  <p>밝은 블록은 반복 실험에서 계속 같은 cluster로 묶인 wafer 묶음이다. 대각선 주변의 블록이 선명하고 외부 영역이 어두울수록 안정성이 좋다.</p>
  <img src="{html.escape(relpath(figure, out))}" alt="FBM grouping coassociation heatmap">

  <h2>Cluster별 안정성</h2>
  <table>
    <tr><th>Cluster</th><th>Size</th><th>내부 동시 배정 평균</th><th>가장 흔들리는 pair</th></tr>
    {cluster_rows(metrics)}
  </table>

  <h2>Acceptance</h2>
  <table>
    <tr><th>항목</th><th>상태</th></tr>
    {acceptance_rows(metrics)}
  </table>

  <h2>산출물</h2>
  <ul>
    <li>Stability Metrics JSON: <code>{html.escape(relpath(metrics_path, out))}</code></li>
    <li>Co-association Figure: <code>{html.escape(relpath(figure, out))}</code></li>
    <li>Input Feature CSV: <code>{html.escape(relpath(features, out))}</code></li>
  </ul>
</body>
</html>
"""


def public_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if not key.startswith("_")}


def main() -> None:
    args = parse_args()
    features = Path(args.features)
    out = Path(args.out)
    metrics_path = Path(args.metrics)
    figure = Path(args.figure)
    rows = load_rows(features)
    if not rows:
        raise SystemExit(f"No rows found in {features}")

    metrics = run_stability(
        rows=rows,
        runs=args.runs,
        clusters=args.clusters,
        top_k=args.top_k,
        feature_rate=args.feature_rate,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    save_heatmap(metrics, figure)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(public_metrics(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_report(metrics, figure, metrics_path, features, out), encoding="utf-8")
    print(f"Wrote stability report: {out}")
    print(f"Wrote stability metrics: {metrics_path}")
    print(f"Wrote co-association figure: {figure}")


if __name__ == "__main__":
    main()
