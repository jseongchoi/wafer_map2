"""Analyze FBM observable features for wafer grouping and similarity search."""

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

from wafermap.data import PATTERN_CLASSES, SyntheticSample, load_sample
from wafermap.evaluation import fit_standardizer, nearest_neighbor_indices
from wafermap.features import compact_observable_feature_names, extract_feature_vector

EVALUATED_CLASSES = ("scratch", "ring", "edge", "local", "shot_grid", "stby_pattern")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/synthetic/fbm_grouping_pilot")
    parser.add_argument("--out", default="outputs/reports/fbm_grouping_report.html")
    parser.add_argument("--metrics", default="outputs/reports/fbm_grouping_metrics.json")
    parser.add_argument("--features", default="outputs/reports/fbm_grouping_features.csv")
    parser.add_argument("--neighbors", default="outputs/reports/fbm_grouping_neighbors.csv")
    parser.add_argument("--figure", default="outputs/figures/fbm_grouping_pca.png")
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def sample_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("synth_*") if path.is_dir())


def class_labels(sample: SyntheticSample) -> dict[str, int]:
    out: dict[str, int] = {}
    for name in EVALUATED_CLASSES:
        idx = PATTERN_CLASSES.index(name)
        out[name] = int(sample.pattern_masks[idx].sum() > 0)
    return out


def observable_feature_names(row: dict[str, Any]) -> list[str]:
    return compact_observable_feature_names(row)


def build_rows(samples: list[SyntheticSample]) -> tuple[list[dict[str, Any]], list[str], np.ndarray, np.ndarray]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        labels = class_labels(sample)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "actual_net_die": sample.metadata["actual_net_die"],
                **{f"label_{name}": value for name, value in labels.items()},
                **extract_feature_vector(sample),
            }
        )
    feature_names = observable_feature_names(rows[0])
    x = np.array([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float32)
    y = np.array([[int(row[f"label_{name}"]) for name in EVALUATED_CLASSES] for row in rows], dtype=np.int32)
    return rows, feature_names, x, y


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, sigma = fit_standardizer(x)
    z = (x - mu) / sigma
    return z, mu.ravel(), sigma.ravel()


def pca_2d(z: np.ndarray) -> tuple[np.ndarray, list[float]]:
    centered = z - z.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    variance = s**2
    total = float(variance.sum())
    explained = [float(v / total) if total else 0.0 for v in variance[:2]]
    return coords.astype(np.float32), explained


def kmeans(z: np.ndarray, clusters: int, seed: int, iterations: int = 80) -> tuple[np.ndarray, np.ndarray]:
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
    return labels, centers


def nearest_neighbors(z: np.ndarray, y: np.ndarray, sample_ids: list[str], top_k: int) -> tuple[list[dict[str, Any]], dict[str, float]]:
    neighbors, distances = nearest_neighbor_indices(z, top_k, standardize_input=False)
    k = neighbors.shape[1]
    rows: list[dict[str, Any]] = []
    jaccards = []
    for idx, sample_id in enumerate(sample_ids):
        query = y[idx].astype(bool)
        for rank, nn in enumerate(neighbors[idx], start=1):
            target = y[nn].astype(bool)
            union = int(np.logical_or(query, target).sum())
            inter = int(np.logical_and(query, target).sum())
            jaccard = float(inter / union) if union else 1.0
            jaccards.append(jaccard)
            rows.append(
                {
                    "sample_id": sample_id,
                    "rank": rank,
                    "neighbor_id": sample_ids[int(nn)],
                    "distance": float(distances[idx, nn]),
                    "label_jaccard_validation_only": jaccard,
                }
            )
    random_jaccards = []
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            left = y[i].astype(bool)
            right = y[j].astype(bool)
            union = int(np.logical_or(left, right).sum())
            inter = int(np.logical_and(left, right).sum())
            random_jaccards.append(float(inter / union) if union else 1.0)
    summary = {
        "top_k": float(k),
        "mean_neighbor_label_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
        "random_pair_label_jaccard": float(np.mean(random_jaccards)) if random_jaccards else 0.0,
    }
    summary["jaccard_lift"] = summary["mean_neighbor_label_jaccard"] / max(
        summary["random_pair_label_jaccard"], 1e-9
    )
    return rows, summary


def cluster_summaries(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    z: np.ndarray,
    y: np.ndarray,
    cluster_labels: np.ndarray,
    centers: np.ndarray,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    sample_ids = [str(row["sample_id"]) for row in rows]
    for cluster_id in sorted(set(int(v) for v in cluster_labels)):
        member_idx = np.where(cluster_labels == cluster_id)[0]
        cluster_z = z[member_idx]
        mean_z = cluster_z.mean(axis=0)
        top_order = np.argsort(np.abs(mean_z))[::-1][:6]
        top_features = [
            {"feature": feature_names[int(idx)], "mean_z": float(mean_z[int(idx)])}
            for idx in top_order
        ]
        label_rates = {
            name: float(y[member_idx, class_idx].mean())
            for class_idx, name in enumerate(EVALUATED_CLASSES)
        }
        distances = np.sqrt(((z[member_idx] - centers[cluster_id]) ** 2).sum(axis=1))
        representative_order = member_idx[np.argsort(distances)[:5]]
        summaries.append(
            {
                "cluster_id": cluster_id,
                "size": int(len(member_idx)),
                "representative_samples": [sample_ids[int(idx)] for idx in representative_order],
                "top_feature_deviations": top_features,
                "validation_label_rates": label_rates,
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_pca_figure(coords: np.ndarray, labels: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=42, edgecolor="#111827", linewidth=0.35)
    ax.set_title("FBM Observable Feature PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.22)
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def feature_label(item: dict[str, Any]) -> str:
    return f"{item['feature']} ({item['mean_z']:+.2f}z)"


def html_cluster_rows(summaries: list[dict[str, Any]]) -> str:
    rows = []
    for summary in summaries:
        labels = ", ".join(
            f"{name}: {value:.2f}" for name, value in summary["validation_label_rates"].items()
        )
        features = ", ".join(feature_label(item) for item in summary["top_feature_deviations"])
        reps = ", ".join(summary["representative_samples"])
        rows.append(
            "<tr>"
            f"<td>{summary['cluster_id']}</td>"
            f"<td>{summary['size']}</td>"
            f"<td>{html.escape(features)}</td>"
            f"<td>{html.escape(labels)}</td>"
            f"<td>{html.escape(reps)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def relpath(target: Path, base_file: Path) -> str:
    return Path(os.path.relpath(Path(target).resolve(), base_file.resolve().parent)).as_posix()


def html_report(metrics: dict[str, Any], figure: Path, features: Path, neighbors: Path, metrics_path: Path, out: Path) -> str:
    summaries = metrics["cluster_summaries"]
    retrieval = metrics["similarity"]
    all_acceptance = all(metrics["acceptance"].values())
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>FBM 정보 추출과 유사 패턴 그룹핑 리포트</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ color: #111827; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; margin: 18px 0; }}
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
  <h1>FBM 정보 추출과 유사 패턴 그룹핑 리포트</h1>
  <p>이 리포트는 Fail Bit Map 자체에서 계산 가능한 observable feature만 사용해 wafer를 그룹핑하고, 유사 wafer 검색이 불량 패턴 관점에서 의미가 있는지 검증한다. Synthetic label은 검증용 요약에만 사용한다.</p>

  <div class="note">
    현재 1차 목표는 공정데이터 조인 전의 FBM 정보 추출이다. 이 리포트는 유사한 불량 패턴 wafer를 묶고, 각 그룹이 어떤 feature 특성을 갖는지 설명하는 데 초점을 둔다.
  </div>

  <div class="summary">
    <div class="card"><div>샘플 수</div><div class="metric">{metrics['sample_count']}</div></div>
    <div class="card"><div>클러스터 수</div><div class="metric">{metrics['cluster_count']}</div></div>
    <div class="card"><div>유사검색 Lift</div><div class="metric">{retrieval['jaccard_lift']:.2f}x</div></div>
    <div class="card"><div>검증 상태</div><div class="metric {'pass' if all_acceptance else 'fail'}">{'PASS' if all_acceptance else 'CHECK'}</div></div>
  </div>

  <h2>PCA View</h2>
  <p>PCA는 observable feature vector를 2차원으로 투영한 것이다. 색상은 k-means cluster이며, 실제 운영에서는 label 없이 이 구조를 보고 유사 wafer를 탐색한다.</p>
  <img src="{html.escape(relpath(figure, out))}" alt="FBM grouping PCA">

  <h2>Cluster Summary</h2>
  <table>
    <tr><th>Cluster</th><th>Size</th><th>Top Feature Deviation</th><th>Validation Label Rate</th><th>Representative Samples</th></tr>
    {html_cluster_rows(summaries)}
  </table>

  <h2>Similarity Search Summary</h2>
  <table>
    <tr><th>지표</th><th>값</th></tr>
    <tr><td>Top K</td><td>{int(retrieval['top_k'])}</td></tr>
    <tr><td>가까운 wafer 간 평균 label Jaccard</td><td>{retrieval['mean_neighbor_label_jaccard']:.4f}</td></tr>
    <tr><td>임의 pair 평균 label Jaccard</td><td>{retrieval['random_pair_label_jaccard']:.4f}</td></tr>
    <tr><td>Jaccard lift</td><td>{retrieval['jaccard_lift']:.4f}</td></tr>
  </table>

  <h2>Acceptance</h2>
  <table>
    <tr><th>항목</th><th>상태</th></tr>
    <tr><td>Observable feature만 사용</td><td class="{'pass' if metrics['acceptance']['observable_features_only'] else 'fail'}">{'PASS' if metrics['acceptance']['observable_features_only'] else 'CHECK'}</td></tr>
    <tr><td>2개 이상 cluster 생성</td><td class="{'pass' if metrics['acceptance']['multiple_clusters'] else 'fail'}">{'PASS' if metrics['acceptance']['multiple_clusters'] else 'CHECK'}</td></tr>
    <tr><td>유사검색 lift 1.10 이상</td><td class="{'pass' if metrics['acceptance']['similarity_lift_above_1_10'] else 'fail'}">{'PASS' if metrics['acceptance']['similarity_lift_above_1_10'] else 'CHECK'}</td></tr>
  </table>

  <h2>산출물</h2>
  <ul>
    <li>Feature CSV: <code>{html.escape(relpath(features, out))}</code></li>
    <li>Neighbor CSV: <code>{html.escape(relpath(neighbors, out))}</code></li>
    <li>Metrics JSON: <code>{html.escape(relpath(metrics_path, out))}</code></li>
    <li>PCA Figure: <code>{html.escape(relpath(figure, out))}</code></li>
  </ul>
</body>
</html>
"""


def make_metrics(
    samples: list[SyntheticSample],
    clusters: int,
    top_k: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows, feature_names, x, y = build_rows(samples)
    z, _, _ = standardize(x)
    coords, explained = pca_2d(z)
    cluster_labels, centers = kmeans(z, clusters, seed)
    sample_ids = [str(row["sample_id"]) for row in rows]
    neighbor_rows, similarity = nearest_neighbors(z, y, sample_ids, top_k)
    summaries = cluster_summaries(rows, feature_names, z, y, cluster_labels, centers)

    enriched_rows = []
    for idx, row in enumerate(rows):
        enriched = dict(row)
        enriched["cluster_id"] = int(cluster_labels[idx])
        enriched["pca_0"] = float(coords[idx, 0])
        enriched["pca_1"] = float(coords[idx, 1])
        enriched_rows.append(enriched)

    metrics = {
        "sample_count": len(samples),
        "feature_count_observable": len(feature_names),
        "cluster_count": int(len(set(cluster_labels.tolist()))),
        "pca_explained_variance": explained,
        "feature_names": feature_names,
        "cluster_summaries": summaries,
        "similarity": similarity,
        "acceptance": {
            "observable_features_only": all(
                not name.startswith("label_") and not name.endswith("_mask_ratio")
                for name in feature_names
            ),
            "multiple_clusters": len(set(cluster_labels.tolist())) >= 2,
            "similarity_lift_above_1_10": similarity["jaccard_lift"] >= 1.10,
        },
    }
    return metrics, enriched_rows, neighbor_rows


def main() -> None:
    args = parse_args()
    data_root = Path(args.data)
    out = Path(args.out)
    metrics_path = Path(args.metrics)
    features = Path(args.features)
    neighbors = Path(args.neighbors)
    figure = Path(args.figure)
    samples = [load_sample(path) for path in sample_dirs(data_root)]
    if not samples:
        raise SystemExit(f"No samples found under {data_root}")

    metrics, rows, neighbor_rows = make_metrics(samples, args.clusters, args.top_k, args.seed)
    coords = np.array([[row["pca_0"], row["pca_1"]] for row in rows], dtype=np.float32)
    labels = np.array([row["cluster_id"] for row in rows], dtype=np.int32)
    write_csv(features, rows)
    write_csv(neighbors, neighbor_rows)
    save_pca_figure(coords, labels, figure)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_report(metrics, figure, features, neighbors, metrics_path, out), encoding="utf-8")
    print(f"Wrote grouping report: {out}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote features: {features}")
    print(f"Wrote neighbors: {neighbors}")
    print(f"Wrote figure: {figure}")


if __name__ == "__main__":
    main()
