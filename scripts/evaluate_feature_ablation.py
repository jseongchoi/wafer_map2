"""Evaluate feature-family ablation and render visual neighbor checks."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.data import load_sample
from wafermap.evaluation import nearest_neighbor_indices, standardize as standardize_matrix
from wafermap.features import compact_observable_feature_names


EXCLUDED_COLUMNS = {"sample_id", "actual_net_die", "cluster_id", "pca_0", "pca_1"}
FAMILY_RULES: dict[str, Callable[[str], bool]] = {
    "global": lambda name: name in {"total_fail_density", "grade_weighted_severity"},
    "stby": lambda name: name == "stby_ratio",
    "stby_spatial": lambda name: name.startswith("stby_polar_"),
    "radial": lambda name: name.startswith("radial_zone_"),
    "angular": lambda name: name.startswith("angular_sector_"),
    "polar_spatial": lambda name: name.startswith("polar_"),
    "edge": lambda name: name.startswith("edge_") or name == "center_density",
    "ring_scratch_morphology": lambda name: name.startswith("ring_") or name.startswith("scratch_"),
    "local_morphology": lambda name: name.startswith("local_"),
    "shot": lambda name: name.startswith("shot_"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="outputs/reports/fbm_grouping_features.csv")
    parser.add_argument("--data", default="data/synthetic/fbm_grouping_pilot")
    parser.add_argument("--out", default="outputs/reports/fbm_feature_ablation_report.html")
    parser.add_argument("--metrics", default="outputs/reports/fbm_feature_ablation_metrics.json")
    parser.add_argument("--ablation-figure", default="outputs/figures/fbm_feature_ablation.png")
    parser.add_argument("--neighbor-gallery", default="outputs/figures/fbm_neighbor_gallery.png")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--gallery-top-k", type=int, default=3)
    parser.add_argument("--gallery-queries", type=int, default=4)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def observable_feature_names(row: dict[str, str]) -> list[str]:
    return compact_observable_feature_names(row, extra_excluded=EXCLUDED_COLUMNS)


def label_columns(row: dict[str, str]) -> list[str]:
    return [key for key in row if key.startswith("label_")]


def standardize(x: np.ndarray) -> np.ndarray:
    return standardize_matrix(x)


def family_for_feature(name: str) -> str:
    for family, rule in FAMILY_RULES.items():
        if rule(name):
            return family
    return "other"


def family_map(feature_names: list[str]) -> dict[str, list[str]]:
    out = {family: [] for family in FAMILY_RULES}
    out["other"] = []
    for name in feature_names:
        out[family_for_feature(name)].append(name)
    return {key: value for key, value in out.items() if value}


def label_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.logical_or(left, right).sum())
    inter = int(np.logical_and(left, right).sum())
    return float(inter / union) if union else 1.0


def neighbor_indices(x: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    return nearest_neighbor_indices(x, top_k)


def retrieval_metrics(x: np.ndarray, y: np.ndarray, top_k: int) -> dict[str, Any]:
    neighbors, distances = neighbor_indices(x, top_k)
    pair_jaccard = []
    for idx in range(len(x)):
        query = y[idx].astype(bool)
        for nn in neighbors[idx]:
            pair_jaccard.append(label_jaccard(query, y[int(nn)].astype(bool)))

    random_jaccard = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            random_jaccard.append(label_jaccard(y[i].astype(bool), y[j].astype(bool)))

    by_class: dict[str, Any] = {}
    for class_idx in range(y.shape[1]):
        labels = y[:, class_idx].astype(bool)
        positives = np.where(labels)[0]
        if len(positives) == 0:
            continue
        precisions = [float(labels[neighbors[idx]].mean()) for idx in positives]
        base_rate = float(labels.mean())
        precision = float(np.mean(precisions))
        by_class[class_idx] = {
            "positive_queries": int(len(positives)),
            "precision_at_k": precision,
            "base_rate": base_rate,
            "lift": precision / max(base_rate, 1e-9),
        }

    mean_jaccard = float(np.mean(pair_jaccard)) if pair_jaccard else 0.0
    baseline_jaccard = float(np.mean(random_jaccard)) if random_jaccard else 0.0
    return {
        "top_k": int(neighbors.shape[1]),
        "mean_neighbor_label_jaccard": mean_jaccard,
        "random_pair_label_jaccard": baseline_jaccard,
        "jaccard_lift": mean_jaccard / max(baseline_jaccard, 1e-9),
        "by_class": by_class,
        "neighbors": neighbors,
        "distances": distances,
    }


def build_arrays(rows: list[dict[str, str]]) -> tuple[list[str], list[str], list[str], np.ndarray, np.ndarray]:
    feature_names = observable_feature_names(rows[0])
    labels = label_columns(rows[0])
    sample_ids = [row["sample_id"] for row in rows]
    x = np.array([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float32)
    y = np.array([[int(row[name]) for name in labels] for row in rows], dtype=np.int32)
    return sample_ids, feature_names, labels, x, y


def make_ablation_metrics(rows: list[dict[str, str]], top_k: int) -> dict[str, Any]:
    sample_ids, feature_names, label_cols, x, y = build_arrays(rows)
    families = family_map(feature_names)
    baseline = retrieval_metrics(x, y, top_k)
    baseline_public = public_retrieval_metrics(baseline, label_cols)

    ablations = []
    for family, family_features in families.items():
        keep_indices = [idx for idx, name in enumerate(feature_names) if name not in family_features]
        if not keep_indices:
            continue
        result = retrieval_metrics(x[:, keep_indices], y, top_k)
        public = public_retrieval_metrics(result, label_cols)
        class_lift_delta = {}
        for label_col in label_cols:
            label = label_col.replace("label_", "")
            base = baseline_public["by_class"].get(label, {}).get("lift", 0.0)
            ablated = public["by_class"].get(label, {}).get("lift", 0.0)
            class_lift_delta[label] = base - ablated
        ablations.append(
            {
                "removed_family": family,
                "removed_feature_count": len(family_features),
                "removed_features": family_features,
                "remaining_feature_count": len(keep_indices),
                "jaccard_lift": public["jaccard_lift"],
                "lift_delta_vs_baseline": baseline_public["jaccard_lift"] - public["jaccard_lift"],
                "mean_neighbor_label_jaccard": public["mean_neighbor_label_jaccard"],
                "by_class": public["by_class"],
                "class_lift_delta_vs_baseline": class_lift_delta,
            }
        )

    ablations.sort(key=lambda item: item["lift_delta_vs_baseline"], reverse=True)
    positive_drops = [item["lift_delta_vs_baseline"] for item in ablations if item["lift_delta_vs_baseline"] > 0]
    acceptance = {
        "observable_features_only": all(
            not name.startswith("label_") and not name.endswith("_mask_ratio") for name in feature_names
        ),
        "baseline_lift_above_1_10": baseline_public["jaccard_lift"] >= 1.10,
        "any_family_removal_reduces_lift": bool(positive_drops),
        "top_drop_above_0_03": bool(positive_drops) and max(positive_drops) >= 0.03,
    }
    return {
        "sample_count": len(rows),
        "feature_count_observable": len(feature_names),
        "label_columns": label_cols,
        "sample_ids": sample_ids,
        "feature_families": families,
        "baseline": baseline_public,
        "ablations": ablations,
        "acceptance": acceptance,
    }


def public_retrieval_metrics(metrics: dict[str, Any], label_cols: list[str]) -> dict[str, Any]:
    by_class = {}
    for idx, value in metrics["by_class"].items():
        label = label_cols[int(idx)].replace("label_", "")
        by_class[label] = value
    return {
        "top_k": metrics["top_k"],
        "mean_neighbor_label_jaccard": metrics["mean_neighbor_label_jaccard"],
        "random_pair_label_jaccard": metrics["random_pair_label_jaccard"],
        "jaccard_lift": metrics["jaccard_lift"],
        "by_class": by_class,
    }


def save_ablation_figure(metrics: dict[str, Any], path: Path) -> None:
    ablations = metrics["ablations"]
    families = [_short_family_label(item["removed_family"]) for item in ablations]
    deltas = [item["lift_delta_vs_baseline"] for item in ablations]
    lifts = [item["jaccard_lift"] for item in ablations]
    baseline_lift = metrics["baseline"]["jaccard_lift"]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = ["#2563eb" if value >= 0 else "#dc2626" for value in deltas]
    axes[0].bar(families, deltas, color=colors)
    axes[0].axhline(0.0, color="#111827", linewidth=0.8)
    axes[0].set_title("Lift Drop When Feature Family Is Removed")
    axes[0].set_ylabel("Baseline lift - ablated lift")
    axes[0].tick_params(axis="x", rotation=25, labelsize=9)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(families, lifts, color="#64748b")
    axes[1].axhline(baseline_lift, color="#2563eb", linestyle="--", label=f"baseline {baseline_lift:.2f}x")
    axes[1].axhline(1.10, color="#9ca3af", linestyle=":", label="1.10x threshold")
    axes[1].set_title("Ablated Top-K Retrieval Lift")
    axes[1].set_ylabel("Label-Jaccard lift")
    axes[1].tick_params(axis="x", rotation=25, labelsize=9)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _short_family_label(name: str) -> str:
    return {
        "ring_scratch_morphology": "ring/scratch",
        "local_morphology": "local morph",
    }.get(name, name)


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
    return labels


def choose_query_indices(x: np.ndarray, limit: int) -> list[int]:
    z = standardize(x)
    labels = kmeans(z, min(limit, 4), seed=17)
    chosen = []
    for cluster_id in sorted(set(int(v) for v in labels)):
        members = np.where(labels == cluster_id)[0]
        center = z[members].mean(axis=0, keepdims=True)
        distances = ((z[members] - center) ** 2).sum(axis=1)
        chosen.append(int(members[int(np.argmin(distances))]))
    return chosen[:limit]


def compact_labels(row: dict[str, str]) -> str:
    active = [key.replace("label_", "") for key, value in row.items() if key.startswith("label_") and int(value) == 1]
    return ", ".join(active) if active else "background"


def render_sample_image(sample_dir: Path) -> np.ndarray:
    sample = load_sample(sample_dir)
    values = np.clip(sample.severity.astype(np.float32), 0, 7) / 7.0
    image = plt.get_cmap("turbo")(values)
    image[(sample.wafer_mask == 0) | ((sample.severity == 0) & (sample.stby_mask == 0))] = (0.0, 0.0, 0.0, 1.0)
    image[sample.stby_mask > 0] = (1.0, 1.0, 1.0, 1.0)
    return image


def save_neighbor_gallery(
    rows: list[dict[str, str]],
    data_root: Path,
    top_k: int,
    query_count: int,
    path: Path,
) -> None:
    sample_ids, feature_names, _, x, y = build_arrays(rows)
    neighbors, distances = neighbor_indices(x, top_k)
    queries = choose_query_indices(x, query_count)
    cols = top_k + 1
    fig, axes = plt.subplots(len(queries), cols, figsize=(3.1 * cols, 3.25 * len(queries)), constrained_layout=True)
    axes = np.atleast_2d(axes)

    for row_pos, query_idx in enumerate(queries):
        indices = [query_idx, *[int(v) for v in neighbors[query_idx, :top_k]]]
        for col_pos, idx in enumerate(indices):
            ax = axes[row_pos, col_pos]
            image = render_sample_image(data_root / sample_ids[idx])
            ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            if col_pos == 0:
                title = f"QUERY\n{sample_ids[idx]}\n{compact_labels(rows[idx])}"
            else:
                jac = label_jaccard(y[query_idx].astype(bool), y[idx].astype(bool))
                title = f"NN {col_pos}\n{sample_ids[idx]}\nJ={jac:.2f}, d={distances[query_idx, idx]:.2f}"
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def relpath(target: Path, base_file: Path) -> str:
    return Path(os.path.relpath(Path(target).resolve(), base_file.resolve().parent)).as_posix()


def ablation_rows(metrics: dict[str, Any]) -> str:
    rows = []
    for item in metrics["ablations"]:
        class_delta = sorted(
            item["class_lift_delta_vs_baseline"].items(),
            key=lambda pair: pair[1],
            reverse=True,
        )[:3]
        class_text = ", ".join(f"{name}: {value:+.2f}" for name, value in class_delta)
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['removed_family'])}</td>"
            f"<td>{item['removed_feature_count']}</td>"
            f"<td>{item['jaccard_lift']:.4f}</td>"
            f"<td>{item['lift_delta_vs_baseline']:+.4f}</td>"
            f"<td>{html.escape(class_text)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def acceptance_rows(metrics: dict[str, Any]) -> str:
    labels = {
        "observable_features_only": "Observable feature만 사용",
        "baseline_lift_above_1_10": "Baseline 유사검색 lift 1.10 이상",
        "any_family_removal_reduces_lift": "제거 시 성능이 떨어지는 feature family 존재",
        "top_drop_above_0_03": "최대 lift drop 0.03 이상",
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


def html_report(
    metrics: dict[str, Any],
    ablation_figure: Path,
    neighbor_gallery: Path,
    metrics_path: Path,
    features: Path,
    out: Path,
) -> str:
    baseline = metrics["baseline"]
    all_acceptance = all(bool(v) for v in metrics["acceptance"].values())
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>FBM Feature Family Ablation 리포트</title>
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
  <h1>FBM Feature Family Ablation 리포트</h1>
  <p>이 검증은 observable feature를 family별로 제거하면서 유사 wafer 검색 품질이 어떻게 바뀌는지 확인한다. Synthetic label은 평가에만 사용하고 feature에는 넣지 않는다.</p>

  <div class="note">
    실무 적용 관점에서는 하나의 black-box embedding보다 feature family별 의미가 설명되는 쪽이 중요하다. 어떤 축을 제거했을 때 성능이 떨어지는지 보면, 이 feature가 실제 FBM 정보를 잡는지 판단할 수 있다.
  </div>

  <div class="summary">
    <div class="card"><div>Sample</div><div class="metric">{metrics['sample_count']}</div></div>
    <div class="card"><div>Observable Feature</div><div class="metric">{metrics['feature_count_observable']}</div></div>
    <div class="card"><div>Baseline Lift</div><div class="metric">{baseline['jaccard_lift']:.2f}x</div></div>
    <div class="card"><div>검증 상태</div><div class="metric {'pass' if all_acceptance else 'fail'}">{'PASS' if all_acceptance else 'CHECK'}</div></div>
  </div>

  <h2>Ablation 결과</h2>
  <img src="{html.escape(relpath(ablation_figure, out))}" alt="FBM feature ablation chart">
  <table>
    <tr><th>제거 family</th><th>제거 feature 수</th><th>제거 후 lift</th><th>baseline 대비 변화</th><th>class lift drop 상위</th></tr>
    {ablation_rows(metrics)}
  </table>

  <h2>Nearest Neighbor 육안 검증</h2>
  <p>각 행의 첫 번째 이미지는 query wafer이고, 오른쪽은 observable feature 거리 기준 top neighbor이다. J는 synthetic validation label Jaccard이며, 실제 운영에서는 이 label 없이 전문가 육안 리뷰와 score 해석으로 검증한다.</p>
  <img src="{html.escape(relpath(neighbor_gallery, out))}" alt="FBM nearest neighbor visual gallery">

  <h2>Acceptance</h2>
  <table>
    <tr><th>항목</th><th>상태</th></tr>
    {acceptance_rows(metrics)}
  </table>

  <h2>산출물</h2>
  <ul>
    <li>Ablation Metrics JSON: <code>{html.escape(relpath(metrics_path, out))}</code></li>
    <li>Ablation Figure: <code>{html.escape(relpath(ablation_figure, out))}</code></li>
    <li>Neighbor Gallery: <code>{html.escape(relpath(neighbor_gallery, out))}</code></li>
    <li>Input Feature CSV: <code>{html.escape(relpath(features, out))}</code></li>
  </ul>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    features = Path(args.features)
    data_root = Path(args.data)
    out = Path(args.out)
    metrics_path = Path(args.metrics)
    ablation_figure = Path(args.ablation_figure)
    neighbor_gallery = Path(args.neighbor_gallery)
    rows = load_rows(features)
    if not rows:
        raise SystemExit(f"No rows found in {features}")

    metrics = make_ablation_metrics(rows, args.top_k)
    save_ablation_figure(metrics, ablation_figure)
    save_neighbor_gallery(rows, data_root, args.gallery_top_k, args.gallery_queries, neighbor_gallery)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        html_report(metrics, ablation_figure, neighbor_gallery, metrics_path, features, out),
        encoding="utf-8",
    )
    print(f"Wrote ablation report: {out}")
    print(f"Wrote ablation metrics: {metrics_path}")
    print(f"Wrote ablation figure: {ablation_figure}")
    print(f"Wrote neighbor gallery: {neighbor_gallery}")


if __name__ == "__main__":
    main()
