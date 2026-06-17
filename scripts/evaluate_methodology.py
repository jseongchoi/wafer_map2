"""Evaluate whether observable wafer features support the target methodology."""

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

from wafermap.data import PATTERN_CLASSES, SyntheticSample, load_sample
from wafermap.evaluation import nearest_neighbor_indices, standardize as standardize_matrix
from wafermap.features import compact_observable_feature_names, extract_feature_vector

ORACLE_SUFFIX = "_mask_ratio"
EVALUATED_CLASSES = ("scratch", "ring", "edge", "local", "shot_grid", "stby_pattern")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/synthetic/methodology_probe")
    parser.add_argument("--out", default="outputs/reports/methodology_validation_report.html")
    parser.add_argument("--metrics", default="outputs/reports/methodology_validation_metrics.json")
    parser.add_argument("--features", default="outputs/reports/methodology_validation_features.csv")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def sample_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("synth_*") if path.is_dir())


def class_labels(sample: SyntheticSample) -> dict[str, int]:
    labels: dict[str, int] = {}
    for name in EVALUATED_CLASSES:
        idx = PATTERN_CLASSES.index(name)
        labels[name] = int(sample.pattern_masks[idx].sum() > 0)
    return labels


def observable_feature_names(rows: list[dict[str, float]]) -> list[str]:
    return compact_observable_feature_names(rows)


def build_table(samples: list[SyntheticSample]) -> tuple[list[dict[str, Any]], list[str], np.ndarray, np.ndarray]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        labels = class_labels(sample)
        row: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "actual_net_die": sample.metadata["actual_net_die"],
            **{f"label_{key}": value for key, value in labels.items()},
            **extract_feature_vector(sample),
        }
        rows.append(row)

    feature_names = observable_feature_names(rows)
    x = np.array([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float32)
    y = np.array([[int(row[f"label_{name}"]) for name in EVALUATED_CLASSES] for row in rows], dtype=np.int32)
    return rows, feature_names, x, y


def write_feature_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def standardize(x: np.ndarray) -> np.ndarray:
    return standardize_matrix(x)


def univariate_discriminability_score(values: np.ndarray, labels: np.ndarray) -> float:
    pos = values[labels == 1]
    neg = values[labels == 0]
    if len(pos) < 2 or len(neg) < 2:
        return 0.0
    grand = float(values.mean())
    between = len(pos) * (float(pos.mean()) - grand) ** 2 + len(neg) * (float(neg.mean()) - grand) ** 2
    within = float(((pos - pos.mean()) ** 2).sum() + ((neg - neg.mean()) ** 2).sum())
    return float(between / max(within / max(len(values) - 2, 1), 1e-9))


def roc_auc(values: np.ndarray, labels: np.ndarray) -> float:
    pos = values[labels == 1]
    neg = values[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    wins = 0.0
    total = 0
    for p_value in pos:
        wins += float((p_value > neg).sum())
        wins += 0.5 * float((p_value == neg).sum())
        total += len(neg)
    return float(wins / max(total, 1))


def discriminability(feature_names: list[str], x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    by_class: dict[str, Any] = {}
    for class_idx, class_name in enumerate(EVALUATED_CLASSES):
        labels = y[:, class_idx]
        positives = int(labels.sum())
        negatives = int(len(labels) - positives)
        if positives < 2 or negatives < 2:
            by_class[class_name] = {
                "status": "skipped",
                "positives": positives,
                "negatives": negatives,
                "reason": "Need at least 2 positive and 2 negative samples.",
            }
            continue

        scores = []
        for feature_idx, feature_name in enumerate(feature_names):
            values = x[:, feature_idx]
            auc = roc_auc(values, labels)
            auc_direction = "high" if auc >= 0.5 else "low"
            scores.append(
                {
                    "feature": feature_name,
                    "discriminability_score": univariate_discriminability_score(values, labels),
                    "auc": max(auc, 1.0 - auc),
                    "direction": auc_direction,
                }
            )
        scores.sort(key=lambda item: (item["auc"], item["discriminability_score"]), reverse=True)
        by_class[class_name] = {
            "status": "evaluated",
            "positives": positives,
            "negatives": negatives,
            "top_features": scores[:8],
        }
    return by_class


def retrieval_metrics(x: np.ndarray, y: np.ndarray, top_k: int) -> dict[str, Any]:
    neighbors, _ = nearest_neighbor_indices(x, top_k)
    k = neighbors.shape[1]
    pair_jaccard = []
    for idx in range(len(x)):
        query = y[idx].astype(bool)
        for nn in neighbors[idx]:
            target = y[nn].astype(bool)
            union = np.logical_or(query, target).sum()
            inter = np.logical_and(query, target).sum()
            pair_jaccard.append(float(inter / union) if union else 1.0)

    random_jaccard = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            left = y[i].astype(bool)
            right = y[j].astype(bool)
            union = np.logical_or(left, right).sum()
            inter = np.logical_and(left, right).sum()
            random_jaccard.append(float(inter / union) if union else 1.0)

    by_class: dict[str, Any] = {}
    for class_idx, class_name in enumerate(EVALUATED_CLASSES):
        labels = y[:, class_idx].astype(bool)
        positives = np.where(labels)[0]
        if len(positives) == 0:
            continue
        precisions = []
        for idx in positives:
            precisions.append(float(labels[neighbors[idx]].mean()))
        base_rate = float(labels.mean())
        precision = float(np.mean(precisions))
        by_class[class_name] = {
            "positive_queries": int(len(positives)),
            "precision_at_k": precision,
            "base_rate": base_rate,
            "lift": precision / max(base_rate, 1e-9),
        }

    mean_jaccard = float(np.mean(pair_jaccard)) if pair_jaccard else 0.0
    baseline_jaccard = float(np.mean(random_jaccard)) if random_jaccard else 0.0
    return {
        "top_k": k,
        "mean_neighbor_label_jaccard": mean_jaccard,
        "random_pair_label_jaccard": baseline_jaccard,
        "jaccard_lift": mean_jaccard / max(baseline_jaccard, 1e-9),
        "by_class": by_class,
    }


def make_metrics(samples: list[SyntheticSample], top_k: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows, feature_names, x, y = build_table(samples)
    disc = discriminability(feature_names, x, y)
    retrieval = retrieval_metrics(x, y, top_k)
    evaluated_classes = [
        key for key, value in disc.items() if value.get("status") == "evaluated"
    ]
    top_auc_values = [
        value["top_features"][0]["auc"]
        for value in disc.values()
        if value.get("status") == "evaluated" and value.get("top_features")
    ]
    acceptance = {
        "has_samples": len(samples) >= 8,
        "uses_observable_features_only": all(
            not name.startswith("label_") and not name.endswith(ORACLE_SUFFIX)
            for name in feature_names
        ),
        "at_least_three_classes_evaluable": len(evaluated_classes) >= 3,
        "retrieval_jaccard_lift_above_1_10": retrieval["jaccard_lift"] >= 1.10,
        "any_class_auc_above_0_70": bool(top_auc_values) and max(top_auc_values) >= 0.70,
    }
    metrics = {
        "sample_count": len(samples),
        "feature_count_observable": len(feature_names),
        "evaluated_classes": evaluated_classes,
        "discriminability": disc,
        "retrieval": retrieval,
        "acceptance": acceptance,
        "methodology_position": {
            "fbm_information_extraction": "primary_goal_now",
            "similarity_grouping": "testable_now",
            "multi_view_defect_scores": "testable_now",
            "process_join_statistics": "later_after_process_metadata_join",
            "segmentation": "next_after_realism_freeze",
            "autoencoder_only": "not_recommended_as_primary_method",
        },
    }
    return metrics, rows


def html_table_rows(metrics: dict[str, Any]) -> str:
    labels = {
        "has_samples": "평가 샘플 수 충분",
        "uses_observable_features_only": "Observable feature만 사용",
        "at_least_three_classes_evaluable": "3개 이상 불량 관점 평가 가능",
        "retrieval_jaccard_lift_above_1_10": "유사검색이 random baseline보다 유의미하게 좋음",
        "any_class_auc_above_0_70": "하나 이상 불량 관점에서 AUC 0.70 이상",
    }
    rows = []
    for key, value in metrics["acceptance"].items():
        cls = "pass" if value else "fail"
        rows.append(
            f"<tr><td>{html.escape(labels.get(key, key))}</td><td class=\"{cls}\">{'PASS' if value else 'CHECK'}</td></tr>"
        )
    return "\n".join(rows)


def class_rows(metrics: dict[str, Any]) -> str:
    rows = []
    for class_name, payload in metrics["discriminability"].items():
        if payload.get("status") != "evaluated":
            rows.append(
                "<tr>"
                f"<td>{html.escape(class_name)}</td><td>평가 제외</td>"
                f"<td>{payload.get('positives', 0)} / {payload.get('negatives', 0)}</td>"
                "<td>-</td><td>-</td><td>-</td>"
                "</tr>"
            )
            continue
        top = payload["top_features"][0]
        retrieval = metrics["retrieval"]["by_class"].get(class_name, {})
        rows.append(
            "<tr>"
            f"<td>{html.escape(class_name)}</td>"
            "<td>평가됨</td>"
            f"<td>{payload['positives']} / {payload['negatives']}</td>"
            f"<td>{html.escape(top['feature'])}</td>"
            f"<td>{top['auc']:.3f}</td>"
            f"<td>{retrieval.get('precision_at_k', 0.0):.3f} / {retrieval.get('base_rate', 0.0):.3f}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def relpath(target: Path, base_file: Path) -> str:
    return Path(os.path.relpath(Path(target).resolve(), base_file.resolve().parent)).as_posix()


def html_report(metrics: dict[str, Any], features: Path, metrics_path: Path, out: Path) -> str:
    all_acceptance = all(metrics["acceptance"].values())
    retrieval = metrics["retrieval"]
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>WaferMap 방법론 검증 리포트</title>
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
    .note {{ background: #fff7ed; border-left: 4px solid #f97316; padding: 12px 14px; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>WaferMap 방법론 검증 리포트</h1>
  <p>이 리포트는 현재 synthetic wafer map이 원래 목표인 <strong>FBM 정보 추출, 유사 불량 wafer 그룹핑, 다양한 불량 관점의 score 정의, 이후 공간 분해 모델</strong>로 이어질 수 있는지 검증한다. Synthetic mask는 검증용 정답으로만 사용하며, feature vector에는 실제 wafer에서도 계산 가능한 observable feature만 포함한다.</p>

  <div class="note">
    현재 결론: 1차 목표는 ANOVA가 아니라 FBM 자체의 정보 추출과 유사 wafer 그룹핑이다. ANOVA나 통계 검정은 나중에 공정/설비/lot/recipe 데이터와 feature를 조인한 뒤 수행하는 후속 분석이다. AutoEncoder 단독 접근은 stby missing chip과 전체 reconstruction error에 과민할 수 있어 1차 방법론으로 두지 않는다.
  </div>

  <div class="summary">
    <div class="card"><div>샘플 수</div><div class="metric">{metrics['sample_count']}</div></div>
    <div class="card"><div>Observable Feature</div><div class="metric">{metrics['feature_count_observable']}</div></div>
    <div class="card"><div>유사검색 Jaccard Lift</div><div class="metric">{retrieval['jaccard_lift']:.2f}x</div></div>
    <div class="card"><div>검증 상태</div><div class="metric {'pass' if all_acceptance else 'fail'}">{'PASS' if all_acceptance else 'CHECK'}</div></div>
  </div>

  <h2>검증 체크</h2>
  <table>
    <tr><th>항목</th><th>상태</th></tr>
    {html_table_rows(metrics)}
  </table>

  <h2>불량 관점별 신호</h2>
  <table>
    <tr><th>불량 관점</th><th>상태</th><th>Pos / Neg</th><th>가장 강한 Observable Feature</th><th>Best AUC</th><th>Precision@K / Base</th></tr>
    {class_rows(metrics)}
  </table>

  <h2>유사 wafer 검색 요약</h2>
  <table>
    <tr><th>지표</th><th>값</th></tr>
    <tr><td>Top K</td><td>{retrieval['top_k']}</td></tr>
    <tr><td>가까운 wafer 간 평균 label Jaccard</td><td>{retrieval['mean_neighbor_label_jaccard']:.4f}</td></tr>
    <tr><td>임의 pair 평균 label Jaccard</td><td>{retrieval['random_pair_label_jaccard']:.4f}</td></tr>
    <tr><td>Jaccard lift</td><td>{retrieval['jaccard_lift']:.4f}</td></tr>
  </table>

  <h2>해석</h2>
  <ol>
    <li>지금의 1차 목표는 FBM에서 불량 정보를 추출하고, 비슷한 불량 패턴을 가진 wafer를 그룹핑하는 것이다.</li>
    <li>Synthetic mask는 검증용 정답으로만 사용하고, feature에는 넣지 않는다.</li>
    <li>Ring, edge, shot-grid, stby는 wafer-level observable feature로도 신호가 잘 잡힌다.</li>
    <li>Scratch와 local은 작은 공간 패턴이라 feature만으로는 약하며, segmentation 또는 morphology feature 보강이 필요하다.</li>
    <li>ANOVA는 공정데이터와 조인한 뒤 수행하는 후속 분석이다.</li>
  </ol>

  <h2>산출물</h2>
  <ul>
    <li>Feature CSV: <code>{html.escape(relpath(features, out))}</code></li>
    <li>Metrics JSON: <code>{html.escape(relpath(metrics_path, out))}</code></li>
  </ul>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    data_root = Path(args.data)
    out = Path(args.out)
    metrics_path = Path(args.metrics)
    features = Path(args.features)
    samples = [load_sample(path) for path in sample_dirs(data_root)]
    if not samples:
        raise SystemExit(f"No samples found under {data_root}")

    metrics, rows = make_metrics(samples, args.top_k)
    write_feature_csv(rows, features)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_report(metrics, features, metrics_path, out), encoding="utf-8")
    print(f"Wrote methodology report: {out}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote features: {features}")


if __name__ == "__main__":
    main()
