"""Summarize filled FBM expert review templates."""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.cli import positive_int
from wafermap.reporting.files import relative_path
from wafermap.reporting import (
    CLOCK_POSITION_MATCHES,
    DOMINANT_DEFECTS,
    MISSED_MAJOR_DEFECT_VALUES,
    NEXT_ACTIONS,
    REVIEW_DECISIONS,
    REVIEW_DEFECT_FAMILIES,
    RETRIEVAL_FAILURE_MODES,
)

REQUIRED_COLUMNS = (
    "query_sample_id",
    "rank",
    "neighbor_sample_id",
    "reviewer_decision",
    "dominant_defect",
    "clock_position_match",
    "missed_major_defect",
    "review_comment",
)
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", default="outputs/reports/expert_review_template.csv")
    parser.add_argument("--out", default="outputs/reports/expert_review_summary.html")
    parser.add_argument("--metrics", default="outputs/reports/expert_review_summary_metrics.json")
    parser.add_argument("--top-k", type=positive_int, default=5)
    return parser.parse_args(argv)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Review CSV has no rows")
    missing = [name for name in REQUIRED_COLUMNS if name not in rows[0]]
    if missing:
        raise ValueError(f"Review CSV missing required columns: {missing}")


def norm(value: str | None) -> str:
    return (value or "").strip().lower()


def parse_rank(row: dict[str, str]) -> int:
    try:
        return int(float(row.get("rank", "")))
    except ValueError:
        return 10**9


def summarize_rows(rows: list[dict[str, str]], top_k: int) -> dict[str, Any]:
    validate_rows(rows)
    decisions = [norm(row.get("reviewer_decision")) for row in rows]
    filled_rows = [row for row, decision in zip(rows, decisions) if decision]
    valid_rows = [row for row in filled_rows if norm(row.get("reviewer_decision")) in REVIEW_DECISIONS]
    invalid_rows = [
        {
            "csv_line": idx,
            "query_sample_id": row.get("query_sample_id", ""),
            "neighbor_sample_id": row.get("neighbor_sample_id", ""),
            "field": "reviewer_decision",
            "value": row.get("reviewer_decision", ""),
        }
        for idx, row in enumerate(rows, start=2)
        if norm(row.get("reviewer_decision")) and norm(row.get("reviewer_decision")) not in REVIEW_DECISIONS
    ]
    invalid_rows.extend(invalid_value_rows(rows, "dominant_defect", DOMINANT_DEFECTS))
    invalid_rows.extend(invalid_value_rows(rows, "query_defect_family", REVIEW_DEFECT_FAMILIES))
    invalid_rows.extend(invalid_value_rows(rows, "neighbor_defect_family", REVIEW_DEFECT_FAMILIES))
    invalid_rows.extend(invalid_value_rows(rows, "clock_position_match", CLOCK_POSITION_MATCHES))
    invalid_rows.extend(invalid_value_rows(rows, "missed_major_defect", MISSED_MAJOR_DEFECT_VALUES))
    invalid_rows.extend(invalid_value_rows(rows, "retrieval_failure_mode", RETRIEVAL_FAILURE_MODES))
    invalid_rows.extend(invalid_value_rows(rows, "next_action", NEXT_ACTIONS))

    decision_counts = Counter(norm(row.get("reviewer_decision")) for row in valid_rows)
    missed_counts = Counter(norm(row.get("missed_major_defect")) for row in valid_rows if norm(row.get("missed_major_defect")))
    clock_applicable = [
        row for row in valid_rows if norm(row.get("clock_position_match")) in {"yes", "partial", "no"}
    ]
    clock_accept_count = sum(1 for row in clock_applicable if norm(row.get("clock_position_match")) in {"yes", "partial"})
    by_query = group_topk_rows(valid_rows, top_k)
    query_count = len({row["query_sample_id"] for row in rows})
    reviewed_query_count = len(by_query)
    query_same = sum(any(norm(row.get("reviewer_decision")) == "same_family" for row in items) for items in by_query.values())
    query_accept = sum(
        any(norm(row.get("reviewer_decision")) in {"same_family", "partial_match"} for row in items)
        for items in by_query.values()
    )
    query_missed = sum(any(norm(row.get("missed_major_defect")) == "yes" for row in items) for items in by_query.values())

    metrics = {
        "total_rows": len(rows),
        "query_count": query_count,
        "filled_review_rows": len(filled_rows),
        "valid_review_rows": len(valid_rows),
        "invalid_value_count": len(invalid_rows),
        "review_completion_rate": safe_div(len(filled_rows), len(rows)),
        "same_family_rate": safe_div(decision_counts["same_family"], len(valid_rows)),
        "partial_match_rate": safe_div(decision_counts["partial_match"], len(valid_rows)),
        "mismatch_rate": safe_div(decision_counts["mismatch"], len(valid_rows)),
        "not_sure_rate": safe_div(decision_counts["not_sure"], len(valid_rows)),
        "accepted_match_rate": safe_div(decision_counts["same_family"] + decision_counts["partial_match"], len(valid_rows)),
        "missed_major_defect_rate": safe_div(missed_counts["yes"], len(valid_rows)),
        "clock_position_accept_rate": safe_div(clock_accept_count, len(clock_applicable)),
        "top_k": top_k,
        "reviewed_query_count": reviewed_query_count,
        "query_topk_same_family_rate": safe_div(query_same, reviewed_query_count),
        "query_topk_accept_rate": safe_div(query_accept, reviewed_query_count),
        "query_missed_major_defect_rate": safe_div(query_missed, reviewed_query_count),
        "decision_counts": dict(decision_counts),
        "dominant_defect_metrics": dominant_defect_metrics(valid_rows),
        "query_defect_family_counts": value_counts(valid_rows, "query_defect_family"),
        "neighbor_defect_family_counts": value_counts(valid_rows, "neighbor_defect_family"),
        "retrieval_failure_mode_counts": value_counts(valid_rows, "retrieval_failure_mode"),
        "next_action_counts": value_counts(valid_rows, "next_action"),
        "next_action_queue": next_action_queue(valid_rows),
        "invalid_values": invalid_rows,
    }
    metrics["interpretation"] = interpretation(metrics)
    return metrics


def invalid_value_rows(rows: list[dict[str, str]], field: str, allowed: tuple[str, ...]) -> list[dict[str, Any]]:
    invalid = []
    for idx, row in enumerate(rows, start=2):
        value = norm(row.get(field))
        if value and value not in allowed:
            invalid.append(
                {
                    "csv_line": idx,
                    "query_sample_id": row.get("query_sample_id", ""),
                    "neighbor_sample_id": row.get("neighbor_sample_id", ""),
                    "field": field,
                    "value": row.get(field, ""),
                }
            )
    return invalid


def group_topk_rows(rows: list[dict[str, str]], top_k: int) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if parse_rank(row) <= top_k:
            groups[row["query_sample_id"]].append(row)
    return groups


def dominant_defect_metrics(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        defect = norm(row.get("dominant_defect")) or "unknown"
        if defect in DOMINANT_DEFECTS:
            grouped[defect].append(row)
    metrics = []
    for defect in sorted(grouped):
        items = grouped[defect]
        decisions = Counter(norm(row.get("reviewer_decision")) for row in items)
        missed_yes = sum(1 for row in items if norm(row.get("missed_major_defect")) == "yes")
        metrics.append(
            {
                "dominant_defect": defect,
                "review_rows": len(items),
                "accepted_match_rate": safe_div(decisions["same_family"] + decisions["partial_match"], len(items)),
                "same_family_rate": safe_div(decisions["same_family"], len(items)),
                "missed_major_defect_rate": safe_div(missed_yes, len(items)),
            }
        )
    return metrics


def value_counts(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    counts = Counter(norm(row.get(field)) for row in rows if norm(row.get(field)))
    return {key: counts[key] for key in sorted(counts)}


def next_action_queue(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        action = norm(row.get("next_action"))
        if not action or action in {"keep_baseline", "not_sure"}:
            continue
        defect = norm(row.get("query_defect_family")) or norm(row.get("dominant_defect")) or "unknown"
        failure = norm(row.get("retrieval_failure_mode")) or "not_sure"
        grouped[(action, defect, failure)].append(row)
    queue = []
    for (action, defect, failure), items in grouped.items():
        missed_yes = sum(1 for row in items if norm(row.get("missed_major_defect")) == "yes")
        queue.append(
            {
                "next_action": action,
                "dominant_defect": defect,
                "retrieval_failure_mode": failure,
                "review_rows": len(items),
                "missed_major_defect_rate": safe_div(missed_yes, len(items)),
                "example_query_sample_id": items[0].get("query_sample_id", ""),
                "example_neighbor_sample_id": items[0].get("neighbor_sample_id", ""),
            }
        )
    return sorted(
        queue,
        key=lambda item: (item["review_rows"], item["missed_major_defect_rate"]),
        reverse=True,
    )


def safe_div(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def interpretation(metrics: dict[str, Any]) -> str:
    if metrics["valid_review_rows"] == 0:
        return "대기: 아직 채워진 전문가 리뷰가 없어 모델 판단을 내릴 수 없다."
    if metrics["accepted_match_rate"] >= 0.7 and metrics["missed_major_defect_rate"] <= 0.1:
        return "유망: top-k 유사검색이 전문가 판단 기준에서도 feature tuning 후보로 쓸 수 있다."
    if metrics["missed_major_defect_rate"] >= 0.25:
        return "주의: query의 주요 불량을 놓치는 비율이 높아 feature 보강 또는 segmentation 검토가 필요하다."
    if metrics["mismatch_rate"] >= 0.5:
        return "주의: mismatch 비율이 높아 현재 feature distance가 현업 유사도와 어긋날 수 있다."
    return "부분 유효: same/partial 판단을 defect family별로 보고 다음 feature tuning 우선순위를 정한다."


def html_report(metrics: dict[str, Any], review_path: Path, metrics_path: Path) -> str:
    defect_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(item['dominant_defect'])}</td>"
        f"<td>{item['review_rows']}</td>"
        f"<td>{item['accepted_match_rate']:.3f}</td>"
        f"<td>{item['same_family_rate']:.3f}</td>"
        f"<td>{item['missed_major_defect_rate']:.3f}</td>"
        "</tr>"
        for item in metrics["dominant_defect_metrics"]
    )
    invalid_rows = "\n".join(
        "<tr>"
        f"<td>{item['csv_line']}</td>"
        f"<td>{html.escape(item['field'])}</td>"
        f"<td>{html.escape(item['value'])}</td>"
        f"<td>{html.escape(item['query_sample_id'])}</td>"
        f"<td>{html.escape(item['neighbor_sample_id'])}</td>"
        "</tr>"
        for item in metrics["invalid_values"]
    )
    action_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(item['next_action'])}</td>"
        f"<td>{html.escape(item['dominant_defect'])}</td>"
        f"<td>{html.escape(item['retrieval_failure_mode'])}</td>"
        f"<td>{item['review_rows']}</td>"
        f"<td>{item['missed_major_defect_rate']:.3f}</td>"
        f"<td>{html.escape(item['example_query_sample_id'])}</td>"
        f"<td>{html.escape(item['example_neighbor_sample_id'])}</td>"
        "</tr>"
        for item in metrics["next_action_queue"]
    )
    failure_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(key)}</td>"
        f"<td>{value}</td>"
        "</tr>"
        for key, value in metrics["retrieval_failure_mode_counts"].items()
    )
    next_action_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(key)}</td>"
        f"<td>{value}</td>"
        "</tr>"
        for key, value in metrics["next_action_counts"].items()
    )
    query_family_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(key)}</td>"
        f"<td>{value}</td>"
        "</tr>"
        for key, value in metrics["query_defect_family_counts"].items()
    )
    neighbor_family_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(key)}</td>"
        f"<td>{value}</td>"
        "</tr>"
        for key, value in metrics["neighbor_defect_family_counts"].items()
    )
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>FBM Expert Review Summary</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ color: #111827; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; margin: 18px 0; }}
    .card {{ border: 1px solid #d8dee9; border-radius: 8px; padding: 14px; background: #f8fafc; }}
    .metric {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px 10px; text-align: left; }}
    th {{ background: #eef2f7; }}
    .note {{ background: #fff7ed; border-left: 4px solid #f97316; padding: 12px 14px; margin: 12px 0; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>FBM Expert Review Summary</h1>
  <div class="note">{html.escape(metrics['interpretation'])}</div>
  <div class="summary">
    <div class="card"><div>Rows reviewed</div><div class="metric">{metrics['valid_review_rows']}/{metrics['total_rows']}</div></div>
    <div class="card"><div>Accepted</div><div class="metric">{metrics['accepted_match_rate']:.3f}</div></div>
    <div class="card"><div>Top-{metrics['top_k']} query accept</div><div class="metric">{metrics['query_topk_accept_rate']:.3f}</div></div>
    <div class="card"><div>Missed major</div><div class="metric">{metrics['missed_major_defect_rate']:.3f}</div></div>
  </div>
  <h2>Overall Metrics</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Review completion rate</td><td>{metrics['review_completion_rate']:.3f}</td></tr>
    <tr><td>Same-family rate</td><td>{metrics['same_family_rate']:.3f}</td></tr>
    <tr><td>Partial-match rate</td><td>{metrics['partial_match_rate']:.3f}</td></tr>
    <tr><td>Mismatch rate</td><td>{metrics['mismatch_rate']:.3f}</td></tr>
    <tr><td>Query top-k same-family rate</td><td>{metrics['query_topk_same_family_rate']:.3f}</td></tr>
    <tr><td>Query top-k accepted rate</td><td>{metrics['query_topk_accept_rate']:.3f}</td></tr>
    <tr><td>Clock-position accept rate</td><td>{metrics['clock_position_accept_rate']:.3f}</td></tr>
    <tr><td>Invalid value count</td><td>{metrics['invalid_value_count']}</td></tr>
  </table>
  <h2>Dominant Defect Metrics</h2>
  <table>
    <tr><th>Dominant defect</th><th>Rows</th><th>Accepted</th><th>Same-family</th><th>Missed major</th></tr>
    {defect_rows}
  </table>
  <h2>Next Action Queue</h2>
  <p>리뷰어가 표시한 실패 유형과 후속 액션을 다음 feature/model 보강 후보로 묶은 표다.</p>
  <table>
    <tr><th>Next action</th><th>Dominant defect</th><th>Failure mode</th><th>Rows</th><th>Missed major</th><th>Example query</th><th>Example neighbor</th></tr>
    {action_rows}
  </table>
  <h2>Query / Neighbor Defect Family Counts</h2>
  <table>
    <tr><th>Query defect family</th><th>Rows</th></tr>
    {query_family_rows}
  </table>
  <table>
    <tr><th>Neighbor defect family</th><th>Rows</th></tr>
    {neighbor_family_rows}
  </table>
  <h2>Failure Mode Counts</h2>
  <table>
    <tr><th>Failure mode</th><th>Rows</th></tr>
    {failure_rows}
  </table>
  <h2>Next Action Counts</h2>
  <table>
    <tr><th>Next action</th><th>Rows</th></tr>
    {next_action_rows}
  </table>
  <h2>Invalid Values</h2>
  <table>
    <tr><th>CSV line</th><th>Field</th><th>Value</th><th>Query</th><th>Neighbor</th></tr>
    {invalid_rows}
  </table>
  <h2>Inputs</h2>
  <ul>
    <li>Review CSV: <code>{html.escape(relative_path(review_path, metrics_path))}</code></li>
    <li>Metrics JSON: <code>{html.escape(metrics_path.name)}</code></li>
  </ul>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    review_path = Path(args.review)
    out_path = Path(args.out)
    metrics_path = Path(args.metrics)
    rows = read_csv(review_path)
    metrics = summarize_rows(rows, top_k=args.top_k)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_report(metrics, review_path, metrics_path), encoding="utf-8")
    print(f"Wrote expert review summary: {out_path}")
    print(f"Wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()
