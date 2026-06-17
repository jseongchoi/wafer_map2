"""Create a human review template for FBM nearest-neighbor results."""

from __future__ import annotations

import argparse
import csv
import html
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.reporting import (
    CLOCK_POSITION_MATCHES,
    DOMINANT_DEFECTS,
    MISSED_MAJOR_DEFECT_VALUES,
    NEXT_ACTIONS,
    REVIEW_DEFECT_FAMILIES,
    RETRIEVAL_FAILURE_MODES,
    REVIEW_DECISIONS,
    TEMPLATE_COLUMNS,
    build_template_rows,
    write_template_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neighbors", default="outputs/reports/real_unlabeled_neighbors.csv")
    parser.add_argument("--template-out", default="outputs/reports/expert_review_template.csv")
    parser.add_argument("--report-out", default="outputs/reports/expert_review_template.html")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_template_csv(path, rows)


def relpath(target: Path, base_file: Path) -> str:
    return Path(os.path.relpath(Path(target).resolve(), base_file.resolve().parent)).as_posix()


def html_report(
    rows: list[dict[str, Any]],
    warnings: list[str],
    neighbors_path: Path,
    template_path: Path,
    report_path: Path,
) -> str:
    preview_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row['query_sample_id']))}</td>"
        f"<td>{html.escape(str(row['rank']))}</td>"
        f"<td>{html.escape(str(row['neighbor_sample_id']))}</td>"
        f"<td>{html.escape(str(row['distance']))}</td>"
        "</tr>"
        for row in rows[:30]
    )
    warning_block = ""
    if warnings:
        warning_items = "\n".join(f"<li>{html.escape(item)}</li>" for item in warnings)
        warning_block = f"<div class=\"note\"><ul>{warning_items}</ul></div>"
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>FBM Expert Review Template</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ color: #111827; }}
    .summary {{ display: grid; grid-template-columns: repeat(3, minmax(160px, 1fr)); gap: 12px; margin: 18px 0; }}
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
  <h1>FBM Expert Review Template</h1>
  <p>이 템플릿은 nearest-neighbor 결과를 전문가가 빠르게 평가하기 위한 최소 형식이다. 실제 wafer 원본, 보안 경로, synthetic oracle label은 포함하지 않는다.</p>
  {warning_block}
  <div class="summary">
    <div class="card"><div>Review cases</div><div class="metric">{len(rows)}</div></div>
    <div class="card"><div>Decision values</div><div class="metric">{len(REVIEW_DECISIONS)}</div></div>
    <div class="card"><div>Defect values</div><div class="metric">{len(DOMINANT_DEFECTS)}</div></div>
  </div>
  <h2>Reviewer Fields</h2>
  <table>
    <tr><th>Field</th><th>Allowed values</th><th>Meaning</th></tr>
    <tr><td><code>reviewer_decision</code></td><td>{', '.join(REVIEW_DECISIONS)}</td><td>query와 neighbor가 같은 계열인지 평가한다.</td></tr>
    <tr><td><code>query_defect_family</code></td><td>{', '.join(REVIEW_DEFECT_FAMILIES)}</td><td>query wafer에서 리뷰어가 본 주된 결함군이다.</td></tr>
    <tr><td><code>neighbor_defect_family</code></td><td>{', '.join(REVIEW_DEFECT_FAMILIES)}</td><td>neighbor wafer에서 리뷰어가 본 주된 결함군이다.</td></tr>
    <tr><td><code>dominant_defect</code></td><td>{', '.join(DOMINANT_DEFECTS)}</td><td>리뷰어가 판단한 주된 불량 축을 남긴다.</td></tr>
    <tr><td><code>clock_position_match</code></td><td>{', '.join(CLOCK_POSITION_MATCHES)}</td><td>12시/3시/edge 방향 같은 위치가 맞는지 평가한다.</td></tr>
    <tr><td><code>missed_major_defect</code></td><td>{', '.join(MISSED_MAJOR_DEFECT_VALUES)}</td><td>검색 결과가 query의 중요한 불량을 놓쳤는지 표시한다.</td></tr>
    <tr><td><code>retrieval_failure_mode</code></td><td>{', '.join(RETRIEVAL_FAILURE_MODES)}</td><td>mismatch 또는 partial match의 원인을 다음 보강 track으로 분류한다.</td></tr>
    <tr><td><code>next_action</code></td><td>{', '.join(NEXT_ACTIONS)}</td><td>리뷰 결과를 feature tuning, location-aware retrieval, segmentation 후보 같은 후속 작업으로 연결한다.</td></tr>
    <tr><td><code>safe_comment</code></td><td>free text</td><td>lot, tool, recipe, chamber, raw path 같은 보안 정보를 쓰지 않는다.</td></tr>
  </table>
  <h2>Preview</h2>
  <table>
    <tr><th>Query</th><th>Rank</th><th>Neighbor</th><th>Distance</th></tr>
    {preview_rows}
  </table>
  <h2>Outputs</h2>
  <ul>
    <li>Input neighbors: <code>{html.escape(relpath(neighbors_path, report_path))}</code></li>
    <li>Review template: <code>{html.escape(relpath(template_path, report_path))}</code></li>
  </ul>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    neighbors_path = Path(args.neighbors)
    template_path = Path(args.template_out)
    report_path = Path(args.report_out)
    neighbor_rows = read_csv(neighbors_path)
    rows, warnings = build_template_rows(neighbor_rows)
    write_csv(template_path, rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html_report(rows, warnings, neighbors_path, template_path, report_path), encoding="utf-8")
    print(f"Wrote expert review template: {template_path}")
    print(f"Wrote expert review template report: {report_path}")


if __name__ == "__main__":
    main()
