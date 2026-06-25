"""HTML report builder for real/unlabeled FBM feature extraction."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from wafermap.reporting.files import relative_path


def html_report(
    sanity_records: list[dict[str, Any]],
    features_out: Path,
    sanity_out: Path,
    neighbors_out: Path | None,
    report_out: Path,
    review_template_out: Path | None = None,
    feature_drift: dict[str, Any] | None = None,
) -> str:
    passed = sum(1 for item in sanity_records if not item["errors"])
    failed = len(sanity_records) - passed
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(item['sample_id'])}</td>"
        f"<td>{'PASS' if not item['errors'] else 'FAIL'}</td>"
        f"<td>{html.escape('; '.join(item['errors']) or '-')}</td>"
        f"<td>{html.escape('; '.join(item['warnings']) or '-')}</td>"
        f"<td>{item['valid_pixel_count']}</td>"
        f"<td>{item['stby_pixel_count']}</td>"
        "</tr>"
        for item in sanity_records
    )
    neighbor_link = (
        f'<li><a href="{html.escape(relative_path(neighbors_out, report_out))}">Nearest-neighbor CSV</a></li>'
        if neighbors_out is not None
        else ""
    )
    review_template_link = (
        f'<li><a href="{html.escape(relative_path(review_template_out, report_out))}">Expert review form CSV</a></li>'
        if review_template_out is not None
        else ""
    )
    drift_section = _feature_drift_section(feature_drift) if feature_drift is not None else ""
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>Real-Unlabeled FBM Feature Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ color: #111827; }}
    .summary {{ display: grid; grid-template-columns: repeat(3, minmax(160px, 1fr)); gap: 12px; margin: 18px 0; }}
    .card {{ border: 1px solid #d8dee9; border-radius: 8px; padding: 14px; background: #f8fafc; }}
    .metric {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f7; }}
    .note {{ background: #fff7ed; border-left: 4px solid #f97316; padding: 12px 14px; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Real-Unlabeled FBM Feature Report</h1>
  <div class="note">이 보고서는 real/unlabeled wafer 입력에서 observable feature와 sanity status를 확인하는 workflow입니다. STBY는 Grade 7 불량이 아니라 unobserved missing-test mask로 분리되어야 합니다.</div>
  <div class="summary">
    <div class="card"><div>Samples</div><div class="metric">{len(sanity_records)}</div></div>
    <div class="card"><div>PASS</div><div class="metric">{passed}</div></div>
    <div class="card"><div>FAIL</div><div class="metric">{failed}</div></div>
  </div>
  <h2>Sanity Checks</h2>
  <table>
    <tr><th>Sample</th><th>Status</th><th>Errors</th><th>Warnings</th><th>Valid Pixels</th><th>Stby Pixels</th></tr>
    {rows}
  </table>
  {drift_section}
  <h2>Outputs</h2>
  <ul>
    <li><a href="{html.escape(relative_path(features_out, report_out))}">Observable feature CSV</a></li>
    <li><a href="{html.escape(relative_path(sanity_out, report_out))}">Sanity JSON</a></li>
    {neighbor_link}
    {review_template_link}
  </ul>
</body>
</html>
"""


def _feature_drift_section(feature_drift: dict[str, Any]) -> str:
    drift_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(item['feature'])}</td>"
        f"<td>{item['query_mean']:.4f}</td>"
        f"<td>{item['reference_mean']:.4f}</td>"
        f"<td>{item['reference_std']:.4f}</td>"
        f"<td>{item['z_delta']:.2f}</td>"
        "</tr>"
        for item in feature_drift.get("top_shifted_features", [])
    )
    return f"""
  <h2>Reference 대비 Feature Drift</h2>
  <p>Reference synthetic feature store와 query feature 평균 차이를 z-score로 요약합니다. 이 값은 real wafer가 synthetic reference 분포와 얼마나 다른지 보는 sanity check이며, 성능 metric은 아닙니다.</p>
  <table>
    <tr><th>Feature</th><th>Query Mean</th><th>Reference Mean</th><th>Reference Std</th><th>Z Delta</th></tr>
    {drift_rows}
  </table>
"""
