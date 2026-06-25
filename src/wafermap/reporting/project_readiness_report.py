"""HTML report builder for project readiness audits."""

from __future__ import annotations

import html
from typing import Any


def html_report(audit: dict[str, Any]) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(stage['stage_id'])}</td>"
        f"<td>{html.escape(stage['title'])}</td>"
        f"<td>{html.escape(stage['status'])}</td>"
        f"<td>{html.escape(stage['goal'])}</td>"
        f"<td>{html.escape('; '.join(stage['missing_required']) or '-')}</td>"
        f"<td>{html.escape('; '.join(stage['missing_evidence']) or '-')}</td>"
        f"<td>{html.escape('; '.join(stage['next_actions']))}</td>"
        "</tr>"
        for stage in audit["stages"]
    )
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>WaferMap Project Readiness Audit</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #202124; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #dadce0; padding: 8px; vertical-align: top; font-size: 13px; }}
    th {{ background: #f1f3f4; text-align: left; }}
    .status {{ font-weight: 700; }}
  </style>
</head>
<body>
  <h1>WaferMap Project Readiness Audit</h1>
  <p class="status">Overall: {html.escape(audit["overall_status"])}</p>
  <p>Generated at {html.escape(audit["generated_at"])}</p>
  <table>
    <thead>
      <tr>
        <th>단계</th><th>제목</th><th>상태</th><th>목표</th>
        <th>빠진 필수 항목</th><th>빠진 근거</th><th>다음 작업</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""
