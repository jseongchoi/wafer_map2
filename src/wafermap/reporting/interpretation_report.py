"""HTML report builder for one-wafer FBM interpretation review."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from wafermap.reporting.files import relative_path
from wafermap.reporting.overlays import OVERLAY_COLORS, OVERLAY_LABELS


def html_report(
    *,
    feature_rows: list[dict[str, str]],
    defect_rows: list[dict[str, Any]],
    sanity_summary: dict[str, Any],
    image_map: dict[str, Path],
    annotated_image_map: dict[str, Path],
    focus_sample_id: str,
    defect_scores_path: Path,
    sanity_path: Path,
    similar_path: Path,
    report_path: Path,
) -> str:
    focus_feature = next(row for row in feature_rows if str(row.get("sample_id", "")) == focus_sample_id)
    focus_rows = _defect_rows_by_sample(defect_rows).get(focus_sample_id, [])
    top_rows = [row for row in focus_rows if float(row.get("score", 0.0)) >= 15.0][:5]
    if not top_rows:
        top_rows = focus_rows[:3]
    status = str(sanity_summary["overall_status"])
    status_class = {"PASS": "pass", "CHECK": "check", "FAIL": "fail"}.get(status, "check")
    score_rows = "\n".join(_score_table_row(row) for row in focus_rows)
    if not score_rows:
        score_rows = '<tr><td colspan="5">표시할 defect 후보가 없습니다.</td></tr>'
    focus_block = _focus_sample_block(
        focus_feature,
        top_rows,
        image_map,
        annotated_image_map,
        report_path,
    )
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FBM One-Wafer Defect Review</title>
  <style>
    body {{ margin: 0; background: #f5f7f8; color: #17211f; font-family: "Segoe UI", "Noto Sans KR", Arial, sans-serif; line-height: 1.58; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 18px 56px; }}
    h1 {{ margin: 0 0 8px; font-size: 34px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 22px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 8px; font-size: 17px; letter-spacing: 0; }}
    a {{ color: #2567a8; text-underline-offset: 3px; }}
    .hero, .section, .focus {{ border: 1px solid #d8e0de; border-radius: 8px; background: #fff; box-shadow: 0 14px 34px rgba(23, 33, 31, 0.08); }}
    .hero {{ padding: 24px; margin-bottom: 18px; }}
    .section {{ padding: 20px; margin: 18px 0; }}
    .focus {{ display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(320px, 0.9fr); gap: 18px; padding: 16px; box-shadow: none; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-top: 16px; }}
    .guide-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    .image-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; align-items: start; }}
    .card {{ border: 1px solid #d8e0de; border-radius: 8px; padding: 14px; background: #fbfcfc; }}
    .metric {{ display: block; font-size: 26px; font-weight: 800; }}
    .muted {{ color: #61716d; }}
    .badge {{ display: inline-flex; border-radius: 999px; padding: 5px 10px; font-weight: 800; font-size: 13px; }}
    .pass {{ background: #e3f2eb; color: #1f6b54; }}
    .check {{ background: #f4ead6; color: #805611; }}
    .fail {{ background: #f7dfdd; color: #9a3d38; }}
    .wafer-img {{ width: 100%; border: 1px solid #d8e0de; border-radius: 8px; background: #17211f; display: block; }}
    .no-image {{ display: grid; min-height: 260px; place-items: center; border: 1px dashed #b8c5c1; border-radius: 8px; color: #61716d; background: #fbfcfc; text-align: center; padding: 12px; }}
    .defects {{ display: grid; grid-template-columns: 1fr; gap: 10px; }}
    .bar {{ height: 9px; border-radius: 999px; background: #e8eeec; overflow: hidden; }}
    .bar span {{ display: block; height: 100%; background: #25745d; }}
    .swatch {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border: 1px solid #d8e0de; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #edf2f1; }}
    code {{ background: #eef3f2; border-radius: 5px; padding: 2px 5px; }}
    @media (max-width: 980px) {{ .focus, .image-grid {{ grid-template-columns: 1fr; }} }}
    @media (max-width: 760px) {{ .summary, .guide-grid {{ grid-template-columns: 1fr; }} h1 {{ font-size: 28px; }} }}
  </style>
</head>
<body>
<main>
  <section class="hero">
    <h1>FBM One-Wafer Defect Review</h1>
    <p class="muted">이 보고서는 여러 웨이퍼를 한꺼번에 결론내리는 문서가 아닙니다. 현재는 <strong>웨이퍼 1장</strong>에서 불량 후보 family를 색상으로 표시하고, 그 표시가 실제 FBM 이미지와 맞는지 검증하기 위한 리뷰 화면입니다.</p>
    <span class="badge {status_class}">Sanity {html.escape(status)}</span>
    <div class="summary">
      <div class="card"><span class="metric">{html.escape(focus_sample_id)}</span><span class="muted">focus wafer</span></div>
      <div class="card"><span class="metric">{len(feature_rows)}</span><span class="muted">CSV에 포함된 전체 wafer</span></div>
      <div class="card"><span class="metric">{sanity_summary['status_counts']['PASS']}</span><span class="muted">sanity PASS</span></div>
      <div class="card"><span class="metric">{len(top_rows)}</span><span class="muted">표시된 defect 후보</span></div>
    </div>
  </section>

  <section class="section">
    <h2>먼저 볼 것</h2>
    <div class="guide-grid">
      <div class="card">
        <h3>1. 색칠된 이미지</h3>
        <p>왼쪽 annotated map에서 색상이 칠해진 영역이 실제 불량 위치와 맞는지 봐야 합니다. 시계 방향 텍스트보다 이 색칠 결과가 더 중요한 판단 기준입니다.</p>
      </div>
      <div class="card">
        <h3>2. 원본 grade preview</h3>
        <p>오른쪽 원본 grade 이미지를 같이 보고, 색칠된 후보가 실제 grade 0~7 분포에서 설득력 있는지 비교합니다.</p>
      </div>
      <div class="card">
        <h3>3. Defect 후보 점수</h3>
        <p>점수는 확률이 아니라 우선순위입니다. 70 이상은 강한 후보, 40~69는 중간 후보, 15~39는 약한 후보로 보면 됩니다.</p>
      </div>
      <div class="card">
        <h3>4. 줘야 하는 피드백</h3>
        <p>색칠 위치가 맞는지, 빠진 패턴이 있는지, defect family 이름이 맞는지, 점수가 과하거나 약한지만 알려주면 다음 보정에 바로 반영할 수 있습니다.</p>
      </div>
    </div>
  </section>

  <section class="section">
    <h2>Focus Wafer</h2>
    {focus_block}
  </section>

  <section class="section">
    <h2>피드백 예시</h2>
    <table>
      <tr><th>확인 항목</th><th>피드백 예시</th><th>이 피드백으로 고치는 것</th></tr>
      <tr><td>색칠 영역이 맞는가</td><td><code>노란 local은 맞지만 초록 ring은 너무 넓다</code></td><td>family별 mask 후보 생성 방식</td></tr>
      <tr><td>빠진 불량이 있는가</td><td><code>보라색 scratch가 있어야 하는데 안 보인다</code></td><td>scratch/line feature와 threshold</td></tr>
      <tr><td>점수가 납득되는가</td><td><code>edge 89는 너무 높고 40 정도가 맞다</code></td><td>score calibration</td></tr>
      <tr><td>STBY를 defect처럼 보는가</td><td><code>하늘색은 실제 불량이 아니라 미측정 영역이다</code></td><td>stby 분리와 defect score 제외 규칙</td></tr>
    </table>
  </section>

  <section class="section">
    <h2>Focus Wafer Score Table</h2>
    <table>
      <tr><th>Defect Family</th><th>Score</th><th>Confidence</th><th>Image Mark</th><th>Evidence</th></tr>
      {score_rows}
    </table>
  </section>

  <section class="section">
    <h2>생성 파일</h2>
    <ul>
      <li><a href="{html.escape(relative_path(defect_scores_path, report_path))}">defect_scores.csv</a>: 전체 wafer의 defect family 점수</li>
      <li><a href="{html.escape(relative_path(sanity_path, report_path))}">sanity_summary.json</a>: 입력 파싱과 mask 상태 점검</li>
      <li><a href="{html.escape(relative_path(similar_path, report_path))}">similar_wafers.csv</a>: reference feature가 있을 때 유사 wafer 후보</li>
    </ul>
  </section>
</main>
</body>
</html>
"""


def _focus_sample_block(
    feature_row: dict[str, str],
    defects: list[dict[str, Any]],
    image_map: dict[str, Path],
    annotated_image_map: dict[str, Path],
    report_path: Path,
) -> str:
    sample_id = str(feature_row.get("sample_id", "unknown"))
    annotated_html = _image_or_placeholder(
        annotated_image_map.get(sample_id),
        report_path,
        f"Annotated defect map for {sample_id}",
        "No annotated defect map available",
    )
    original_html = _image_or_placeholder(
        image_map.get(sample_id),
        report_path,
        f"Wafer preview for {sample_id}",
        "No wafer preview available",
    )
    cards = "\n".join(_defect_card(item) for item in defects)
    if not cards:
        cards = '<div class="card"><h3>No dominant defect</h3><p class="muted">15점 이상 defect 후보가 없습니다.</p></div>'
    return f"""<article class="focus">
  <div>
    <h3>{html.escape(sample_id)}</h3>
    <div class="image-grid">
      <div>
        <p class="muted"><strong>색칠된 후보 영역</strong></p>
        {annotated_html}
      </div>
      <div>
        <p class="muted"><strong>원본 grade preview</strong></p>
        {original_html}
      </div>
    </div>
  </div>
  <div>
    <h3>Defect 후보</h3>
    <div class="defects">{cards}</div>
  </div>
</article>"""


def _image_or_placeholder(path: Path | None, report_path: Path, alt: str, message: str) -> str:
    if path is None:
        return f'<div class="no-image">{html.escape(message)}<br><span class="muted">manifest 또는 raw PNG 입력으로 실행해야 이미지가 생성됩니다.</span></div>'
    return f'<img class="wafer-img" src="{html.escape(relative_path(path, report_path))}" alt="{html.escape(alt)}">'


def _defect_card(item: dict[str, Any]) -> str:
    family = str(item["defect_family"])
    color = OVERLAY_COLORS.get(family, "#8e8e93")
    label = OVERLAY_LABELS.get(family, family)
    score = float(item["score"])
    return f"""<div class="card">
  <h3><span class="swatch" style="background: {html.escape(color)}"></span>{html.escape(label)} · {score:.1f}</h3>
  <div class="bar"><span style="width: {max(0.0, min(100.0, score)):.1f}%"></span></div>
  <p class="muted">{html.escape(str(item["confidence"]))} confidence · 이미지에서 같은 색 영역을 확인하세요.</p>
  <p>{html.escape(str(item["evidence"]))}</p>
</div>"""


def _score_table_row(row: dict[str, Any]) -> str:
    family = str(row["defect_family"])
    color = OVERLAY_COLORS.get(family, "#8e8e93")
    label = OVERLAY_LABELS.get(family, family)
    return (
        "<tr>"
        f"<td><span class=\"swatch\" style=\"background: {html.escape(color)}\"></span>{html.escape(label)}</td>"
        f"<td><strong>{float(row['score']):.1f}</strong></td>"
        f"<td>{html.escape(str(row['confidence']))}</td>"
        f"<td>{html.escape(str(row['location']))}</td>"
        f"<td>{html.escape(str(row['evidence']))}</td>"
        "</tr>"
    )


def _defect_rows_by_sample(defect_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in defect_rows:
        grouped.setdefault(str(row.get("sample_id", "")), []).append(row)
    return grouped
