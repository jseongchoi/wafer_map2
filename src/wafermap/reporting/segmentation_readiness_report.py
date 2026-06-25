"""HTML report builder for segmentation readiness metrics."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from wafermap.reporting.files import relative_path


def html_report(metrics: dict[str, Any], gallery: Path, manifest: Path, metrics_path: Path, out: Path) -> str:
    scratch = metrics["scratch_risk"]
    conclusion = (
        "scratch는 ring/local/stby와 자주 겹치므로 wafer-level retrieval feature만으로 분리하기 어렵다. "
        "다음 단계는 synthetic mask 기반 multi-label segmentation 또는 scratch-specific representation으로 가는 것이 맞다."
    )
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>FBM Segmentation Readiness 중간점검</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2933; line-height: 1.55; }}
    h1, h2 {{ color: #111827; }}
    .note {{ background: #eef6ff; border-left: 4px solid #2563eb; padding: 12px 14px; margin: 14px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f7; }}
    img {{ width: 100%; max-width: 1400px; border: 1px solid #d8dee9; border-radius: 8px; background: white; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>FBM Segmentation Readiness 중간점검</h1>
  <p>목적은 GPU 학습 전에 synthetic target mask가 어떤 class 불균형과 overlap risk를 갖는지 확인하고, scratch/local을 다음 모델 단계로 넘길 근거를 정리하는 것입니다.</p>
  <div class="note">Synthetic mask는 학습/검증 target입니다. 실제 inference feature에는 <code>pattern_masks</code>, <code>label_*</code>, <code>*_mask_ratio</code>를 넣지 않습니다.</div>

  <h2>Executive Summary</h2>
  <ul>
    <li>데이터셋: {metrics['sample_count']} samples, train {metrics['split_counts']['train']} / val {metrics['split_counts']['val']}</li>
    <li>입력 semantic channels: {html.escape(', '.join(metrics['input_channels']))}</li>
    <li>target channels: {html.escape(', '.join(metrics['target_channels']))}</li>
    <li>{html.escape(conclusion)}</li>
  </ul>

  <h2>Scratch Risk Check</h2>
  <table>
    <tr><th>Metric</th><th>Value</th><th>해석</th></tr>
    <tr><td>scratch positive samples</td><td>{scratch['scratch_positive_samples']}</td><td>scratch target 학습 후보 수</td></tr>
    <tr><td>scratch + ring co-occurrence</td><td>{_fmt(scratch['scratch_ring_cooccurrence_rate'])}</td><td>ring과 겹치면 wafer-level feature가 scratch를 놓치기 쉽습니다.</td></tr>
    <tr><td>scratch + local co-occurrence</td><td>{_fmt(scratch['scratch_local_cooccurrence_rate'])}</td><td>작은 blob과 같이 보이면 검색 기준별 결과가 흔들릴 수 있습니다.</td></tr>
    <tr><td>scratch + stby co-occurrence</td><td>{_fmt(scratch['scratch_stby_cooccurrence_rate'])}</td><td>scratch 시작점 또는 충돌점이 stby로 가려질 가능성입니다.</td></tr>
    <tr><td>scratch pixels hidden by stby mean/p95</td><td>{_fmt(scratch['scratch_pixels_hidden_by_stby_mean'])} / {_fmt(scratch['scratch_pixels_hidden_by_stby_p95'])}</td><td>stby가 scratch 관측을 얼마나 가리는지 보는 synthetic proxy입니다.</td></tr>
  </table>

  <h2>Class Balance</h2>
  <table>
    <tr><th>Class</th><th>Positive samples</th><th>Presence</th><th>Mean pixel ratio</th><th>Median positive</th><th>P95 positive</th><th>Suggested pos weight</th></tr>
    {_summary_rows(metrics['class_summary'])}
  </table>

  <h2>Overlap Top Pairs</h2>
  <table>
    <tr><th>Pair</th><th>Sample co-occurrence</th><th>Mean overlap pixel ratio</th></tr>
    {_overlap_rows(metrics['overlap_summary'])}
  </table>

  <h2>Mask Gallery</h2>
  <p>각 행은 대표 sample입니다. 왼쪽은 input severity이고, 오른쪽은 scratch/ring/local/stby target mask overlay입니다.</p>
  <img src="{html.escape(relative_path(gallery, out))}" alt="segmentation readiness gallery">

  <h2>다음 확인 단계</h2>
  <ol>
    <li>local은 현재 morphology baseline 결과를 expert review form에 연결합니다.</li>
    <li>scratch는 이 manifest를 이용해 작은 U-Net/SegFormer 계열 multi-label segmentation으로 넘깁니다.</li>
    <li>학습 target은 class별 sigmoid mask이고 overlap은 허용합니다.</li>
    <li>성공 기준은 전체 mIoU가 아니라 scratch recall, stby-hidden scratch recall, local small-blob recall입니다.</li>
  </ol>

  <h2>Outputs</h2>
  <ul>
    <li>Manifest CSV: <code>{html.escape(relative_path(manifest, out))}</code></li>
    <li>Metrics JSON: <code>{html.escape(relative_path(metrics_path, out))}</code></li>
    <li>Gallery: <code>{html.escape(relative_path(gallery, out))}</code></li>
  </ul>
</body>
</html>
"""


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _summary_rows(rows: list[dict[str, Any]]) -> str:
    out = []
    for row in rows:
        out.append(
            "<tr>"
            f"<td>{html.escape(str(row['class']))}</td>"
            f"<td>{row['positive_samples']}</td>"
            f"<td>{_fmt(row['sample_presence_rate'])}</td>"
            f"<td>{_fmt(row['mean_pixel_ratio'])}</td>"
            f"<td>{_fmt(row['median_positive_ratio'])}</td>"
            f"<td>{_fmt(row['p95_positive_ratio'])}</td>"
            f"<td>{_fmt(row['suggested_pos_weight_capped'])}</td>"
            "</tr>"
        )
    return "\n".join(out)


def _overlap_rows(rows: list[dict[str, Any]], limit: int = 10) -> str:
    out = []
    for row in rows[:limit]:
        out.append(
            "<tr>"
            f"<td>{html.escape(str(row['pair']))}</td>"
            f"<td>{_fmt(row['cooccurrence_rate'])}</td>"
            f"<td>{_fmt(row['mean_overlap_pixel_ratio'])}</td>"
            "</tr>"
        )
    return "\n".join(out)
