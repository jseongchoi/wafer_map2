"""Project-level HTML report for the pattern-asset synthetic data pipeline."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from wafermap.assets import (
    FAMILY_DATA_SOURCES,
    PRIMARY_ASSET_FAMILIES,
    PROCEDURAL_FAMILIES,
    TARGET_FAMILIES,
)
from wafermap.reporting.files import relative_path


def project_report_html(payload: dict[str, Any]) -> str:
    readiness = payload["readiness_metrics"]
    segmentation = payload["segmentation_metrics"]
    embedding = payload["embedding_metrics"]
    unet = payload.get("unet_metrics", {})
    outputs = payload["outputs"]
    assets = payload["assets"]
    report_out = Path(payload["report_out"])
    target_channels = readiness.get("target_channels", [])
    retrieval = embedding["retrieval"]
    missing_targets = [
        row["class"]
        for row in readiness.get("class_summary", [])
        if int(row.get("positive_samples", 0)) == 0
    ]
    asset_counts = {family: 0 for family in TARGET_FAMILIES}
    for asset in assets:
        asset_family = str(asset.get("family", ""))
        asset_counts[asset_family] = asset_counts.get(asset_family, 0) + 1
    annotation_missing = [name for name in PRIMARY_ASSET_FAMILIES if asset_counts.get(name, 0) == 0]
    procedural_missing = [name for name in missing_targets if name in PROCEDURAL_FAMILIES]
    asset_priority = ", ".join(annotation_missing) if annotation_missing else "primary asset coverage OK"
    procedural_status = (
        ", ".join(procedural_missing) + " 생성 설정 확인"
        if procedural_missing
        else "scratch fallback, edge, shot_grid, random 라벨이 코드로 생성됨"
    )
    procedural_families = ", ".join(payload.get("procedural_families", [])) or "none"
    loss_delta = segmentation["loss"]["initial"] - segmentation["loss"]["final"]
    unet_status = "ready" if unet.get("torch_available") else "blocked: PyTorch not installed"
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WaferMap Hybrid Synthetic Data Report</title>
  <style>
    body {{ margin:0; background:#eef2f1; color:#17211f; font-family:"Segoe UI","Noto Sans KR",Arial,sans-serif; line-height:1.62; }}
    main {{ max-width:1220px; margin:0 auto; padding:28px 18px 64px; }}
    h1 {{ margin:0 0 8px; font-size:32px; }}
    h2 {{ margin:30px 0 12px; font-size:22px; }}
    h3 {{ margin:20px 0 8px; font-size:16px; }}
    .muted {{ color:#63716d; }}
    .band {{ background:#fff; border:1px solid #d4ddda; border-radius:8px; padding:16px; margin:14px 0; }}
    .grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }}
    .metric {{ background:#fff; border:1px solid #d4ddda; border-radius:8px; padding:13px; }}
    .metric strong {{ display:block; font-size:24px; }}
    table {{ border-collapse:collapse; width:100%; background:#fff; margin:12px 0 22px; font-size:14px; }}
    th,td {{ border:1px solid #d4ddda; padding:8px 10px; text-align:left; vertical-align:top; }}
    th {{ background:#e6ecea; }}
    code {{ background:#e6ecea; border-radius:4px; padding:2px 5px; }}
    a {{ color:#12614e; }}
    .ok {{ color:#12614e; font-weight:700; }}
    .todo {{ color:#9a5c00; font-weight:700; }}
    .danger {{ color:#9a3d38; font-weight:700; }}
    @media (max-width:860px) {{ .grid {{ grid-template-columns:1fr 1fr; }} }}
    @media (max-width:560px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>WaferMap Hybrid Synthetic Data Report</h1>
  <p class="muted">목표: 실제 FBM wafer map에서 defect pattern을 family별 pixel mask, 위치/면적/score, embedding 기반 유사 wafer 검색으로 연결하는 딥러닝 학습 파이프라인을 만든다.</p>

  <section class="grid">
    <div class="metric"><strong>{payload['asset_count']}</strong><span>저장된 human pattern asset</span></div>
    <div class="metric"><strong>{payload['composed_count']}</strong><span>생성된 hybrid synthetic wafer</span></div>
    <div class="metric"><strong>{len(target_channels)}</strong><span>defect target channels</span></div>
    <div class="metric"><strong>{retrieval['lift_vs_baseline']:.2f}x</strong><span>embedding top-1 lift</span></div>
  </section>

  <h2>1. 이번 설계 변경의 핵심</h2>
  <div class="band">
    <p>모든 defect family를 사람이 누끼 따는 방식으로 만들면 안 됩니다. <code>random</code>은 본질적으로 산발 noise/background라서 누끼 기준이 불안정합니다. <code>edge</code>와 <code>shot_grid</code>도 wafer 좌표, edge 거리, chip/shot 반복 규칙으로 합성할 수 있습니다.</p>
    <p><code>scratch</code>는 실제 모양이 중요하므로 human asset이 최종 기준이지만, 사람이 아직 충분히 누끼 따지 못한 상태에서는 모델 학습 루프가 멈추면 안 됩니다. 그래서 현재는 procedural scratch를 cold-start fallback으로 생성하고, 나중에 실제 scratch asset이 쌓이면 그쪽으로 더 전환하는 구조로 갑니다.</p>
    <p>따라서 합성 데이터 생성 책임을 두 갈래로 나눴습니다. 사람이 해야 하는 것은 실제 형태/texture가 중요한 <code>local</code>, <code>scratch</code>, <code>ring</code>입니다. 코드가 만드는 것은 cold-start <code>scratch</code>와 규칙/통계가 명확한 <code>edge</code>, <code>shot_grid</code>, <code>random</code>입니다. 모두 최종적으로는 동일한 multi-label mask target으로 저장됩니다.</p>
  </div>

  <h2>2. Family별 라벨 생성 방법론</h2>
  <table>
    <tr><th>Family</th><th>라벨 생성 방식</th><th>왜 이렇게 하는가</th><th>현재 상태</th></tr>
    {family_source_rows()}
  </table>

  <h2>3. 현재 실행 설정</h2>
  <table>
    <tr><th>항목</th><th>값</th><th>의미</th></tr>
    <tr><td>Base sample</td><td><code>{html.escape(str(payload['base_sample_dir']))}</code></td><td>합성 wafer를 붙일 기본 wafer map입니다.</td></tr>
    <tr><td>Human assets</td><td><code>{html.escape(str(payload['assets_root']))}</code></td><td>사람이 에디터로 저장한 pattern asset library입니다.</td></tr>
    <tr><td>Procedural families</td><td><code>{html.escape(procedural_families)}</code></td><td>코드가 직접 mask와 grade를 생성하는 family입니다.</td></tr>
    <tr><td>Placement mode</td><td><code>{html.escape(str(payload['placement_mode']))}</code></td><td>human asset은 원본 절대좌표 signature를 보존하는 배치가 기본입니다.</td></tr>
    <tr><td>Target channels</td><td><code>{html.escape(', '.join(target_channels))}</code></td><td>STBY는 target에서 제외하고 defect family만 학습합니다.</td></tr>
    <tr><td>U-Net training status</td><td><code>{html.escape(unet_status)}</code></td><td>실제 딥러닝 segmentation 학습 entrypoint 준비 상태입니다.</td></tr>
  </table>

  <h2>4. 실행 결과 해석</h2>
  <table>
    <tr><th>지표</th><th>값</th><th>해석</th></tr>
    <tr><td>Readiness samples</td><td>{readiness['sample_count']}</td><td>이번 manifest에 들어간 synthetic wafer 수입니다.</td></tr>
    <tr><td>Train / Val split</td><td>{readiness['split_counts']['train']} / {readiness['split_counts']['val']}</td><td>smoke 학습/검증 분리입니다.</td></tr>
    <tr><td>Segmentation loss</td><td>{segmentation['loss']['initial']:.4f} -> {segmentation['loss']['final']:.4f}</td><td>라벨 연결이 학습 가능한 형태인지 보는 최소 검증입니다. Delta {loss_delta:.4f}</td></tr>
    <tr><td>Embedding top-1 Jaccard</td><td>{retrieval['top1_mean_jaccard']:.4f}</td><td>가까운 embedding wafer가 같은 family 조합을 갖는지 보는 smoke 지표입니다.</td></tr>
    <tr><td>Embedding baseline Jaccard</td><td>{retrieval['baseline_mean_jaccard']:.4f}</td><td>무작위 비교 기준입니다. 현재는 asset 다양성이 작으면 쉽게 나올 수 있습니다.</td></tr>
  </table>

  <h2>5. Family Coverage</h2>
  <div class="band">
    <p><strong>사람 작업 우선순위:</strong> {html.escape(asset_priority)}</p>
    <p><strong>코드 생성 상태:</strong> {html.escape(procedural_status)}</p>
  </div>
  <table>
    <tr><th>Family</th><th>Positive samples</th><th>Presence rate</th><th>Mean pixel ratio</th><th>판정</th></tr>
    {coverage_rows(readiness.get('class_summary', []))}
  </table>

  <h2>6. 딥러닝 모델로 이어지는 구조</h2>
  <div class="band">
    <p>이 파이프라인의 목적은 라벨 있는 synthetic data를 만드는 것입니다. 모델은 이 데이터를 입력으로 받아 family별 probability mask를 출력합니다. 그 mask에서 area ratio, centroid, radial position, edge distance, component count, family score를 계산하고, encoder embedding은 유사 wafer top-k 검색에 사용합니다.</p>
    <p>즉 지금 만든 것은 에디터 자체가 최종 제품이 아닙니다. 딥러닝 모델을 학습시키기 위한 라벨 생성 공장입니다. 에디터는 사람이 필요한 family만 빠르게 보강하는 도구이고, 나머지는 코드 합성기가 담당합니다.</p>
  </div>

  <h2>7. 논문/기술 근거</h2>
  <table>
    <tr><th>근거</th><th>프로젝트 반영</th></tr>
    {research_rows()}
  </table>

  <h2>8. 다음 작업 순서</h2>
  <ol>
    <li><code>scratch</code>, <code>local</code>, <code>ring</code> human asset을 실제 wafer에서 더 모읍니다. 현재 우선순위는 <code>{html.escape(asset_priority)}</code>입니다.</li>
    <li><code>scratch</code>는 procedural fallback으로 학습을 시작하되, 실제 human scratch asset이 들어오면 그쪽으로 realism을 보정합니다.</li>
    <li><code>edge</code>, <code>shot_grid</code>, <code>random</code>은 누끼가 아니라 procedural generator의 realism을 조정합니다.</li>
    <li>hybrid synthetic manifest를 train/val/test로 고정하고, family별 최소 positive sample 수를 보장합니다.</li>
    <li>small U-Net 또는 SegFormer 계열 모델을 붙여 family별 mask prediction을 학습합니다.</li>
    <li>현재 학습 환경에 PyTorch가 없으면 <code>scripts/train_unet_segmentation.py</code>의 dependency check report를 보고, torch 설치 후 같은 manifest로 학습을 실행합니다.</li>
    <li><code>scripts/export_unet_predictions.py</code>로 모델 prediction을 <code>fbm_prediction_masks/v1</code>로 export해 에디터에서 사람이 수정하는 active learning loop로 연결합니다.</li>
    <li>encoder embedding을 저장하고 cosine/FAISS 기반 유사 wafer top-k 검색을 붙입니다.</li>
  </ol>

  <h2>9. Output Links</h2>
  <ul>
    <li><a href="{html.escape(relative_path(outputs['readiness_report'], report_out))}">Segmentation readiness report</a></li>
    <li><a href="{html.escape(relative_path(outputs['segmentation_report'], report_out))}">Segmentation smoke report</a></li>
    <li><a href="{html.escape(relative_path(outputs['embedding_report'], report_out))}">Embedding smoke report</a></li>
    <li><a href="{html.escape(relative_path(outputs['unet_report'], report_out))}">U-Net segmentation training report</a></li>
    <li><a href="{html.escape(relative_path(outputs['embeddings_csv'], report_out))}">Embedding vectors CSV</a></li>
  </ul>
</main>
</body>
</html>
"""


def coverage_rows(rows: list[dict[str, Any]]) -> str:
    out = []
    for row in rows:
        family = str(row["class"])
        positives = int(row.get("positive_samples", 0))
        source = FAMILY_DATA_SOURCES.get(family, "unknown")
        if positives > 0:
            verdict = "학습 후보 있음"
            verdict_class = "ok"
        elif family in PRIMARY_ASSET_FAMILIES:
            verdict = "우선 누끼 필요"
            verdict_class = "danger"
        elif family in PROCEDURAL_FAMILIES:
            verdict = "코드 생성 설정 확인"
            verdict_class = "todo"
        else:
            verdict = "확인 필요"
            verdict_class = "todo"
        out.append(
            "<tr>"
            f"<td>{html.escape(family)}<br><span class=\"muted\">{html.escape(source)}</span></td>"
            f"<td>{positives}</td>"
            f"<td>{float(row.get('sample_presence_rate', 0.0)):.3f}</td>"
            f"<td>{float(row.get('mean_pixel_ratio', 0.0)):.6f}</td>"
            f"<td class=\"{verdict_class}\">{html.escape(verdict)}</td>"
            "</tr>"
        )
    return "\n".join(out)


def family_source_rows() -> str:
    explanations = {
        "local": ("human asset primary", "실제 blob 모양, grade texture, 군집 형태가 중요해서 사람이 딴 asset이 가장 가치 있습니다."),
        "scratch": ("human asset primary + procedural fallback", "얇고 긴 선형 패턴은 실제 wafer 누끼가 최종 기준이지만, cold-start 학습을 위해 radial/spin-arc scratch를 코드로 보강합니다."),
        "ring": ("human asset primary", "얇은 ring, partial ring, 원형/반경의 실제 공정 signature를 담습니다."),
        "edge": ("procedural primary, asset optional", "wafer edge 거리와 angular sector rule로 안정적인 mask를 만들 수 있습니다."),
        "shot_grid": ("procedural primary, asset optional", "chip/shot 반복 좌표로 합성할 수 있고, 실제 shot layout 정보가 있으면 더 정교해집니다."),
        "random": ("procedural only", "사람 누끼 대상이 아니라 산발 fail/noise baseline입니다."),
    }
    rows = []
    for family in TARGET_FAMILIES:
        method, reason = explanations[family]
        source = FAMILY_DATA_SOURCES.get(family, "")
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(family)}</code></td>"
            f"<td>{html.escape(method)}</td>"
            f"<td>{html.escape(reason)}</td>"
            f"<td>{html.escape(source)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def research_rows() -> str:
    rows = [
        (
            '<a href="https://ieeexplore.ieee.org/document/10122621/">Semantic Segmentation-Based Wafer Map Mixed-Type Defect Pattern Recognition</a>',
            "최종 출력은 단일 class가 아니라 family별 pixel mask로 정의하는 근거입니다.",
        ),
        (
            '<a href="https://ieeexplore.ieee.org/document/8263132/">Wafer Map Defect Pattern Classification and Image Retrieval Using CNN</a>',
            "encoder embedding으로 유사 wafer top-k 검색을 구성하는 근거입니다.",
        ),
        (
            '<a href="https://ieeexplore.ieee.org/document/8368296/">Classification of Mixed-Type Defect Patterns in Wafer Bin Maps Using CNNs</a>',
            "mixed-type defect를 multi-label 관점으로 다루어야 한다는 근거입니다.",
        ),
        (
            '<a href="https://pubs.aip.org/aip/adv/article/14/4/045329/3283648/An-efficient-deep-learning-framework-for-mixed">Efficient deep learning framework for mixed-type wafer map DPR</a>',
            "초기 모델은 작고 검증 가능한 encoder/segmentation 구조부터 시작하는 것이 실용적입니다.",
        ),
    ]
    return "\n".join(f"<tr><td>{source}</td><td>{html.escape(impact)}</td></tr>" for source, impact in rows)
