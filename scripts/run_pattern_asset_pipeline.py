"""Run the Pattern Asset learning-readiness pipeline end to end."""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.assets import (
    DEFAULT_PROCEDURAL_FAMILIES,
    FAMILY_DATA_SOURCES,
    FAMILY_LABELS,
    PRIMARY_ASSET_FAMILIES,
    PROCEDURAL_FAMILIES,
    TARGET_FAMILIES,
    scan_pattern_assets,
)
from wafermap.data import load_sample


RESEARCH_NOTES: tuple[dict[str, str], ...] = (
    {
        "topic": "mixed-type defect에는 multi-label/분리 관점이 필요",
        "source": "Kyeong & Kim, Classification of Mixed-Type Defect Patterns in Wafer Bin Maps Using CNNs, IEEE TSM 2018",
        "url": "https://ieeexplore.ieee.org/document/8368296/",
        "impact": "우리 목표가 단일 class 분류가 아니라 family별 mask/score로 가야 하는 근거",
    },
    {
        "topic": "wafer map classification과 image retrieval에 CNN feature 사용",
        "source": "Nakazawa & Kulkarni, Wafer Map Defect Pattern Classification and Image Retrieval Using CNN, IEEE TSM 2018",
        "url": "https://ieeexplore.ieee.org/document/8263132/",
        "impact": "embedding vector 기반 유사 wafer top-k 검색 방향의 직접 근거",
    },
    {
        "topic": "mixed-type wafer defect recognition에 semantic segmentation 적용",
        "source": "Yan et al., Semantic Segmentation-Based Wafer Map Mixed-Type Defect Pattern Recognition, IEEE TCAD 2023",
        "url": "https://ieeexplore.ieee.org/document/10122621/",
        "impact": "family별 pixel mask를 모델 출력으로 정의하는 방향의 직접 근거",
    },
    {
        "topic": "가벼운 CNN/attention으로 mixed-type wafer recognition 비용 절감",
        "source": "AIP Advances, An efficient deep learning framework for mixed-type wafer map defect pattern recognition, 2024",
        "url": "https://pubs.aip.org/aip/adv/article/14/4/045329/3283648/An-efficient-deep-learning-framework-for-mixed",
        "impact": "초기 모델은 거대 모델보다 작고 검증 가능한 encoder부터 시작하는 것이 합리적",
    },
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sample-dir", default="data/synthetic/fbm_grouping_scale_pilot/synth_000000")
    parser.add_argument("--assets-root", default="data/pattern_assets")
    parser.add_argument("--composed-dir", default="data/synthetic/asset_composed")
    parser.add_argument("--work-dir", default="outputs/pattern_asset_pipeline")
    parser.add_argument("--report-out", default="outputs/reports/pattern_asset_project_report.html")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--assets-per-wafer", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--placement-mode", choices=("source_jitter", "random_valid"), default="source_jitter")
    parser.add_argument("--jitter-pixels", type=int, default=48)
    parser.add_argument(
        "--procedural-families",
        default=",".join(DEFAULT_PROCEDURAL_FAMILIES),
        help="Comma-separated code-generated families. Use none to disable.",
    )
    parser.add_argument("--output-size", type=int, default=48)
    parser.add_argument("--embedding-dim", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args(argv)


def load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Cannot load script: {path}")
    spec.loader.exec_module(module)
    return module


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    composer = load_script("compose_synthetic_from_assets.py")
    readiness = load_script("build_segmentation_readiness.py")
    segmentation_smoke = load_script("train_segmentation_smoke.py")
    embedding_smoke = load_script("train_embedding_smoke.py")
    unet_train = load_script("train_unet_segmentation.py")

    assets_root = resolve_repo_path(args.assets_root)
    base_sample_dir = resolve_repo_path(args.base_sample_dir)
    composed_dir = resolve_repo_path(args.composed_dir)
    work_dir = resolve_repo_path(args.work_dir)
    report_out = resolve_repo_path(args.report_out)
    work_dir.mkdir(parents=True, exist_ok=True)

    procedural_families = composer.parse_family_list(args.procedural_families)
    assets = composer.load_assets(assets_root, require_assets=not procedural_families)
    base = load_sample(base_sample_dir)
    rng = random.Random(args.seed)
    composed_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(args.count):
        sample_id = f"asset_composed_{idx:06d}"
        sample = composer.compose_sample(
            base,
            assets,
            rng,
            args.assets_per_wafer,
            sample_id,
            placement_mode=args.placement_mode,
            jitter_pixels=args.jitter_pixels,
            procedural_families=procedural_families,
        )
        composer.write_sample(sample, composed_dir / sample_id)

    manifest = work_dir / "asset_segmentation_manifest.csv"
    readiness_metrics = work_dir / "asset_segmentation_readiness_metrics.json"
    readiness_report = work_dir / "asset_segmentation_readiness.html"
    gallery = work_dir / "asset_segmentation_gallery.png"
    readiness_outputs = readiness.build_outputs(
        SimpleNamespace(
            data=str(composed_dir),
            out=str(readiness_report),
            metrics=str(readiness_metrics),
            manifest=str(manifest),
            gallery=str(gallery),
            val_fraction=0.2,
            split_seed=args.seed,
            max_gallery_rows=6,
            overlap_stride=4,
        )
    )

    segmentation_report = work_dir / "asset_segmentation_smoke.html"
    segmentation_metrics = work_dir / "asset_segmentation_smoke_metrics.json"
    segmentation_figure = work_dir / "asset_segmentation_smoke_loss.png"
    segmentation_smoke.main(
        [
            "--manifest",
            str(manifest),
            "--out",
            str(segmentation_report),
            "--metrics",
            str(segmentation_metrics),
            "--figure",
            str(segmentation_figure),
            "--output-size",
            str(args.output_size),
            "--max-train-samples",
            "8",
            "--max-val-samples",
            "4",
            "--steps",
            "6",
        ]
    )

    embedding_report = work_dir / "asset_embedding_smoke.html"
    embedding_metrics = work_dir / "asset_embedding_smoke_metrics.json"
    embeddings_csv = work_dir / "asset_embedding_vectors.csv"
    embedding_smoke.main(
        [
            "--manifest",
            str(manifest),
            "--out",
            str(embedding_report),
            "--metrics",
            str(embedding_metrics),
            "--embeddings-out",
            str(embeddings_csv),
            "--output-size",
            str(args.output_size),
            "--embedding-dim",
            str(args.embedding_dim),
            "--top-k",
            str(args.top_k),
            "--max-train-samples",
            "16",
            "--max-val-samples",
            "8",
        ]
    )

    unet_report = work_dir / "asset_unet_segmentation.html"
    unet_metrics = work_dir / "asset_unet_segmentation_metrics.json"
    unet_model = work_dir / "asset_unet_segmentation.pt"
    unet_train.main(
        [
            "--manifest",
            str(manifest),
            "--out",
            str(unet_report),
            "--metrics",
            str(unet_metrics),
            "--model-out",
            str(unet_model),
            "--output-size",
            str(args.output_size),
            "--check-deps",
        ]
    )

    payload = {
        "assets": scan_pattern_assets(assets_root),
        "asset_count": len(scan_pattern_assets(assets_root)),
        "composed_count": int(args.count),
        "assets_root": assets_root,
        "base_sample_dir": base_sample_dir,
        "composed_dir": composed_dir,
        "work_dir": work_dir,
        "report_out": report_out,
        "readiness_outputs": readiness_outputs,
        "readiness_metrics": json.loads(readiness_metrics.read_text(encoding="utf-8")),
        "segmentation_metrics": json.loads(segmentation_metrics.read_text(encoding="utf-8")),
        "embedding_metrics": json.loads(embedding_metrics.read_text(encoding="utf-8")),
        "unet_metrics": json.loads(unet_metrics.read_text(encoding="utf-8")),
        "outputs": {
            "manifest": manifest,
            "readiness_report": readiness_report,
            "segmentation_report": segmentation_report,
            "embedding_report": embedding_report,
            "embeddings_csv": embeddings_csv,
            "unet_report": unet_report,
            "unet_metrics": unet_metrics,
            "project_report": report_out,
        },
        "placement_mode": args.placement_mode,
        "jitter_pixels": int(args.jitter_pixels),
        "procedural_families": list(procedural_families),
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(clean_project_report_html(payload), encoding="utf-8")
    return payload


def project_report_html(payload: dict[str, Any]) -> str:
    readiness = payload["readiness_metrics"]
    segmentation = payload["segmentation_metrics"]
    embedding = payload["embedding_metrics"]
    assets = payload["assets"]
    outputs = payload["outputs"]
    report_parent = Path(payload["report_out"]).parent
    target_channels = readiness.get("target_channels", [])
    segmentation_loss_delta = segmentation["loss"]["initial"] - segmentation["loss"]["final"]
    retrieval = embedding["retrieval"]
    missing_targets = [
        row["class"]
        for row in readiness.get("class_summary", [])
        if int(row.get("positive_samples", 0)) == 0
    ]
    coverage_message = (
        "현재 asset library가 모든 family를 덮지 못합니다. 다음 annotation 우선순위는 "
        + ", ".join(missing_targets)
        if missing_targets
        else "현재 실행 데이터에는 모든 target family가 최소 1개 이상 포함되어 있습니다."
    )
    annotation_missing_targets = [name for name in missing_targets if name in PRIMARY_ASSET_FAMILIES]
    procedural_missing_targets = [name for name in missing_targets if name in PROCEDURAL_FAMILIES]
    coverage_message = (
        "사람 누끼가 필요한 primary asset family 중 아직 비어 있는 항목: "
        + ", ".join(annotation_missing_targets)
        if annotation_missing_targets
        else "사람 누끼가 필요한 primary asset family는 이번 실행 데이터 안에 최소 1개 이상 포함되어 있습니다."
    )
    procedural_message = (
        "코드 생성 family 중 이번 실행에서 양성 sample이 없던 항목: "
        + ", ".join(procedural_missing_targets)
        if procedural_missing_targets
        else "edge, shot_grid, random 같은 procedural family는 이번 실행에서 코드로 라벨이 생성되었습니다."
    )
    asset_priority_message = (
        ", ".join(annotation_missing_targets) if annotation_missing_targets else "primary asset coverage OK"
    )
    retrieval_message = (
        "현재 embedding top-1이 baseline보다 낮습니다. 이는 모델 문제가 아니라 현재 asset이 ring/local에 편중되어 label 다양성이 부족하다는 신호입니다."
        if retrieval["lift_vs_baseline"] < 1.0
        else "현재 embedding top-1은 baseline보다 같은 label family를 더 잘 찾습니다."
    )
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WaferMap Pattern Asset Project Report</title>
  <style>
    body {{ margin: 0; background: #eef2f1; color: #17211f; font-family: "Segoe UI", "Noto Sans KR", Arial, sans-serif; line-height: 1.62; }}
    main {{ max-width: 1220px; margin: 0 auto; padding: 28px 18px 64px; }}
    h1 {{ margin: 0 0 8px; font-size: 32px; }}
    h2 {{ margin: 30px 0 12px; font-size: 22px; }}
    h3 {{ margin: 20px 0 8px; font-size: 16px; }}
    .muted {{ color: #63716d; }}
    .band {{ background: #fff; border: 1px solid #d4ddda; border-radius: 8px; padding: 16px; margin: 14px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .metric {{ background: #fff; border: 1px solid #d4ddda; border-radius: 8px; padding: 13px; }}
    .metric strong {{ display: block; font-size: 24px; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff; margin: 12px 0 22px; font-size: 14px; }}
    th, td {{ border: 1px solid #d4ddda; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #e6ecea; }}
    code {{ background: #e6ecea; border-radius: 4px; padding: 2px 5px; }}
    a {{ color: #12614e; }}
    .ok {{ color: #12614e; font-weight: 700; }}
    .todo {{ color: #9a5c00; font-weight: 700; }}
    .danger {{ color: #9a3d38; font-weight: 700; }}
    @media (max-width: 860px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} }}
    @media (max-width: 560px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>WaferMap Pattern Asset Project Report</h1>
  <p class="muted">이 문서는 현재 프로젝트가 본질 목표인 defect mask 검출, 위치/형태 수치화, wafer-level score, embedding retrieval로 가고 있는지 확인하기 위한 실행 보고서입니다.</p>

  <section class="grid">
    <div class="metric"><strong>{payload['asset_count']}</strong><span>saved pattern assets</span></div>
    <div class="metric"><strong>{payload['composed_count']}</strong><span>asset-composed wafers</span></div>
    <div class="metric"><strong>{len(target_channels)}</strong><span>defect target channels</span></div>
    <div class="metric"><strong>{retrieval['lift_vs_baseline']:.2f}x</strong><span>embedding top-1 lift</span></div>
  </section>

  <h2>1. 목표 정의</h2>
  <div class="band">
    <p>우리가 만들 것은 단순 wafer class 분류기가 아닙니다. 최종 시스템은 입력 wafer grade 0~7 map을 보고, defect family별 pixel mask를 예측하고, 그 mask에서 위치/면적/중심/edge 거리/score를 계산하며, encoder embedding으로 유사 wafer top-k를 찾는 구조입니다.</p>
    <p>현재 구현은 그 큰 목표의 첫 학습 루프입니다. 사람이 누끼 딴 pattern asset을 합성 wafer에 붙이고, 그 합성 결과로 segmentation target과 embedding retrieval smoke check를 생성합니다.</p>
  </div>

  <h2>2. 현재 실행 단계</h2>
  <table>
    <tr><th>단계</th><th>상태</th><th>산출물</th><th>해석</th></tr>
    <tr><td>Pattern Asset 저장</td><td class="ok">구현됨</td><td><code>{html.escape(str(payload['assets_root']))}</code></td><td>실제 wafer에서 사람이 family별 pixel mask를 누끼로 저장한다.</td></tr>
    <tr><td>Asset 합성</td><td class="ok">구현됨</td><td><code>{html.escape(str(payload['composed_dir']))}</code></td><td><code>{html.escape(payload['placement_mode'])}</code> 배치로 절대좌표 의미를 보존한다.</td></tr>
    <tr><td>Segmentation manifest</td><td class="ok">구현됨</td><td><code>{html.escape(str(outputs['manifest']))}</code></td><td>target channel은 {html.escape(', '.join(target_channels))}이며 STBY는 제외됐다.</td></tr>
    <tr><td>Segmentation smoke</td><td class="ok">구현됨</td><td><code>{html.escape(str(outputs['segmentation_report']))}</code></td><td>NumPy sigmoid baseline으로 input/target/loss 배관을 검증했다. Loss delta: {segmentation_loss_delta:.4f}</td></tr>
    <tr><td>Embedding retrieval</td><td class="ok">구현됨</td><td><code>{html.escape(str(outputs['embedding_report']))}</code></td><td>PCA baseline embedding으로 cosine-style top-k 검색 배관을 검증했다.</td></tr>
    <tr><td>Scratch line assist</td><td class="ok">구현됨</td><td><code>Pattern Asset Builder</code></td><td>대충 칠한 scratch seed의 주방향을 계산해 선형 mask를 확장한다. 실제 wafer 검증은 필요하다.</td></tr>
    <tr><td>Model prediction → editor 재표시</td><td class="todo">부분 구현</td><td><code>fbm_prediction_masks/v1</code></td><td>에디터가 prediction JSON을 editable mask로 불러올 수 있다. 다음은 모델 prediction exporter가 필요하다.</td></tr>
  </table>

  <h2>3. 사용자가 지금 보면 되는 것</h2>
  <table>
    <tr><th>볼 것</th><th>질문</th></tr>
    <tr><td>Pattern Asset Library</td><td>누끼가 family별로 맞는가? ring 하나가 쪼개지지 않았는가?</td></tr>
    <tr><td>Segmentation Readiness</td><td>학습 target channel이 STBY를 제외하고 defect family만 잡고 있는가?</td></tr>
    <tr><td>Segmentation Smoke</td><td>학습 배관이 정상적으로 loss를 줄이는가? 특정 family target pixel이 0으로 비어 있지 않은가?</td></tr>
    <tr><td>Embedding Smoke</td><td>embedding top-k가 baseline보다 label이 비슷한 wafer를 더 잘 찾는가?</td></tr>
  </table>

  <h2>4. 이번 실행 결과</h2>
  <table>
    <tr><th>항목</th><th>값</th></tr>
    <tr><td>Readiness samples</td><td>{readiness['sample_count']}</td></tr>
    <tr><td>Train / Val split</td><td>{readiness['split_counts']['train']} / {readiness['split_counts']['val']}</td></tr>
    <tr><td>Target channels</td><td>{html.escape(', '.join(readiness['target_channels']))}</td></tr>
    <tr><td>Segmentation initial/final loss</td><td>{segmentation['loss']['initial']:.4f} / {segmentation['loss']['final']:.4f}</td></tr>
    <tr><td>Embedding top-1 mean Jaccard</td><td>{retrieval['top1_mean_jaccard']:.4f}</td></tr>
    <tr><td>Embedding baseline mean Jaccard</td><td>{retrieval['baseline_mean_jaccard']:.4f}</td></tr>
    <tr><td>Embedding top-k best mean Jaccard</td><td>{retrieval['topk_best_mean_jaccard']:.4f}</td></tr>
  </table>
  <div class="band">
    <h3>데이터 커버리지 판정</h3>
    <p>{html.escape(coverage_message)}</p>
    <p>{html.escape(procedural_message)}</p>
    <p>{html.escape(retrieval_message)}</p>
  </div>

  <h2>5. Family Coverage</h2>
  <table>
    <tr><th>Family</th><th>Positive samples</th><th>Presence rate</th><th>Mean pixel ratio</th><th>판정</th></tr>
    {coverage_rows(readiness.get('class_summary', []))}
  </table>

  <h2>6. 논문/기술 근거</h2>
  <table>
    <tr><th>주제</th><th>출처</th><th>프로젝트 반영</th></tr>
    {research_rows()}
  </table>

  <h2>7. 다음 구현 순서</h2>
  <ol>
    <li>사람 누끼는 primary asset family부터 추가한다. 현재 우선순위: {html.escape(asset_priority_message)}</li>
    <li>Smart Fit과 Trace Scratch Line을 실제 wafer 5~20장으로 검증한다.</li>
    <li>segmentation smoke prediction을 <code>fbm_prediction_masks/v1</code> JSON으로 export한다.</li>
    <li>asset-composed dataset manifest를 train/val/test로 확장하고 family별 최소 양을 보장한다.</li>
    <li>PyTorch small U-Net 또는 SegFormer 계열 실제 모델 학습 스크립트를 붙인다.</li>
    <li>모델 encoder embedding을 저장하고 FAISS 또는 cosine index로 real wafer top-k 검색을 붙인다.</li>
  </ol>

  <h2>8. Output Links</h2>
  <ul>
    <li><a href="{html.escape(relhref(outputs['readiness_report'], report_parent))}">Segmentation readiness report</a></li>
    <li><a href="{html.escape(relhref(outputs['segmentation_report'], report_parent))}">Segmentation smoke report</a></li>
    <li><a href="{html.escape(relhref(outputs['embedding_report'], report_parent))}">Embedding smoke report</a></li>
    <li><a href="{html.escape(relhref(outputs['embeddings_csv'], report_parent))}">Embedding vectors CSV</a></li>
  </ul>
</main>
</body>
</html>
"""


def research_rows() -> str:
    rows = []
    for item in RESEARCH_NOTES:
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['topic'])}</td>"
            f"<td><a href=\"{html.escape(item['url'])}\">{html.escape(item['source'])}</a></td>"
            f"<td>{html.escape(item['impact'])}</td>"
            "</tr>"
        )
    return "\n".join(rows)


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


def relhref(target: str | Path, base_dir: str | Path) -> str:
    return os.path.relpath(Path(target).resolve(), Path(base_dir).resolve()).replace("\\", "/")


def clean_project_report_html(payload: dict[str, Any]) -> str:
    readiness = payload["readiness_metrics"]
    segmentation = payload["segmentation_metrics"]
    embedding = payload["embedding_metrics"]
    unet = payload.get("unet_metrics", {})
    outputs = payload["outputs"]
    assets = payload["assets"]
    report_parent = Path(payload["report_out"]).parent
    target_channels = readiness.get("target_channels", [])
    retrieval = embedding["retrieval"]
    missing_targets = [
        row["class"]
        for row in readiness.get("class_summary", [])
        if int(row.get("positive_samples", 0)) == 0
    ]
    asset_counts = {family: 0 for family in TARGET_FAMILIES}
    for asset in assets:
        asset_counts[str(asset.get("family", ""))] = asset_counts.get(str(asset.get("family", "")), 0) + 1
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
    <p><code>scratch</code>는 실제 모양이 중요하므로 human asset이 최종 기준이지만, 사람이 아직 충분히 누끼 따지 못한 상태에서도 모델 학습 루프가 멈추면 안 됩니다. 그래서 현재는 procedural scratch를 cold-start fallback으로 생성하고, 나중에 실제 scratch asset이 쌓이면 그쪽을 더 신뢰하는 구조로 갑니다.</p>
    <p>따라서 합성 데이터 생성 책임을 두 갈래로 나눴습니다. 사람이 따야 하는 것은 실제 형태/texture가 중요한 <code>local</code>, <code>scratch</code>, <code>ring</code>입니다. 코드가 만드는 것은 cold-start <code>scratch</code>와 규칙/통계가 명확한 <code>edge</code>, <code>shot_grid</code>, <code>random</code>입니다. 둘 다 최종적으로는 동일한 multi-label mask target으로 저장됩니다.</p>
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
    <tr><td>Embedding baseline Jaccard</td><td>{retrieval['baseline_mean_jaccard']:.4f}</td><td>무작위 비교 기준입니다. 현재는 asset 다양성이 작으면 낮게 나올 수 있습니다.</td></tr>
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
    <p>즉 지금 만든 것은 에디터 자체가 최종 제품이 아니라, 딥러닝 모델을 학습시키기 위한 라벨 생성 공장입니다. 에디터는 사람이 필요한 family만 빠르게 보강하는 도구이고, 나머지는 코드 합성기가 담당합니다.</p>
  </div>

  <h2>7. 논문/기술 근거</h2>
  <table>
    <tr><th>근거</th><th>프로젝트 반영</th></tr>
    {clean_research_rows()}
  </table>

  <h2>8. 다음 작업 순서</h2>
  <ol>
    <li><code>scratch</code>, <code>local</code>, <code>ring</code> human asset을 실제 wafer에서 더 모읍니다. 현재 우선순위는 <code>{html.escape(asset_priority)}</code>입니다.</li>
    <li><code>scratch</code>는 procedural fallback으로 학습을 시작하되, 실제 human scratch asset이 들어오면 그쪽으로 realism을 보정합니다.</li>
    <li><code>edge</code>, <code>shot_grid</code>, <code>random</code>은 누끼가 아니라 procedural generator의 realism을 조정합니다.</li>
    <li>hybrid synthetic manifest를 train/val/test로 고정하고, family별 최소 positive sample 수를 보장합니다.</li>
    <li>small U-Net 또는 SegFormer 계열 모델을 붙여 family별 mask prediction을 학습합니다.</li>
    <li>현재 학습 환경에는 PyTorch가 없으면 <code>scripts/train_unet_segmentation.py</code>의 dependency check report를 보고, torch 설치 후 같은 manifest로 학습을 실행합니다.</li>
    <li>모델 prediction을 <code>fbm_prediction_masks/v1</code>로 export해 에디터에서 사람이 수정하는 active learning loop로 연결합니다.</li>
    <li>encoder embedding을 저장하고 cosine/FAISS 기반 유사 wafer top-k 검색을 붙입니다.</li>
  </ol>

  <h2>9. Output Links</h2>
  <ul>
    <li><a href="{html.escape(relhref(outputs['readiness_report'], report_parent))}">Segmentation readiness report</a></li>
    <li><a href="{html.escape(relhref(outputs['segmentation_report'], report_parent))}">Segmentation smoke report</a></li>
    <li><a href="{html.escape(relhref(outputs['embedding_report'], report_parent))}">Embedding smoke report</a></li>
    <li><a href="{html.escape(relhref(outputs['unet_report'], report_parent))}">U-Net segmentation training report</a></li>
    <li><a href="{html.escape(relhref(outputs['embeddings_csv'], report_parent))}">Embedding vectors CSV</a></li>
  </ul>
</main>
</body>
</html>
"""


def family_source_rows() -> str:
    explanations = {
        "local": ("human asset primary", "실제 blob 모양, grade texture, 군집 형태가 중요해서 사람이 딴 asset이 가장 가치 있습니다."),
        "scratch": ("human asset primary + procedural fallback", "얇고 끊긴 선형 패턴은 실제 wafer 누끼가 최종 기준이지만, cold-start 학습을 위해 radial/spin-arc scratch를 코드로 보강합니다."),
        "ring": ("human asset primary", "끊긴 ring, partial ring, 두께/반경이 실제 공정 signature를 담습니다."),
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


def clean_research_rows() -> str:
    rows = [
        (
            '<a href="https://ieeexplore.ieee.org/document/10122621/">Semantic Segmentation-Based Wafer Map Mixed-Type Defect Pattern Recognition</a>',
            "최종 출력을 단일 class가 아니라 family별 pixel mask로 정의하는 근거입니다.",
        ),
        (
            '<a href="https://ieeexplore.ieee.org/document/8263132/">Wafer Map Defect Pattern Classification and Image Retrieval Using CNN</a>',
            "encoder embedding으로 유사 wafer top-k 검색을 구성하는 근거입니다.",
        ),
        (
            '<a href="https://ieeexplore.ieee.org/document/8368296/">Classification of Mixed-Type Defect Patterns in Wafer Bin Maps Using CNNs</a>',
            "mixed-type defect는 multi-label 관점으로 다루어야 한다는 근거입니다.",
        ),
        (
            '<a href="https://pubs.aip.org/aip/adv/article/14/4/045329/3283648/An-efficient-deep-learning-framework-for-mixed">Efficient deep learning framework for mixed-type wafer map DPR</a>',
            "초기 모델은 작고 검증 가능한 encoder/segmentation 구조부터 시작하는 게 실용적입니다.",
        ),
    ]
    return "\n".join(f"<tr><td>{source}</td><td>{html.escape(impact)}</td></tr>" for source, impact in rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_pipeline(args)
    print(f"Wrote project report: {payload['outputs']['project_report']}")
    print(f"Wrote manifest: {payload['outputs']['manifest']}")
    print(f"Wrote embedding vectors: {payload['outputs']['embeddings_csv']}")


if __name__ == "__main__":
    main()
