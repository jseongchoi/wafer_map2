"""Create a project progress briefing HTML from the latest review metrics."""

from __future__ import annotations

import argparse
import html
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="outputs/reports/final_review_metrics.json")
    parser.add_argument("--report", default="outputs/reports/final_review_report.html")
    parser.add_argument("--gallery", default="outputs/figures/final_review_gallery.png")
    parser.add_argument("--features", default="outputs/reports/final_review_features.csv")
    parser.add_argument("--methodology-report", default="outputs/reports/methodology_validation_report.html")
    parser.add_argument("--methodology-metrics", default="outputs/reports/methodology_validation_metrics.json")
    parser.add_argument("--grouping-report", default="outputs/reports/fbm_grouping_report.html")
    parser.add_argument("--grouping-metrics", default="outputs/reports/fbm_grouping_metrics.json")
    parser.add_argument("--stability-report", default="outputs/reports/fbm_grouping_stability_report.html")
    parser.add_argument("--stability-metrics", default="outputs/reports/fbm_grouping_stability_metrics.json")
    parser.add_argument("--sweep-report", default="outputs/reports/fbm_grouping_parameter_sweep_report.html")
    parser.add_argument("--sweep-metrics", default="outputs/reports/fbm_grouping_parameter_sweep_metrics.json")
    parser.add_argument("--ablation-report", default="outputs/reports/fbm_feature_ablation_report.html")
    parser.add_argument("--ablation-metrics", default="outputs/reports/fbm_feature_ablation_metrics.json")
    parser.add_argument("--scale-doc", default="docs/project_overview.md")
    parser.add_argument("--scale-grouping-report", default="outputs/reports/fbm_grouping_scale_report.html")
    parser.add_argument("--scale-grouping-metrics", default="outputs/reports/fbm_grouping_scale_metrics.json")
    parser.add_argument("--scale-stability-report", default="outputs/reports/fbm_grouping_scale_stability_report.html")
    parser.add_argument("--scale-stability-metrics", default="outputs/reports/fbm_grouping_scale_stability_metrics.json")
    parser.add_argument("--scale-sweep-report", default="outputs/reports/fbm_grouping_scale_parameter_sweep_report.html")
    parser.add_argument("--scale-sweep-metrics", default="outputs/reports/fbm_grouping_scale_parameter_sweep_metrics.json")
    parser.add_argument("--scale-ablation-report", default="outputs/reports/fbm_feature_ablation_scale_report.html")
    parser.add_argument("--scale-ablation-metrics", default="outputs/reports/fbm_feature_ablation_scale_metrics.json")
    parser.add_argument("--confidence-report", default="outputs/reports/fbm_retrieval_confidence_scale_report.html")
    parser.add_argument("--out", default="outputs/reports/project_progress_briefing.html")
    return parser.parse_args()


def read_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def relpath(target: Path, base_file: Path) -> str:
    return Path(os.path.relpath(Path(target).resolve(), base_file.resolve().parent)).as_posix()


def status_class(status: str) -> str:
    if status == "완료":
        return "done"
    if status == "진행 중":
        return "progress"
    if status == "시작":
        return "started"
    return "pending"


def phase_rows(
    metrics: dict[str, Any],
    methodology_metrics: dict[str, Any],
    grouping_metrics: dict[str, Any],
    stability_metrics: dict[str, Any],
    sweep_metrics: dict[str, Any],
    ablation_metrics: dict[str, Any],
    scale_grouping_metrics: dict[str, Any],
    scale_stability_metrics: dict[str, Any],
    scale_sweep_metrics: dict[str, Any],
    scale_ablation_metrics: dict[str, Any],
) -> list[dict[str, str]]:
    acceptance = metrics.get("acceptance", {})
    all_acceptance = bool(acceptance) and all(bool(v) for v in acceptance.values())
    method_acceptance = methodology_metrics.get("acceptance", {})
    method_all_acceptance = bool(method_acceptance) and all(bool(v) for v in method_acceptance.values())
    active_grouping_metrics = scale_grouping_metrics or grouping_metrics
    active_stability_metrics = scale_stability_metrics or stability_metrics
    active_sweep_metrics = scale_sweep_metrics or sweep_metrics
    active_ablation_metrics = scale_ablation_metrics or ablation_metrics
    phase1_evidence = "Synthetic generator, review gallery, metrics, validation 산출물이 존재한다."
    if all_acceptance:
        phase1_evidence += " 최신 acceptance check가 모두 통과했다."
    phase2_progress = "78%" if scale_ablation_metrics else ("72%" if ablation_metrics else ("60%" if active_sweep_metrics else ("52%" if active_stability_metrics else ("45%" if active_grouping_metrics else ("35%" if methodology_metrics else "20%")))))
    phase2_evidence = "Wafer-level observable feature extraction이 가능하며 final_review_features.csv가 생성된다."
    if methodology_metrics:
        lift = methodology_metrics.get("retrieval", {}).get("jaccard_lift", 0.0)
        phase2_evidence = (
            "Observable feature 기반 방법론 검증이 존재한다. "
            f"유사 wafer 검색 label-Jaccard lift는 random baseline 대비 {lift:.2f}x이다."
        )
        if method_all_acceptance:
            phase2_evidence += " 방법론 acceptance check가 모두 통과했다."
    if grouping_metrics:
        lift = grouping_metrics.get("similarity", {}).get("jaccard_lift", 0.0)
        clusters = grouping_metrics.get("cluster_count", 0)
        phase2_evidence = (
            f"Observable feature 기반 grouping pilot이 존재한다. "
            f"{clusters}개 cluster와 유사검색 label-Jaccard lift {lift:.2f}x를 확인했다."
        )
    if scale_grouping_metrics:
        sample_count = scale_grouping_metrics.get("sample_count", 0)
        lift = scale_grouping_metrics.get("similarity", {}).get("jaccard_lift", 0.0)
        clusters = scale_grouping_metrics.get("cluster_count", 0)
        phase2_evidence = (
            f"155장 scale pilot 기준 Observable feature grouping을 검증했다. "
            f"sample {sample_count}장, {clusters}개 cluster, 유사검색 label-Jaccard lift {lift:.2f}x이다."
        )
    if active_stability_metrics:
        separation = active_stability_metrics.get("coassociation_separation", 0.0)
        overlap = active_stability_metrics.get("mean_nearest_neighbor_overlap", 0.0)
        phase2_evidence += f" CPU-only 안정성 검증에서 co-association separation {separation:.2f}, nearest-neighbor overlap {overlap:.2f}를 확인했다."
    if active_sweep_metrics:
        recommended_k = active_sweep_metrics.get("recommended_cluster_count_cpu_pilot", 0)
        lifts = [item.get("jaccard_lift", 0.0) for item in active_sweep_metrics.get("retrieval_top_k_sweep", [])]
        mean_lift = sum(lifts) / len(lifts) if lifts else 0.0
        phase2_evidence += f" 파라미터 스윕 기준 CPU pilot 권장 cluster 수는 K={recommended_k}, top-k 평균 lift는 {mean_lift:.2f}x이다."
    if active_ablation_metrics:
        baseline_lift = active_ablation_metrics.get("baseline", {}).get("jaccard_lift", 0.0)
        ablations = active_ablation_metrics.get("ablations", [])
        top_drop = ablations[0] if ablations else {}
        phase2_evidence += (
            f" Feature family ablation에서 baseline lift {baseline_lift:.2f}x를 확인했고, "
            f"가장 큰 전체 lift drop은 {top_drop.get('removed_family', 'n/a')} 제거 시 {top_drop.get('lift_delta_vs_baseline', 0.0):.2f}이다."
        )
    phase2_next = (
        "155장 scale batch를 기준으로 real-unlabeled inference workflow를 설계하고, scratch/local 보강을 진행한다."
        if scale_grouping_metrics
        else "K=4 coarse group, nearest-neighbor review, defect score ranking을 기준으로 더 큰 batch 검증과 scratch/local 보강을 진행한다."
    )

    return [
        {
            "phase": "Phase 0",
            "name": "문제 정의와 설계",
            "status": "완료",
            "progress": "100%",
            "evidence": "문제 정의, schema, taxonomy, validation protocol, modeling strategy, physical assumption 문서가 존재한다.",
            "next": "Expert feedback이 들어올 때마다 문서를 최신 상태로 유지한다.",
        },
        {
            "phase": "Phase 1",
            "name": "Synthetic Generator와 현실성 Calibration",
            "status": "진행 중",
            "progress": "80%",
            "evidence": phase1_evidence,
            "next": "Preset을 사용해 현실성, overlap, edge 강도, stby 빈도, shot-relative defect subtlety를 조정한다.",
        },
        {
            "phase": "Phase 2",
            "name": "FBM 정보 추출과 유사 패턴 그룹핑",
            "status": "진행 중" if active_grouping_metrics else "시작",
            "progress": phase2_progress,
            "evidence": phase2_evidence,
            "next": phase2_next,
        },
        {
            "phase": "Phase 3",
            "name": "Synthetic-Label Segmentation Baseline",
            "status": "대기",
            "progress": "0%",
            "evidence": "Pattern mask는 존재하지만 segmentation 학습은 아직 시작하지 않았다.",
            "next": "Synthetic realism이 수용된 뒤 작은 multi-label segmentation baseline을 학습한다.",
        },
        {
            "phase": "Phase 4",
            "name": "Real-Unlabeled Adaptation",
            "status": "대기",
            "progress": "0%",
            "evidence": "Real wafer data는 data/raw 또는 사내 파일 경로에서 읽는 흐름으로 정리한다.",
            "next": "real data batch에서 feature extraction/inference와 review report가 바로 이어지도록 workflow를 정리한다.",
        },
        {
            "phase": "Phase 5",
            "name": "Advanced Modeling",
            "status": "대기",
            "progress": "0%",
            "evidence": "Advanced encoder와 domain adaptation은 아직 보류한다.",
            "next": "Feature usefulness와 synthetic-to-real realism이 확인된 뒤 시작한다.",
        },
    ]


def acceptance_table(metrics: dict[str, Any]) -> str:
    acceptance = metrics.get("acceptance", {})
    if not acceptance:
        return "<p>Metrics 파일을 찾지 못했다.</p>"
    labels = {
        "all_samples_internal_valid": "모든 sample 내부 검증 통과",
        "no_flow_patterns": "Flow-like pattern 없음",
        "pattern_classes_no_flow": "Pattern class에 flow 없음",
        "pattern_classes_include_ring": "Ring class 포함",
        "pattern_classes_include_shot_grid": "Shot-grid class 포함",
        "grade0_present_all_samples": "모든 sample에 Grade 0 존재",
        "edge_fail_density_higher_majority": "대부분 sample에서 edge fail density가 center보다 높음",
        "edge_chip_outer_face_higher_majority": "대부분 edge chip에서 outer face fail이 inner face보다 높음",
        "local_modes_cover_single_double_triple": "Local blob mode coverage 확보",
        "stby_present_all_samples": "모든 sample에 stby 존재",
        "origin_coupled_stby_present": "Defect origin-hidden stby 존재",
        "shot_grid_present": "Shot-grid pattern 존재",
    }
    rows = []
    for key, value in acceptance.items():
        cls = "pass" if value else "fail"
        label = "PASS" if value else "FAIL"
        rows.append(
            "<tr>"
            f"<td>{html.escape(labels.get(str(key), str(key)))}</td>"
            f"<td class=\"{cls}\">{label}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def phase_table(
    metrics: dict[str, Any],
    methodology_metrics: dict[str, Any],
    grouping_metrics: dict[str, Any],
    stability_metrics: dict[str, Any],
    sweep_metrics: dict[str, Any],
    ablation_metrics: dict[str, Any],
    scale_grouping_metrics: dict[str, Any],
    scale_stability_metrics: dict[str, Any],
    scale_sweep_metrics: dict[str, Any],
    scale_ablation_metrics: dict[str, Any],
) -> str:
    rows = []
    for item in phase_rows(
        metrics,
        methodology_metrics,
        grouping_metrics,
        stability_metrics,
        sweep_metrics,
        ablation_metrics,
        scale_grouping_metrics,
        scale_stability_metrics,
        scale_sweep_metrics,
        scale_ablation_metrics,
    ):
        cls = status_class(item["status"])
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['phase'])}</td>"
            f"<td>{html.escape(item['name'])}</td>"
            f"<td class=\"{cls}\">{html.escape(item['status'])}</td>"
            f"<td>{html.escape(item['progress'])}</td>"
            f"<td>{html.escape(item['evidence'])}</td>"
            f"<td>{html.escape(item['next'])}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def html_report(
    metrics: dict[str, Any],
    methodology_metrics: dict[str, Any],
    grouping_metrics: dict[str, Any],
    stability_metrics: dict[str, Any],
    sweep_metrics: dict[str, Any],
    ablation_metrics: dict[str, Any],
    scale_grouping_metrics: dict[str, Any],
    scale_stability_metrics: dict[str, Any],
    scale_sweep_metrics: dict[str, Any],
    scale_ablation_metrics: dict[str, Any],
    report: Path,
    gallery: Path,
    features: Path,
    methodology_report: Path,
    grouping_report: Path,
    stability_report: Path,
    sweep_report: Path,
    ablation_report: Path,
    overview_doc: Path,
    scale_grouping_report: Path,
    scale_stability_report: Path,
    scale_sweep_report: Path,
    scale_ablation_report: Path,
    confidence_report: Path,
    metrics_path: Path,
    out: Path,
) -> str:
    sample_count = metrics.get("sample_count", 0)
    pattern_counts = metrics.get("pattern_counts", {})
    shot_modes = metrics.get("shot_mode_counts", {})
    active_grouping_metrics = scale_grouping_metrics or grouping_metrics
    active_sweep_metrics = scale_sweep_metrics or sweep_metrics
    retrieval_lift = methodology_metrics.get("retrieval", {}).get("jaccard_lift", 0.0)
    grouping_lift = active_grouping_metrics.get("similarity", {}).get("jaccard_lift", retrieval_lift)
    recommended_k = active_sweep_metrics.get("recommended_cluster_count_cpu_pilot", "-")
    scale_sample_count = scale_grouping_metrics.get("sample_count", 0)
    edge_count = metrics.get("edge_fail_density_higher_count", 0)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    all_acceptance = bool(metrics.get("acceptance")) and all(metrics["acceptance"].values())
    current_position = (
        "Phase 2 진행 중: 155장 scale pilot 기준 FBM feature 유사검색, coarse grouping, stability, ablation 검증 단계."
        if scale_grouping_metrics
        else "Phase 2 진행 중: FBM observable feature 기반 유사 wafer 검색, coarse grouping, ablation 검증 단계."
    )

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>WaferMap 프로젝트 진행 브리핑</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ color: #111827; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; margin: 18px 0; }}
    .card {{ border: 1px solid #d8dee9; border-radius: 8px; padding: 14px; background: #f8fafc; }}
    .metric {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f7; }}
    .pass, .done {{ color: #096b3b; font-weight: 700; }}
    .fail {{ color: #b42318; font-weight: 700; }}
    .progress {{ color: #9a5b00; font-weight: 700; }}
    .started {{ color: #1d4ed8; font-weight: 700; }}
    .pending {{ color: #6b7280; font-weight: 700; }}
    .note {{ background: #eef6ff; border-left: 4px solid #2563eb; padding: 12px 14px; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
    a {{ color: #1d4ed8; }}
  </style>
</head>
<body>
  <h1>WaferMap 프로젝트 진행 브리핑</h1>
  <p>생성 시각: <code>{generated_at}</code>. 현재 위치: <strong>{html.escape(current_position)}</strong></p>

  <div class="note">
    현재 프로젝트는 기본 문제 정의와 generator scaffold를 넘어섰다. 지금의 핵심은 FBM 자체에서 정보를 추출하고, 유사한 불량 패턴의 wafer를 그룹핑할 수 있는지 검증하는 것이다.
  </div>

  <div class="summary">
    <div class="card"><div>Review Sample</div><div class="metric">{sample_count}</div></div>
    <div class="card"><div>Scale Pilot</div><div class="metric">{scale_sample_count or '-'}</div></div>
    <div class="card"><div>Acceptance</div><div class="metric {'pass' if all_acceptance else 'fail'}">{'PASS' if all_acceptance else 'CHECK'}</div></div>
    <div class="card"><div>Edge &gt; Center</div><div class="metric">{edge_count}/{sample_count}</div></div>
    <div class="card"><div>그룹핑 검색 Lift</div><div class="metric">{grouping_lift:.2f}x</div></div>
    <div class="card"><div>권장 K</div><div class="metric">{recommended_k}</div></div>
  </div>

  <h2>Phase 진행 상태</h2>
  <table>
    <tr><th>Phase</th><th>이름</th><th>상태</th><th>진행률</th><th>근거</th><th>다음 작업</th></tr>
    {phase_table(metrics, methodology_metrics, grouping_metrics, stability_metrics, sweep_metrics, ablation_metrics, scale_grouping_metrics, scale_stability_metrics, scale_sweep_metrics, scale_ablation_metrics)}
  </table>

  <h2>최신 Review Metric</h2>
  <table>
    <tr><th>항목</th><th>상태</th></tr>
    {acceptance_table(metrics)}
  </table>

  <h2>관측된 Pattern Count</h2>
  <table>
    <tr><th>항목</th><th>값</th></tr>
    <tr><td>Pattern counts</td><td>{html.escape(json.dumps(pattern_counts, ensure_ascii=False))}</td></tr>
    <tr><td>Shot-grid modes</td><td>{html.escape(json.dumps(shot_modes, ensure_ascii=False))}</td></tr>
  </table>

  <h2>Review Preset</h2>
  <table>
    <tr><th>Preset</th><th>목적</th></tr>
    <tr><td><code>configs/synth/presets/review_balanced.json</code></td><td>기본 realism review mix.</td></tr>
    <tr><td><code>configs/synth/presets/review_edge_heavy.json</code></td><td>Edge baseline과 edge-chip face gradient를 강하게 보는 설정.</td></tr>
    <tr><td><code>configs/synth/presets/review_shot_relative.json</code></td><td>Shot-relative lower-left와 edge-band behavior를 강하게 보는 설정.</td></tr>
  </table>

  <h2>산출물</h2>
  <ul>
    <li><a href="{html.escape(relpath(report, out))}">Final review report</a></li>
    <li><a href="{html.escape(relpath(methodology_report, out))}">방법론 검증 리포트</a></li>
    <li><a href="{html.escape(relpath(grouping_report, out))}">FBM 그룹핑 리포트</a></li>
    <li><a href="{html.escape(relpath(stability_report, out))}">FBM 그룹핑 안정성 리포트</a></li>
    <li><a href="{html.escape(relpath(sweep_report, out))}">FBM 그룹핑 파라미터 스윕 리포트</a></li>
    <li><a href="{html.escape(relpath(ablation_report, out))}">FBM feature family ablation 리포트</a></li>
    <li><a href="{html.escape(relpath(overview_doc, out))}">Project overview 문서</a></li>
    <li><a href="{html.escape(relpath(scale_grouping_report, out))}">Scale FBM 그룹핑 리포트</a></li>
    <li><a href="{html.escape(relpath(scale_stability_report, out))}">Scale FBM 안정성 리포트</a></li>
    <li><a href="{html.escape(relpath(scale_sweep_report, out))}">Scale FBM 파라미터 스윕 리포트</a></li>
    <li><a href="{html.escape(relpath(scale_ablation_report, out))}">Scale FBM feature family ablation 리포트</a></li>
    <li><a href="{html.escape(relpath(confidence_report, out))}">Scale retrieval confidence 리포트</a></li>
    <li><a href="{html.escape(relpath(gallery, out))}">Final review gallery</a></li>
    <li><a href="{html.escape(relpath(features, out))}">Feature CSV</a></li>
    <li><a href="{html.escape(relpath(metrics_path, out))}">Metrics JSON</a></li>
  </ul>

  <h2>추천 다음 작업</h2>
  <ol>
    <li>현재 고정한 155장 scale pilot을 기준으로 real-unlabeled inference workflow를 설계한다.</li>
    <li>Scratch/local처럼 작은 공간 불량은 morphology feature와 segmentation baseline으로 더 보강한다.</li>
    <li>필요하면 generator runtime을 더 줄인 뒤 200장 이상 full batch를 재시도한다. 현재 노트북 CPU에서는 155장이 실용적인 scale check다.</li>
    <li>real wafer batch에서 feature extraction과 nearest-neighbor review를 실행하는 sanity-check workflow를 정리한다.</li>
  </ol>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    report = Path(args.report)
    gallery = Path(args.gallery)
    features = Path(args.features)
    methodology_report = Path(args.methodology_report)
    grouping_report = Path(args.grouping_report)
    stability_report = Path(args.stability_report)
    sweep_report = Path(args.sweep_report)
    ablation_report = Path(args.ablation_report)
    overview_doc = Path(args.scale_doc)
    scale_grouping_report = Path(args.scale_grouping_report)
    scale_stability_report = Path(args.scale_stability_report)
    scale_sweep_report = Path(args.scale_sweep_report)
    scale_ablation_report = Path(args.scale_ablation_report)
    confidence_report = Path(args.confidence_report)
    out = Path(args.out)
    metrics = read_metrics(metrics_path)
    methodology_metrics = read_metrics(Path(args.methodology_metrics))
    grouping_metrics = read_metrics(Path(args.grouping_metrics))
    stability_metrics = read_metrics(Path(args.stability_metrics))
    sweep_metrics = read_metrics(Path(args.sweep_metrics))
    ablation_metrics = read_metrics(Path(args.ablation_metrics))
    scale_grouping_metrics = read_metrics(Path(args.scale_grouping_metrics))
    scale_stability_metrics = read_metrics(Path(args.scale_stability_metrics))
    scale_sweep_metrics = read_metrics(Path(args.scale_sweep_metrics))
    scale_ablation_metrics = read_metrics(Path(args.scale_ablation_metrics))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        html_report(
            metrics,
            methodology_metrics,
            grouping_metrics,
            stability_metrics,
            sweep_metrics,
            ablation_metrics,
            scale_grouping_metrics,
            scale_stability_metrics,
            scale_sweep_metrics,
            scale_ablation_metrics,
            report,
            gallery,
            features,
            methodology_report,
            grouping_report,
            stability_report,
            sweep_report,
            ablation_report,
            overview_doc,
            scale_grouping_report,
            scale_stability_report,
            scale_sweep_report,
            scale_ablation_report,
            confidence_report,
            metrics_path,
            out,
        ),
        encoding="utf-8",
    )
    print(f"Wrote briefing: {out}")


if __name__ == "__main__":
    main()
