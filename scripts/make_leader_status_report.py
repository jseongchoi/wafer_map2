"""Build a leader-facing status and refactor audit report."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "reports" / "leader_status_and_refactor_audit.html"


def read_json(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def read_json_if_exists(path: str) -> dict[str, Any] | None:
    full_path = ROOT / path
    if not full_path.exists():
        return None
    return json.loads(full_path.read_text(encoding="utf-8"))


def relpath(target: str | Path, base_file: Path = OUT) -> str:
    return Path(os.path.relpath(Path(target).resolve(), base_file.resolve().parent)).as_posix()


def metric_card(title: str, value: str, note: str = "") -> str:
    return (
        '<div class="card">'
        f"<div class=\"label\">{html.escape(title)}</div>"
        f"<div class=\"metric\">{html.escape(value)}</div>"
        f"<div class=\"small\">{html.escape(note)}</div>"
        "</div>"
    )


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def class_rows(confidence: dict[str, Any]) -> str:
    rows = []
    for item in confidence["class_metrics"]:
        cls = item["class"]
        lift = float(item["lift"])
        strength = "강함" if lift >= 1.35 else "보통" if lift >= 1.10 else "약함"
        rows.append(
            "<tr>"
            f"<td>{html.escape(cls)}</td>"
            f"<td>{item['positive_count']}</td>"
            f"<td>{item['precision_at_k']:.3f}</td>"
            f"<td>{lift:.2f}x</td>"
            f"<td>{strength}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def cluster_rows(grouping: dict[str, Any]) -> str:
    rows = []
    for item in grouping["cluster_summaries"]:
        features = ", ".join(f"{f['feature']} ({f['mean_z']:+.2f}z)" for f in item["top_feature_deviations"][:4])
        reps = ", ".join(item["representative_samples"][:3])
        rows.append(
            "<tr>"
            f"<td>{item['cluster_id']}</td>"
            f"<td>{item['size']}</td>"
            f"<td>{html.escape(features)}</td>"
            f"<td>{html.escape(reps)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def refactor_table() -> str:
    findings = [
        (
            "데이터 의미 분리",
            "양호",
            "Grade 0, none-wafer, valid-test, stby가 tensor/mask로 분리되어 있다. Synthetic oracle feature도 inference feature에서 제외된다.",
            "유지. real parser에서도 같은 semantic 기준을 강제한다.",
        ),
        (
            "테스트 커버리지",
            "양호",
            "synthetic generator, feature, retrieval, real-unlabeled workflow, expert review protocol 테스트가 존재하고 전체 테스트가 통과했다.",
            "다음 real adapter가 생기면 stby/Grade0/none-wafer fixture를 추가한다.",
        ),
        (
            "Generator 크기",
            "주의",
            "src/wafermap/synth/generator.py가 약 730라인으로 커졌다. 다만 지금은 defect realism이 계속 바뀌는 단계라 조기 분리는 비용이 크다.",
            "holdout 검증 후 geometry, pattern draw, stby placement 정도로 분리한다.",
        ),
        (
            "Real loader 크기",
            "주의",
            "extract_real_unlabeled_features.py가 loader, validator, nearest-neighbor, HTML report를 함께 가진 MVP 구조다.",
            "실제 보안 npz 1회 검증 후 validator와 report helper를 src로 승격한다.",
        ),
        (
            "Report scripts 중복",
            "허용",
            "여러 script가 relpath/html style/read_csv 같은 작은 helper를 반복한다.",
            "보고서 형식이 안정되면 reporting 공통 모듈로 묶는다. 지금은 과한 추상화 금지.",
        ),
        (
            "Generated outputs",
            "양호",
            "outputs/**, data/synthetic/**는 .gitignore 대상이고 재생성 산출물로 관리된다.",
            "HTML 보고서는 의사결정용 산출물로 유지하되 source of truth는 scripts/docs/configs로 둔다.",
        ),
        (
            "other_sample",
            "주의",
            "사용자 reference sample/실험 코드가 별도 폴더에 남아 있다. 현재 pipeline에는 직접 연결하지 않는다.",
            "reference 목적을 문서에 남기거나, 확정되면 configs/synth preset로 흡수한다.",
        ),
    ]
    return "\n".join(
        "<tr>"
        f"<td>{html.escape(area)}</td>"
        f"<td>{html.escape(status)}</td>"
        f"<td>{html.escape(finding)}</td>"
        f"<td>{html.escape(action)}</td>"
        "</tr>"
        for area, status, finding, action in findings
    )


def build_html() -> str:
    confidence = read_json("outputs/reports/fbm_retrieval_confidence_scale_metrics.json")
    grouping = read_json("outputs/reports/fbm_grouping_scale_metrics.json")
    review = read_json("outputs/reports/expert_review_summary_metrics.json")
    holdout = read_json_if_exists("outputs/reports/fbm_retrieval_confidence_holdout_metrics.json")
    holdout_interest = read_json_if_exists("outputs/reports/fbm_interest_retrieval_holdout_metrics.json")
    feature_retrieval = read_json_if_exists("outputs/reports/fbm_defect_feature_retrieval_scale_metrics.json")
    holdout_feature_retrieval = read_json_if_exists("outputs/reports/fbm_defect_feature_retrieval_holdout_metrics.json")
    resize = read_json_if_exists("outputs/reports/fbm_resize_benchmark_scale_metrics.json")

    lift = float(confidence["jaccard_lift"])
    ci = confidence["bootstrap"]
    p_value = float(confidence["permutation_p_value"])
    pca_ev = grouping["pca_explained_variance"]
    top_k = confidence["top_k"]

    cards = "\n".join(
        [
            metric_card("Scale samples", str(confidence["sample_count"]), "현재 CPU pilot batch"),
            metric_card("Observable features", str(confidence["feature_count_observable"]), "real inference에 쓸 수 있는 feature"),
            metric_card("Top-k retrieval lift", f"{lift:.2f}x", f"95% CI {ci['lift_ci_low']:.2f}x - {ci['lift_ci_high']:.2f}x"),
            metric_card("Permutation p-value", f"{p_value:.3f}", "synthetic validation 기준"),
            metric_card(
                "Holdout lift",
                f"{holdout['jaccard_lift']:.2f}x" if holdout else "pending",
                "stress smoke 기준" if holdout else "아직 미실행",
            ),
            metric_card(
                "Feature-key lift",
                f"{feature_retrieval['summary_by_target_kind']['feature_key']['mean_lift']:.2f}x"
                if feature_retrieval
                else "pending",
                "structured defect target 기준" if feature_retrieval else "아직 미실행",
            ),
            metric_card(
                "Resize-only lift",
                f"{resize['representations']['semantic_pool_64']['jaccard_lift']:.2f}x"
                if resize
                else "pending",
                "semantic pool 64 기준" if resize else "아직 미실행",
            ),
            metric_card("Cluster count", str(grouping["cluster_count"]), "hard label이 아니라 coarse review group"),
            metric_card("Expert review rows", f"{review['filled_review_rows']}/{review['total_rows']}", "아직 사람이 채운 리뷰는 없음"),
        ]
    )
    images = [
        ("Synthetic realism gallery", ROOT / "outputs" / "figures" / "final_review_gallery.png"),
        ("Nearest-neighbor visual check", ROOT / "outputs" / "figures" / "fbm_neighbor_gallery_scale.png"),
        ("Observable feature PCA", ROOT / "outputs" / "figures" / "fbm_grouping_scale_pca.png"),
        ("Feature family ablation", ROOT / "outputs" / "figures" / "fbm_feature_ablation_scale.png"),
    ]
    image_blocks = "\n".join(
        f"<section><h2>{html.escape(title)}</h2><img src=\"{html.escape(relpath(path))}\" alt=\"{html.escape(title)}\"></section>"
        for title, path in images
    )
    holdout_section = ""
    if holdout and holdout_interest:
        interest_rows = "\n".join(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{item.get('lift', 0.0):.2f}x</td>"
            f"<td>{item.get('precision_at_k', item.get('mean_neighbor_label_jaccard', 0.0)):.3f}</td>"
            "</tr>"
            for name, item in holdout_interest["criteria"].items()
            if name != "overall"
        )
        holdout_section = f"""
  <h2>Holdout Smoke</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Holdout samples</td><td>{holdout['sample_count']}</td></tr>
    <tr><td>Top-k retrieval lift</td><td>{holdout['jaccard_lift']:.2f}x</td></tr>
    <tr><td>Bootstrap 95% CI</td><td>{holdout['bootstrap']['lift_ci_low']:.2f}x - {holdout['bootstrap']['lift_ci_high']:.2f}x</td></tr>
    <tr><td>Permutation p-value</td><td>{holdout['permutation_p_value']:.3f}</td></tr>
  </table>
  <table>
    <tr><th>Interest criterion</th><th>Lift</th><th>Precision/Jaccard</th></tr>
    {interest_rows}
  </table>
  <section><h2>Holdout Interest Gallery</h2><img src="../figures/fbm_interest_neighbor_gallery_holdout.png" alt="Holdout interest gallery"></section>
"""
    feature_retrieval_section = ""
    if feature_retrieval:
        def feature_kind_rows(label: str, metrics: dict[str, Any]) -> str:
            return "\n".join(
                "<tr>"
                f"<td>{html.escape(label)}</td>"
                f"<td>{html.escape(kind)}</td>"
                f"<td>{item['target_count']}</td>"
                f"<td>{item['mean_precision_at_k']:.3f}</td>"
                f"<td>{item['mean_hit_rate_at_k']:.3f}</td>"
                f"<td>{item['mean_lift']:.2f}x</td>"
                "</tr>"
                for kind, item in metrics["summary_by_target_kind"].items()
            )

        rows = feature_kind_rows("scale", feature_retrieval)
        if holdout_feature_retrieval:
            rows += "\n" + feature_kind_rows("holdout", holdout_feature_retrieval)
        feature_retrieval_section = f"""
  <h2>Defect Feature Target Retrieval</h2>
  <p>구조화 defect feature target을 채점 기준으로 두고, 검색에는 observable feature만 사용한 결과다.</p>
  <table>
    <tr><th>Dataset</th><th>Target kind</th><th>Target count</th><th>Mean P@K</th><th>Mean Hit@K</th><th>Mean lift</th></tr>
    {rows}
  </table>
  <section><h2>Defect Feature Retrieval Gallery</h2><img src="../figures/fbm_defect_feature_retrieval_scale_gallery.png" alt="Defect feature retrieval gallery"></section>
"""
    resize_section = ""
    if resize:
        resize_rows = "\n".join(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{item['dimension']}</td>"
            f"<td>{item['jaccard_lift']:.2f}x</td>"
            f"<td>{item['mean_neighbor_label_jaccard']:.3f}</td>"
            f"<td>{item['class_metrics']['scratch']['lift']:.2f}x</td>"
            f"<td>{item['class_metrics']['local']['lift']:.2f}x</td>"
            "</tr>"
            for name, item in resize["representations"].items()
        )
        resize_section = f"""
  <h2>Resize / Aggregation Benchmark</h2>
  <p>리사이즈 표현을 단독 nearest-neighbor representation으로 쓸 수 있는지 확인한 결과다.</p>
  <table>
    <tr><th>Representation</th><th>Dim</th><th>Lift</th><th>Neighbor Jaccard</th><th>Scratch lift</th><th>Local lift</th></tr>
    {resize_rows}
  </table>
  <section><h2>Resize Benchmark Gallery</h2><img src="../figures/fbm_resize_benchmark_scale_gallery.png" alt="Resize benchmark gallery"></section>
"""
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>WaferMap Leader Status and Refactor Audit</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color: #1f2933; margin: 32px; line-height: 1.55; }}
    h1, h2, h3 {{ color: #111827; }}
    .summary {{ display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 12px; margin: 18px 0 28px; }}
    .card {{ border: 1px solid #d8dee9; border-radius: 8px; padding: 14px; background: #f8fafc; }}
    .label {{ color: #52606d; font-size: 13px; }}
    .metric {{ color: #102a43; font-size: 26px; font-weight: 700; margin-top: 4px; }}
    .small {{ color: #66788a; font-size: 12px; margin-top: 4px; }}
    .note {{ background: #fff7ed; border-left: 4px solid #f97316; padding: 12px 14px; margin: 14px 0; }}
    .ok {{ background: #ecfdf5; border-left-color: #10b981; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f7; }}
    img {{ width: 100%; max-width: 1280px; border: 1px solid #d8dee9; border-radius: 8px; background: white; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>WaferMap Leader Status and Refactor Audit</h1>
  <p>이 보고서는 현재 FBM 프로젝트가 원래 목표인 <strong>유사 wafer 검색, defect score, 전문가 리뷰 절차</strong>로 제대로 가고 있는지 리더 관점에서 점검한 것이다.</p>
  <div class="note ok">판정: synthetic scale pilot은 유망하다. 다만 real wafer 성능은 아직 검증 전이며, scratch/local 계열은 wafer-level feature만으로는 약해서 morphology 또는 segmentation 보강 후보로 유지한다.</div>
  <div class="summary">{cards}</div>

  <h2>핵심 결론</h2>
  <table>
    <tr><th>질문</th><th>현재 답</th></tr>
    <tr><td>원래 목표대로 가고 있는가?</td><td>그렇다. AutoEncoder 단독 접근에서 벗어나 semantic mask, observable feature, 유사 wafer 검색, expert review protocol까지 연결됐다.</td></tr>
    <tr><td>수치적으로 유망한가?</td><td>155장 synthetic scale pilot에서 top-{top_k} label-Jaccard lift가 {lift:.2f}x이고, bootstrap 95% CI는 {ci['lift_ci_low']:.2f}x - {ci['lift_ci_high']:.2f}x다.</td></tr>
    <tr><td>어떤 defect가 강한가?</td><td>shot_grid, edge, stby 계열은 triage 후보로 유효하다. ring은 보통, scratch/local은 약하다.</td></tr>
    <tr><td>지금 네가 판단해야 하는 것은?</td><td>이미지 현실성, top-k neighbor가 현업 눈에 비슷한지, 그리고 review label schema가 업무 판단과 맞는지다.</td></tr>
    <tr><td>PCA 설명력</td><td>PC1 {pca_ev[0]:.3f}, PC2 {pca_ev[1]:.3f}. 2D 그림은 구조 확인용이지 최종 성능 지표가 아니다.</td></tr>
  </table>

  <h2>Class별 검색 성능</h2>
  <table>
    <tr><th>Class</th><th>Positive count</th><th>Precision@K</th><th>Lift</th><th>판정</th></tr>
    {class_rows(confidence)}
  </table>

  {feature_retrieval_section}

  {resize_section}

  {holdout_section}

  <h2>Coarse Group 해석</h2>
  <table>
    <tr><th>Cluster</th><th>Size</th><th>Top feature deviations</th><th>Representative samples</th></tr>
    {cluster_rows(grouping)}
  </table>

  {image_blocks}

  <h2>리팩토링 점검</h2>
  <table>
    <tr><th>영역</th><th>상태</th><th>점검 내용</th><th>권장 조치</th></tr>
    {refactor_table()}
  </table>

  <h2>다음 확인 단계</h2>
  <ol>
    <li>이 보고서의 synthetic gallery와 neighbor gallery를 보고 현실성/유사성 판단을 남긴다.</li>
    <li><code>outputs/reports/expert_review_template.csv</code>의 일부 row를 채워서 accepted/mismatch 기준을 만든다.</li>
    <li>실제 보안 환경에서 semantic npz export가 가능한지 확인한다.</li>
    <li>그 다음 holdout synthetic stress test로 generator 과적합 여부를 확인한다.</li>
  </ol>
</body>
</html>
"""


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_html(), encoding="utf-8")
    print(f"Wrote leader report: {OUT}")


if __name__ == "__main__":
    main()
