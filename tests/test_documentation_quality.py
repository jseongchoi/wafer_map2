from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


USER_FACING_DOCS = [
    ROOT / "README.md",
    ROOT / "scripts" / "README.md",
    DOCS / "index.html",
    DOCS / "README.md",
    DOCS / "core_direction.md",
    DOCS / "end_to_end_workflow.md",
    DOCS / "architecture.md",
    DOCS / "operator_manual.md",
    DOCS / "segmentation_tool_workflow.md",
    DOCS / "glossary.md",
    DOCS / "fbm_data_flow_guide.md",
    DOCS / "fbm_pattern_asset_pipeline.md",
    DOCS / "semiconductor_ai_review.md",
    DOCS / "project_overview.md",
    DOCS / "label_data_guidelines.md",
    DOCS / "training_data_contract.md",
    DOCS / "similar_map_retrieval_guide.md",
    DOCS / "defect_severity_scoring_guide.md",
    DOCS / "documentation_quality_audit.md",
    DOCS / "experiment_history.md",
    DOCS / "roadmap.md",
    DOCS / "data_schema.md",
    DOCS / "real_png_operator_runbook.md",
    DOCS / "real_unlabeled_workflow.md",
    DOCS / "expert_review_protocol.md",
    DOCS / "real_wafer_review_checklist.md",
    DOCS / "enterprise_readiness_assessment.md",
    DOCS / "modeling_strategy.md",
    DOCS / "validation_protocol.md",
    DOCS / "pattern_taxonomy.md",
    DOCS / "legacy_pattern_asset_editor.md",
]

GENERATED_HTML_DOCS = [
    DOCS / "pages" / "label_data_guidelines.html",
    DOCS / "pages" / "training_data_contract.html",
    DOCS / "pages" / "modeling_strategy.html",
    DOCS / "pages" / "similar_map_retrieval_guide.html",
    DOCS / "pages" / "defect_severity_scoring_guide.html",
    DOCS / "pages" / "end_to_end_workflow.html",
    DOCS / "pages" / "operator_manual.html",
    DOCS / "pages" / "pattern_taxonomy.html",
    DOCS / "pages" / "fbm_pattern_asset_pipeline.html",
    DOCS / "pages" / "documentation_quality_audit.html",
    DOCS / "pages" / "scripts_README.html",
]


MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
HTML_HREF_RE = re.compile(r"""href=["']([^"']+)["']""")
MOJIBAKE_RE = re.compile(r"\ufffd")


def test_core_documentation_files_exist() -> None:
    missing = [path.relative_to(ROOT).as_posix() for path in USER_FACING_DOCS if not path.exists()]
    assert missing == []


def test_generated_html_document_pages_exist() -> None:
    missing = [path.relative_to(ROOT).as_posix() for path in GENERATED_HTML_DOCS if not path.exists()]
    assert missing == []


def test_markdown_file_links_resolve() -> None:
    broken: list[str] = []
    for markdown_path in USER_FACING_DOCS:
        if markdown_path.suffix.lower() != ".md":
            continue
        text = markdown_path.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK_RE.findall(text):
            target = raw_target.split("#", 1)[0].strip()
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (markdown_path.parent / target).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                continue
            if not resolved.exists():
                broken.append(f"{markdown_path.relative_to(ROOT).as_posix()} -> {raw_target}")
    assert broken == []


def test_html_file_links_resolve() -> None:
    broken: list[str] = []
    html_paths = [DOCS / "index.html", *sorted((DOCS / "pages").glob("*.html"))]
    for html_path in html_paths:
        text = html_path.read_text(encoding="utf-8")
        for raw_target in HTML_HREF_RE.findall(text):
            target = raw_target.split("#", 1)[0].strip()
            if not target or "://" in target:
                continue
            resolved = (html_path.parent / target).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                continue
            if not resolved.exists():
                broken.append(f"{html_path.relative_to(ROOT).as_posix()} -> {raw_target}")
    assert broken == []


def test_documentation_home_links_to_styled_html_pages() -> None:
    index_html = (DOCS / "index.html").read_text(encoding="utf-8")
    local_targets = [
        target.split("#", 1)[0].strip()
        for target in HTML_HREF_RE.findall(index_html)
        if target and "://" not in target
    ]
    markdown_targets = [target for target in local_targets if target.endswith(".md")]
    assert markdown_targets == []
    assert "pages/label_data_guidelines.html" in local_targets
    assert "pages/training_data_contract.html" in local_targets


def test_core_documentation_has_no_replacement_mojibake_codepoints() -> None:
    offenders: list[str] = []
    for path in USER_FACING_DOCS:
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if MOJIBAKE_RE.search(line):
                offenders.append(f"{path.relative_to(ROOT).as_posix()}:{line_no}")
                break
    assert offenders == []


def test_core_documentation_is_korean_and_example_driven() -> None:
    core_docs = [
        DOCS / "core_direction.md",
        DOCS / "end_to_end_workflow.md",
        DOCS / "operator_manual.md",
        DOCS / "segmentation_tool_workflow.md",
        DOCS / "fbm_data_flow_guide.md",
        DOCS / "fbm_pattern_asset_pipeline.md",
        DOCS / "validation_protocol.md",
        DOCS / "modeling_strategy.md",
        DOCS / "similar_map_retrieval_guide.md",
        DOCS / "defect_severity_scoring_guide.md",
        DOCS / "documentation_quality_audit.md",
    ]

    weak_docs: list[str] = []
    for path in core_docs:
        text = path.read_text(encoding="utf-8")
        korean_count = len(re.findall(r"[가-힣]", text))
        example_signals = len(re.findall(r"```|예시|json|powershell", text, flags=re.IGNORECASE))
        if korean_count < 250 or example_signals < 3:
            weak_docs.append(path.relative_to(ROOT).as_posix())

    assert weak_docs == []


def test_documentation_guides_current_project_direction() -> None:
    docs_index = (DOCS / "README.md").read_text(encoding="utf-8")
    core_direction = (DOCS / "core_direction.md").read_text(encoding="utf-8")
    workflow = (DOCS / "end_to_end_workflow.md").read_text(encoding="utf-8")
    overview = (DOCS / "project_overview.md").read_text(encoding="utf-8")
    roadmap = (DOCS / "roadmap.md").read_text(encoding="utf-8")
    pipeline = (DOCS / "fbm_pattern_asset_pipeline.md").read_text(encoding="utf-8")
    data_flow = (DOCS / "fbm_data_flow_guide.md").read_text(encoding="utf-8")
    architecture = (DOCS / "architecture.md").read_text(encoding="utf-8")
    operator_manual = (DOCS / "operator_manual.md").read_text(encoding="utf-8")
    segmentation_workflow = (DOCS / "segmentation_tool_workflow.md").read_text(encoding="utf-8")
    scripts_map = (ROOT / "scripts" / "README.md").read_text(encoding="utf-8")
    experiment_history = (DOCS / "experiment_history.md").read_text(encoding="utf-8")
    glossary = (DOCS / "glossary.md").read_text(encoding="utf-8")
    roadmap_html = (DOCS / "index.html").read_text(encoding="utf-8")
    label_guidelines = (DOCS / "label_data_guidelines.md").read_text(encoding="utf-8")
    training_contract = (DOCS / "training_data_contract.md").read_text(encoding="utf-8")
    modeling_strategy = (DOCS / "modeling_strategy.md").read_text(encoding="utf-8")
    retrieval_guide = (DOCS / "similar_map_retrieval_guide.md").read_text(encoding="utf-8")
    severity_guide = (DOCS / "defect_severity_scoring_guide.md").read_text(encoding="utf-8")
    taxonomy = (DOCS / "pattern_taxonomy.md").read_text(encoding="utf-8")
    audit = (DOCS / "documentation_quality_audit.md").read_text(encoding="utf-8")

    assert "core_direction.md" in docs_index
    assert "end_to_end_workflow.md" in docs_index
    assert "FBM maps" in core_direction
    assert "defect generation" in core_direction
    assert "multi-defect synthetic maps" in core_direction
    assert "multi-defect segmentation training and validation" in core_direction
    assert "real-data pattern asset extraction" in core_direction
    assert "불량 하나씩 vertical slice로 닫습니다" in core_direction
    assert "신규 defect 발견" in core_direction
    assert "family = 모델 target" in core_direction
    assert "subtype = 초기에는 분석/리뷰 metadata" in core_direction
    assert "전체 실행 흐름" in workflow
    assert "run_segmentation_tool.py" in workflow
    assert "compose_synthetic_from_assets.py" in workflow
    assert "train_unet_segmentation.py" in workflow
    assert "export_unet_predictions.py" in workflow
    assert "fbm_prediction_masks/v1" in workflow
    assert "architecture.md" in docs_index
    assert "operator_manual.md" in docs_index
    assert "segmentation_tool_workflow.md" in docs_index
    assert "fbm_pattern_asset_pipeline.md" in docs_index
    assert "semiconductor_ai_review.md" in docs_index
    assert "label_data_guidelines.md" in docs_index
    assert "training_data_contract.md" in docs_index
    assert "similar_map_retrieval_guide.md" in docs_index
    assert "defect_severity_scoring_guide.md" in docs_index
    assert "fbm_data_flow_guide.md" in docs_index
    assert "project_overview.md" in docs_index
    assert "roadmap.md" in docs_index
    assert "glossary.md" in docs_index
    assert "real_png_operator_runbook.md" in docs_index
    assert "experiment_history.md" in docs_index
    assert "실행 명령 지도" in docs_index
    assert "제품 경계" in architecture
    assert "패키지 경계" in architecture
    assert "src/wafermap" in architecture
    assert "scripts/README.md" in architecture
    assert "작업자 매뉴얼" in operator_manual
    assert "Troubleshooting" in operator_manual
    assert "Release Checklist" in operator_manual
    assert "현재 주력 Segmentation Pipeline" in scripts_map
    assert "export_unet_predictions.py" in scripts_map
    assert "호환성" in scripts_map
    assert "연구/이력용 평가" in scripts_map
    assert "segmentation tool" in overview
    assert "run_segmentation_tool.py" in overview
    assert "pattern asset library" in overview
    assert "합성 데이터" in overview
    assert "train_unet_segmentation.py" in overview
    assert "export_unet_predictions.py" in overview
    assert "Direct Segmentation Tool" in roadmap
    assert "Small U-Net Training" in roadmap
    assert "Active Learning" in roadmap
    assert "Family별 지속 학습 루프" in roadmap
    assert "definition_ready" in roadmap
    assert "human asset primary" in pipeline
    assert "procedural fallback" in pipeline
    assert "export_unet_predictions.py" in pipeline
    assert "run_segmentation_tool.py" in segmentation_workflow
    assert "--prediction-json" in segmentation_workflow
    assert "export_unet_predictions.py" in segmentation_workflow
    assert "data/pattern_assets" in segmentation_workflow
    assert "data/pattern_assets" in data_flow
    assert "data/synthetic/asset_composed" in data_flow
    assert "asset_segmentation_manifest.csv" in data_flow
    assert "coordinate-aware small U-Net" in data_flow
    assert "export_unet_predictions.py" in data_flow
    assert "학습 데이터 규격" in data_flow
    assert "readiness" in pipeline
    assert "resize-only representation" in experiment_history
    assert "patch proposal" in experiment_history
    assert "Segmentation Smoke Test" in experiment_history
    assert "Phase 4" in roadmap_html
    assert "이 파일부터 보면 됩니다" in roadmap_html
    assert "학습 데이터 규격" in roadmap_html
    assert "구현 위치 지도" in roadmap_html
    assert "core_direction.md" in roadmap_html
    assert "end_to_end_workflow.md" in roadmap_html
    assert "label_data_guidelines.md" in roadmap_html
    assert "training_data_contract.md" in roadmap_html
    assert "modeling_strategy.html" in roadmap_html
    assert "similar_map_retrieval_guide.html" in roadmap_html
    assert "defect_severity_scoring_guide.html" in roadmap_html
    assert "pages/glossary.html" in roadmap_html
    assert "documentation_quality_audit.html" in roadmap_html
    assert "`severity`" in glossary
    assert "`retrieval_failure_mode`" in glossary
    assert "bbox_xywh = 어디를 볼지 알려주는 사각형 힌트" in label_guidelines
    assert "mask.png  = U-Net이 실제로 학습하는 정답" in label_guidelines
    assert "local blob mask는 이렇게 만듭니다" in label_guidelines
    assert "threshold_assisted_brush" in label_guidelines
    assert "subtype_status" in label_guidelines
    assert "full_mask[y:y+h, x:x+w]" in label_guidelines
    assert "parametric_mask" in label_guidelines
    assert "local 안의 subtype은 처음에는 metadata로 둡니다" in taxonomy
    assert "metadata_only" in taxonomy
    assert "target_channel" in taxonomy
    assert "초기: 해석 가능한 feature 기반 nearest-neighbor" in retrieval_guide
    assert "2차 재정렬" in retrieval_guide
    assert "subtype은 target channel로 승격하기 전에도 검색 metadata" in retrieval_guide
    assert "encoder embedding" in retrieval_guide
    assert "confidence와 severity는 다릅니다" in severity_guide
    assert "defect_severity/v1" in severity_guide
    assert "final_severity_score" in severity_guide
    assert "severity_bucket" in severity_guide
    assert "rule-based score" in severity_guide
    assert "arrays.npz" in training_contract
    assert "pattern_masks" in training_contract
    assert "severity_mean" in training_contract
    assert "target = pattern_mask & wafer_mask & valid_test_mask" in training_contract
    assert "BCEWithLogitsLoss" in modeling_strategy
    assert "sigmoid multi-label segmentation" in modeling_strategy
    assert "X.shape = [12, output_size, output_size]" in modeling_strategy
    assert "Y.shape = [6, output_size, output_size]" in modeling_strategy
    assert "문서별 검증 결과" in audit
    assert "대표 불량 패턴" in audit
    assert "U-Net 구조" in audit


def test_core_direction_avoids_removed_workflows() -> None:
    active_docs = [
        ROOT / "README.md",
        DOCS / "README.md",
        DOCS / "core_direction.md",
        DOCS / "end_to_end_workflow.md",
        DOCS / "project_overview.md",
        DOCS / "segmentation_tool_workflow.md",
        DOCS / "fbm_pattern_asset_pipeline.md",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8").lower() for path in active_docs)

    forbidden = [
        "cv" + "at",
        "nap" + "ari",
        "sec" + "urity",
        "access" + "-control",
        "prove" + "nance",
        "sha" + "256",
    ]
    offenders = [term for term in forbidden if term in combined]
    assert offenders == []


def test_real_png_operator_guide_has_required_execution_and_share_items() -> None:
    runbook = (DOCS / "real_png_operator_runbook.md").read_text(encoding="utf-8")

    required_phrases = [
        "scripts/analyze_png_raw_folders.py",
        "--geometry-json",
        "outputs/manifests/real_png_batch_manifest.json",
        "outputs/reports/real_png_batch",
        "scripts/summarize_expert_review.py",
        "actual_net_die=0",
    ]
    missing = [phrase for phrase in required_phrases if phrase not in runbook]
    assert missing == []
