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


MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
HTML_HREF_RE = re.compile(r"""href=["']([^"']+)["']""")
MOJIBAKE_RE = re.compile(r"\ufffd")


def test_core_documentation_files_exist() -> None:
    missing = [path.relative_to(ROOT).as_posix() for path in USER_FACING_DOCS if not path.exists()]
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
    html_path = DOCS / "index.html"
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


def test_core_documentation_has_no_replacement_mojibake_codepoints() -> None:
    offenders: list[str] = []
    for path in USER_FACING_DOCS:
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if MOJIBAKE_RE.search(line):
                offenders.append(f"{path.relative_to(ROOT).as_posix()}:{line_no}")
                break
    assert offenders == []


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

    assert "core_direction.md" in docs_index
    assert "end_to_end_workflow.md" in docs_index
    assert "FBM maps" in core_direction
    assert "defect generation" in core_direction
    assert "multi-defect synthetic maps" in core_direction
    assert "multi-defect segmentation training and validation" in core_direction
    assert "real-data pattern asset extraction" in core_direction
    assert "End-To-End Workflow" in workflow
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
    assert "fbm_data_flow_guide.md" in docs_index
    assert "project_overview.md" in docs_index
    assert "roadmap.md" in docs_index
    assert "glossary.md" in docs_index
    assert "real_png_operator_runbook.md" in docs_index
    assert "experiment_history.md" in docs_index
    assert "scripts command map" in docs_index
    assert "Product Boundary" in architecture
    assert "Package Boundaries" in architecture
    assert "src/wafermap" in architecture
    assert "scripts/README.md" in architecture
    assert "Operator Manual" in operator_manual
    assert "Troubleshooting" in operator_manual
    assert "Release Checklist" in operator_manual
    assert "Primary In-Repo Segmentation Pipeline" in scripts_map
    assert "export_unet_predictions.py" in scripts_map
    assert "Compatibility" in scripts_map
    assert "Research / Historical Evaluation" in scripts_map
    assert "in-repo segmentation tool" in overview
    assert "run_segmentation_tool.py" in overview
    assert "run_pattern_asset_editor.py" in overview
    assert "hybrid synthetic data" in overview
    assert "train_unet_segmentation.py" in overview
    assert "export_unet_predictions.py" in overview
    assert "Direct Segmentation Tool" in roadmap
    assert "Small U-Net Training" in roadmap
    assert "Active Learning" in roadmap
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
    assert "resize-only representation" in experiment_history
    assert "patch proposal" in experiment_history
    assert "Segmentation Smoke Test" in experiment_history
    assert "Phase 4" in roadmap_html
    assert "core_direction.md" in roadmap_html
    assert "end_to_end_workflow.md" in roadmap_html
    assert "glossary.md" in roadmap_html
    assert "`severity`" in glossary
    assert "`retrieval_failure_mode`" in glossary


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
