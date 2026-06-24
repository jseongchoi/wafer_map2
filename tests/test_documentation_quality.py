from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


USER_FACING_DOCS = [
    ROOT / "README.md",
    DOCS / "index.html",
    DOCS / "README.md",
    DOCS / "glossary.md",
    DOCS / "fbm_pattern_asset_pipeline.md",
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
    overview = (DOCS / "project_overview.md").read_text(encoding="utf-8")
    roadmap = (DOCS / "roadmap.md").read_text(encoding="utf-8")
    pipeline = (DOCS / "fbm_pattern_asset_pipeline.md").read_text(encoding="utf-8")
    experiment_history = (DOCS / "experiment_history.md").read_text(encoding="utf-8")
    glossary = (DOCS / "glossary.md").read_text(encoding="utf-8")
    roadmap_html = (DOCS / "index.html").read_text(encoding="utf-8")

    assert "fbm_pattern_asset_pipeline.md" in docs_index
    assert "project_overview.md" in docs_index
    assert "roadmap.md" in docs_index
    assert "glossary.md" in docs_index
    assert "real_png_operator_runbook.md" in docs_index
    assert "experiment_history.md" in docs_index
    assert "hybrid synthetic data" in overview
    assert "train_unet_segmentation.py" in overview
    assert "Pattern Asset Builder" in roadmap
    assert "Small U-Net" in roadmap
    assert "active learning" in roadmap
    assert "procedural fallback" in pipeline
    assert "resize-only representation" in experiment_history
    assert "patch proposal" in experiment_history
    assert "Segmentation Smoke Test" in experiment_history
    assert "Phase 4" in roadmap_html
    assert "glossary.md" in roadmap_html
    assert "`severity`" in glossary
    assert "`retrieval_failure_mode`" in glossary


def test_real_png_operator_guide_has_required_execution_and_share_items() -> None:
    runbook = (DOCS / "real_png_operator_runbook.md").read_text(encoding="utf-8")

    required_phrases = [
        "scripts/analyze_png_raw_folders.py",
        "--production-run",
        "--geometry-json",
        "outputs/reports/real_png_batch",
        "outputs/private/*_manifest.json",
        "scripts/audit_project_readiness.py",
        "scripts/summarize_expert_review.py",
        "actual_net_die=0",
    ]
    missing = [phrase for phrase in required_phrases if phrase not in runbook]
    assert missing == []
