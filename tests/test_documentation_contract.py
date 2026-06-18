from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


CORE_DOCS = [
    ROOT / "README.md",
    DOCS / "index.html",
    DOCS / "README.md",
    DOCS / "glossary.md",
    DOCS / "project_overview.md",
    DOCS / "experiment_history.md",
    DOCS / "roadmap.md",
    DOCS / "data_schema.md",
    DOCS / "real_unlabeled_workflow.md",
    DOCS / "expert_review_protocol.md",
    DOCS / "real_wafer_review_checklist.md",
    DOCS / "modeling_strategy.md",
    DOCS / "validation_protocol.md",
    DOCS / "pattern_taxonomy.md",
]


MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
HTML_HREF_RE = re.compile(r"""href=["']([^"']+)["']""")


def test_core_documentation_files_exist() -> None:
    missing = [path.relative_to(ROOT).as_posix() for path in CORE_DOCS if not path.exists()]
    assert missing == []


def test_markdown_file_links_resolve() -> None:
    broken: list[str] = []
    for markdown_path in CORE_DOCS:
        text = markdown_path.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK_RE.findall(text):
            target = raw_target.split("#", 1)[0].strip()
            if not target or "://" in target:
                continue
            if target.startswith("mailto:"):
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


def test_documentation_guides_experiment_history() -> None:
    docs_index = (DOCS / "README.md").read_text(encoding="utf-8")
    overview = (DOCS / "project_overview.md").read_text(encoding="utf-8")
    experiment_history = (DOCS / "experiment_history.md").read_text(encoding="utf-8")
    glossary = (DOCS / "glossary.md").read_text(encoding="utf-8")
    roadmap_html = (DOCS / "index.html").read_text(encoding="utf-8")

    assert "index.html" in docs_index
    assert "glossary.md" in docs_index
    assert "experiment_history.md" in docs_index
    assert "실험과 판단 기록" in overview
    assert "resize-only representation" in experiment_history
    assert "patch proposal" in experiment_history
    assert "Segmentation Smoke Test" in experiment_history
    assert "라벨 없는 실제 Wafer 처리 절차" in experiment_history
    assert "라벨 없는 실제 wafer 적용 준비 단계" in roadmap_html
    assert "Phase 4" in roadmap_html
    assert "처음 보는 용어와 변수" in roadmap_html
    assert "glossary.md" in roadmap_html
    assert "`severity`" in glossary
    assert "`retrieval_failure_mode`" in glossary
