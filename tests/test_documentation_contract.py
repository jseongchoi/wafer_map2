from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


CORE_DOCS = [
    ROOT / "README.md",
    DOCS / "README.md",
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


def test_documentation_guides_experiment_history() -> None:
    docs_index = (DOCS / "README.md").read_text(encoding="utf-8")
    overview = (DOCS / "project_overview.md").read_text(encoding="utf-8")
    experiment_history = (DOCS / "experiment_history.md").read_text(encoding="utf-8")

    assert "experiment_history.md" in docs_index
    assert "Experiment History" in overview
    assert "resize-only representation" in experiment_history
    assert "patch proposal" in experiment_history
    assert "Segmentation Smoke" in experiment_history
    assert "Real-Unlabeled Workflow" in experiment_history
