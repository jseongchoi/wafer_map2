from __future__ import annotations

import html
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PAGES = DOCS / "pages"


DOC_SOURCES = [*sorted(DOCS.glob("*.md")), ROOT / "README.md", ROOT / "scripts" / "README.md"]
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
STRONG_RE = re.compile(r"\*\*([^*]+)\*\*")
EM_RE = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")


def output_name(source: Path) -> str:
    if source == ROOT / "README.md":
        return "root_README.html"
    if source == ROOT / "scripts" / "README.md":
        return "scripts_README.html"
    if source.name == "README.md":
        return "docs_README.html"
    return f"{source.stem}.html"


SOURCE_TO_OUTPUT = {source.resolve(): output_name(source) for source in DOC_SOURCES}


def slugify(text: str, used: set[str]) -> str:
    slug = re.sub(r"[^0-9A-Za-z가-힣_-]+", "-", text.strip()).strip("-").lower()
    slug = slug or "section"
    base = slug
    index = 2
    while slug in used:
        slug = f"{base}-{index}"
        index += 1
    used.add(slug)
    return slug


def split_target(target: str) -> tuple[str, str]:
    if "#" not in target:
        return target, ""
    path, anchor = target.split("#", 1)
    return path, f"#{anchor}"


def rewrite_target(source: Path, target: str) -> str:
    path_part, anchor = split_target(target.strip())
    if not path_part or "://" in path_part or path_part.startswith("mailto:"):
        return target

    resolved = (source.parent / path_part).resolve()
    output = SOURCE_TO_OUTPUT.get(resolved)
    if output:
        return f"{output}{anchor}"
    if resolved == (DOCS / "index.html").resolve():
        return f"../index.html{anchor}"
    return target


def render_inline(source: Path, text: str) -> str:
    placeholders: list[str] = []

    def keep_code(match: re.Match[str]) -> str:
        placeholders.append(f"<code>{html.escape(match.group(1))}</code>")
        return f"\x00{len(placeholders) - 1}\x00"

    escaped = html.escape(text)
    escaped = INLINE_CODE_RE.sub(keep_code, escaped)

    def replace_link(match: re.Match[str]) -> str:
        label = match.group(1)
        target = html.unescape(match.group(2))
        rewritten = rewrite_target(source, target)
        return f'<a href="{html.escape(rewritten, quote=True)}">{label}</a>'

    escaped = LINK_RE.sub(replace_link, escaped)
    escaped = STRONG_RE.sub(r"<strong>\1</strong>", escaped)
    escaped = EM_RE.sub(r"<em>\1</em>", escaped)

    for index, replacement in enumerate(placeholders):
        escaped = escaped.replace(f"\x00{index}\x00", replacement)
    return escaped


def is_table_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)


def render_table(source: Path, rows: list[str]) -> str:
    parsed = [[cell.strip() for cell in row.strip().strip("|").split("|")] for row in rows]
    header = parsed[0]
    body = parsed[2:] if len(parsed) > 1 and is_table_separator(rows[1]) else parsed[1:]
    parts = ["<div class=\"table-wrap\"><table>"]
    parts.append("<thead><tr>")
    parts.extend(f"<th>{render_inline(source, cell)}</th>" for cell in header)
    parts.append("</tr></thead>")
    if body:
        parts.append("<tbody>")
        for row in body:
            parts.append("<tr>")
            parts.extend(f"<td>{render_inline(source, cell)}</td>" for cell in row)
            parts.append("</tr>")
        parts.append("</tbody>")
    parts.append("</table></div>")
    return "".join(parts)


def extract_title(markdown: str, fallback: str) -> str:
    for line in markdown.splitlines():
        match = HEADING_RE.match(line)
        if match and len(match.group(1)) == 1:
            return match.group(2).strip()
    return fallback


def render_markdown(source: Path, markdown: str) -> tuple[str, list[tuple[int, str, str]]]:
    lines = markdown.splitlines()
    html_parts: list[str] = []
    toc: list[tuple[int, str, str]] = []
    used_ids: set[str] = set()
    paragraph: list[str] = []
    in_code = False
    code_lang = ""
    code_lines: list[str] = []
    list_type: str | None = None
    blockquote: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            html_parts.append(f"<p>{render_inline(source, ' '.join(paragraph).strip())}</p>")
            paragraph = []

    def flush_list() -> None:
        nonlocal list_type
        if list_type:
            html_parts.append(f"</{list_type}>")
            list_type = None

    def flush_blockquote() -> None:
        nonlocal blockquote
        if blockquote:
            html_parts.append(f"<blockquote>{render_inline(source, ' '.join(blockquote).strip())}</blockquote>")
            blockquote = []

    index = 0
    while index < len(lines):
        line = lines[index]

        if in_code:
            if line.startswith("```"):
                language_class = f" language-{html.escape(code_lang)}" if code_lang else ""
                code = html.escape("\n".join(code_lines))
                html_parts.append(f"<pre><code class=\"{language_class.strip()}\">{code}</code></pre>")
                in_code = False
                code_lang = ""
                code_lines = []
            else:
                code_lines.append(line)
            index += 1
            continue

        if line.startswith("```"):
            flush_paragraph()
            flush_list()
            flush_blockquote()
            in_code = True
            code_lang = line.strip("`").strip()
            index += 1
            continue

        if not line.strip():
            flush_paragraph()
            flush_list()
            flush_blockquote()
            index += 1
            continue

        if line.strip() in {"---", "***", "___"}:
            flush_paragraph()
            flush_list()
            flush_blockquote()
            html_parts.append("<hr>")
            index += 1
            continue

        if line.lstrip().startswith(">"):
            flush_paragraph()
            flush_list()
            blockquote.append(line.lstrip()[1:].strip())
            index += 1
            continue

        heading = HEADING_RE.match(line)
        if heading:
            flush_paragraph()
            flush_list()
            flush_blockquote()
            level = len(heading.group(1))
            text = heading.group(2).strip()
            section_id = slugify(re.sub(r"`([^`]+)`", r"\1", text), used_ids)
            if level <= 3:
                toc.append((level, text, section_id))
            html_parts.append(
                f'<h{level} id="{html.escape(section_id)}">{render_inline(source, text)}</h{level}>'
            )
            index += 1
            continue

        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        if "|" in line and is_table_separator(next_line):
            flush_paragraph()
            flush_list()
            flush_blockquote()
            table_rows = [line, next_line]
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                table_rows.append(lines[index])
                index += 1
            html_parts.append(render_table(source, table_rows))
            continue

        unordered = re.match(r"^\s*[-*]\s+(.+)$", line)
        ordered = re.match(r"^\s*\d+\.\s+(.+)$", line)
        if unordered or ordered:
            flush_paragraph()
            flush_blockquote()
            desired = "ul" if unordered else "ol"
            if list_type != desired:
                flush_list()
                list_type = desired
                html_parts.append(f"<{list_type}>")
            item = (unordered or ordered).group(1)
            html_parts.append(f"<li>{render_inline(source, item)}</li>")
            index += 1
            continue

        flush_list()
        flush_blockquote()
        paragraph.append(line.strip())
        index += 1

    flush_paragraph()
    flush_list()
    flush_blockquote()
    if in_code:
        code = html.escape("\n".join(code_lines))
        html_parts.append(f"<pre><code>{code}</code></pre>")
    return "\n".join(html_parts), toc


def page_template(source: Path, title: str, body: str, toc: list[tuple[int, str, str]]) -> str:
    rel_source = source.relative_to(ROOT).as_posix()
    toc_items = "\n".join(
        f'<a class="toc-level-{level}" href="#{html.escape(section_id)}">{html.escape(text)}</a>'
        for level, text, section_id in toc
    )
    if not toc_items:
        toc_items = '<span class="toc-empty">목차가 없습니다.</span>'
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} - WaferMap 문서</title>
  <link rel="stylesheet" href="doc_site.css">
</head>
<body>
  <header class="site-header">
    <div>
      <a class="home-link" href="../index.html">WaferMap 문서 표지판</a>
      <h1>{html.escape(title)}</h1>
      <p>{html.escape(rel_source)}</p>
    </div>
  </header>
  <main class="layout">
    <aside class="toc">
      <strong>목차</strong>
      {toc_items}
    </aside>
    <article class="content">
      {body}
    </article>
  </main>
</body>
</html>
"""


def build() -> None:
    PAGES.mkdir(parents=True, exist_ok=True)
    for source in DOC_SOURCES:
        markdown = source.read_text(encoding="utf-8")
        title = extract_title(markdown, source.stem)
        body, toc = render_markdown(source, markdown)
        target = PAGES / output_name(source)
        target.write_text(page_template(source, title, body, toc), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    build()
