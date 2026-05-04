"""
File loaders for the local RAG: PDF, TeX, HTML, Markdown, plain text.

Each loader returns a list of (section_title, text) tuples. The chunker
in `chunker.py` then splits these into embedding-sized fragments while
preserving the section title as metadata.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Plain text / Markdown
# ---------------------------------------------------------------------------
def load_text(path: Path) -> list[tuple[str, str]]:
    """Markdown / plain text: split on `^# Title` headings."""
    text = path.read_text(encoding="utf-8", errors="replace")
    sections: list[tuple[str, str]] = []
    current_title = path.stem
    current_lines: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^(#+)\s+(.+)\s*$", line)
        if m:
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = m.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return [(t, b) for t, b in sections if b]


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def load_html(path: Path) -> list[tuple[str, str]]:
    """HTML: strip tags, split on `<h1>..<h6>` boundaries."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    # crude but dependency-free
    raw = re.sub(r"<script.*?</script>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = re.sub(r"<style.*?</style>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    parts = re.split(r"<h[1-6][^>]*>", raw, flags=re.IGNORECASE)
    sections: list[tuple[str, str]] = []
    if parts:
        # first chunk has no preceding heading
        first = re.sub(r"<[^>]+>", " ", parts[0])
        first = re.sub(r"\s+", " ", first).strip()
        if first:
            sections.append((path.stem, first))
        for chunk in parts[1:]:
            # split off the heading text up to </h?>
            m = re.match(r"(.*?)</h[1-6]>(.*)", chunk, flags=re.DOTALL | re.IGNORECASE)
            if m:
                title = re.sub(r"<[^>]+>", "", m.group(1)).strip() or path.stem
                body = re.sub(r"<[^>]+>", " ", m.group(2))
                body = re.sub(r"\s+", " ", body).strip()
                if body:
                    sections.append((title, body))
    return sections


# ---------------------------------------------------------------------------
# TeX / LaTeX
# ---------------------------------------------------------------------------
_SECTION_RE = re.compile(
    r"\\(chapter|section|subsection|subsubsection|paragraph)\*?\s*\{([^{}]*)\}",
    flags=re.MULTILINE,
)


def load_tex(path: Path) -> list[tuple[str, str]]:
    """TeX: split on \\section{...} (and chapter/subsection/...) boundaries.

    Strip comments and the most common math-environment delimiters; we keep
    the math intact for retrieval — physicist queries often hit equation text.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    # strip LaTeX comments
    raw = re.sub(r"(?<!\\)%.*?\n", "\n", raw)
    # find all section starts
    matches = list(_SECTION_RE.finditer(raw))
    sections: list[tuple[str, str]] = []
    if not matches:
        sections.append((path.stem, raw.strip()))
        return [(t, b) for t, b in sections if b]
    # preamble before first section
    preamble = raw[: matches[0].start()].strip()
    if preamble:
        sections.append((path.stem + " (preamble)", preamble))
    for i, m in enumerate(matches):
        title = m.group(2).strip() or path.stem
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[start:end].strip()
        if body:
            sections.append((title, body))
    return sections


# ---------------------------------------------------------------------------
# PDF (via pdfplumber, optional dep)
# ---------------------------------------------------------------------------
def load_pdf(path: Path) -> list[tuple[str, str]]:
    """PDF: extract text page-by-page; group into sections by detecting
    likely heading lines (uppercase / numbered / short lines)."""
    try:
        import pdfplumber  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pdfplumber not available — install with `pip install -e .[rag]`"
        ) from exc

    sections: list[tuple[str, str]] = []
    current_title = path.stem
    current_lines: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if _looks_like_heading(stripped):
                    if current_lines:
                        sections.append((current_title,
                                         " ".join(current_lines).strip()))
                    current_title = stripped[:100]
                    current_lines = []
                else:
                    current_lines.append(stripped)
    if current_lines:
        sections.append((current_title, " ".join(current_lines).strip()))
    return [(t, b) for t, b in sections if b]


_HEADING_RE = re.compile(
    r"^("
    r"\d+(\.\d+)*\.?\s+\S.*"     # "1.2.3 Section title"
    r"|[IVX]+\.\s+\S.*"          # "IV. Roman section"
    r"|[A-Z][A-Z\s\-:]{4,}\S?"   # ALL CAPS HEADING
    r")$"
)


def _looks_like_heading(line: str) -> bool:
    if len(line) > 120:
        return False
    return bool(_HEADING_RE.match(line))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
LOADERS = {
    ".pdf": load_pdf,
    ".tex": load_tex,
    ".html": load_html,
    ".htm": load_html,
    ".md": load_text,
    ".markdown": load_text,
    ".txt": load_text,
    ".rst": load_text,
}


def load_any(path: Path) -> list[tuple[str, str]]:
    suffix = path.suffix.lower()
    loader = LOADERS.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported extension: {suffix} ({path.name})")
    return loader(path)


def walk_supported(folder: Path) -> Iterable[Path]:
    """Yield all files under `folder` whose extension is supported."""
    for ext in LOADERS:
        yield from folder.rglob(f"*{ext}")
