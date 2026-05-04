"""
Section → chunk splitter. Chunks are sized to fit comfortably in a small
embedding model's context window (≈ 512 tokens ≈ 2000 chars), with a
configurable overlap to preserve sentences that span boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .loaders import load_any, walk_supported


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str           # absolute path
    section_title: str
    chunk_index: int      # 0-based within the (source, section)


def _split_into_chunks(text: str, max_chars: int, overlap: int) -> list[str]:
    """Greedy paragraph-aware splitter."""
    if len(text) <= max_chars:
        return [text]
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for para in paragraphs:
        # If a single paragraph is larger than max_chars, split by sentence-ish
        if len(para) > max_chars:
            for sent in _sentence_split(para):
                if buf_len + len(sent) > max_chars and buf:
                    chunks.append("\n".join(buf))
                    buf = _tail_for_overlap(buf, overlap)
                    buf_len = sum(len(s) for s in buf)
                buf.append(sent)
                buf_len += len(sent) + 1
            continue
        if buf_len + len(para) > max_chars and buf:
            chunks.append("\n".join(buf))
            buf = _tail_for_overlap(buf, overlap)
            buf_len = sum(len(s) for s in buf)
        buf.append(para)
        buf_len += len(para) + 1
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def _tail_for_overlap(buf: list[str], overlap: int) -> list[str]:
    """Return the last few items from buf totaling ≤ `overlap` chars."""
    out: list[str] = []
    total = 0
    for s in reversed(buf):
        if total + len(s) > overlap:
            break
        out.insert(0, s)
        total += len(s) + 1
    return out


_SENT_RE = None  # lazy compile


def _sentence_split(text: str) -> list[str]:
    import re
    global _SENT_RE
    if _SENT_RE is None:
        _SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\])")
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def chunk_file(path: Path, *, max_chars: int = 1800,
               overlap: int = 200) -> list[Chunk]:
    """Load and chunk a single file."""
    sections = load_any(path)
    chunks: list[Chunk] = []
    for title, body in sections:
        for i, piece in enumerate(_split_into_chunks(body, max_chars, overlap)):
            chunks.append(Chunk(text=piece, source=str(path),
                                section_title=title, chunk_index=i))
    return chunks


def chunk_folder(folder: Path, *, max_chars: int = 1800,
                 overlap: int = 200) -> list[Chunk]:
    """Walk `folder`, chunk every supported file."""
    all_chunks: list[Chunk] = []
    for path in walk_supported(folder):
        try:
            all_chunks.extend(chunk_file(path, max_chars=max_chars, overlap=overlap))
        except Exception as exc:
            # don't let one bad file break the whole index
            print(f"[rag] WARNING: failed to load {path}: {exc}")
    return all_chunks
