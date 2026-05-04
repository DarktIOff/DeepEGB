"""Local RAG over EGB-inflation literature.

Implementation:
- Walk a directory of PDFs / TeX / HTML / Markdown / plain text.
- Chunk by section using format-specific loaders (`loaders.py`).
- Embed with `sentence-transformers` (default BGE-small, 384-dim).
- Hybrid retrieval: dense FAISS + sparse BM25, weighted by α (default 0.6).
- Persistent on-disk index under `~/.deepegb/rag_index/` (override via env).
"""
from __future__ import annotations

from .chunker import Chunk, chunk_file, chunk_folder
from .index import (
    DEFAULT_INDEX_DIR,
    DEFAULT_MODEL,
    IndexMeta,
    build_index,
    embed_texts,
    index_exists,
    load_index,
)
from .retrieve import RetrievalHit, format_hits_for_llm, hybrid_retrieve

__all__ = [
    "Chunk",
    "chunk_file",
    "chunk_folder",
    "DEFAULT_INDEX_DIR",
    "DEFAULT_MODEL",
    "IndexMeta",
    "build_index",
    "embed_texts",
    "index_exists",
    "load_index",
    "RetrievalHit",
    "format_hits_for_llm",
    "hybrid_retrieve",
]
