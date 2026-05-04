"""RAG over EGB-inflation literature (v2 — stubbed for MVP).

Planned implementation:
- Walk a directory of PDFs (default: ~/University/PhD/PhD/papers).
- Chunk by section using `pdfplumber`.
- Embed with `sentence-transformers` (e.g. all-MiniLM-L6-v2 or bge-small).
- Hybrid retrieval: dense FAISS + sparse BM25, as in DeepInflation.
- Expose a `retrieve_literature(query, k=5)` tool to the main agent.

For now, importing this submodule is a no-op so the rest of the package can
load without the optional `rag` extras.
"""
from __future__ import annotations

__all__: list[str] = []
