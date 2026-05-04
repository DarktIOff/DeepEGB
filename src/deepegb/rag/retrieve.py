"""
Hybrid retrieval (dense FAISS + sparse BM25) over the local index.

Final score = α · dense_score + (1 − α) · sparse_score, with both rescaled
to [0, 1] across the candidate pool.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .chunker import Chunk
from .index import (
    DEFAULT_INDEX_DIR,
    DEFAULT_MODEL,
    embed_texts,
    load_index,
)


@dataclass
class RetrievalHit:
    chunk: Chunk
    score: float
    dense_score: float
    sparse_score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.chunk.text,
            "source": self.chunk.source,
            "section_title": self.chunk.section_title,
            "chunk_index": self.chunk.chunk_index,
            "score": self.score,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
        }


def _normalise(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def hybrid_retrieve(
    query: str,
    *,
    index_dir: Path = DEFAULT_INDEX_DIR,
    k: int = 8,
    pool: int = 64,
    alpha: float = 0.6,
    embedding_model: str = DEFAULT_MODEL,
) -> list[RetrievalHit]:
    """Retrieve top-k chunks for a query using hybrid dense+sparse scoring.

    Parameters
    ----------
    k       : final number of results returned.
    pool    : candidate pool size from each retriever before merging.
    alpha   : weight on dense score (1.0 = pure dense, 0.0 = pure BM25).
    """
    idx = load_index(index_dir)
    chunks: list[Chunk] = idx["chunks"]
    if not chunks:
        return []

    # Dense
    q_vec = embed_texts([query], model_name=embedding_model)[0]
    dense_scores_all, dense_idx = idx["faiss"].search(
        q_vec.reshape(1, -1).astype(np.float32),
        min(pool, len(chunks)),
    )
    dense_scores_all = dense_scores_all[0]
    dense_idx = dense_idx[0]

    # Sparse
    bm25 = idx["bm25"]
    sparse_scores_all = bm25.get_scores(query.lower().split())
    sparse_top = np.argsort(sparse_scores_all)[::-1][:pool]

    # Pool: union of both
    pool_set = set(int(i) for i in dense_idx if i >= 0) | set(int(i) for i in sparse_top)
    pool_idx = np.array(sorted(pool_set), dtype=int)

    # Lookup scores
    dense_lookup = {int(i): float(s) for i, s in zip(dense_idx, dense_scores_all)
                    if i >= 0}
    dense_per = np.array([dense_lookup.get(int(i), 0.0) for i in pool_idx])
    sparse_per = np.array([sparse_scores_all[i] for i in pool_idx])

    # Normalise + merge
    d_norm = _normalise(dense_per)
    s_norm = _normalise(sparse_per)
    merged = alpha * d_norm + (1.0 - alpha) * s_norm

    order = np.argsort(merged)[::-1][:k]
    hits: list[RetrievalHit] = []
    for rank in order:
        chunk_i = int(pool_idx[rank])
        hits.append(RetrievalHit(
            chunk=chunks[chunk_i],
            score=float(merged[rank]),
            dense_score=float(d_norm[rank]),
            sparse_score=float(s_norm[rank]),
        ))
    return hits


def format_hits_for_llm(hits: list[RetrievalHit], max_chars_per_hit: int = 1200) -> str:
    """Render hits as a compact context block for the LLM."""
    lines: list[str] = []
    for i, h in enumerate(hits, 1):
        path = Path(h.chunk.source).name
        snippet = h.chunk.text
        if len(snippet) > max_chars_per_hit:
            snippet = snippet[:max_chars_per_hit] + " [...]"
        lines.append(
            f"[{i}] {path} — § {h.chunk.section_title}  "
            f"(score={h.score:.3f})\n{snippet}"
        )
    return "\n\n".join(lines)
