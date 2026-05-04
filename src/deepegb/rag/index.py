"""
Hybrid FAISS+BM25 index for the local RAG.

Disk layout under `index_dir`:
    chunks.json        — list of {text, source, section_title, chunk_index}
    embeddings.npy     — float32, shape (n_chunks, d_embed)
    faiss.index        — FAISS HNSW index
    bm25.pkl           — pickled rank_bm25.BM25Okapi
    metadata.json      — {model_name, n_chunks, d_embed, …}
"""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .chunker import Chunk, chunk_folder

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"     # 384-dim, ~30 MB, good quality/speed
DEFAULT_INDEX_DIR = Path.home() / ".deepegb" / "rag_index"


@dataclass
class IndexMeta:
    embedding_model: str
    n_chunks: int
    d_embed: int
    source_folder: str
    last_updated_iso: str

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def _load_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer  # lazy
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model_name: str = DEFAULT_MODEL,
                batch_size: int = 64) -> np.ndarray:
    embedder = _load_embedder(model_name)
    vecs = embedder.encode(
        texts, batch_size=batch_size, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    return vecs.astype(np.float32)


# ---------------------------------------------------------------------------
# Build / save / load
# ---------------------------------------------------------------------------
def build_index(
    folder: Path,
    *,
    index_dir: Path = DEFAULT_INDEX_DIR,
    embedding_model: str = DEFAULT_MODEL,
    max_chars: int = 1800,
    overlap: int = 200,
    progress: bool = True,
) -> IndexMeta:
    """Walk `folder`, chunk + embed everything, save FAISS+BM25 index."""
    import datetime as _dt
    folder = Path(folder).expanduser().resolve()
    index_dir = Path(index_dir).expanduser()
    index_dir.mkdir(parents=True, exist_ok=True)

    if progress:
        print(f"[rag] indexing {folder} → {index_dir}")
    chunks = chunk_folder(folder, max_chars=max_chars, overlap=overlap)
    if not chunks:
        raise RuntimeError(f"No supported files found in {folder}")
    if progress:
        print(f"[rag] {len(chunks)} chunks; embedding with {embedding_model} …")

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts, model_name=embedding_model)
    d_embed = int(embeddings.shape[1])

    # FAISS index (cosine sim ≡ inner product on normalised vectors)
    import faiss  # lazy
    faiss_index = faiss.IndexHNSWFlat(d_embed, 32)   # HNSW, M=32
    faiss_index.metric_type = faiss.METRIC_INNER_PRODUCT
    faiss_index.add(embeddings)

    # BM25
    from rank_bm25 import BM25Okapi
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    # Persist
    chunks_path = index_dir / "chunks.json"
    chunks_path.write_text(json.dumps([asdict(c) for c in chunks]),
                            encoding="utf-8")
    np.save(index_dir / "embeddings.npy", embeddings)
    faiss.write_index(faiss_index, str(index_dir / "faiss.index"))
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    meta = IndexMeta(
        embedding_model=embedding_model,
        n_chunks=len(chunks),
        d_embed=d_embed,
        source_folder=str(folder),
        last_updated_iso=_dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )
    (index_dir / "metadata.json").write_text(json.dumps(meta.as_dict(), indent=2))
    if progress:
        print(f"[rag] saved index ({meta.n_chunks} chunks, dim={d_embed}) → {index_dir}")
    return meta


def load_index(index_dir: Path = DEFAULT_INDEX_DIR):
    """Load chunks, embeddings, FAISS+BM25 from disk. Returns a dict."""
    index_dir = Path(index_dir).expanduser()
    if not (index_dir / "metadata.json").exists():
        raise FileNotFoundError(
            f"No RAG index at {index_dir}. Run `deepegb rag index <folder>` first.")
    meta = json.loads((index_dir / "metadata.json").read_text())
    chunks_raw = json.loads((index_dir / "chunks.json").read_text())
    chunks = [Chunk(**c) for c in chunks_raw]
    embeddings = np.load(index_dir / "embeddings.npy")
    import faiss
    faiss_index = faiss.read_index(str(index_dir / "faiss.index"))
    with open(index_dir / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    return {
        "meta": meta, "chunks": chunks, "embeddings": embeddings,
        "faiss": faiss_index, "bm25": bm25,
    }


def index_exists(index_dir: Path = DEFAULT_INDEX_DIR) -> bool:
    return (Path(index_dir).expanduser() / "metadata.json").exists()
