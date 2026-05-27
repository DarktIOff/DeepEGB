"""CosmoRAG tools for the DeepEGB agent.

Wraps CosmoRAG's retrieval and ingestion functions as plain Python callables
so they can be registered with either the Agno-based agent or the native
Claude API backend.

Both direct Python import (preferred, no process overhead) and MCP subprocess
modes are supported; the wrappers here use direct import.  The functions are
safe to call even when CosmoRAG is not installed — they return a structured
TOOL_ERROR JSON string instead of raising, which the agent handles gracefully.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


def _import_err(fn: str, exc: Exception) -> str:
    return json.dumps(
        {
            "error": "TOOL_ERROR:cosmorag_not_available",
            "message": f"CosmoRAG is not installed or could not be imported: {exc}",
            "suggestion": (
                "Run: pip install -e /path/to/CosmoRAG'[mcp]'  "
                "or set COSMORAG_HOME if the database lives in a custom location."
            ),
            "do_not_retry": True,
        },
        indent=2,
    )


def _runtime_err(fn: str, exc: Exception) -> str:
    return json.dumps(
        {
            "error": "TOOL_ERROR:cosmorag_runtime_failure",
            "message": f"{type(exc).__name__}: {exc}",
            "do_not_retry": False,
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Search / retrieval
# ---------------------------------------------------------------------------

def cosmorag_search_tool(query: str, top_k: int = 10) -> str:
    """Search the CosmoRAG cosmology-paper knowledge base for relevant passages.

    Performs hybrid retrieval (FTS5 + token overlap + equation match + optional
    embedding similarity + cross-encoder reranking) over all arXiv papers
    currently indexed in CosmoRAG. Returns formatted passages with paper ID,
    section location, LaTeX equations, and relevance score.

    PREFER this over retrieve_literature_tool when CosmoRAG has the relevant
    papers indexed — CosmoRAG's equation-aware retrieval outperforms plain BM25
    for physics queries containing formulas or equation labels.

    Parameters
    ----------
    query : Natural-language or equation-containing search query. Examples:
            "Gauss-Bonnet coupling function slow-roll equations",
            "tensor power spectrum EGB inflation formula",
            "ACT DR6 constraints on spectral index"
    top_k : Number of results to return (default 10, max 25).

    Returns
    -------
    Formatted list of matching passages with paper metadata, section location,
    LaTeX equations, and relevance scores. Or a TOOL_ERROR JSON on failure.
    """
    try:
        from cosmorag.mcp_server import cosmorag_search  # noqa: PLC0415
        return cosmorag_search(query=query, top_k=top_k)
    except ImportError as exc:
        return _import_err("cosmorag_search_tool", exc)
    except Exception as exc:  # noqa: BLE001
        return _runtime_err("cosmorag_search_tool", exc)


def cosmorag_list_papers_tool() -> str:
    """List all arXiv papers currently indexed in the CosmoRAG knowledge base.

    Returns title, arXiv ID, authors, year, and categories for every paper
    in the local CosmoRAG database, ordered by ingestion date (newest first).
    Call at the start of a research session to see which literature is
    available before querying.

    Returns
    -------
    Formatted table of all indexed papers with metadata.
    """
    try:
        from cosmorag.mcp_server import cosmorag_list_papers  # noqa: PLC0415
        return cosmorag_list_papers()
    except ImportError as exc:
        return _import_err("cosmorag_list_papers_tool", exc)
    except Exception as exc:  # noqa: BLE001
        return _runtime_err("cosmorag_list_papers_tool", exc)


def cosmorag_get_paper_tool(arxiv_id: str) -> str:
    """Retrieve full metadata for a single paper from CosmoRAG by arXiv ID.

    Parameters
    ----------
    arxiv_id : The arXiv paper identifier, e.g. "2401.12345" or "hep-th/9805136".

    Returns
    -------
    Title, authors, year, categories, abstract, and local file paths for the
    requested paper. Returns an error message if the paper is not indexed.
    """
    try:
        from cosmorag.mcp_server import cosmorag_get_paper  # noqa: PLC0415
        return cosmorag_get_paper(arxiv_id=arxiv_id.strip())
    except ImportError as exc:
        return _import_err("cosmorag_get_paper_tool", exc)
    except Exception as exc:  # noqa: BLE001
        return _runtime_err("cosmorag_get_paper_tool", exc)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def cosmorag_add_paper_tool(arxiv_id: str) -> str:
    """Download, parse, chunk, and index a single arXiv paper into CosmoRAG.

    The fast path (download → parse → chunk → SQLite) runs synchronously and
    completes in a few seconds — the paper becomes FTS5-searchable immediately.
    Embedding runs in a serialised background worker so no timeout occurs.
    Use cosmorag_embedding_status_tool() to monitor background embedding.

    Parameters
    ----------
    arxiv_id : arXiv identifier, e.g. "2401.12345" or "hep-th/9805136".

    Returns
    -------
    Ingestion status: chunk count, equation count, and embedding queue
    confirmation.
    """
    try:
        from cosmorag.mcp_server import cosmorag_add_arxiv  # noqa: PLC0415
        return cosmorag_add_arxiv(arxiv_id=arxiv_id.strip())
    except ImportError as exc:
        return _import_err("cosmorag_add_paper_tool", exc)
    except Exception as exc:  # noqa: BLE001
        return _runtime_err("cosmorag_add_paper_tool", exc)


def cosmorag_add_papers_tool(arxiv_ids: list[str]) -> str:
    """Download, parse, chunk, and index multiple arXiv papers into CosmoRAG.

    Fast path (download → parse → chunk → SQLite) runs per-paper in sequence;
    papers become FTS5-searchable immediately.  Embeddings for the whole batch
    are processed by a single background worker under an exclusive lock (one
    GPU instance for the full batch — no VRAM contention).
    Use cosmorag_embedding_status_tool() for live progress and ETA.

    Parameters
    ----------
    arxiv_ids : List of arXiv identifiers, e.g. ["2401.12345", "2301.02035"].

    Returns
    -------
    Per-paper ingestion status and batch-embedding worker confirmation.
    """
    try:
        from cosmorag.mcp_server import cosmorag_add_papers  # noqa: PLC0415
        return cosmorag_add_papers(arxiv_ids=[aid.strip() for aid in arxiv_ids])
    except ImportError as exc:
        return _import_err("cosmorag_add_papers_tool", exc)
    except Exception as exc:  # noqa: BLE001
        return _runtime_err("cosmorag_add_papers_tool", exc)


def cosmorag_embedding_status_tool() -> str:
    """Check the live progress and ETA for CosmoRAG's background embedding worker.

    Reads the status JSON written by the embedding worker after each completed
    paper. Shows: running vs idle, current paper being embedded, N/total done,
    elapsed time, average per-paper time, and estimated time remaining.

    Returns
    -------
    Structured progress report with ETA, or "idle" if no embedding job is
    running.
    """
    try:
        from cosmorag.mcp_server import cosmorag_embedding_status  # noqa: PLC0415
        return cosmorag_embedding_status()
    except ImportError as exc:
        return _import_err("cosmorag_embedding_status_tool", exc)
    except Exception as exc:  # noqa: BLE001
        return _runtime_err("cosmorag_embedding_status_tool", exc)


# ---------------------------------------------------------------------------
# Availability helpers
# ---------------------------------------------------------------------------

def is_cosmorag_available() -> bool:
    """Return True if the cosmorag package is importable."""
    try:
        import cosmorag  # noqa: F401
        return True
    except ImportError:
        return False


def all_cosmorag_tools() -> list:
    """Return all CosmoRAG tool callables for registration with an agent."""
    return [
        cosmorag_search_tool,
        cosmorag_list_papers_tool,
        cosmorag_get_paper_tool,
        cosmorag_add_paper_tool,
        cosmorag_add_papers_tool,
        cosmorag_embedding_status_tool,
    ]
