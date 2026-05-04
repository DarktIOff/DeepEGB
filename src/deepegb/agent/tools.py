"""Agno-compatible tool functions exposed to the LLM.

Each function is a plain Python callable that Agno will introspect by type
annotation + docstring to build its tool schema. We keep them small and
side-effect-free (apart from the plot-saving tool).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from ..analysis import analyze_egb_model, analyze_egb_relic_gw, plot_egb_model
from ..search import SearchConfig, run_joint_search


def search_egb_potentials(
    target_ns: float,
    target_r: float,
    sigma_ns: float = 0.003,
    sigma_r: float = 0.05,
    N_pivot: float = 55.0,
    niterations: int = 30,
    populations: int = 30,
    maxsize: int = 25,
    runs_dir: str = "runs/agent",
) -> str:
    """Run a joint symbolic-regression search for V(φ) and ξ(φ) in EGB
    inflation that match given target observables.

    Parameters
    ----------
    target_ns : Target value of the scalar spectral index n_s.
    target_r  : Target value of the tensor-to-scalar ratio r.
    sigma_ns  : 1σ uncertainty on n_s (used in the χ² loss).
    sigma_r   : 1σ uncertainty on r.
    N_pivot   : Number of e-folds before end of inflation at which to evaluate.
    niterations / populations / maxsize : PySR hyperparameters.

    Returns
    -------
    A JSON string with up to 5 best (V, ξ) candidates ranked by χ².
    """
    cfg = SearchConfig(
        target_ns=target_ns, sigma_ns=sigma_ns,
        target_r=target_r, sigma_r=sigma_r,
        N_pivot=N_pivot,
        niterations=niterations, populations=populations,
        maxsize=maxsize,
        runs_dir=runs_dir,
    )
    results = run_joint_search(cfg)
    payload = [r.as_dict() for r in results[:5]]
    return json.dumps({"candidates": payload, "config": cfg.to_dict()}, default=str, indent=2)


def analyze_egb_model_tool(
    V_expr: str,
    xi_expr: str = "0",
    N: float = 55.0,
) -> str:
    """Compute (n_s, r, ε, φ_N, φ_end) for a given EGB inflation model.

    Parameters
    ----------
    V_expr : Sympy-style string for V(φ). Example: "phi**2".
    xi_expr: Sympy-style string for ξ(φ). Example: "0.1*exp(-0.5*phi)".
    N      : e-folds before end of inflation.
    """
    return json.dumps(analyze_egb_model(V_expr, xi_expr, N=N), indent=2, default=str)


def plot_egb_model_tool(
    V_expr: str,
    xi_expr: str = "0",
    N: float = 55.0,
    out_path: str = "outputs/egb_diagnostic.png",
) -> str:
    """Render a 6-panel diagnostic plot for an EGB inflation model and save
    it to disk. Returns the saved path as a string.
    """
    p = plot_egb_model(V_expr, xi_expr, N=N, out_path=out_path)
    return f"Saved diagnostic plot to: {p}"


def relic_gw_spectrum_tool(
    V_expr: str,
    xi_expr: str = "0",
    N: float = 55.0,
    T_reh_GeV: float = 1.0e15,
    n_decades: float = 10.0,
    n_k: int = 24,
) -> str:
    """Compute the relic GW energy density Ω_GW(f) h² across the LISA / PTA /
    DECIGO / ET frequency bands using full background EOMs + Mukhanov-Sasaki.

    Parameters
    ----------
    V_expr     : V(φ) expression in `phi`.
    xi_expr    : ξ(φ) expression. Pass "0" for the GR limit.
    N          : pivot e-folds before end of inflation.
    T_reh_GeV  : reheating temperature in GeV (controls g_* in transfer).
    n_decades  : log-decades of k around pivot.
    n_k        : number of k samples (more = slower but smoother).
    """
    return json.dumps(
        analyze_egb_relic_gw(V_expr, xi_expr, N=N, n_decades=n_decades,
                              n_k=n_k, T_reh_GeV=T_reh_GeV),
        indent=2, default=str,
    )


def retrieve_literature_tool(query: str, k: int = 5) -> str:
    """Search the local RAG index for chunks of EGB-inflation literature
    relevant to a natural-language query. Returns formatted hits with
    source filename, section title, and similarity score.

    Build the index first with `deepegb rag index <folder>`. The default
    folder is `~/University/PhD/PhD/papers` or whatever
    `DEEPEGB_RAG_PATH` is set to.

    Parameters
    ----------
    query : free-text query (e.g., "Starobinsky inflation slow-roll formulas",
            "ACT DR6 constraints on EGB models").
    k     : number of chunks to return (default 5; cap 20).
    """
    try:
        from ..rag import format_hits_for_llm, hybrid_retrieve, index_exists
    except ImportError:
        return json.dumps({
            "error": "RAG dependencies not installed. "
                     "Run: pip install -e '.[rag]'"
        })
    if not index_exists():
        return json.dumps({
            "error": "No local RAG index. Build one with "
                     "`deepegb rag index <folder>` first."
        })
    k = max(1, min(int(k), 20))
    hits = hybrid_retrieve(query, k=k)
    return format_hits_for_llm(hits)


# ----------------------------------------------------------------------------
# Helper to register tools with an Agno agent.
# ----------------------------------------------------------------------------
def all_tools(include_rag: bool = True) -> list:
    tools = [
        search_egb_potentials,
        analyze_egb_model_tool,
        plot_egb_model_tool,
        relic_gw_spectrum_tool,
    ]
    if include_rag:
        tools.append(retrieve_literature_tool)
    return tools
