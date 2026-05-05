"""Agno-compatible tool functions exposed to the LLM.

Each function is a plain Python callable that Agno will introspect by type
annotation + docstring to build its tool schema. We keep them small and
side-effect-free (apart from the plot-saving tool).

Error contract
--------------
Every tool returns a JSON string. On failure it returns a JSON object
with a top-level `"error"` key whose value starts with `TOOL_ERROR:`.
The system prompt tells the agent: NEVER retry a TOOL_ERROR call —
either fix the inputs or fall back to a known-good model.
"""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any, Optional

from ..analysis import analyze_egb_model, analyze_egb_relic_gw, plot_egb_model
from ..search import SearchConfig, run_joint_search, run_joint_search_subprocess


def _tool_error(category: str, msg: str, *, suggestion: str = "") -> str:
    return json.dumps({
        "error": f"TOOL_ERROR:{category}",
        "message": msg,
        "suggestion": suggestion,
        "do_not_retry": True,
    }, indent=2)


def search_egb_potentials(
    target_ns: float,
    target_r: float,
    sigma_ns: float = 0.003,
    sigma_r: float = 0.05,
    N_pivot: float = 55.0,
    niterations: int = 30,
    populations: int = 30,
    maxsize: int = 25,
    enforce_egb: bool = True,
    runs_dir: str = "runs/agent",
) -> str:
    """Run a joint symbolic-regression search for V(φ) and ξ(φ) in EGB
    inflation that match given target observables.

    Parameters
    ----------
    target_ns : Target value of the scalar spectral index n_s.
    target_r  : Target value of the tensor-to-scalar ratio r. For an
                inequality constraint like "r < 0.01", set target_r=0
                and sigma_r=0.005 — that gives χ²<1 whenever r < 0.005.
    sigma_ns  : 1σ uncertainty on n_s (used in the χ² loss).
    sigma_r   : 1σ uncertainty on r.
    N_pivot   : Number of e-folds before end of inflation at which to evaluate.
    niterations / populations / maxsize : PySR hyperparameters.
    enforce_egb : When True (DEFAULT), reject ξ ≡ 0 candidates (the GR
                  limit). DO NOT pass False unless the user explicitly
                  asks for a GR baseline as part of the search — DeepEGB
                  is for EGB inflation, not vanilla GR.

    Returns
    -------
    JSON string with up to 5 best (V, ξ) candidates ranked by χ², or a
    structured TOOL_ERROR on failure.  DO NOT retry on TOOL_ERROR.
    """
    import sys

    def _log(msg: str) -> None:
        print(f"[DeepEGB] {msg}", file=sys.stderr, flush=True)

    _log(f"search_egb_potentials called: ns={target_ns}, r={target_r}")
    _log(f"stdout type={type(sys.stdout).__name__}, stderr type={type(sys.stderr).__name__}")
    import threading
    _log(f"current thread: {threading.current_thread().name}, is_main={threading.current_thread() is threading.main_thread()}")
    cfg = SearchConfig(
        target_ns=target_ns, sigma_ns=sigma_ns,
        target_r=target_r, sigma_r=sigma_r,
        N_pivot=N_pivot,
        niterations=niterations, populations=populations,
        maxsize=maxsize,
        enforce_egb=enforce_egb,
        use_julia_loss="auto",
        runs_dir=runs_dir,
    )
    try:
        results = run_joint_search_subprocess(cfg, progress_cb=_log)
    except (ImportError, RuntimeError) as exc:
        msg = str(exc)
        if "pysr" in msg.lower() or "julia" in msg.lower():
            return _tool_error(
                "missing_dependency",
                f"PySR or Julia not installed: {exc}",
                suggestion="Run `pip install pysr` and `python -c \"import pysr; pysr.install()\"`."
            )
        return _tool_error(
            "search_failure",
            f"{type(exc).__name__}: {exc}",
            suggestion="Inspect the message; try smaller niterations/populations.",
        )
    except TypeError as exc:
        # PySR API mismatch (e.g. removed kwarg) → unrecoverable here.
        return _tool_error(
            "pysr_api_mismatch",
            f"PySR rejected a constructor argument: {exc}",
            suggestion="Update DeepEGB or pin to the matching pysr version. "
                       "Fall back to analyze_egb_model_tool on a known model."
        )
    except Exception as exc:                                       # noqa: BLE001
        return _tool_error(
            "search_failure",
            f"{type(exc).__name__}: {exc}",
            suggestion="Try smaller niterations/populations, or analyze a "
                       "known Starobinsky-family model with analyze_egb_model_tool."
        )
    if not results:
        return _tool_error(
            "no_candidates",
            "Search produced no candidates.",
            suggestion="Loosen target sigmas, raise niterations, or analyze "
                       "a known model directly."
        )
    payload = [r.as_dict() for r in results[:5]]
    return json.dumps({"candidates": payload, "config": cfg.to_dict()},
                      default=str, indent=2)


def analyze_egb_model_tool(
    V_expr: str,
    xi_expr: str = "0",
    N: float = 55.0,
) -> str:
    """Compute observables (n_s, n_T, r, α_s, P_S, P_T, c_T², c_S², ε, δ₁,
    consistency_r_minus_8nT) for a given EGB inflation model.

    Parameters
    ----------
    V_expr : Sympy-style string for V(φ). Example: "phi**2".
    xi_expr: Sympy-style string for ξ(φ). Example: "0.1*exp(-0.5*phi)".
    N      : e-folds before end of inflation. Default 55.

    Returns
    -------
    JSON dict with all production observables, or TOOL_ERROR on failure.
    """
    try:
        return json.dumps(analyze_egb_model(V_expr, xi_expr, N=N),
                          indent=2, default=str)
    except Exception as exc:                                       # noqa: BLE001
        return _tool_error(
            "analyze_failure",
            f"{type(exc).__name__}: {exc}",
            suggestion="Check the V/ξ expression syntax (use `phi`, sympy "
                       "form), and that N is positive."
        )


def plot_egb_model_tool(
    V_expr: str,
    xi_expr: str = "0",
    N: float = 55.0,
    out_path: str = "outputs/egb_diagnostic.png",
) -> str:
    """Render a 6-panel diagnostic plot for an EGB inflation model and save
    it to disk. Returns the saved path as a string."""
    try:
        p = plot_egb_model(V_expr, xi_expr, N=N, out_path=out_path)
        return json.dumps({"saved_path": p})
    except Exception as exc:                                       # noqa: BLE001
        return _tool_error(
            "plot_failure",
            f"{type(exc).__name__}: {exc}",
            suggestion="Check that the trajectory is non-trivial: use "
                       "analyze_egb_model_tool first to verify the model "
                       "produces finite slow-roll observables."
        )


def relic_gw_spectrum_tool(
    V_expr: str,
    xi_expr: str = "0",
    N: float = 55.0,
    T_reh_GeV: float = 1.0e15,
    n_decades: float = 10.0,
    n_k: int = 24,
) -> str:
    """Compute the relic GW energy density Ω_GW(f) h² across the relic-GW
    frequency band using full background EOMs + Mukhanov-Sasaki + a
    radiation/matter-dominated transfer function.

    Parameters
    ----------
    V_expr     : V(φ) expression in `phi`.
    xi_expr    : ξ(φ) expression. Pass "0" for the GR limit.
    N          : pivot e-folds before end of inflation.
    T_reh_GeV  : reheating temperature in GeV (controls g_* in transfer).
    n_decades  : log-decades of k around pivot.
    n_k        : number of k samples (more = slower but smoother).
    """
    try:
        return json.dumps(
            analyze_egb_relic_gw(V_expr, xi_expr, N=N, n_decades=n_decades,
                                 n_k=n_k, T_reh_GeV=T_reh_GeV),
            indent=2, default=str,
        )
    except Exception as exc:                                       # noqa: BLE001
        return _tool_error(
            "relic_gw_failure",
            f"{type(exc).__name__}: {exc}",
            suggestion="Verify the model with analyze_egb_model_tool first; "
                       "the relic-GW pipeline needs ≥ N+5 e-folds of slow-roll."
        )


def diagnose_egb_model_tool(
    V_expr: str,
    xi_expr: str = "0",
    *,
    N: float = 55.0,
    target_ns: float = 0.965,
    target_r: float = 0.0,
    sigma_ns: float = 0.005,
    sigma_r: float = 0.018,
) -> str:
    """Diagnose WHY an EGB inflation model is failing or producing odd
    observables. Returns a structured report with:

      * Whether φ_end / φ_pivot were found.
      * Soft penalty value and qualitative reasons.
      * Atomized χ² breakdown (per-component contributions).
      * Concrete suggestions for how to fix the model.

    Use this whenever a previous tool call returned NaN observables, a
    huge χ², or a "background_failure" message. The breakdown will tell
    you which TERM (n_s, r, A_s, Ω_GW@1mHz, etc.) is dominating, so you
    can adjust the offending part of V or ξ instead of giving up.

    Parameters
    ----------
    V_expr, xi_expr : Sympy strings in `phi`. Pass "0" for ξ to test GR.
    N               : Pivot e-folds before end of inflation.
    target_ns/r, sigma_ns/r : χ² targets to score against.
    """
    from ..physics import diagnose_model, chi2_full_with_breakdown
    from ..search.pysr_search import expressions_to_model
    try:
        model = expressions_to_model(V_expr, xi_expr)
    except Exception as exc:
        return _tool_error("expression_parse",
                           f"Could not parse V/ξ: {exc}",
                           suggestion="Use Sympy syntax with `phi` as the field.")
    diag = diagnose_model(model, N_pivot=N)
    obs = None
    if diag.get("observables_valid"):
        from ..physics import compute_observables_full
        obs = compute_observables_full(model, N_pivot=N)
        bd = chi2_full_with_breakdown(
            obs, target_ns=target_ns, sigma_ns=sigma_ns,
            target_r=target_r, sigma_r=sigma_r, model=model,
        )
        diag["chi2_breakdown"] = bd.as_dict()
        # Top contributors
        top = bd.dominant_components(3)
        if top:
            diag["dominant_chi2"] = [
                {"component": k, "contribution": v} for k, v in top
            ]
    return json.dumps(diag, indent=2, default=str)


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
        diagnose_egb_model_tool,
    ]
    if include_rag:
        tools.append(retrieve_literature_tool)
    return tools
