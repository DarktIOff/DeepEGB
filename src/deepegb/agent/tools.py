"""Agno-compatible tool functions exposed to the LLM.

Each function is a plain Python callable that Agno will introspect by type
annotation + docstring to build its tool schema. We keep them small and
side-effect-free (apart from the plot-saving tool).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..analysis import analyze_egb_model, plot_egb_model
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
    """Render a 4-panel diagnostic plot for an EGB inflation model and save
    it to disk. Returns the saved path as a string.
    """
    p = plot_egb_model(V_expr, xi_expr, N=N, out_path=out_path)
    return f"Saved diagnostic plot to: {p}"


# ----------------------------------------------------------------------------
# Helper to register tools with an Agno agent.
# ----------------------------------------------------------------------------
def all_tools() -> list:
    return [search_egb_potentials, analyze_egb_model_tool, plot_egb_model_tool]
