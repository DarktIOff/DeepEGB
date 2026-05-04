"""
Joint symbolic regression for V(φ) and ξ(φ) in EGB inflation.

Strategy: **two-pass joint search**.
  a. SR for V(φ) with ξ ≡ 0 (Einstein-frame baseline).
  b. With each top-K V candidate frozen, SR for ξ(φ) that improves the χ²
     further. The winner is the (V, ξ) pair with lowest combined loss.

The χ² is computed by `chi2_full` (slow-roll closed-form) or
`chi2_relic_gw` (full Mukhanov-Sasaki + Ω_GW transfer). Both come from
`physics/egb_perturbations.py`. The legacy leading-order kernel
(`r = 16ε`, etc.) has been removed; production runs only.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import sympy as sp

from ..physics import (
    EGBModel,
    chi2_full,
    chi2_relic_gw,
    compute_observables_full,
)

try:
    from pysr import PySRRegressor
except ImportError:  # PySR may not be installed in lightweight environments
    PySRRegressor = None  # type: ignore


# ---------------------------------------------------------------------------
# Config + result types
# ---------------------------------------------------------------------------
@dataclass
class SearchConfig:
    # Mandatory targets
    target_ns: float = 0.974
    sigma_ns: float = 0.003
    target_r: float = 0.0
    sigma_r: float = 0.018

    # Optional production-grade targets (None ⇒ excluded from χ²)
    target_lnAs: float | None = None       # ln(10¹⁰ A_s); Planck ≈ 3.044
    sigma_lnAs: float = 0.014
    target_alphas: float | None = None     # running of n_s
    sigma_alphas: float = 0.013
    target_nT: float | None = None         # tensor spectral index
    sigma_nT: float = 0.1
    target_cT2: float | None = None        # tensor sound-speed squared
    sigma_cT2: float = 0.05

    N_pivot: float = 55.0

    # Loss kernel:
    #   "production"     — slow-roll closed-form (background EOMs + full
    #                      perturbation kernel with c_T², c_S²). ~10 ms / call,
    #                      default for n_s/r searches.
    #   "production_gw"  — full background ODE + Mukhanov-Sasaki for the
    #                      target k modes + relic-GW transfer.
    #                      ~0.5–2 s / call; required when omega_gw_targets
    #                      or omega_gw_band_min are set.
    loss_kind: str = "production"

    # Relic-GW targets (only used when loss_kind == "production_gw").
    # Each entry: (frequency_in_Hz, target_Omega_GW_h2, sigma_Omega_GW_h2).
    # Interpreted as a target on log10(Ω_GW h²); sigma is taken in log10 space
    # using the relative formula sigma_log = (sigma/target)/ln(10).
    omega_gw_targets: tuple[tuple[float, float, float], ...] = ()
    # (f_lo_Hz, f_hi_Hz, target_min_Omega_GW_h2): adds a band-floor penalty
    # that increases when the spectrum drops below target_min anywhere in
    # [f_lo, f_hi]. Useful for "make this loud enough to detect at LISA".
    omega_gw_band_min: tuple[float, float, float] | None = None
    T_reh_GeV: float | None = 1.0e15

    # PySR hyperparameters
    niterations: int = 40
    populations: int = 35
    population_size: int = 33
    maxsize: int = 25
    parsimony: float = 1.0e-3

    binary_operators: tuple[str, ...] = ("+", "-", "*", "/")
    unary_operators: tuple[str, ...] = ("exp", "log", "sqrt", "tanh")

    # Search strategy
    mode: str = "two_pass"   # "two_pass" or "joint"
    top_k_V: int = 5         # how many V candidates to retain in two-pass mode

    # Sampling grid for X data fed to PySR (PySR fits an objective via
    # `loss_function` so X values are largely irrelevant; we still need them).
    n_samples: int = 96
    phi_sample_range: tuple[float, float] = (-10.0, 10.0)

    phi_search_range: tuple[float, float] = (-15.0, 15.0)
    phi_search_grid: int = 4001

    runs_dir: str | Path = "runs"

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class SearchResult:
    V_expr: str
    xi_expr: str
    chi2: float
    # Core
    n_s: float
    r: float
    epsilon: float
    phi_N: float
    phi_end: float
    elapsed_s: float
    # Production-grade extras (NaN if computed via the leading-order kernel)
    n_T: float = float("nan")
    alpha_s: float = float("nan")
    P_S: float = float("nan")
    P_T: float = float("nan")
    c_T2: float = float("nan")
    delta1: float = float("nan")
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {**self.__dict__}


# ---------------------------------------------------------------------------
# Sympy → callable
# ---------------------------------------------------------------------------
def _sympy_to_callable(expr_str: str) -> Callable[[np.ndarray], np.ndarray]:
    """Parse a string expression in φ to a vectorised callable."""
    phi = sp.Symbol("phi", real=True)
    try:
        expr = sp.sympify(expr_str, locals={"phi": phi})
    except Exception as e:
        raise ValueError(f"Could not parse expression {expr_str!r}: {e}")
    return sp.lambdify(phi, expr, modules=["numpy"])


def expressions_to_model(V_expr: str, xi_expr: str = "0", *, name: str = "model") -> EGBModel:
    return EGBModel(
        V=_sympy_to_callable(V_expr),
        xi=_sympy_to_callable(xi_expr),
        name=name,
        description=f"V={V_expr};  ξ={xi_expr}",
    )


# ---------------------------------------------------------------------------
# χ² loss as a function of expression strings
# ---------------------------------------------------------------------------
def chi2_for_expressions(
    V_expr: str,
    xi_expr: str,
    cfg: SearchConfig,
) -> float:
    """Evaluate χ² for a (V, ξ) expression pair using the production kernel."""
    try:
        model = expressions_to_model(V_expr, xi_expr)
        if cfg.loss_kind == "production_gw":
            return chi2_relic_gw(
                model,
                target_ns=cfg.target_ns, sigma_ns=cfg.sigma_ns,
                target_r=cfg.target_r, sigma_r=cfg.sigma_r,
                target_lnAs=cfg.target_lnAs, sigma_lnAs=cfg.sigma_lnAs,
                omega_gw_targets=list(cfg.omega_gw_targets) or None,
                omega_gw_band_min=cfg.omega_gw_band_min,
                N_pivot=cfg.N_pivot,
                T_reh_GeV=cfg.T_reh_GeV,
            )
        # production (default): slow-roll closed-form with full perturbations
        obs_full = compute_observables_full(
            model,
            N_pivot=cfg.N_pivot,
            phi_range=cfg.phi_search_range,
            n_grid=cfg.phi_search_grid,
        )
        return chi2_full(
            obs_full,
            target_ns=cfg.target_ns, sigma_ns=cfg.sigma_ns,
            target_r=cfg.target_r, sigma_r=cfg.sigma_r,
            target_lnAs=cfg.target_lnAs, sigma_lnAs=cfg.sigma_lnAs,
            target_alphas=cfg.target_alphas, sigma_alphas=cfg.sigma_alphas,
            target_nT=cfg.target_nT, sigma_nT=cfg.sigma_nT,
            target_cT2=cfg.target_cT2, sigma_cT2=cfg.sigma_cT2,
        )
    except Exception:
        return 1.0e6


def observables_for_result(V_expr: str, xi_expr: str, cfg: SearchConfig) -> dict:
    """Production-grade observables for a (V, ξ) pair (slow-roll closed-form
    is fine here — used only for ranking, not for the loss)."""
    try:
        model = expressions_to_model(V_expr, xi_expr)
        o = compute_observables_full(
            model, N_pivot=cfg.N_pivot,
            phi_range=cfg.phi_search_range, n_grid=cfg.phi_search_grid,
        )
        return o.as_dict()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# PySR custom loss: PySR sees y_i = f(x_i) regression, but we override the
# loss function so the y values are unused. We still pass dummy X, y so the
# scikit-style API is happy.
# ---------------------------------------------------------------------------
def _make_pysr(cfg: SearchConfig, kind: str, fixed_other: Optional[str] = None) -> "PySRRegressor":
    if PySRRegressor is None:
        raise RuntimeError("PySR is not installed. `pip install pysr` and run `pysr.install()`.")

    # We embed the *current* candidate (V or ξ) into a Julia closure. PySR
    # gets passed a `loss_function` (Julia source string) that calls back into
    # our Python χ² evaluator via JuliaCall would be ideal — but for the MVP
    # we use PySR's `extra_sympy_mappings` and `loss_function` is approximated
    # in Julia as MSE on a dummy target. The actual χ² is then re-evaluated
    # in Python on the hall_of_fame after the run finishes. This means PySR
    # explores by raw fitting of a "dummy" curve but we re-rank using physics.
    #
    # For thesis-grade work, replace this with a true Julia loss kernel
    # (see `physics/kernel.jl` and the `custom_loss` example in PySR docs).

    return PySRRegressor(
        niterations=cfg.niterations,
        populations=cfg.populations,
        population_size=cfg.population_size,
        maxsize=cfg.maxsize,
        parsimony=cfg.parsimony,
        binary_operators=list(cfg.binary_operators),
        unary_operators=list(cfg.unary_operators),
        model_selection="best",
        progress=False,
        deterministic=False,
        random_state=0,
        verbosity=0,
        # Save scratch under runs/
        equation_file=str(Path(cfg.runs_dir) / f"hall_of_fame_{kind}.csv"),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_joint_search(
    cfg: SearchConfig,
    *,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> list[SearchResult]:
    """Run a joint search for (V, ξ) and return ranked candidates by χ²."""
    if PySRRegressor is None:
        raise RuntimeError("PySR is not installed.")

    runs_dir = Path(cfg.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    log = progress_cb or (lambda s: None)
    t0 = time.time()

    # We use a synthetic "regression target" that biases PySR toward smooth,
    # bounded functions in φ. The true ranking is by EGB-physics χ².
    rng = np.random.default_rng(0)
    phi = np.linspace(*cfg.phi_sample_range, cfg.n_samples)
    X = phi.reshape(-1, 1)
    # A gentle plateau-like target gets the search going; not load-bearing.
    y_seed = np.tanh(0.3 * phi) ** 2 * 1e-10

    # ---- Pass 1: search for V with ξ ≡ 0 ----
    log("[1/2] PySR search for V(φ) with ξ ≡ 0 …")
    pysr_V = _make_pysr(cfg, kind="V")
    pysr_V.fit(X, y_seed)
    V_candidates = _hall_of_fame_strings(pysr_V, top_k=cfg.top_k_V)

    # Re-rank V candidates by EGB χ² (with ξ = 0).
    V_ranked = sorted(
        ((vstr, chi2_for_expressions(vstr, "0", cfg)) for vstr in V_candidates),
        key=lambda it: it[1],
    )
    log(f"      Top V (with ξ=0): " + ", ".join(
        f"{v[:30]}…  χ²={c:.3g}" for v, c in V_ranked[:3]))

    # ---- Pass 2: for each top V, search ξ(φ) ----
    results: list[SearchResult] = []
    for v_idx, (V_str, _) in enumerate(V_ranked[: cfg.top_k_V]):
        log(f"[2/2] PySR search for ξ(φ) given V = {V_str!r} ({v_idx+1}/{cfg.top_k_V}) …")
        pysr_xi = _make_pysr(cfg, kind=f"xi_v{v_idx}")
        pysr_xi.fit(X, y_seed * 0.5 + 1e-12)
        xi_candidates = _hall_of_fame_strings(pysr_xi, top_k=cfg.top_k_V)
        xi_candidates = ["0", *xi_candidates]   # keep the GR baseline as a candidate

        for xi_str in xi_candidates:
            chi2 = chi2_for_expressions(V_str, xi_str, cfg)
            obs_dict = observables_for_result(V_str, xi_str, cfg)
            if not obs_dict:
                continue

            def _get(k, default=float("nan")):
                v = obs_dict.get(k, default)
                return float(v) if v is not None else default

            results.append(SearchResult(
                V_expr=V_str,
                xi_expr=xi_str,
                chi2=chi2,
                n_s=_get("n_s"),
                r=_get("r"),
                epsilon=_get("epsilon"),
                phi_N=_get("phi_N"),
                phi_end=_get("phi_end"),
                elapsed_s=time.time() - t0,
                n_T=_get("n_T"),
                alpha_s=_get("alpha_s"),
                P_S=_get("P_S"),
                P_T=_get("P_T"),
                c_T2=_get("c_T2"),
                delta1=_get("delta1"),
                extra={"V_index": v_idx, "loss_kind": cfg.loss_kind},
            ))

    results.sort(key=lambda r: (math.inf if not math.isfinite(r.chi2) else r.chi2))
    log(f"Found {len(results)} candidates; best χ² = "
        f"{results[0].chi2 if results else float('nan'):.3g}")
    return results


def _hall_of_fame_strings(reg: "PySRRegressor", top_k: int = 5) -> list[str]:
    try:
        eqs = reg.equations_  # PySR exposes a DataFrame
    except Exception:
        return []
    if eqs is None or len(eqs) == 0:
        return []
    # Prefer simpler equations: PySR sorts by complexity in `equations_`.
    out: list[str] = []
    for _, row in eqs.iterrows():
        s = str(row.get("equation", "")).strip()
        if s and s not in out:
            out.append(s)
        if len(out) >= top_k:
            break
    return out
