"""High-level analyze tool: takes string expressions, returns observables.

Uses the **production** EGB perturbation kernel (egb_perturbations.py):
n_s, n_T, r, α_s, P_S, P_T, c_T², c_S² with the GB-corrected slow-roll
parameters and tensor / scalar sound speeds. For relic-GW analysis,
`analyze_egb_relic_gw` integrates the full background EOMs and the
tensor Mukhanov-Sasaki equation across a frequency band.

The legacy leading-order toy kernel has been removed.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..config.defaults import DEFAULTS
from ..physics import (
    compute_N_pivot_from_model,
    compute_observables_full,
    diagnose_model,
    egb_consistency_metric,
    integrate_background_robust,
    integrate_with_pivot,
    k_inflation_to_today_Mpc_inv,
    k_pivot_from_traj,
    relic_gw_spectrum,
    scalar_power_spectrum,
    tensor_power_spectrum,
)
from ..search.pysr_search import expressions_to_model


def _resolve_N_auto(model, N, phi_range, T_reh_GeV=None):
    """Return (N_pivot, N_was_auto) after optional self-consistent computation."""
    N_was_auto = N is None
    if N is None:
        kw = dict(
            phi_range=phi_range,
            N_min=DEFAULTS.physics.N_pivot_min,
            N_max=DEFAULTS.physics.N_pivot_max,
        )
        if T_reh_GeV is not None:
            kw["T_reh_GeV"] = T_reh_GeV
        N = compute_N_pivot_from_model(model, **kw)
    return float(N), N_was_auto


def analyze_egb_model(
    V_expr: str,
    xi_expr: str | None = None,
    *,
    N: float | None = None,
    phi_range: tuple[float, float] | None = None,
    n_grid: int = 10001,
) -> dict[str, Any]:
    """Compute observables (n_s, n_T, r, α_s, P_S, P_T, c_T², c_S²) for an EGB
    model given as Sympy-style strings in `phi`.

    Parameters
    ----------
    V_expr    : V(φ) expression. Use ``phi`` as the field.
    xi_expr   : ξ(φ) expression. Default: nontrivial EGB coupling from config.
    N         : Number of e-folds before end of inflation at the pivot.
    phi_range : Bracket for φ scanning. Default from centralized config.
    n_grid    : Resolution of the φ scan.
    """
    if xi_expr is None:
        xi_expr = DEFAULTS.default_xi_expr
    if phi_range is None:
        phi_range = DEFAULTS.phi_range
    model = expressions_to_model(V_expr, xi_expr)
    N, N_was_auto = _resolve_N_auto(model, N, phi_range)
    full = compute_observables_full(model, N_pivot=N, phi_range=phi_range, n_grid=n_grid)
    cons_metric = egb_consistency_metric(full)
    return {
        "V_expr": V_expr, "xi_expr": xi_expr, "N_pivot": N,
        "N_was_auto": N_was_auto,
        "method": "production",
        **full.as_dict(),
        "valid": full.is_valid,
        "egb_consistency": cons_metric["egb_consistency"],
        "c_T2_deviation": cons_metric["c_T2_deviation"],
        "delta1_magnitude": cons_metric["delta1_magnitude"],
        "ln10_As": float(__import__("math").log(1e10 * full.P_S)) if full.P_S > 0 else None,
    }


def analyze_egb_relic_gw(
    V_expr: str,
    xi_expr: str | None = None,
    *,
    N: float | None = None,
    n_decades: float | None = None,
    n_k: int | None = None,
    T_reh_GeV: float | None = None,
    phi_range: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Compute the relic GW spectrum Ω_GW(f) h² for an EGB model.

    Workflow:
      1. Solve full background EOMs (`egb_background.integrate_with_pivot`).
      2. Sample comoving wavenumbers around k_pivot covering n_decades each
         side (log-spaced).
      3. Integrate the tensor Mukhanov-Sasaki equation for each mode.
      4. Apply the radiation/matter-domination transfer function and
         convert k → today's frequency (Hz).

    n_decades and n_k policy: defaults come from the centralized config.
    n_decades=8 covers the LISA–DECIGO band (Watanabe & Komatsu 2006,
    astro-ph/0604176); 10 gives the full RD+MD transition.
    T_reh_GeV controls the g_* thermal correction in the transfer function
    (Kuroyanagi et al. 2015, arXiv:1407.4785).

    Returns a dict with arrays for k, P_T(k), Ω_GW h²(k), 𝒯²(k), and
    f_today (Hz).
    """
    if xi_expr is None:
        xi_expr = DEFAULTS.default_xi_expr
    if phi_range is None:
        phi_range = DEFAULTS.phi_range
    if n_decades is None:
        n_decades = DEFAULTS.n_decades
    if n_k is None:
        n_k = DEFAULTS.n_k
    if T_reh_GeV is None:
        T_reh_GeV = DEFAULTS.T_reh_GeV
    model = expressions_to_model(V_expr, xi_expr)
    N, N_was_auto = _resolve_N_auto(model, N, phi_range, T_reh_GeV=T_reh_GeV)
    obs = compute_observables_full(model, N_pivot=N, phi_range=phi_range)
    traj, ladder_log = integrate_background_robust(
        model, N_pivot=N, phi_range=phi_range,
        obs=obs if obs.is_valid else None,
    )
    if traj is None:
        diag = diagnose_model(model, N_pivot=N, phi_range=phi_range)
        return {
            "V_expr": V_expr, "xi_expr": xi_expr, "valid": False,
            "error": "background integration failed across the full retry "
                     "ladder (see `ladder_log` and `diagnosis` for the "
                     "actionable reason)",
            "ladder_log": ladder_log,
            "diagnosis": diag,
        }
    k_pivot = k_pivot_from_traj(traj, N_pivot=N)
    k_arr = k_pivot * np.logspace(-n_decades / 2, n_decades / 2, n_k)
    spec = relic_gw_spectrum(model, k_arr, traj=traj, N_pivot=N, T_reh_GeV=T_reh_GeV)
    return {
        "V_expr": V_expr,
        "xi_expr": xi_expr,
        "N_pivot": N,
        "N_was_auto": N_was_auto,
        "T_reh_GeV": T_reh_GeV,
        "k_inflation": k_arr.tolist(),
        "k_today_Mpc_inv": k_inflation_to_today_Mpc_inv(
            k_arr, H_pivot=float(traj.H[int(np.argmin(np.abs(traj.N - (traj.N_end - N))))]),
            a_pivot=float(traj.a[int(np.argmin(np.abs(traj.N - (traj.N_end - N))))]),
        ).tolist(),
        "f_today_Hz": (spec.f_today.tolist() if spec.f_today is not None else None),
        "P_T": spec.P_T.tolist(),
        "transfer_sq": spec.transfer_sq.tolist(),
        "Omega_GW_h2": spec.Omega_GW_h2.tolist(),
        "k_pivot_inflation": k_pivot,
        "valid": bool(np.isfinite(spec.Omega_GW_h2).any()),
    }
