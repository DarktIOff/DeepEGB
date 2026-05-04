"""High-level analyze tool: takes string expressions, returns observables.

Uses the **production-grade** EGB perturbation module (egb_perturbations.py),
which returns n_s, n_T, r, α_s, P_S, P_T, c_T², c_S² with the GB-corrected
forms of the slow-roll parameters and tensor sound speed.

For relic-GW analysis, see `analyze_egb_relic_gw` which integrates the
Mukhanov–Sasaki equations (egb_modes.py) over the full background
trajectory (egb_background.py) and computes Ω_GW(f) h² today (relic_gw.py).

Set ``leading_order=True`` to fall back to the simple (1/2 Q V'/V²) toy
kernel; only useful for cross-checks.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..physics import (
    analyze_model,
    compute_observables_full,
    integrate_with_pivot,
    k_inflation_to_today_Mpc_inv,
    k_pivot_from_traj,
    relic_gw_spectrum,
    scalar_power_spectrum,
    tensor_power_spectrum,
)
from ..search.pysr_search import expressions_to_model


def analyze_egb_model(
    V_expr: str,
    xi_expr: str = "0",
    *,
    N: float = 55.0,
    phi_range: tuple[float, float] = (-15.0, 15.0),
    n_grid: int = 4001,
    leading_order: bool = False,
) -> dict[str, Any]:
    """Compute observables (n_s, n_T, r, α_s, P_S, P_T, c_T², c_S²) for an EGB
    model given as Sympy-style strings in `phi`.

    Parameters
    ----------
    V_expr        : V(φ) expression. Use ``phi`` as the field.
    xi_expr       : ξ(φ) expression. Pass ``"0"`` for the GR limit.
    N             : Number of e-folds before end of inflation at the pivot.
    phi_range     : Bracket for φ scanning.
    n_grid        : Resolution of the φ scan.
    leading_order : If True, fall back to the simple (1/2 Q V'/V²) kernel
                    (no c_T, n_T, P_S, P_T amplitudes). Only for cross-checks.
    """
    model = expressions_to_model(V_expr, xi_expr)
    if leading_order:
        obs = analyze_model(model, N_target=N, phi_range=phi_range, n_grid=n_grid)
        return {
            "V_expr": V_expr, "xi_expr": xi_expr, "N": N,
            "method": "leading_order",
            **obs.as_dict(),
            "valid": obs.is_valid,
        }
    full = compute_observables_full(model, N_pivot=N, phi_range=phi_range, n_grid=n_grid)
    return {
        "V_expr": V_expr, "xi_expr": xi_expr, "N": N,
        "method": "production",
        **full.as_dict(),
        "valid": full.is_valid,
        # convenience derived quantities:
        "ln10_As": float(__import__("math").log(1e10 * full.P_S)) if full.P_S > 0 else None,
        "consistency_r_minus_8nT": (full.r / (-8 * full.n_T)) if full.n_T not in (0.0,) and full.n_T == full.n_T else None,
    }


def analyze_egb_relic_gw(
    V_expr: str,
    xi_expr: str = "0",
    *,
    N: float = 55.0,
    n_decades: float = 6.0,
    n_k: int = 24,
    T_reh_GeV: float | None = 1.0e15,
    phi_range: tuple[float, float] = (-15.0, 15.0),
) -> dict[str, Any]:
    """Compute the relic GW spectrum Ω_GW(f) h² for an EGB model.

    Workflow:
      1. Solve full background EOMs (`egb_background.integrate_with_pivot`).
      2. Sample comoving wavenumbers around k_pivot covering n_decades each
         side (log-spaced).
      3. Integrate the tensor Mukhanov-Sasaki equation for each mode.
      4. Apply the radiation/matter-domination transfer function and
         convert k → today's frequency (Hz).

    Returns a dict with arrays for k, P_T(k), Ω_GW h²(k), 𝒯²(k), and
    f_today (Hz).
    """
    model = expressions_to_model(V_expr, xi_expr)
    traj = integrate_with_pivot(model, N_pivot=N, phi_range=phi_range)
    if traj is None:
        return {"V_expr": V_expr, "xi_expr": xi_expr, "valid": False,
                "error": "background integration failed"}
    k_pivot = k_pivot_from_traj(traj, N_pivot=N)
    k_arr = k_pivot * np.logspace(-n_decades / 2, n_decades / 2, n_k)
    spec = relic_gw_spectrum(model, k_arr, traj=traj, N_pivot=N, T_reh_GeV=T_reh_GeV)
    return {
        "V_expr": V_expr,
        "xi_expr": xi_expr,
        "N_pivot": N,
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
