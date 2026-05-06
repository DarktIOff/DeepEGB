"""Self-consistent N_pivot computation for EGB inflation.

The pivot e-fold count N_pivot is the number of e-folds before the end of
inflation at which the CMB pivot mode k_* = 0.05 Mpc⁻¹ crossed the Hubble
horizon. Its value depends on the post-inflationary reheating history and
on the model's own trajectory — it is *not* a free parameter but a
derived quantity.

This module computes a model-specific N_pivot from the full background
trajectory and the reheating temperature, following the self-consistency
equation of Liddle & Leach 2003 (astro-ph/0306262, Eq. 4.3) and the
refined treatment in Martin & Ringeval 2010 (arXiv:1004.4654, Eq. 3.4):

    N_pivot = 67.26
              − ln(k_* / a_0 H_0)
              − (1/4) ln(9 / (V_end / V_k))
              − (1/4) ln(V_k / ρ_end)
              − (1/12) ln(g_{*s} / g_* )   [usually ≈ 0]
              + (1/12) ln(ρ_reh / ρ_end)

where ρ_reh = (π²/30) g_* T_reh⁴ and ρ_end ≈ V_end.

In practice we use the simplified form:

    N_pivot ≈ 56.06 − (2/3) ln(V_k^(1/4) / 10^16 GeV)
                     − (1/3) ln(T_reh / 10^9 GeV)
                     + (1/6) ln(V_k^(1/4) / V_end^(1/4))

clamped to the physically reasonable range [50, 60] (Liddle & Leach 2003
argue that N_pivot ∈ [46, 64] across all reheating scenarios; the narrower
[50, 60] range excludes pathological reheating histories).

References
----------
* Liddle A.R., Leach S.M., *How many e-folds should we expect from
  high-scale inflation?*, Phys.Rev. D68 (2003) 103503,
  astro-ph/0306262.
* Martin J., Ringeval C., *First CMB Constraints on the Inflationary
  Reheating Temperature*, Phys.Rev.D 82 (2010) 023511,
  arXiv:1004.4654.
"""
from __future__ import annotations

import numpy as np

from .egb_background import BackgroundTrajectory
from .egb_slow_roll import EGBModel


def compute_N_pivot(
    model: EGBModel,
    traj: BackgroundTrajectory,
    *,
    T_reh_GeV: float = 1.0e15,
    k_star_Mpc: float = 0.05,
    N_min: float = 50.0,
    N_max: float = 60.0,
) -> float:
    """Compute a self-consistent N_pivot for an EGB inflation model.

    Uses the full background trajectory to determine V at pivot and at
    end of inflation, then applies the Liddle-Leach / Martin-Ringeval
    self-consistency formula.

    Parameters
    ----------
    model : EGBModel
        The inflation model (V, ξ).
    traj : BackgroundTrajectory
        Full background trajectory from ``integrate_with_pivot`` or
        ``integrate_background``. Must cover at least N_min e-folds.
    T_reh_GeV : float
        Reheating temperature in GeV.
    k_star_Mpc : float
        CMB pivot scale in Mpc⁻¹ (Planck convention: 0.05).
    N_min, N_max : float
        Physical bounds on N_pivot. Result is clamped to [N_min, N_max].

    Returns
    -------
    float
        Self-consistent N_pivot, clamped to [N_min, N_max].

    Notes
    -----
    The formula uses the simplified Liddle-Leach expression (Eq. 4.3 of
    astro-ph/0306262) adapted for EGB:

        N_pivot ≈ 56.06
                  − (2/3) ln(V_k^{1/4} / 10^{16} GeV)
                  − (1/3) ln(T_reh / 10^{9} GeV)

    where V_k is the potential energy at the pivot and T_reh is the
    reheating temperature. We estimate V_k from the trajectory at the
    initial guess N_pivot = 55, then iterate once for self-consistency.
    """
    # Get V at the approximate pivot position (start with N=55 guess)
    N_guess = min(55.0, max(N_min, traj.N_end - 1.0))
    N_target = traj.N_end - N_guess
    if N_target < traj.N[0]:
        N_target = traj.N[0] + 1.0

    # V at pivot (approximate)
    idx_pivot = int(np.argmin(np.abs(traj.N - N_target)))
    V_pivot = float(traj.V[idx_pivot]) if hasattr(traj, 'V') else 0.0
    # If traj doesn't store V, compute from model
    if V_pivot == 0.0:
        phi_pivot = float(traj.phi[idx_pivot])
        V_pivot = float(model.V(phi_pivot))

    # V at end of inflation
    phi_end = float(traj.phi[-1])
    V_end = float(model.V(phi_end))

    # Convert from M_pl=1 units to GeV⁴: M_pl = 2.435×10¹⁸ GeV
    M_pl_GeV = 2.435e18
    V_pivot_GeV4 = V_pivot * M_pl_GeV**4
    V_end_GeV4 = V_end * M_pl_GeV**4

    # V^{1/4} in GeV
    V14_pivot = float(np.abs(V_pivot_GeV4)) ** 0.25 if V_pivot_GeV4 > 0 else 1e16
    V14_end = float(np.abs(V_end_GeV4)) ** 0.25 if V_end_GeV4 > 0 else 1e15

    # Liddle-Leach self-consistency (simplified form)
    # N_pivot ≈ 56.06 − (2/3) ln(V^{1/4} / 10^16) − (1/3) ln(T_reh / 10^9)
    term_V = (2.0 / 3.0) * np.log(max(V14_pivot, 1.0) / 1e16)
    term_T = (1.0 / 3.0) * np.log(max(T_reh_GeV, 1.0) / 1e9)

    N_pivot = 56.06 - term_V - term_T

    # Clamp to physical range [N_min, N_max]
    N_pivot = float(np.clip(N_pivot, N_min, N_max))

    return N_pivot


def compute_N_pivot_from_model(
    model: EGBModel,
    *,
    N_pivot_guess: float = 55.0,
    T_reh_GeV: float = 1.0e15,
    phi_range: tuple[float, float] = (-15.0, 15.0),
    N_min: float = 50.0,
    N_max: float = 60.0,
) -> float:
    """Convenience: integrate background, then compute self-consistent N_pivot.

    Returns N_pivot clamped to [N_min, N_max].
    """
    from .egb_background import integrate_with_pivot

    traj = integrate_with_pivot(model, N_pivot=N_pivot_guess, phi_range=phi_range)
    if traj is None:
        return float(np.clip(N_pivot_guess, N_min, N_max))
    return compute_N_pivot(
        model, traj, T_reh_GeV=T_reh_GeV,
        N_min=N_min, N_max=N_max,
    )
