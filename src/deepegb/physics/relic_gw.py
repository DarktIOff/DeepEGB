"""
Relic gravitational-wave background from EGB inflation.

Given the primordial tensor power spectrum P_T(k) computed by the Mukhanov–
Sasaki integrator in `egb_modes.py`, we compute the energy density of the
stochastic GW background today,

    Ω_GW(k) h² = (1/24) · Ω_R h² · P_T(k) · 𝒯(k)²,

where Ω_R h² ≈ 4.18×10⁻⁵ is the present-day radiation density (photons +
neutrinos, post-electron-positron annihilation), and 𝒯(k) is the transfer
function accounting for the post-inflation evolution.

Modes that re-enter the Hubble horizon during radiation domination
contribute the bulk of the spectrum and are well-approximated by

    𝒯(k)² ≈ ½ · (g_∗(T_in)/g_∗(T_0))·(g_{∗s,0}/g_{∗s}(T_in))^{4/3}

For modes re-entering during matter domination (k < k_eq ≈ 0.01 Mpc⁻¹):

    𝒯(k)² ≈ ½ · (k_eq/k)²    (standard suppression)

References
----------
* Watanabe Y., Komatsu E., *Improved calculation of the primordial GW
  spectrum*, Phys.Rev.D **73** (2006) 123515,
  [arXiv:astro-ph/0604176](https://arxiv.org/abs/astro-ph/0604176).
* Boyle L.A., Steinhardt P.J., *Probing the early universe with inflationary
  gravitational waves*, Phys.Rev.D **77** (2008) 063504,
  [arXiv:astro-ph/0512014](https://arxiv.org/abs/astro-ph/0512014).
* Kuroyanagi S. et al., *Probing reheating with stochastic gravitational
  waves*, JCAP 02 (2015) 003,
  [arXiv:1407.4785](https://arxiv.org/abs/1407.4785).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .egb_background import BackgroundTrajectory, integrate_with_pivot
from .egb_modes import tensor_power_spectrum, k_pivot_from_traj
from .egb_slow_roll import EGBModel

# ---------------------------------------------------------------------------
# Cosmological constants today
# ---------------------------------------------------------------------------
OMEGA_R_H2 = 4.18e-5         # Ω_R h² (photons + neutrinos)
G_STAR_0 = 3.36              # effective relativistic dof today
G_STAR_S_0 = 3.91            # effective entropy dof today
K_EQ_MPC = 0.01              # comoving wavenumber at matter-radiation eq, Mpc⁻¹
HUBBLE_TODAY = 1.0           # 1/h units; observable Ω h² is what we report

# Conversion: comoving k (in inflation M_pl=1 units) ↔ physical f today
# requires the inflation energy scale and reheating history.
# For instant reheating to RD:
#     f_today (Hz) ≈ 0.65×10⁻¹⁵ × (k/Mpc⁻¹)
# We expose conversion helpers below.

CMB_PIVOT_KSTAR_MPC = 0.05   # CMB pivot scale (per Planck convention)


@dataclass(frozen=True)
class GWSpectrum:
    """Container for the relic GW spectrum across a range of k."""

    k: np.ndarray                # comoving wavenumbers (model units)
    P_T: np.ndarray              # primordial tensor power
    Omega_GW_h2: np.ndarray      # density parameter today
    transfer_sq: np.ndarray      # transfer function squared
    f_today: np.ndarray | None = None  # physical frequency today, Hz (if conv known)


# ---------------------------------------------------------------------------
# g_* approximations (relativistic dof as function of temperature)
# ---------------------------------------------------------------------------
def g_star_at_T(T_GeV: float | np.ndarray) -> float | np.ndarray:
    """Effective relativistic dof g_*(T) as a piecewise approximation.

    Coarse-grained — sufficient for the transfer function of the SGWB (which
    only depends on g_* through the (g_*/g_{*,0})^(1/3) factor below).
    Captures the main thresholds: e⁻e⁺, μ, π, QCD, ν decoupling, top.
    """
    T = np.asarray(T_GeV, dtype=float)
    g = np.full_like(T, 106.75)   # SM total
    g = np.where(T < 175.0, 96.25, g)        # below top
    g = np.where(T < 80.0, 86.25, g)         # below W,Z
    g = np.where(T < 0.2, 17.25, g)          # below QCD (rough)
    g = np.where(T < 0.1, 14.25, g)          # below muons
    g = np.where(T < 5.0e-4, 10.75, g)       # below electrons (roughly), ν not yet decoupled
    g = np.where(T < 1.0e-4, 3.36, g)        # post e⁺e⁻ annihilation
    return g if T.shape else float(g)


# ---------------------------------------------------------------------------
# Transfer function
# ---------------------------------------------------------------------------
def transfer_function_sq(k_today_Mpc_inv: np.ndarray,
                         T_in_GeV: np.ndarray | None = None) -> np.ndarray:
    """Transfer-function squared 𝒯² at comoving k (Mpc⁻¹).

    Pieces:
      * RD modes (k > k_eq):  𝒯² = ½ · (g_*(T_in)/g_*(T_0)) ·
                                     (g_{*s,0}/g_{*s}(T_in))^(4/3)
      * MD modes (k < k_eq):  ½ · (k_eq/k)²  (suppression)

    For the RD piece if T_in_GeV is None, we use g_∗ = g_∗,0 (no thermal
    suppression — appropriate when the inflation energy scale is unknown).
    """
    k = np.asarray(k_today_Mpc_inv, dtype=float)
    transfer_sq = np.full_like(k, 0.5)

    if T_in_GeV is not None:
        Tin = np.asarray(T_in_GeV, dtype=float)
        gstar_in = g_star_at_T(Tin)
        gstars_in = gstar_in
        ratio = (gstar_in / G_STAR_0) * (G_STAR_S_0 / gstars_in) ** (4.0 / 3.0)
        transfer_sq = 0.5 * ratio * np.ones_like(k)
    md_mask = k < K_EQ_MPC
    transfer_sq = np.where(md_mask, 0.5 * (K_EQ_MPC / np.maximum(k, 1e-30)) ** 2,
                           transfer_sq)
    return transfer_sq


# ---------------------------------------------------------------------------
# Frequency / wavenumber conversion
# ---------------------------------------------------------------------------
def k_inflation_to_today_Mpc_inv(
    k_inflation: np.ndarray,
    *,
    H_pivot: float,
    a_pivot: float,
    a_today_over_a_end: float = 1.0,
    k_pivot_today_Mpc_inv: float = CMB_PIVOT_KSTAR_MPC,
    k_pivot_inflation: float | None = None,
) -> np.ndarray:
    """Map inflation comoving k (model units) to today's k in Mpc⁻¹.

    The simplest practical anchor: the inflationary mode that crossed the
    Hubble horizon N_pivot e-folds before end of inflation maps to the CMB
    pivot k_∗ = 0.05 Mpc⁻¹ today.  This sets the overall normalisation; we
    then propagate k_inflation → k_today by the same constant factor.
    """
    if k_pivot_inflation is None:
        k_pivot_inflation = a_pivot * H_pivot
    return np.asarray(k_inflation, dtype=float) * (
        k_pivot_today_Mpc_inv / k_pivot_inflation
    )


def k_today_Mpc_inv_to_freq_Hz(k_Mpc_inv: np.ndarray) -> np.ndarray:
    """Convert k (Mpc⁻¹) today to a physical frequency in Hz.

        f = k c / (2π)   with  c = (Mpc/s) × (Mpc⁻¹) / (2π)
    Numerically  1 Mpc⁻¹ ≈ 0.65×10⁻¹⁵ Hz.  Note the slight scale-factor
    quirk: this assumes the comoving k is computed today (a_0 = 1).
    """
    return 0.65e-15 * np.asarray(k_Mpc_inv, dtype=float)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def relic_gw_spectrum(
    model: EGBModel,
    k_array: np.ndarray,
    *,
    traj: BackgroundTrajectory | None = None,
    N_pivot: float = 55.0,
    T_reh_GeV: float | None = None,
    map_to_today: bool = True,
    k_pivot_today_Mpc_inv: float = CMB_PIVOT_KSTAR_MPC,
) -> GWSpectrum:
    """Compute the relic GW energy density Ω_GW(k) h² today.

    Parameters
    ----------
    model        : EGBModel (V, ξ).
    k_array      : comoving wavenumbers (in inflation model units).
    traj         : pre-integrated background trajectory (optional).
    N_pivot      : pivot e-folds before end of inflation.
    T_reh_GeV    : reheating temperature in GeV (for thermal suppression of
                   the transfer function). If None, omit the g_* correction.
    map_to_today : whether to compute a today-frame frequency for each k.
    """
    if traj is None:
        traj = integrate_with_pivot(model, N_pivot=N_pivot)
        if traj is None:
            return GWSpectrum(k_array, np.full_like(k_array, np.nan),
                              np.full_like(k_array, np.nan),
                              np.full_like(k_array, np.nan), None)
    P_T, _ = tensor_power_spectrum(model, k_array, traj=traj, N_pivot=N_pivot)

    # Map to today's k (Mpc⁻¹) using the pivot anchor
    pivot_idx = int(np.argmin(np.abs(traj.N - (traj.N_end - N_pivot))))
    H_pivot = float(traj.H[pivot_idx])
    a_pivot = float(traj.a[pivot_idx])
    k_today_Mpc = k_inflation_to_today_Mpc_inv(
        k_array, H_pivot=H_pivot, a_pivot=a_pivot,
        k_pivot_today_Mpc_inv=k_pivot_today_Mpc_inv,
    )

    if T_reh_GeV is not None:
        Tin_arr = np.full_like(k_array, T_reh_GeV, dtype=float)
    else:
        Tin_arr = None
    transfer_sq = transfer_function_sq(k_today_Mpc, T_in_GeV=Tin_arr)

    Omega_GW_h2 = (OMEGA_R_H2 / 24.0) * P_T * transfer_sq

    f_Hz = k_today_Mpc_inv_to_freq_Hz(k_today_Mpc) if map_to_today else None
    return GWSpectrum(k=k_array, P_T=P_T,
                      Omega_GW_h2=Omega_GW_h2,
                      transfer_sq=transfer_sq,
                      f_today=f_Hz)


# ---------------------------------------------------------------------------
# (Detector catalogue moved to `detectors.py` — import from there.)
# ---------------------------------------------------------------------------
