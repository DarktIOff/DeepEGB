"""Physics kernels for EGB inflation.

Single production stack:
  - egb_slow_roll.py    EGBModel + φ-trajectory utilities (no toy observables)
  - egb_background.py   Full Friedmann + KG ODE integration (solve_ivp)
  - egb_n3lo.py          Analytic N3LO observables (Green's-function exact
                         coefficients; production analytic path)
  - egb_perturbations.py Observable pipeline (n3lo → MS → slow-roll fallback)
  - egb_modes.py         Mukhanov-Sasaki mode integration (P_T, P_S exact)
  - egb_uaa.py           Uniform-asymptotic-approximation observables
                         (cross-check only — carries a ~0.15% method residual)
  - n_pivot.py           Self-consistent N_pivot computation (Liddle-Leach)
  - relic_gw.py          Ω_GW(f) h² with detector sensitivity overlay
  - detectors.py         Catalogue of GW experiments and their Ω_GW floors
"""
from .egb_slow_roll import (
    EGBModel,
    end_of_inflation,
)
from .egb_n3lo import (
    compute_observables_n3lo,
    reduce_sector,
    sector_grids,
)
from .egb_perturbations import (
    FullObservables,
    background_at,
    background_along,
    chi2_full,
    chi2_full_with_breakdown,
    chi2_relic_gw,
    chi2_relic_gw_with_breakdown,
    compute_c_S2,
    compute_c_T2,
    compute_observables_full,
    egb_consistency_metric,
    integrate_background_robust,
    power_spectra_at,
)
from .diagnostics import (
    Chi2Breakdown,
    chi2_full_breakdown,
    chi2_omega_gw_breakdown,
    diagnose_model,
    soft_invalid_penalty,
)
from .egb_background import (
    BackgroundTrajectory,
    integrate_background,
    integrate_with_pivot,
    hubble_from_constraint,
)
from .egb_modes import (
    tensor_power_spectrum,
    scalar_power_spectrum,
    k_pivot_from_traj,
)
from .n_pivot import (
    compute_N_pivot,
    compute_N_pivot_from_model,
)
from .relic_gw import (
    GWSpectrum,
    relic_gw_spectrum,
    k_inflation_to_today_Mpc_inv,
    k_today_Mpc_inv_to_freq_Hz,
)
from .detectors import (
    Detector,
    DETECTORS,
    detector_by_name,
    detectors_in_band,
    sensitivity_at,
)
from .normalize import (
    NormalizationResult,
    PLANCK_A_S,
    PLANCK_LN_10_10_A_S,
    normalize_egb_model,
)

__all__ = [
    # Core data types and trajectory
    "EGBModel",
    "end_of_inflation",
    # Analytic N3LO observables (production analytic path)
    "compute_observables_n3lo",
    "reduce_sector",
    "sector_grids",
    # Slow-roll closed-form observables (production)
    "FullObservables",
    "background_at",
    "background_along",
    "chi2_full",
    "chi2_full_with_breakdown",
    "chi2_relic_gw",
    "chi2_relic_gw_with_breakdown",
    "compute_c_S2",
    "compute_c_T2",
    "compute_observables_full",
    "egb_consistency_metric",
    "integrate_background_robust",
    "power_spectra_at",
    # Diagnostics
    "Chi2Breakdown",
    "chi2_full_breakdown",
    "chi2_omega_gw_breakdown",
    "diagnose_model",
    "soft_invalid_penalty",
    # Full background EOM integration
    "BackgroundTrajectory",
    "integrate_background",
    "integrate_with_pivot",
    "hubble_from_constraint",
    # Mukhanov-Sasaki
    "tensor_power_spectrum",
    "scalar_power_spectrum",
    "k_pivot_from_traj",
    # Self-consistent N_pivot (Liddle-Leach 2003, Martin-Ringeval 2010)
    "compute_N_pivot",
    "compute_N_pivot_from_model",
    # Relic GW spectrum
    "GWSpectrum",
    "relic_gw_spectrum",
    "k_inflation_to_today_Mpc_inv",
    "k_today_Mpc_inv_to_freq_Hz",
    # Detectors
    "Detector",
    "DETECTORS",
    "detector_by_name",
    "detectors_in_band",
    "sensitivity_at",
    # Amplitude normalisation
    "NormalizationResult",
    "PLANCK_A_S",
    "PLANCK_LN_10_10_A_S",
    "normalize_egb_model",
]
