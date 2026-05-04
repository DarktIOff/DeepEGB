"""
Tests for the full-kernel modules:
  - egb_background.py  (full EOM integration with solve_ivp)
  - egb_modes.py       (Mukhanov-Sasaki mode integration)
  - relic_gw.py        (Ω_GW(f) h² spectrum today)

The MS integrator is benchmarked against the slow-roll closed-form kernel
in `egb_perturbations.py`, which itself is benchmarked against textbook
GR limits in `tests/test_egb_perturbations.py`.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from deepegb.physics import (
    EGBModel,
    compute_observables_full,
    integrate_with_pivot,
    k_pivot_from_traj,
    relic_gw_spectrum,
    scalar_power_spectrum,
    tensor_power_spectrum,
)


@pytest.fixture(scope="module")
def starobinsky():
    V0 = 1e-10
    return EGBModel(
        V=lambda p: V0 * (1.0 - np.exp(-np.sqrt(2 / 3) * p)) ** 2,
        xi=lambda p: 0.0 * p,
        name="Starobinsky GR",
    )


@pytest.fixture(scope="module")
def traj(starobinsky):
    t = integrate_with_pivot(starobinsky, N_pivot=55.0, phi_range=(0.1, 8.0))
    assert t is not None, "background integration failed"
    return t


# ---------------------------------------------------------------------------
# Background integration
# ---------------------------------------------------------------------------
def test_background_reaches_end_of_inflation(traj):
    assert abs(traj.eps1[-1] - 1.0) < 5.0e-3, traj.eps1[-1]


def test_background_pivot_field_value(traj):
    """At N_end - 55, φ should be ≈ 5.35 for Starobinsky (textbook)."""
    idx = int(np.argmin(np.abs(traj.N - (traj.N_end - 55.0))))
    assert abs(traj.phi[idx] - 5.35) < 0.1, traj.phi[idx]


def test_background_eps_positive_during_inflation(traj):
    # ε₁ should be positive for the entire inflationary trajectory
    inflationary = traj.eps1[:-5]   # last few points are at ε ≈ 1
    assert (inflationary > 0).all()


# ---------------------------------------------------------------------------
# Mukhanov-Sasaki vs slow-roll
# ---------------------------------------------------------------------------
def test_MS_tensor_matches_slow_roll(traj, starobinsky):
    """At the pivot, MS-integrated P_T should agree with the slow-roll
    closed-form prediction to within a few percent."""
    sr = compute_observables_full(starobinsky, N_pivot=55.0, phi_range=(0.1, 8.0))
    k_pivot = k_pivot_from_traj(traj, N_pivot=55.0)
    P_T_MS, _ = tensor_power_spectrum(starobinsky, np.array([k_pivot]), traj=traj)
    assert abs(P_T_MS[0] / sr.P_T - 1.0) < 0.05, (P_T_MS[0], sr.P_T)


def test_MS_scalar_matches_slow_roll(traj, starobinsky):
    sr = compute_observables_full(starobinsky, N_pivot=55.0, phi_range=(0.1, 8.0))
    k_pivot = k_pivot_from_traj(traj, N_pivot=55.0)
    P_S_MS, _ = scalar_power_spectrum(starobinsky, np.array([k_pivot]), traj=traj)
    assert abs(P_S_MS[0] / sr.P_S - 1.0) < 0.05, (P_S_MS[0], sr.P_S)


def test_MS_n_s_via_finite_difference(traj, starobinsky):
    """Compute n_s from MS by finite differencing log P_S over k."""
    k_pivot = k_pivot_from_traj(traj, N_pivot=55.0)
    k_arr = k_pivot * np.exp(np.array([-1.0, 0.0, 1.0]))
    P_S_MS, _ = scalar_power_spectrum(starobinsky, k_arr, traj=traj)
    n_s_MS = 1.0 + (np.log(P_S_MS[2]) - np.log(P_S_MS[0])) / 2.0
    expected = 1.0 - 2.0 / 55.0    # ≈ 0.96364 for Starobinsky
    assert abs(n_s_MS - expected) < 0.005, (n_s_MS, expected)


def test_MS_consistency_relation_in_GR(traj, starobinsky):
    """In GR, the MS-derived r and n_T should satisfy r = -8 n_T."""
    k_pivot = k_pivot_from_traj(traj, N_pivot=55.0)
    k_arr = k_pivot * np.exp(np.array([-0.5, 0.0, 0.5]))
    P_T, _ = tensor_power_spectrum(starobinsky, k_arr, traj=traj)
    P_S, _ = scalar_power_spectrum(starobinsky, k_arr, traj=traj)
    n_T = (np.log(P_T[2]) - np.log(P_T[0])) / 1.0
    r = P_T[1] / P_S[1]
    ratio = r / (-8.0 * n_T) if n_T != 0 else float("nan")
    assert abs(ratio - 1.0) < 0.1, ratio


# ---------------------------------------------------------------------------
# Relic GW
# ---------------------------------------------------------------------------
def test_relic_gw_finite_in_RD_band(traj, starobinsky):
    k_pivot = k_pivot_from_traj(traj, N_pivot=55.0)
    k_arr = k_pivot * np.logspace(2, 4, 5)   # well into RD regime
    spec = relic_gw_spectrum(starobinsky, k_arr, traj=traj, T_reh_GeV=1e15)
    assert (spec.Omega_GW_h2 > 0).all()
    assert np.isfinite(spec.Omega_GW_h2).all()


def test_relic_gw_starobinsky_amplitude(traj, starobinsky):
    """Starobinsky predicts Ω_GW h² ~ 1e-18 in the RD band — a tiny amplitude
    but the right order of magnitude (well below LISA / DECIGO sensitivity)."""
    k_pivot = k_pivot_from_traj(traj, N_pivot=55.0)
    k_arr = np.array([k_pivot * 1e3])   # well into RD
    spec = relic_gw_spectrum(starobinsky, k_arr, traj=traj, T_reh_GeV=1e15)
    assert 1e-20 < spec.Omega_GW_h2[0] < 1e-16, spec.Omega_GW_h2[0]


def test_relic_gw_MD_suppression(traj, starobinsky):
    """Modes that re-enter during MD (k < k_eq today ≈ 0.01 Mpc⁻¹) get a
    (k_eq/k)² boost in 𝒯². We test directly via the transfer function."""
    from deepegb.physics import k_inflation_to_today_Mpc_inv
    from deepegb.physics.relic_gw import transfer_function_sq, K_EQ_MPC

    # Construct a today-frame k well below k_eq:
    k_today_lo = K_EQ_MPC * 0.1
    k_today_hi = K_EQ_MPC * 100.0
    Tlo = transfer_function_sq(np.array([k_today_lo]))
    Thi = transfer_function_sq(np.array([k_today_hi]))
    assert Tlo[0] > Thi[0], (Tlo[0], Thi[0])
    # And the MD suppression factor should be approx (k_eq/k)² × ½:
    assert math.isclose(Tlo[0], 0.5 * (1.0 / 0.1) ** 2, rel_tol=0.05)
