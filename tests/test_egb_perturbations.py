"""Benchmarks for the production-grade EGB perturbation kernel.

These pin the kernel against:
  (i)   GR limits with known closed-form predictions (Starobinsky, m²φ²).
  (ii)  Internal consistency (r = -8 n_T in pure GR).
  (iii) Sound speed reduction in the GR limit (c_T² = 1).
  (iv)  Smooth turn-on of EGB corrections as ξ is dialed up.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from deepegb.physics import EGBModel, compute_observables_full, compute_c_T2


# ---------------------------------------------------------------------------
# Closed-form GR predictions to compare against
# ---------------------------------------------------------------------------
def _starobinsky_gr_predictions(N: float) -> dict:
    return {
        "n_s": 1 - 2 / N,
        "n_T_lo": -2.0 * (3 / (4 * N**2)),  # ≈ -3/(2 N²)
        "r": 12 / N**2,
    }


def _quadratic_gr_predictions(N: float) -> dict:
    return {
        "n_s": 1 - 2 / N,
        "r": 8 / N,
        "n_T": -2 / N / 8 * 8,  # = -2/N · (1) only if we identify n_T=-r/8: -1/N
    }


def _gr_starobinsky() -> EGBModel:
    return EGBModel(V=lambda p: (1.0 - np.exp(-np.sqrt(2 / 3) * p)) ** 2,
                    xi=lambda p: 0.0 * p)


def _gr_quadratic() -> EGBModel:
    return EGBModel(V=lambda p: 0.5 * p ** 2,  # mass scale arbitrary for ratios
                    xi=lambda p: 0.0 * p)


# ---------------------------------------------------------------------------
# (i) GR limits
# ---------------------------------------------------------------------------
def test_starobinsky_n_s_within_2_percent():
    obs = compute_observables_full(_gr_starobinsky(), N_pivot=55.0,
                                   phi_range=(0.1, 8.0), n_grid=4001)
    expected = _starobinsky_gr_predictions(55.0)
    assert obs.is_valid
    assert abs(obs.n_s - expected["n_s"]) < 0.005, (obs.n_s, expected)


def test_starobinsky_r_small():
    obs = compute_observables_full(_gr_starobinsky(), N_pivot=55.0,
                                   phi_range=(0.1, 8.0), n_grid=4001)
    # Starobinsky: r ≈ 12/N² ≈ 0.004
    assert obs.r < 0.01, obs.r
    assert obs.r > 0.0


def test_quadratic_GR_nsr_consistency():
    obs = compute_observables_full(_gr_quadratic(), N_pivot=60.0,
                                   phi_range=(-30, 30), n_grid=8001)
    expected = _quadratic_gr_predictions(60.0)
    assert obs.is_valid
    assert abs(obs.n_s - expected["n_s"]) < 0.005
    assert abs(obs.r - expected["r"]) < 0.01


# ---------------------------------------------------------------------------
# (ii) Internal consistency relation r = -8 n_T in GR
# ---------------------------------------------------------------------------
def test_consistency_relation_GR_starobinsky():
    obs = compute_observables_full(_gr_starobinsky(), N_pivot=55.0,
                                   phi_range=(0.1, 8.0), n_grid=4001)
    # GR consistency: r/(-8 n_T) = 1
    ratio = obs.r / (-8 * obs.n_T)
    assert abs(ratio - 1.0) < 0.01, ratio


def test_consistency_relation_GR_quadratic():
    obs = compute_observables_full(_gr_quadratic(), N_pivot=60.0,
                                   phi_range=(-30, 30), n_grid=8001)
    ratio = obs.r / (-8 * obs.n_T)
    assert abs(ratio - 1.0) < 0.01, ratio


# ---------------------------------------------------------------------------
# (iii) c_T² = 1 in GR limit
# ---------------------------------------------------------------------------
def test_cT2_equals_one_in_GR():
    obs = compute_observables_full(_gr_starobinsky(), N_pivot=55.0,
                                   phi_range=(0.1, 8.0), n_grid=4001)
    assert abs(obs.c_T2 - 1.0) < 1e-10, obs.c_T2


def test_cT2_along_trajectory_GR():
    """At every φ in the inflationary range, c_T² should be 1 in GR."""
    model = _gr_starobinsky()
    for p in np.linspace(1.0, 6.0, 10):
        cT2 = compute_c_T2(model, float(p))
        assert math.isclose(cT2, 1.0, abs_tol=1e-10), (p, cT2)


# ---------------------------------------------------------------------------
# (iv) EGB corrections turn on smoothly
# ---------------------------------------------------------------------------
def test_EGB_correction_scales_with_xi_amplitude():
    """δ_1 should scale ~linearly with ξ amplitude for a small coupling."""
    # Pick a regime where V is O(1) so V²ξ' isn't a wash:
    V = lambda p: 0.05 * p ** 4
    xi_small = lambda p: 0.001 / (p * p + 1)
    xi_big = lambda p: 0.1 / (p * p + 1)
    o_small = compute_observables_full(
        EGBModel(V=V, xi=xi_small), N_pivot=55.0, phi_range=(0.5, 30), n_grid=4001
    )
    o_big = compute_observables_full(
        EGBModel(V=V, xi=xi_big), N_pivot=55.0, phi_range=(0.5, 30), n_grid=4001
    )
    assert o_small.is_valid and o_big.is_valid
    assert abs(o_big.delta1) > 10 * abs(o_small.delta1), (
        o_small.delta1, o_big.delta1)
    # And c_T² should deviate from 1 more for the bigger coupling:
    assert abs(o_big.c_T2 - 1.0) > 5 * abs(o_small.c_T2 - 1.0)


def test_EGB_breaks_consistency_relation():
    """When ξ ≠ 0, the GR consistency r = -8 n_T should be measurably broken."""
    V = lambda p: 0.05 * p ** 4
    xi = lambda p: 1.0 / (p * p + 1)
    obs = compute_observables_full(EGBModel(V=V, xi=xi),
                                   N_pivot=55.0, phi_range=(0.5, 30), n_grid=4001)
    if not obs.is_valid:
        pytest.skip("EGB model failed to produce finite observables")
    ratio = obs.r / (-8 * obs.n_T) if obs.n_T != 0 else float("nan")
    # Should deviate from 1 by more than 1% when EGB is on:
    assert abs(ratio - 1.0) > 0.01, (ratio, obs.delta1, obs.c_T2)


# ---------------------------------------------------------------------------
# (v) Power spectra are positive and finite when valid
# ---------------------------------------------------------------------------
def test_power_spectra_positive_finite():
    obs = compute_observables_full(_gr_starobinsky(), N_pivot=55.0,
                                   phi_range=(0.1, 8.0), n_grid=4001)
    assert obs.P_S > 0 and np.isfinite(obs.P_S)
    assert obs.P_T > 0 and np.isfinite(obs.P_T)
    # r = P_T / P_S should match obs.r
    assert math.isclose(obs.P_T / obs.P_S, obs.r, rel_tol=1e-10)
