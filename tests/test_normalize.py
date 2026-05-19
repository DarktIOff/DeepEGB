"""Tests for the COBE/Planck amplitude normalisation."""
from __future__ import annotations

import math

import numpy as np
import pytest

from deepegb.physics import normalize_egb_model, PLANCK_A_S


# ---------------------------------------------------------------------------
# Pure-GR Starobinsky baseline
# ---------------------------------------------------------------------------
def test_starobinsky_gr_normalizes_to_planck():
    """The default Starobinsky shape with V₀=1 has P_S of O(0.01); after
    normalisation P_S should equal Planck's 2.1e-9 to high precision."""
    res = normalize_egb_model(
        "(1 - exp(-sqrt(2/3)*phi))**2", "0",
        N=55.0, phi_range=(0.1, 8.0),
    )
    assert res.valid
    # Within 0.1% of target
    assert math.isclose(res.P_S_after, PLANCK_A_S, rel_tol=1e-3), (
        res.P_S_after, PLANCK_A_S)
    # And ln(10^10 A_s) ≈ 3.044
    assert abs(res.ln_10_10_A_s_after - 3.044) < 0.05


def test_normalize_preserves_slow_roll_invariants():
    """The whole point of V → λV, ξ → ξ/λ is that ε, n_s, r are invariant."""
    res = normalize_egb_model(
        "(1 - exp(-sqrt(2/3)*phi))**2", "0",
        N=55.0, phi_range=(0.1, 8.0),
    )
    assert res.valid
    assert math.isclose(res.n_s_before, res.n_s_after, rel_tol=1e-3), (
        res.n_s_before, res.n_s_after)
    assert math.isclose(res.r_before, res.r_after, rel_tol=2e-2), (
        res.r_before, res.r_after)
    assert math.isclose(res.epsilon_before, res.epsilon_after, rel_tol=2e-2)
    # c_T² is trivially 1 in GR; check it's still 1
    if np.isfinite(res.c_T2_before):
        assert math.isclose(res.c_T2_before, res.c_T2_after, rel_tol=1e-3)


def test_lambda_factor_matches_amplitude_ratio():
    """λ = P_target / P_before by construction."""
    res = normalize_egb_model(
        "(1 - exp(-sqrt(2/3)*phi))**2", "0",
        N=55.0, phi_range=(0.1, 8.0),
    )
    assert res.valid
    expected_lambda = PLANCK_A_S / res.P_S_before
    assert math.isclose(res.lambda_factor, expected_lambda, rel_tol=1e-6), (
        res.lambda_factor, expected_lambda)


# ---------------------------------------------------------------------------
# EGB case
# ---------------------------------------------------------------------------
def test_normalize_egb_quartic_with_pole_coupling():
    """A live-EGB model: V=λφ⁴, ξ=α/(φ²+1). Verify normalisation works
    and δ_1 is preserved (key EGB diagnostic)."""
    res = normalize_egb_model(
        "0.05*phi**4", "0.1/(phi**2 + 1)",
        N=55.0, phi_range=(0.5, 30.0),
    )
    assert res.valid
    # P_S matches target
    assert math.isclose(res.P_S_after, PLANCK_A_S, rel_tol=1e-3)
    # δ_1 should be preserved (slow-roll invariant of the rescaling)
    # Note: the rescaling is exact only at leading order; we allow some drift.
    if abs(res.delta1_before) > 1e-6:
        assert math.isclose(res.delta1_before, res.delta1_after, rel_tol=5e-2)
    # n_s also invariant
    assert math.isclose(res.n_s_before, res.n_s_after, rel_tol=2e-3)


def test_normalized_xi_is_zero_when_xi_was_zero():
    """xi='0' stays exactly '0' after normalisation (no division)."""
    res = normalize_egb_model("phi**2", "0", N=60.0, phi_range=(-30, 30))
    assert res.valid
    assert res.xi_expr_normalized == "0"


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------
def test_invalid_model_returns_invalid_result():
    """A constant V has no inflationary trajectory; normalisation should
    gracefully report invalid rather than crash."""
    res = normalize_egb_model("1.0", "0", N=55.0)
    assert not res.valid
    assert any("observables" in n.lower() or "p_s" in n.lower()
               for n in res.notes)
