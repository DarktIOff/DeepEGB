"""Validation of the analytic N3LO observable engine (egb_n3lo.py).

Pins the Green's-function analytic path against:
  (i)   the GR limit (Starobinsky, m²φ²) with known predictions;
  (ii)  the Mukhanov–Sasaki mode integrator (both sectors, GR and EGB);
  (iii) the exact background flow identity φ̇²/H² = 2ε₁ − δ₁ − δ₁ε₁ + δ₁δ₂
        (regression test for the Ḣ-equation coefficient fix);
  (iv)  exact sound speeds (c_S², c_T² → 1 in GR; finite in EGB even for
        steep ξ(φ), where the legacy approximation diverged).
"""
from __future__ import annotations

import numpy as np
import pytest

from deepegb.physics import EGBModel, integrate_with_pivot
from deepegb.physics.egb_n3lo import (
    _analytic_grids,
    compute_observables_n3lo,
)
from deepegb.physics.egb_perturbations import (
    _observables_from_trajectory,
    compute_observables_full,
)

V0 = 1.0e-10


@pytest.fixture(scope="module")
def gr_starobinsky():
    return EGBModel(
        V=lambda p: V0 * (1.0 - np.exp(-np.sqrt(2 / 3) * p)) ** 2,
        xi=lambda p: 0.0 * p, name="staro-GR")


@pytest.fixture(scope="module")
def egb_quad_strong():
    # δ₁ ~ 1e-3 at the pivot: GB effects at the per-mille level.
    return EGBModel(V=lambda p: 0.5e-10 * p ** 2,
                    xi=lambda p: 1.0e9 / (p ** 2 + 1.0), name="quad-EGB")


@pytest.fixture(scope="module")
def obs_staro(gr_starobinsky):
    return compute_observables_n3lo(gr_starobinsky, N_pivot=55.0,
                                    phi_range=(0.1, 8.0))


@pytest.fixture(scope="module")
def obs_quad(egb_quad_strong):
    return compute_observables_n3lo(egb_quad_strong, N_pivot=55.0,
                                    phi_range=(-15.0, 15.0))


# ---------------------------------------------------------------------------
# (i) GR limits
# ---------------------------------------------------------------------------
def test_gr_starobinsky_n_s(obs_staro):
    assert obs_staro is not None
    # 1 − 2/N with N_eff ≈ 55–57; generous band around the textbook value
    assert 0.962 < obs_staro.n_s < 0.967


def test_gr_starobinsky_r(obs_staro):
    # 12/N² ≈ 0.0035–0.0040
    assert 0.0030 < obs_staro.r < 0.0042


def test_gr_consistency_relation(obs_staro):
    # r = −8 n_T up to the genuine NLO correction ≈ C ε₂ ≈ −0.73·0.037
    assert obs_staro.egb_consistency == pytest.approx(1.0, abs=0.06)


def test_gr_sound_speeds_unity(obs_staro):
    assert obs_staro.c_S2 == pytest.approx(1.0, abs=1e-10)
    assert obs_staro.c_T2 == pytest.approx(1.0, abs=1e-10)


def test_gr_alpha_s_third_order(obs_staro):
    # α_s ≈ −2ε₁ε₂ − ε₂ε₃ ≈ −2/N² at this pivot (≈ −6.3e-4)
    assert -8.0e-4 < obs_staro.alpha_s < -4.5e-4


# ---------------------------------------------------------------------------
# (ii) cross-validation against Mukhanov–Sasaki mode integration
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("which", ["staro", "quad"])
def test_n3lo_matches_MS(which, gr_starobinsky, egb_quad_strong,
                          obs_staro, obs_quad):
    model, rng, obs_a = {
        "staro": (gr_starobinsky, (0.1, 8.0), obs_staro),
        "quad": (egb_quad_strong, (-15.0, 15.0), obs_quad),
    }[which]
    assert obs_a is not None
    obs_ms = _observables_from_trajectory(model, 55.0, rng, 0.5)
    assert obs_ms is not None
    # Amplitudes: MS integrator accuracy is ~0.1%
    assert obs_a.P_S == pytest.approx(obs_ms.P_S, rel=3e-3)
    assert obs_a.P_T == pytest.approx(obs_ms.P_T, rel=3e-3)
    assert obs_a.r == pytest.approx(obs_ms.r, rel=5e-3)
    # Tilts: absolute agreement well below ACT DR6 σ(n_s) = 0.0034
    assert obs_a.n_s == pytest.approx(obs_ms.n_s, abs=5e-4)
    assert obs_a.n_T == pytest.approx(obs_ms.n_T, abs=5e-4)


def test_pipeline_default_uses_n3lo(gr_starobinsky, obs_staro):
    obs = compute_observables_full(gr_starobinsky, N_pivot=55.0,
                                   phi_range=(0.1, 8.0))
    assert obs.is_valid
    assert obs.n_s == pytest.approx(obs_staro.n_s, abs=1e-12)


# ---------------------------------------------------------------------------
# (iii) exact background flow identity (Ḣ-equation regression)
# ---------------------------------------------------------------------------
def test_flow_identity_phidot(egb_quad_strong):
    traj = integrate_with_pivot(egb_quad_strong, N_pivot=55.0,
                                phi_range=(-15.0, 15.0))
    assert traj is not None
    ag = _analytic_grids(egb_quad_strong, traj)
    i = int(np.argmin(np.abs(traj.N - (traj.N_end - 55.0))))
    H, e1, d1 = ag["H"][i], ag["eps1"][i], ag["delta1"][i]
    d2 = ag["ddelta1_dN"][i] / d1
    lhs = (ag["phidot"][i] / H) ** 2
    rhs = 2 * e1 - d1 - d1 * e1 + d1 * d2
    assert abs(d1) > 1e-4          # the model genuinely exercises the GB term
    assert lhs == pytest.approx(rhs, rel=1e-10)


# ---------------------------------------------------------------------------
# (iv) sound speeds in EGB
# ---------------------------------------------------------------------------
def test_egb_sound_speeds_finite_and_near_unity(obs_quad):
    assert obs_quad is not None
    # deviations are O(δ₁) (tensor) and O(δ₁²/ε₁, δ₁ε₁) (scalar)
    assert abs(obs_quad.c_T2 - 1.0) < 0.05
    assert abs(obs_quad.c_S2 - 1.0) < 0.05
    assert obs_quad.c_S2 > 0 and obs_quad.c_T2 > 0


def test_exact_cS2_steep_xi_regression():
    """Steep ξ(φ) with tiny δ₁: the legacy c_S² formula gave ≈ −12 here;
    the exact expression must stay at 1 − O(δ₁²)."""
    from deepegb.physics import compute_c_S2
    m = EGBModel(V=lambda p: V0 * (1.0 - np.exp(-np.sqrt(2 / 3) * p)) ** 2,
                 xi=lambda p: 1e6 * np.exp(-p))
    obs = compute_observables_n3lo(m, N_pivot=55.0, phi_range=(0.1, 8.0))
    assert obs is not None
    assert obs.c_S2 == pytest.approx(1.0, abs=1e-6)
    # slow-roll reporter path as well
    c = compute_c_S2(m, obs.phi_N)
    assert np.isfinite(c) and abs(c - 1.0) < 1e-4


def test_egb_breaks_consistency(obs_quad):
    """δ₁ ≠ 0 must shift r/(−8 n_T) away from the GR-like value by O(δ₁/ε₁)."""
    assert obs_quad is not None
    assert abs(obs_quad.delta1) > 1e-4
    assert np.isfinite(obs_quad.egb_consistency)
