"""Sanity checks for the EGB slow-roll kernel.

These don't need PySR or Agno installed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from deepegb.physics import EGBModel, analyze_model


# ---------------------------------------------------------------------------
# 1. Quadratic chaotic inflation (GR limit, ξ=0)
#    For V = m^2 phi^2 / 2, slow-roll predicts:
#       N(phi) ≈ phi^2 / 4  (in M_pl=1 units)
#       phi_N  = 2 sqrt(N)
#       n_s = 1 - 2/N,    r = 8/N
#    These should match our kernel within ~few %.
# ---------------------------------------------------------------------------
def test_quadratic_GR_limit_nsr():
    m = 1e-5  # arbitrary; observables are scale-free
    model = EGBModel(V=lambda p: 0.5 * m * m * p ** 2,
                     xi=lambda p: 0.0 * p,
                     name="m^2 phi^2 / 2")
    obs = analyze_model(model, N_target=60.0,
                        phi_range=(-30.0, 30.0), n_grid=8001)
    assert obs.is_valid, f"failed: {obs}"
    # Expected n_s = 1 - 2/N, r = 8/N at N=60
    ns_exp = 1.0 - 2.0 / 60.0
    r_exp = 8.0 / 60.0
    assert abs(obs.n_s - ns_exp) < 0.01, (obs.n_s, ns_exp)
    assert abs(obs.r - r_exp) < 0.03, (obs.r, r_exp)


# ---------------------------------------------------------------------------
# 2. With ξ ≡ 0 the kernel must reduce to standard GR slow-roll.
#    Specifically Q = V_,phi exactly.
# ---------------------------------------------------------------------------
def test_Q_reduces_to_Vphi_when_xi_zero():
    model = EGBModel(V=lambda p: (1.0 - np.exp(-np.sqrt(2/3) * p)) ** 2,
                     xi=lambda p: 0.0 * p)
    phis = np.linspace(0.5, 6.0, 50)
    for p in phis:
        Q = model.Q(p)
        Vp = model.V_phi(p)
        assert math.isclose(Q, Vp, rel_tol=1e-6, abs_tol=1e-12), (p, Q, Vp)


# ---------------------------------------------------------------------------
# 3. Starobinsky (ξ=0): V = (1 - exp(-sqrt(2/3) phi))^2.
#    Expected: n_s ≈ 1 - 2/N, r ≈ 12/N^2 (small).
# ---------------------------------------------------------------------------
def test_starobinsky_GR_limit():
    model = EGBModel(V=lambda p: (1.0 - np.exp(-np.sqrt(2/3) * p)) ** 2,
                     xi=lambda p: 0.0 * p)
    obs = analyze_model(model, N_target=55.0, phi_range=(0.1, 8.0), n_grid=4001)
    assert obs.is_valid
    # Starobinsky: n_s = 1 - 2/N
    assert abs(obs.n_s - (1 - 2/55)) < 0.005
    # r ~ 12/N^2 = 12/3025 ≈ 0.004; we accept a band.
    assert obs.r < 0.02, obs.r


# ---------------------------------------------------------------------------
# 4. Adding ξ(φ) ≠ 0 should *change* the predictions in a smooth, finite way.
# ---------------------------------------------------------------------------
def test_xi_nonzero_shifts_observables():
    V = lambda p: 0.5 * (p ** 2)
    obs0 = analyze_model(
        EGBModel(V=V, xi=lambda p: 0.0 * p),
        N_target=55.0, phi_range=(-30.0, 30.0), n_grid=4001,
    )
    obs1 = analyze_model(
        EGBModel(V=V, xi=lambda p: 0.05 * np.exp(-0.4 * p)),
        N_target=55.0, phi_range=(-30.0, 30.0), n_grid=4001,
    )
    assert obs0.is_valid and obs1.is_valid
    assert obs0.n_s != obs1.n_s, "ξ(φ) had no effect on n_s"
