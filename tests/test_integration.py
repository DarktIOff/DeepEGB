"""
End-to-end integration test.

Exercises every layer of DeepEGB on the same EGB inflation model and checks
the layers are consistent with each other:

  1. EGBModel + slow-roll φ_end finder
  2. Full background ODE (egb_background.integrate_with_pivot)
  3. Slow-roll closed-form perturbation kernel (compute_observables_full)
  4. Mukhanov-Sasaki tensor modes (tensor_power_spectrum)
  5. Mukhanov-Sasaki scalar modes (scalar_power_spectrum)
  6. Relic GW spectrum + transfer function (relic_gw_spectrum)
  7. Detector catalogue + sensitivity overlay (DETECTORS, sensitivity_at)
  8. χ² losses (chi2_full, chi2_relic_gw)
  9. Analyze + plot tools
 10. Search-config validation

The RAG and Agno agent layers depend on optional heavy deps (sentence-
transformers, agno+OpenAI client) and live in their own tests; this file
keeps coverage on the physics+tools layer that ships in the default install.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deepegb.physics import (
    DETECTORS,
    EGBModel,
    chi2_full,
    chi2_relic_gw,
    compute_observables_full,
    detectors_in_band,
    end_of_inflation,
    integrate_with_pivot,
    k_pivot_from_traj,
    relic_gw_spectrum,
    scalar_power_spectrum,
    sensitivity_at,
    tensor_power_spectrum,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _model_starobinsky_GR():
    return EGBModel(
        V=lambda p: 1e-10 * (1.0 - np.exp(-np.sqrt(2 / 3) * p)) ** 2,
        xi=lambda p: 0.0 * p,
        name="Starobinsky (GR)",
    )


def _model_egb_strong():
    """A quartic V × pole-coupling ξ chosen so δ₁ ≈ 0.01 at N=55, big enough
    that the EGB sector visibly modifies c_T² but small enough to keep
    the slow-roll trajectory well-defined."""
    return EGBModel(
        V=lambda p: 0.05 * p ** 4,
        xi=lambda p: 0.1 / (p * p + 1.0),
        name="quartic V × weak pole-coupling ξ",
    )


# ---------------------------------------------------------------------------
# Layer-by-layer integration check
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gr_traj():
    model = _model_starobinsky_GR()
    return model, integrate_with_pivot(model, N_pivot=55.0, phi_range=(0.1, 8.0))


def test_layer_1_phi_end(gr_traj):
    model, _ = gr_traj
    phi_end = end_of_inflation(model, phi_range=(0.1, 8.0))
    assert phi_end is not None
    assert 0.5 < phi_end < 1.5    # textbook Starobinsky φ_end ≈ 0.94


def test_layer_2_full_background(gr_traj):
    _, traj = gr_traj
    assert traj is not None
    # Inflation should last at least 55 e-folds before ε=1
    assert traj.N_end > 55.0
    # ε must approach 1 monotonically toward the end
    assert traj.eps1[-1] > 0.95


def test_layer_3_closed_form_observables(gr_traj):
    model, _ = gr_traj
    obs = compute_observables_full(model, N_pivot=55.0, phi_range=(0.1, 8.0))
    assert obs.is_valid
    # Starobinsky: n_s ≈ 0.964, r ≈ 0.004, c_T² = 1 in GR
    assert abs(obs.n_s - (1 - 2 / 55)) < 0.005
    assert abs(obs.r - 12 / 55**2) < 0.005
    assert abs(obs.c_T2 - 1.0) < 1e-9
    # In GR limit c_S² = 1 exactly (no GB correction):
    assert abs(obs.c_S2 - 1.0) < 1e-6


def test_layer_4_5_MS_modes_match_closed_form(gr_traj):
    model, traj = gr_traj
    k_pivot = k_pivot_from_traj(traj, N_pivot=55.0)
    obs = compute_observables_full(model, N_pivot=55.0, phi_range=(0.1, 8.0))
    P_T_MS, _ = tensor_power_spectrum(model, np.array([k_pivot]), traj=traj)
    P_S_MS, _ = scalar_power_spectrum(model, np.array([k_pivot]), traj=traj)
    # MS amplitudes should match closed-form within a few percent
    assert abs(P_T_MS[0] / obs.P_T - 1.0) < 0.05
    assert abs(P_S_MS[0] / obs.P_S - 1.0) < 0.05


def test_layer_6_relic_gw_spectrum_in_RD(gr_traj):
    model, traj = gr_traj
    k_pivot = k_pivot_from_traj(traj, N_pivot=55.0)
    k_arr = k_pivot * np.logspace(2, 4, 5)        # well into RD
    spec = relic_gw_spectrum(model, k_arr, traj=traj, T_reh_GeV=1e15)
    assert (spec.Omega_GW_h2 > 0).all()
    assert np.isfinite(spec.Omega_GW_h2).all()
    # Order-of-magnitude check for Starobinsky in RD:
    # P_T ≈ 6e-12 × Ω_R/24 × g_*-correction (~0.4) × 0.5 ⇒ ~2e-18
    assert (spec.Omega_GW_h2 < 1e-15).all()
    assert (spec.Omega_GW_h2 > 1e-22).all()


def test_layer_7_detector_catalogue_covers_all_bands():
    # Make sure we have detectors for every relic-GW band of interest:
    bands = {
        "CMB": (1e-18, 1e-15),
        "PTA": (1e-9, 1e-7),
        "LISA": (1e-4, 1e-1),
        "DECIGO": (1e-2, 1e1),
        "ground": (1e1, 1e3),
    }
    for label, (lo, hi) in bands.items():
        ds = detectors_in_band(lo, hi)
        assert len(ds) > 0, f"No detector covers {label} band [{lo}, {hi}] Hz"


def test_layer_7_sensitivity_curve_has_reasonable_values():
    # At LISA's peak frequency the floor should be ~ Ω_GW h² ~ 1e-13
    sens = sensitivity_at(np.array([3e-3]), probes=("space",))
    assert "LISA" in sens
    # near peak, value should be ~ floor
    assert 1e-14 < sens["LISA"][0] < 1e-12


def test_layer_8_chi2_full_perfect_fit_low(gr_traj):
    model, _ = gr_traj
    obs = compute_observables_full(model, N_pivot=55.0, phi_range=(0.1, 8.0))
    chi2 = chi2_full(obs, target_ns=obs.n_s, sigma_ns=0.01,
                     target_r=obs.r, sigma_r=0.005)
    assert chi2 < 1.0e-6      # exact match


def test_layer_8_chi2_full_n_s_shift_quadratic(gr_traj):
    model, _ = gr_traj
    obs = compute_observables_full(model, N_pivot=55.0, phi_range=(0.1, 8.0))
    chi2_a = chi2_full(obs, target_ns=obs.n_s + 0.01, sigma_ns=0.01,
                        target_r=obs.r, sigma_r=0.05)
    chi2_b = chi2_full(obs, target_ns=obs.n_s + 0.02, sigma_ns=0.01,
                        target_r=obs.r, sigma_r=0.05)
    # χ² should be ~1 for 1σ shift, ~4 for 2σ shift
    assert 0.8 < chi2_a < 1.5
    assert 3.0 < chi2_b < 5.0


def test_layer_8_chi2_relic_gw_with_LISA_target(gr_traj):
    model, _ = gr_traj
    chi2 = chi2_relic_gw(
        model,
        target_ns=0.965, sigma_ns=0.005,
        target_r=0.0, sigma_r=0.05,
        omega_gw_targets=[(1e-3, 1e-12, 5e-13)],
        N_pivot=55.0, T_reh_GeV=1e15,
    )
    # Starobinsky predicts ~2e-18 at LISA, target is 1e-12 ⇒ huge χ²
    assert chi2 > 1e2
    assert chi2 < 1e7        # but not the invalid_penalty


# ---------------------------------------------------------------------------
# Tool layer
# ---------------------------------------------------------------------------
def test_layer_9_analyze_tool_returns_full_observables():
    from deepegb.analysis import analyze_egb_model
    out = analyze_egb_model(
        "1e-10*(1 - exp(-sqrt(2/3)*phi))**2", "0", N=55.0,
        phi_range=(0.1, 8.0),
    )
    assert out["valid"]
    # Production fields all present:
    for key in ("n_s", "n_T", "r", "alpha_s", "P_S", "P_T", "c_T2", "c_S2",
                "epsilon", "delta1", "phi_N", "phi_end"):
        assert key in out, f"missing {key}"
        assert np.isfinite(out[key])
    # GR consistency relation
    cons = out["consistency_r_minus_8nT"]
    assert cons is not None and abs(cons - 1.0) < 0.05


def test_layer_9_plot_tool_writes_diagnostic(tmp_path):
    from deepegb.analysis import plot_egb_model
    p = plot_egb_model(
        "1e-10*(1 - exp(-sqrt(2/3)*phi))**2", "0",
        N=55.0, out_path=str(tmp_path / "diag.png"),
        phi_range=(0.1, 8.0),
    )
    assert Path(p).exists()
    assert Path(p).stat().st_size > 1000


def test_layer_9_relic_gw_plot_with_detector_overlay(tmp_path):
    from deepegb.analysis import plot_relic_gw_spectrum
    p = plot_relic_gw_spectrum(
        "1e-10*(1 - exp(-sqrt(2/3)*phi))**2", "0",
        N=55.0, n_decades=10.0, n_k=12,
        T_reh_GeV=1e15,
        out_path=str(tmp_path / "relic.png"),
        phi_range=(0.1, 8.0),
    )
    assert Path(p).exists()
    # Should be substantial since we draw 17 detector curves
    assert Path(p).stat().st_size > 5000


# ---------------------------------------------------------------------------
# Search config integration
# ---------------------------------------------------------------------------
def test_layer_10_search_config_validates():
    from deepegb.search import SearchConfig
    cfg = SearchConfig(
        target_ns=0.965, target_r=0.0,
        omega_gw_targets=((1e-3, 1e-12, 5e-13),),
        loss_kind="production_gw",
    )
    assert cfg.loss_kind == "production_gw"
    assert cfg.omega_gw_targets[0][0] == 1e-3
    # to_dict round-trips:
    d = cfg.to_dict()
    assert d["loss_kind"] == "production_gw"


def test_layer_10_no_legacy_loss_kind():
    """The leading-order toy must be unreachable from the public API."""
    from deepegb.search import SearchConfig
    cfg = SearchConfig(target_ns=0.965, target_r=0.0,
                       loss_kind="leading_order")  # not a production option
    # The string is set, but the loss function will treat it as the
    # default production kernel (not crash). Ensure that's the behaviour.
    from deepegb.search.pysr_search import chi2_for_expressions
    chi2 = chi2_for_expressions("phi**2", "0", cfg)
    assert np.isfinite(chi2)


# ---------------------------------------------------------------------------
# EGB modifies observables (sanity that the EGB sector is "live")
# ---------------------------------------------------------------------------
def test_EGB_correction_changes_observables():
    gr = compute_observables_full(_model_starobinsky_GR(), N_pivot=55.0,
                                   phi_range=(0.1, 8.0))
    egb = compute_observables_full(_model_egb_strong(), N_pivot=55.0,
                                    phi_range=(0.5, 30.0))
    assert gr.is_valid and egb.is_valid
    # GR has c_T² = 1 exactly; in the EGB model c_T² should deviate
    assert abs(gr.c_T2 - 1.0) < 1e-9
    assert abs(egb.c_T2 - 1.0) > 1e-4, egb.c_T2
    # GR has δ₁ = 0; EGB has |δ₁| > 0
    assert abs(gr.delta1) < 1e-12
    assert abs(egb.delta1) > 1e-4, egb.delta1
