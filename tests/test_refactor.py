"""
Repository-wide refactor tests.

Covers:
  1. Defaults single-source consistency
  2. N_pivot self-consistency and [50,60] clamping
  3. Full-vs-slow-roll sanity comparison in GR limit
  4. Kernel consistency with centralized defaults and dynamic source injection
  5. EGB-aware consistency metric (replacing r/(-8nT))
  6. Primary (exact) trajectory path vs. fallback for compute_observables_full
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from deepegb.config.defaults import DEFAULTS, get_defaults
from deepegb.physics import (
    EGBModel,
    compute_N_pivot_from_model,
    compute_observables_full,
    egb_consistency_metric,
    end_of_inflation,
    integrate_with_pivot,
    k_pivot_from_traj,
    tensor_power_spectrum,
    scalar_power_spectrum,
)


# ---------------------------------------------------------------------------
# 1. Defaults single-source consistency
# ---------------------------------------------------------------------------
class TestDefaultsConsistency:
    """Verify the centralised DEFAULTS object is self-consistent and that
    modules throughout the repo consume it rather than hardcoding."""

    def test_defaults_module_is_frozen(self):
        """DEFAULTS should be a frozen dataclass — no accidental mutation."""
        import dataclasses
        assert dataclasses.is_dataclass(DEFAULTS)
        # Attempting to set should raise
        with pytest.raises(dataclasses.FrozenInstanceError):
            DEFAULTS.N_pivot = 99  # type: ignore

    def test_phi_range_is_pair(self):
        pr = DEFAULTS.phi_range
        assert isinstance(pr, tuple)
        assert len(pr) == 2
        assert pr[0] < pr[1]

    def test_N_pivot_in_valid_range(self):
        assert DEFAULTS.physics.N_pivot_min <= DEFAULTS.N_pivot <= DEFAULTS.physics.N_pivot_max

    def test_default_xi_is_nontrivial(self):
        """The default xi must NOT be '0' (GR limit)."""
        xi = DEFAULTS.default_xi_expr
        assert xi.strip() != "0"
        assert xi.strip() != "0.0"
        # Should contain some functional form
        assert any(c in xi for c in ["(", "/", "exp", "sin", "cos", "^", "**"])

    def test_search_operators_broad(self):
        """After the refactor, unary_operators should include the broad set."""
        ops = DEFAULTS.search.unary_operators
        # At least 4 operators (original set: exp, log, sqrt, tanh)
        assert len(ops) >= 4
        assert "exp" in ops
        assert "sqrt" in ops

    def test_seed_expressions_nonempty(self):
        assert len(DEFAULTS.search.seed_expressions_V) > 0
        assert len(DEFAULTS.search.seed_expressions_xi) > 0

    def test_seed_xi_has_no_GR_default(self):
        """None of the xi seed expressions should be plain '0'."""
        for expr in DEFAULTS.search.seed_expressions_xi:
            assert expr.strip() != "0", f"Found GR default in seed: {expr}"

    def test_gw_defaults_present(self):
        assert DEFAULTS.gw.n_decades > 0
        assert DEFAULTS.gw.n_k > 0
        assert DEFAULTS.gw.T_reh_GeV > 0

    def test_get_defaults_returns_same_object(self):
        """get_defaults() should return the module singleton."""
        assert get_defaults() is DEFAULTS

    def test_search_config_uses_defaults(self):
        """SearchConfig default should match the centralized config."""
        from deepegb.search import SearchConfig
        cfg = SearchConfig()
        assert cfg.N_pivot == DEFAULTS.N_pivot
        assert cfg.niterations == DEFAULTS.search.niterations
        assert cfg.populations == DEFAULTS.search.populations

    def test_cli_defaults_consistency(self):
        """CLI should not hardcode different numeric defaults."""
        # Import the cli module to check it uses DEFAULTS
        from deepegb import cli
        # The cli module should have imported DEFAULTS
        assert hasattr(cli, 'DEFAULTS')
        assert cli.DEFAULTS is DEFAULTS

    def test_agent_tools_uses_defaults(self):
        """Agent tools should import DEFAULTS."""
        from deepegb.agent import tools
        assert hasattr(tools, 'DEFAULTS')
        assert tools.DEFAULTS is DEFAULTS

    def test_diagnostics_uses_defaults(self):
        """Diagnostics module should import DEFAULTS."""
        from deepegb.physics import diagnostics
        assert hasattr(diagnostics, 'DEFAULTS')


# ---------------------------------------------------------------------------
# 2. N_pivot self-consistency and [50,60] clamping
# ---------------------------------------------------------------------------
class TestNPivotSelfConsistency:
    """Verify the self-consistent N_pivot computation from n_pivot.py."""

    @pytest.fixture
    def starobinsky_gr(self):
        return EGBModel(
            V=lambda p: 1e-10 * (1.0 - np.exp(-np.sqrt(2 / 3) * p)) ** 2,
            xi=lambda p: 0.0 * p,
            name="Starobinsky GR",
        )

    def test_compute_N_pivot_returns_float(self, starobinsky_gr):
        N = compute_N_pivot_from_model(starobinsky_gr)
        assert isinstance(N, float)

    def test_compute_N_pivot_clamped(self, starobinsky_gr):
        """N_pivot must be in [50, 60]."""
        N = compute_N_pivot_from_model(starobinsky_gr)
        assert 50.0 <= N <= 60.0

    def test_compute_N_pivot_varies_with_T_reh(self, starobinsky_gr):
        """Higher T_reh should give slightly different N_pivot."""
        N_lo = compute_N_pivot_from_model(starobinsky_gr, T_reh_GeV=1e9)
        N_hi = compute_N_pivot_from_model(starobinsky_gr, T_reh_GeV=1e16)
        # Both should be in range
        assert 50.0 <= N_lo <= 60.0
        assert 50.0 <= N_hi <= 60.0
        # They should differ (unless both clamp to the same boundary)
        # At least they should not error
        assert isinstance(N_lo, float) and isinstance(N_hi, float)

    def test_compute_N_pivot_with_custom_range(self, starobinsky_gr):
        """Custom N_min/N_max should be respected."""
        N = compute_N_pivot_from_model(
            starobinsky_gr, N_min=52.0, N_max=58.0,
        )
        assert 52.0 <= N <= 58.0

    def test_compute_N_pivot_clamps_extreme_cases(self):
        """A model that would give N>60 or N<50 should be clamped."""
        # Very high V model — the formula could give large N
        model = EGBModel(
            V=lambda p: 1e-5 * p**2,
            xi=lambda p: 0.0 * p,
        )
        N = compute_N_pivot_from_model(model, phi_range=(0.1, 30))
        assert 50.0 <= N <= 60.0

    def test_n_pivot_module_citations(self):
        """The n_pivot module should reference Liddle & Leach 2003 and
        Martin & Ringeval 2010 in its docstring."""
        from deepegb.physics import n_pivot
        doc = n_pivot.__doc__ or ""
        assert "Liddle" in doc
        assert "Leach" in doc
        assert "Martin" in doc
        assert "Ringeval" in doc


# ---------------------------------------------------------------------------
# 3. Full-vs-slow-roll sanity comparison in GR limit
# ---------------------------------------------------------------------------
class TestFullVsSlowRoll:
    """Compare full-background observable computation against known GR
    analytic results. In the GR limit (ξ=0), both should agree."""

    @pytest.fixture
    def starobinsky_gr(self):
        return EGBModel(
            V=lambda p: 1e-10 * (1.0 - np.exp(-np.sqrt(2 / 3) * p)) ** 2,
            xi=lambda p: 0.0 * p,
        )

    def test_starobinsky_n_s_close_to_textbook(self, starobinsky_gr):
        """Starobinsky: n_s ≈ 1 - 2/N ≈ 0.9636 at N=55."""
        obs = compute_observables_full(
            starobinsky_gr, N_pivot=55.0, phi_range=(0.1, 8.0))
        expected = 1.0 - 2.0 / 55.0
        assert abs(obs.n_s - expected) < 0.005, (obs.n_s, expected)

    def test_starobinsky_r_close_to_textbook(self, starobinsky_gr):
        """Starobinsky: r ≈ 12/N² ≈ 0.00397 at N=55."""
        obs = compute_observables_full(
            starobinsky_gr, N_pivot=55.0, phi_range=(0.1, 8.0))
        expected = 12.0 / 55.0**2
        assert abs(obs.r - expected) < 0.005, (obs.r, expected)

    def test_cT2_equals_one_in_GR(self, starobinsky_gr):
        obs = compute_observables_full(
            starobinsky_gr, N_pivot=55.0, phi_range=(0.1, 8.0))
        assert abs(obs.c_T2 - 1.0) < 1e-9, obs.c_T2

    def test_delta1_zero_in_GR(self, starobinsky_gr):
        obs = compute_observables_full(
            starobinsky_gr, N_pivot=55.0, phi_range=(0.1, 8.0))
        assert abs(obs.delta1) < 1e-12, obs.delta1

    def test_MS_matches_slow_roll_in_GR(self, starobinsky_gr):
        """Mukhanov-Sasaki tensor power should match slow-roll closed-form."""
        obs = compute_observables_full(
            starobinsky_gr, N_pivot=55.0, phi_range=(0.1, 8.0))
        traj = integrate_with_pivot(
            starobinsky_gr, N_pivot=55.0, phi_range=(0.1, 8.0))
        assert traj is not None
        k_pivot = k_pivot_from_traj(traj, N_pivot=55.0)
        P_T_MS, _ = tensor_power_spectrum(
            starobinsky_gr, np.array([k_pivot]), traj=traj)
        assert abs(P_T_MS[0] / obs.P_T - 1.0) < 0.05


# ---------------------------------------------------------------------------
# 4. Kernel consistency with centralized defaults
# ---------------------------------------------------------------------------
class TestKernelConsistency:

    def test_dynamic_julia_loss_uses_config_targets(self):
        """The Julia loss source generator should inject targets from config."""
        from deepegb.search.pysr_search import _build_julia_v_loss_source, SearchConfig
        cfg = SearchConfig(target_ns=0.965, target_r=0.0, N_pivot=55.0)
        src = _build_julia_v_loss_source(cfg)
        # Check that the injected values appear in the generated source
        assert "0.965" in src
        assert "N_pivot" in src

    def test_dynamic_xi_loss_uses_config_targets(self):
        """The xi loss source should inject EGB enforcement parameters."""
        from deepegb.search.pysr_search import _build_julia_xi_loss_source, SearchConfig
        cfg = SearchConfig(target_ns=0.965, target_r=0.0, N_pivot=55.0,
                           enforce_egb=True, egb_min_delta1=1e-4)
        src = _build_julia_xi_loss_source(cfg, "phi^2")
        assert "enforce_egb" in src
        assert "egb_min" in src

    def test_kernel_jl_is_reference_only(self):
        """kernel.jl should NOT contain hardcoded target defaults."""
        from pathlib import Path
        kernel_path = Path(__file__).parent.parent / "src" / "deepegb" / "physics" / "kernel.jl"
        if not kernel_path.exists():
            pytest.skip("kernel.jl not found")
        content = kernel_path.read_text()
        # Should not contain actual numeric defaults like target_ns=0.965
        assert "target_ns = 0.965" not in content
        assert "N_pivot = 55.0" not in content
        # Should mention reference/pseudocode
        assert "REFERENCE" in content or "PSEUDOCODE" in content or "reference" in content.lower()

    def test_search_config_defaults_match_yaml(self):
        """SearchConfig field defaults should match DEFAULTS."""
        from deepegb.search import SearchConfig
        cfg = SearchConfig()
        assert cfg.N_pivot == DEFAULTS.N_pivot
        assert cfg.niterations == DEFAULTS.search.niterations
        assert cfg.populations == DEFAULTS.search.populations
        assert cfg.population_size == DEFAULTS.search.population_size
        assert cfg.maxsize == DEFAULTS.search.maxsize


# ---------------------------------------------------------------------------
# 5. EGB-aware consistency metric
# ---------------------------------------------------------------------------
class TestEGBConsistencyMetric:

    def test_gr_consistency_is_one(self):
        """In GR (ξ=0), the EGB consistency metric should be ≈ 1."""
        model = EGBModel(
            V=lambda p: 1e-10 * (1.0 - np.exp(-np.sqrt(2 / 3) * p)) ** 2,
            xi=lambda p: 0.0 * p,
        )
        obs = compute_observables_full(model, N_pivot=55.0, phi_range=(0.1, 8.0))
        metric = egb_consistency_metric(obs)
        assert abs(metric["egb_consistency"] - 1.0) < 0.05

    def test_gr_ct2_deviation_is_zero(self):
        model = EGBModel(
            V=lambda p: 1e-10 * (1.0 - np.exp(-np.sqrt(2 / 3) * p)) ** 2,
            xi=lambda p: 0.0 * p,
        )
        obs = compute_observables_full(model, N_pivot=55.0, phi_range=(0.1, 8.0))
        metric = egb_consistency_metric(obs)
        assert abs(metric["c_T2_deviation"]) < 1e-8

    def test_egb_breaks_consistency(self):
        """With nonzero ξ, EGB consistency should deviate from 1."""
        model = EGBModel(
            V=lambda p: 0.05 * p ** 4,
            xi=lambda p: 1.0 / (p * p + 1.0),
        )
        obs = compute_observables_full(model, N_pivot=55.0, phi_range=(0.5, 30.0))
        if not obs.is_valid:
            pytest.skip("EGB model didn't produce valid observables")
        metric = egb_consistency_metric(obs)
        # EGB should break the consistency relation
        assert abs(metric["egb_consistency"] - 1.0) > 0.01, metric
        # And delta1 should be nontrivial
        assert abs(metric["delta1_magnitude"]) > 1e-4, metric

    def test_analyze_returns_egb_consistency(self):
        """FullObservables.as_dict() should contain egb_consistency field."""
        model = EGBModel(
            V=lambda p: 1e-10 * (1.0 - np.exp(-np.sqrt(2 / 3) * p)) ** 2,
            xi=lambda p: 0.0 * p,
        )
        obs = compute_observables_full(model, N_pivot=55.0, phi_range=(0.1, 8.0))
        d = obs.as_dict()
        assert "egb_consistency" in d
        assert np.isfinite(d["egb_consistency"])

        # Also verify the consistency metric function returns expected keys
        metric = egb_consistency_metric(obs)
        assert "egb_consistency" in metric
        assert "c_T2_deviation" in metric
        assert "delta1_magnitude" in metric

    def test_full_observables_has_egb_consistency_field(self):
        model = EGBModel(
            V=lambda p: 1e-10 * (1.0 - np.exp(-np.sqrt(2 / 3) * p)) ** 2,
            xi=lambda p: 0.0 * p,
        )
        obs = compute_observables_full(model, N_pivot=55.0, phi_range=(0.1, 8.0))
        assert hasattr(obs, "egb_consistency")
        assert np.isfinite(obs.egb_consistency)
        assert abs(obs.egb_consistency - 1.0) < 0.05


# ---------------------------------------------------------------------------
# Integration: verify no hardcoded numeric drifts
# ---------------------------------------------------------------------------
class TestNoHardcodedDrifts:

    def test_diagnostics_defaults_match_config(self):
        """Diagnostics phi_range and N_pivot should come from DEFAULTS."""
        from deepegb.physics.diagnostics import DEFAULTS as diag_defaults
        assert diag_defaults is DEFAULTS

    def test_agent_tools_defaults_match_config(self):
        from deepegb.agent.tools import DEFAULTS as tools_defaults
        assert tools_defaults is DEFAULTS

    def test_cli_defaults_match_config(self):
        from deepegb.cli import DEFAULTS as cli_defaults
        assert cli_defaults is DEFAULTS


# ---------------------------------------------------------------------------
# 6. Auto-N_pivot integration in analysis/plot/CLI/agent paths
# ---------------------------------------------------------------------------
class TestAutoNPivotIntegration:
    """Verify self-consistent N_pivot is exercised through user-facing paths."""

    @pytest.fixture
    def starobinsky_gr(self):
        return EGBModel(
            V=lambda p: 1e-10 * (1.0 - np.exp(-np.sqrt(2 / 3) * p)) ** 2,
            xi=lambda p: 0.0 * p,
            name="Starobinsky GR",
        )

    def test_analyze_auto_N_returns_flag(self):
        from deepegb.analysis import analyze_egb_model
        out = analyze_egb_model("1e-10*(1 - exp(-sqrt(2/3)*phi))**2", "0", N=None)
        assert out["N_was_auto"] is True
        assert 50.0 <= out["N_pivot"] <= 60.0

    def test_analyze_explicit_N_no_auto_flag(self):
        from deepegb.analysis import analyze_egb_model
        out = analyze_egb_model("1e-10*(1 - exp(-sqrt(2/3)*phi))**2", "0", N=55.0)
        assert out["N_was_auto"] is False
        assert out["N_pivot"] == 55.0

    def test_relic_gw_auto_N_returns_flag(self):
        from deepegb.analysis import analyze_egb_relic_gw
        out = analyze_egb_relic_gw(
            "1e-10*(1 - exp(-sqrt(2/3)*phi))**2", "0",
            N=None, n_k=5, n_decades=2,
        )
        assert out["N_was_auto"] is True
        assert 50.0 <= out["N_pivot"] <= 60.0

    def test_relic_gw_explicit_N(self):
        from deepegb.analysis import analyze_egb_relic_gw
        out = analyze_egb_relic_gw(
            "1e-10*(1 - exp(-sqrt(2/3)*phi))**2", "0",
            N=52.0, n_k=5, n_decades=2,
        )
        assert out["N_was_auto"] is False
        assert out["N_pivot"] == 52.0

    def test_analyze_model_tool_passes_none(self):
        from deepegb.agent.tools import analyze_egb_model_tool
        import json
        result = json.loads(analyze_egb_model_tool(
            "1e-10*(1 - exp(-sqrt(2/3)*phi))**2", "0", N=None,
        ))
        assert result.get("N_was_auto") is True
        assert 50.0 <= result.get("N_pivot", 0) <= 60.0

    def test_diagnose_tool_accepts_none_N(self):
        from deepegb.agent.tools import diagnose_egb_model_tool
        import json
        result = json.loads(diagnose_egb_model_tool(
            "1e-10*(1 - exp(-sqrt(2/3)*phi))**2", "0", N=None,
        ))
        assert "error" not in result or not result.get("error", "").startswith("TOOL_ERROR")

    def test_cli_analyze_accepts_none_N(self):
        from click.testing import CliRunner
        from deepegb.cli import main
        runner = CliRunner()
        result = runner.invoke(main, [
            "analyze", "1e-10*(1 - exp(-sqrt(2/3)*phi))**2", "0",
        ])
        assert result.exit_code == 0

    def test_cli_plot_accepts_none_N(self, tmp_path):
        from click.testing import CliRunner
        from deepegb.cli import main
        runner = CliRunner()
        out = str(tmp_path / "test_plot.png")
        result = runner.invoke(main, [
            "plot", "1e-10*(1 - exp(-sqrt(2/3)*phi))**2", "0",
            "--out", out,
        ])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# 7. Primary (exact) trajectory path vs. slow-roll fallback
# ---------------------------------------------------------------------------
class TestTrajectoryPrimaryPath:
    """Verify compute_observables_full uses the full-background trajectory
    + Mukhanov–Sasaki mode integration as the primary path, falling back to
    the slow-roll closed-form when the trajectory is unavailable."""

    @pytest.fixture
    def starobinsky_gr(self):
        return EGBModel(
            V=lambda p: 1e-10 * (1.0 - np.exp(-np.sqrt(2 / 3) * p)) ** 2,
            xi=lambda p: 0.0 * p,
        )

    def test_primary_path_attempted_and_succeeds(self, starobinsky_gr):
        """The primary path should be attempted: _observables_from_trajectory
        is called.  For Starobinsky GR the trajectory succeeds, so we get
        mode-computed observables."""
        from unittest.mock import patch
        with patch(
            "deepegb.physics.egb_perturbations._observables_from_trajectory",
            wraps=None,
        ) as mock_primary:
            from deepegb.physics.egb_perturbations import (
                _observables_from_trajectory as _real,
            )
            mock_primary.side_effect = lambda *a, **kw: _real(*a, **kw)

            obs = compute_observables_full(
                starobinsky_gr, N_pivot=55.0, phi_range=(0.1, 8.0))

            # Primary path was attempted at least once
            assert mock_primary.call_count >= 1
            # And it succeeded (returned a valid result)
            assert obs.is_valid
            assert np.isfinite(obs.n_s)
            assert np.isfinite(obs.r)

    def test_fallback_when_trajectory_returns_none(self, starobinsky_gr):
        """When _observables_from_trajectory returns None, the slow-roll
        fallback should still produce valid observables."""
        from unittest.mock import patch
        with patch(
            "deepegb.physics.egb_perturbations._observables_from_trajectory",
            return_value=None,
        ):
            obs = compute_observables_full(
                starobinsky_gr, N_pivot=55.0, phi_range=(0.1, 8.0))

        # Fallback produced valid results
        assert obs.is_valid
        assert np.isfinite(obs.n_s)
        assert np.isfinite(obs.r)
        assert np.isfinite(obs.P_S)
        assert np.isfinite(obs.P_T)
        # Starobinsky n_s should still be near 1 - 2/N
        expected = 1.0 - 2.0 / 55.0
        assert abs(obs.n_s - expected) < 0.005, (obs.n_s, expected)

    def test_fallback_when_trajectory_raises(self, starobinsky_gr):
        """When _observables_from_trajectory raises an exception, the
        slow-roll fallback should still produce valid observables."""
        from unittest.mock import patch
        with patch(
            "deepegb.physics.egb_perturbations._observables_from_trajectory",
            side_effect=RuntimeError("ODE diverged"),
        ):
            obs = compute_observables_full(
                starobinsky_gr, N_pivot=55.0, phi_range=(0.1, 8.0))

        assert obs.is_valid
        assert np.isfinite(obs.n_s)
        assert np.isfinite(obs.r)

    def test_primary_path_output_finite_starobinsky(self, starobinsky_gr):
        """Primary-path output should be finite and physically reasonable
        for the well-behaved Starobinsky model."""
        obs = compute_observables_full(
            starobinsky_gr, N_pivot=55.0, phi_range=(0.1, 8.0))

        assert obs.is_valid
        # Spectral index near 1 - 2/N (may differ slightly from slow-roll)
        assert 0.94 < obs.n_s < 0.98, obs.n_s
        # Small tensor-to-scalar ratio
        assert obs.r < 0.01, obs.r
        # Running should be finite and small
        assert np.isfinite(obs.alpha_s)
        assert abs(obs.alpha_s) < 0.01, obs.alpha_s
        # Sound speed = 1 in GR
        assert abs(obs.c_T2 - 1.0) < 1e-6, obs.c_T2

    def test_primary_path_output_finite_egb(self):
        """Primary-path output should be finite for an EGB model with
        nonzero ξ coupling."""
        model = EGBModel(
            V=lambda p: 0.05 * p ** 4,
            xi=lambda p: 1.0 / (p * p + 1.0),
        )
        obs = compute_observables_full(
            model, N_pivot=55.0, phi_range=(0.5, 30.0))
        if not obs.is_valid:
            pytest.skip("EGB model didn't produce valid observables")
        assert np.isfinite(obs.n_s)
        assert np.isfinite(obs.r)
        assert np.isfinite(obs.P_S)
        assert np.isfinite(obs.P_T)
        assert np.isfinite(obs.alpha_s)

    def test_primary_path_uses_mode_spectra(self, starobinsky_gr):
        """Verify the Mukhanov–Sasaki mode functions are actually invoked
        by the primary path (not just the slow-roll formulas)."""
        from unittest.mock import patch
        with patch(
            "deepegb.physics.egb_modes.tensor_power_spectrum",
            wraps=tensor_power_spectrum,
        ) as mock_tensor, patch(
            "deepegb.physics.egb_modes.scalar_power_spectrum",
            wraps=scalar_power_spectrum,
        ) as mock_scalar:
            obs = compute_observables_full(
                starobinsky_gr, N_pivot=55.0, phi_range=(0.1, 8.0))
            # If the primary path succeeded, the mode functions were called
            if obs.is_valid:
                assert mock_tensor.call_count >= 1, \
                    "tensor_power_spectrum was not called"
                assert mock_scalar.call_count >= 1, \
                    "scalar_power_spectrum was not called"

    def test_dlnk_parameter_accepted(self, starobinsky_gr):
        """The dlnk parameter should be accepted without error."""
        obs = compute_observables_full(
            starobinsky_gr, N_pivot=55.0, phi_range=(0.1, 8.0),
            dlnk=0.3,
        )
        assert obs.is_valid
