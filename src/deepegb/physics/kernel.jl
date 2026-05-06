# EGB slow-roll χ² loss — REFERENCE IMPLEMENTATION ONLY.
#
# This file documents the physics logic line-by-line. The ACTUAL Julia loss
# passed to PySR is generated dynamically by
# `search/pysr_search.py::_build_julia_v_loss_source()` and
# `search/pysr_search.py::_build_julia_xi_loss_source()`, which inject
# runtime targets (n_s, r, N_pivot, egb_min_delta1, enforce_egb) from the
# centralised config (`deepegb.config.defaults.DEFAULTS`).
#
# DO NOT add hardcoded numeric targets to this file. All defaults come from
# configs/default.yaml via the Python config layer.
#
# CONVENTIONS (M_pl = 1)
# ----------------------
# In the GR limit ξ ≡ 0, the slow-roll quantities reduce to
#   Q(φ)     = V'(φ)
#   ε(φ)     = (1/2)(V'/V)²
#   N(φ)     = ∫_{φ_end}^{φ} V/V' dψ
#   ε(φ_end) = 1
# Observables at horizon crossing N e-folds before end:
#   n_s = 1 − 6 ε + 2 η,   η = V''/V
#   r   = 16 ε
#
# With nonzero ξ(φ), pass 2 uses the full EGB force:
#   Q(φ) = V' + (4/3) V² ξ'
#   ε(φ) = (1/2) Q V' / V²
# and adds a GR-limit penalty when |δ₁| = |−(4/3) ξ' Q| is small.
#
# All derivatives are computed by central differences on the φ-grid that
# we co-opt as `dataset.X`. PySR's `eval_tree_array` evaluates V(φ) at all
# grid points in one shot.
#
# SOFT PENALTIES
# --------------
# A hard "1e6 if anything failed" floor robs the genetic search of
# gradient. We return graded penalties (1e3..2e3) for partly-broken
# candidates so PySR can climb back into the valid region.

# ---------------------------------------------------------------------------
# The dynamic Julia source generators in pysr_search.py produce functions
# equivalent to the pseudocode below. Target values are injected at runtime
# from the centralised config, NOT hardcoded here.
# ---------------------------------------------------------------------------

# PSEUDOCODE (not executed — see pysr_search.py for the actual source):
#
# function egb_chi2_v_only(V_grid, Vp_grid, Vpp_grid, phi_grid;
#                          target_ns, sigma_ns, target_r, sigma_r, N_pivot)
#     # 1. V positivity — soft penalty proportional to badness.
#     # 2. ε(φ) = (1/2)(V'/V)² (GR limit with ξ=0).
#     # 3. φ_end via first sign-change of (ε − 1).
#     # 4. N(φ) via cumulative trapezoid of V/V'.
#     # 5. φ_pivot on slow-roll side (ε < 1, N = N_pivot).
#     # 6. Observables: n_s = 1 − 6ε + 2η, r = 16ε.
#     # 7. χ² = ((n_s - target_ns)/sigma_ns)² + ((r - target_r)/sigma_r)²
# end
#
# function egb_chi2_with_xi(V_grid, xi_grid, Vp_grid, xip_grid, phi_grid;
#                           target_ns, sigma_ns, target_r, sigma_r,
#                           N_pivot, enforce_egb, egb_min)
#     # Full EGB: Q = V' + (4/3) V² ξ', ε = (1/2) Q V'/V²
#     # GR-limit penalty: δ₁ = -(4/3) ξ' Q; penalise if |δ₁| < egb_min
# end
