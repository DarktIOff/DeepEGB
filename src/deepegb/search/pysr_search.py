"""
Joint symbolic regression for V(φ) and ξ(φ) in EGB inflation.

Strategy: **two-pass joint search**.
  a. SR for V(φ) with ξ ≡ 0 (Einstein-frame baseline).
  b. With each top-K V candidate frozen, SR for ξ(φ) that improves the χ²
     further. The winner is the (V, ξ) pair with lowest combined loss.

The χ² is computed by `chi2_full` (slow-roll closed-form) or
`chi2_relic_gw` (full Mukhanov-Sasaki + Ω_GW transfer). Both come from
`physics/egb_perturbations.py`. The legacy leading-order kernel
(`r = 16ε`, etc.) has been removed; production runs only.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import sympy as sp

from ..physics import (
    EGBModel,
    chi2_full,
    chi2_relic_gw,
    compute_observables_full,
)

try:
    from pysr import PySRRegressor
except ImportError:  # PySR may not be installed in lightweight environments
    PySRRegressor = None  # type: ignore
except Exception:  # Julia init errors, network issues, etc.
    PySRRegressor = None  # type: ignore


# ---------------------------------------------------------------------------
# Sympy → Julia source conversion
# ---------------------------------------------------------------------------
def _sympy_to_julia(expr_str: str) -> str:
    """Convert a Sympy expression string in `phi` (or `x0`) to a scalar Julia expr.

    The conversion goes through `sympy.julia_code` for correctness, then
    a small post-pass strips the vectorisation (`.*`, `./`, `.^`) so the
    output is a pure scalar expression that can be wrapped in `@inline f(p) = ...`.
    Fallback: regex `**` → `^`.

    Accepts both `phi` and `x0` as the field variable — PySR returns
    hall-of-fame expressions using `x0` (its internal feature name).
    """
    import re

    import sympy as sp
    phi = sp.Symbol("phi", real=True)
    try:
        expr = sp.sympify(expr_str, locals={"phi": phi, "x0": phi})
        from sympy.printing.julia import julia_code
        out = julia_code(expr)
        # Strip dot-broadcasts (we want scalar Julia, not vectorised)
        out = re.sub(r"\.([\^\*/+\-])", r"\1", out)
        return out
    except Exception:
        s = re.sub(r"\*\*", "^", str(expr_str))
        # Fallback: also replace x0 with phi in raw text
        s = re.sub(r"\bx0\b", "phi", s)
        return s


# ---------------------------------------------------------------------------
# Julia physics-χ² loss source generator (see physics/kernel.jl for the
# documented reference).
# ---------------------------------------------------------------------------
def _build_julia_v_loss_source(
    cfg: "SearchConfig",
    *,
    n_grid: int = 1001,
) -> str:
    """Generate the Julia source for PySR's `loss_function=` argument.

    Targets are substituted from the SearchConfig into a `let` block that
    captures them as locals. PySR compiles the string once at construction
    time; per-evaluation cost is just the closure call.

    The slow-roll-with-ξ=0 derivation is identical to
    `physics/kernel.jl::egb_chi2_v_only`; we inline it here because PySR's
    `loss_function` doesn't easily import external modules.
    """
    return f"""
function deepegb_v_loss(tree, dataset::Dataset{{T,L}}, options) where {{T,L}}
    target_ns = T({cfg.target_ns})
    sigma_ns  = T({cfg.sigma_ns})
    target_r  = T({cfg.target_r})
    sigma_r   = T({cfg.sigma_r})
    N_pivot   = T({cfg.N_pivot})

    n = size(dataset.X, 2)
    if n < 4
        return L(1e6)
    end
    phi = view(dataset.X, 1, :)
    dphi = phi[2] - phi[1]

    # Evaluate V(φ) on the grid via PySR's tree evaluator.
    V_vec, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(1e6)
    end

    # Soft V-positivity penalty.
    n_bad = 0
    @inbounds for i in 1:n
        if !isfinite(V_vec[i]) || V_vec[i] <= 0
            n_bad += 1
        end
    end
    if n_bad > 0
        frac = T(n_bad) / T(n)
        return L(1.0e3 * (1.0 + 10.0 * frac))
    end

    # Central-difference V', V''.
    Vp = similar(V_vec)
    Vpp = similar(V_vec)
    @inbounds Vp[1] = (V_vec[2] - V_vec[1]) / dphi
    @inbounds Vp[end] = (V_vec[end] - V_vec[end-1]) / dphi
    @inbounds Vpp[1] = (V_vec[3] - 2*V_vec[2] + V_vec[1]) / (dphi*dphi)
    @inbounds Vpp[end] = (V_vec[end] - 2*V_vec[end-1] + V_vec[end-2]) / (dphi*dphi)
    @inbounds for i in 2:n-1
        Vp[i] = (V_vec[i+1] - V_vec[i-1]) / (2*dphi)
        Vpp[i] = (V_vec[i+1] - 2*V_vec[i] + V_vec[i-1]) / (dphi*dphi)
    end

    # ε(φ) = (1/2)(V'/V)².
    eps_arr = similar(V_vec)
    @inbounds for i in 1:n
        eps_arr[i] = T(0.5) * (Vp[i] / V_vec[i])^2
    end

    # φ_end: first sign-change of (ε − 1).
    end_idx = 0
    phi_end = T(0.0)
    @inbounds for i in 1:n-1
        a = eps_arr[i] - T(1.0)
        b = eps_arr[i+1] - T(1.0)
        if isfinite(a) && isfinite(b) && a * b < 0
            phi_end = phi[i] - a * dphi / (b - a)
            end_idx = i
            break
        end
    end
    if end_idx == 0
        eps_found = false
        eps_min = T(0)
        @inbounds for i in 1:n
            val = eps_arr[i]
            if isfinite(val)
                if !eps_found || val < eps_min
                    eps_min = val
                    eps_found = true
                end
            end
        end
        if !eps_found
            return L(2.0e3)
        end
        # Graded penalty: smaller |ε−1| ⇒ closer to having an end-of-inflation.
        return L(2.0e3 * (1.0 + abs(log10(max(abs(eps_min - T(1.0)), T(1e-30))))))
    end

    # N(φ) cumulative trapezoid of V/V' (= V/Q with ξ=0).
    integrand = V_vec ./ Vp
    cum_int = zeros(T, n)
    @inbounds for i in 2:n
        cum_int[i] = cum_int[i-1] + T(0.5) * (integrand[i] + integrand[i-1]) * dphi
    end
    cum_at_end = cum_int[end_idx] +
        (cum_int[end_idx+1] - cum_int[end_idx]) * (phi_end - phi[end_idx]) / dphi
    N_grid = cum_int .- cum_at_end

    # φ_pivot (ε < 1, N = N_pivot).
    pivot_idx = 0
    @inbounds for i in 1:n-1
        a = N_grid[i] - N_pivot
        b = N_grid[i+1] - N_pivot
        if a * b < 0 && eps_arr[i] < T(1.0) && eps_arr[i+1] < T(1.0)
            pivot_idx = i
            break
        end
    end
    if pivot_idx == 0
        finite_found = false
        N_max = T(0)
        @inbounds for i in 1:n
            val = N_grid[i]
            if isfinite(val)
                if !finite_found || val > N_max
                    N_max = val
                    finite_found = true
                end
            end
        end
        if !finite_found
            return L(1.5e3 * 2.0)
        end
        deficit = max(T(0.0), N_pivot - N_max)
        return L(1.5e3 * (1.0 + deficit / N_pivot))
    end
    if pivot_idx == 1 || pivot_idx == n
        return L(1.5e3)
    end

    eps_pivot = eps_arr[pivot_idx]
    eta_pivot = Vpp[pivot_idx] / V_vec[pivot_idx]

    # Slow-roll observables.
    n_s_pred = T(1.0) - T(6.0) * eps_pivot + T(2.0) * eta_pivot
    r_pred   = T(16.0) * eps_pivot

    chi2 = ((n_s_pred - target_ns) / sigma_ns)^2 +
           ((r_pred   - target_r ) / sigma_r )^2

    return L(chi2)
end
"""


def _build_julia_v_loss_with_xi_source(cfg: "SearchConfig", xi_expr: str) -> str:
    """Generate Julia loss source for the V search with ξ HARDCODED.

    Used in coordinate-descent rounds: we have a candidate ξ and want to
    refine V against it. Mirror of `_build_julia_xi_loss_source` with
    the roles swapped — the SR tree is V, and ξ is inlined.
    """
    xi_julia = _sympy_to_julia(xi_expr)
    enforce_egb = "true" if cfg.enforce_egb else "false"
    return f"""
function deepegb_v_loss_with_xi(tree, dataset::Dataset{{T,L}}, options) where {{T,L}}
    target_ns = T({cfg.target_ns})
    sigma_ns  = T({cfg.sigma_ns})
    target_r  = T({cfg.target_r})
    sigma_r   = T({cfg.sigma_r})
    N_pivot   = T({cfg.N_pivot})
    egb_min   = T({cfg.egb_min_delta1})
    enforce_egb = {enforce_egb}

    n = size(dataset.X, 2)
    if n < 4
        return L(1e6)
    end
    phi = view(dataset.X, 1, :)
    dphi = phi[2] - phi[1]

    # ξ(φ) hardcoded from previous round.
    @inline xi_user(phi) = {xi_julia}
    xi_vec = xi_user.(phi)

    # Evaluate V(φ) on grid via PySR.
    V_vec, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(1e6)
    end

    # V positivity: graded penalty.
    n_bad = 0
    @inbounds for i in 1:n
        if !isfinite(V_vec[i]) || V_vec[i] <= 0
            n_bad += 1
        end
    end
    if n_bad > 0
        frac = T(n_bad) / T(n)
        return L(1.0e3 * (1.0 + 10.0 * frac))
    end

    # Central diffs.
    Vp = similar(V_vec); Vpp = similar(V_vec); xip = similar(V_vec)
    @inbounds Vp[1] = (V_vec[2] - V_vec[1]) / dphi
    @inbounds Vp[end] = (V_vec[end] - V_vec[end-1]) / dphi
    @inbounds Vpp[1] = (V_vec[3] - 2*V_vec[2] + V_vec[1]) / (dphi*dphi)
    @inbounds Vpp[end] = (V_vec[end] - 2*V_vec[end-1] + V_vec[end-2]) / (dphi*dphi)
    @inbounds xip[1] = (xi_vec[2] - xi_vec[1]) / dphi
    @inbounds xip[end] = (xi_vec[end] - xi_vec[end-1]) / dphi
    @inbounds for i in 2:n-1
        Vp[i]  = (V_vec[i+1] - V_vec[i-1]) / (2*dphi)
        Vpp[i] = (V_vec[i+1] - 2*V_vec[i] + V_vec[i-1]) / (dphi*dphi)
        xip[i] = (xi_vec[i+1] - xi_vec[i-1]) / (2*dphi)
    end

    # Q, ε.
    Q = similar(V_vec); eps_arr = similar(V_vec)
    @inbounds for i in 1:n
        Q[i] = Vp[i] + T(4/3) * V_vec[i]^2 * xip[i]
        eps_arr[i] = T(0.5) * Q[i] * Vp[i] / (V_vec[i]^2)
    end

    # φ_end.
    end_idx = 0
    phi_end = T(0)
    @inbounds for i in 1:n-1
        a = eps_arr[i] - T(1)
        b = eps_arr[i+1] - T(1)
        if isfinite(a) && isfinite(b) && a*b < 0
            phi_end = phi[i] - a*dphi/(b - a)
            end_idx = i
            break
        end
    end
    if end_idx == 0
        eps_found = false
        eps_min = T(0)
        @inbounds for i in 1:n
            val = eps_arr[i]
            if isfinite(val)
                if !eps_found || val < eps_min
                    eps_min = val
                    eps_found = true
                end
            end
        end
        if !eps_found
            return L(2.0e3)
        end
        return L(2.0e3 * (1.0 + abs(log10(max(abs(eps_min - T(1)), T(1e-30))))))
    end

    # N(φ).
    integrand = V_vec ./ Q
    cum_int = zeros(T, n)
    @inbounds for i in 2:n
        cum_int[i] = cum_int[i-1] + T(0.5)*(integrand[i] + integrand[i-1])*dphi
    end
    cum_at_end = cum_int[end_idx] +
        (cum_int[end_idx+1] - cum_int[end_idx]) * (phi_end - phi[end_idx]) / dphi
    N_grid = cum_int .- cum_at_end

    # φ_pivot.
    pivot_idx = 0
    @inbounds for i in 1:n-1
        a = N_grid[i] - N_pivot
        b = N_grid[i+1] - N_pivot
        if a*b < 0 && eps_arr[i] < T(1) && eps_arr[i+1] < T(1)
            pivot_idx = i
            break
        end
    end
    if pivot_idx == 0
        finite_found = false
        N_max = T(0)
        @inbounds for i in 1:n
            val = N_grid[i]
            if isfinite(val)
                if !finite_found || val > N_max
                    N_max = val
                    finite_found = true
                end
            end
        end
        if !finite_found
            return L(1.5e3 * 2.0)
        end
        deficit = max(T(0), N_pivot - N_max)
        return L(1.5e3 * (1.0 + deficit / N_pivot))
    end
    if pivot_idx == 1 || pivot_idx == n
        return L(1.5e3)
    end

    eps_pivot = eps_arr[pivot_idx]
    Q_pivot   = Q[pivot_idx]
    V_pivot   = V_vec[pivot_idx]
    deps_dphi = (eps_arr[pivot_idx+1] - eps_arr[pivot_idx-1]) / (2*dphi)

    n_s_pred = T(1) - T(2)*eps_pivot + (Q_pivot/(eps_pivot*V_pivot)) * deps_dphi
    r_pred   = T(16) * eps_pivot

    chi2 = ((n_s_pred - target_ns) / sigma_ns)^2 +
           ((r_pred   - target_r ) / sigma_r )^2

    if enforce_egb
        delta1 = -T(4/3) * xip[pivot_idx] * Q_pivot
        d1abs  = abs(delta1)
        chi2 += T(2.0e3) * exp(-d1abs / max(T(0.5) * egb_min, T(1e-30)))
    end

    return L(chi2)
end
"""


def _build_julia_xi_loss_source(cfg: "SearchConfig", V_expr: str) -> str:
    """Generate Julia loss source for the ξ search with V hardcoded.

    The full EGB physics χ² is computed with a non-trivial ξ(φ), using
    the SR tree as ξ(φ) and the inlined `V_user(p)` as V(φ).

    Includes the GR-limit penalty: when `enforce_egb=True` and
    |δ_1(φ_pivot)| < `egb_min_delta1`, an exp-decay penalty pushes the
    search out of the ξ → 0 basin.
    """
    V_julia = _sympy_to_julia(V_expr)
    enforce_egb = "true" if cfg.enforce_egb else "false"
    return f"""
function deepegb_xi_loss(tree, dataset::Dataset{{T,L}}, options) where {{T,L}}
    target_ns = T({cfg.target_ns})
    sigma_ns  = T({cfg.sigma_ns})
    target_r  = T({cfg.target_r})
    sigma_r   = T({cfg.sigma_r})
    N_pivot   = T({cfg.N_pivot})
    egb_min   = T({cfg.egb_min_delta1})
    enforce_egb = {enforce_egb}

    n = size(dataset.X, 2)
    if n < 4
        return L(1e6)
    end
    phi = view(dataset.X, 1, :)
    dphi = phi[2] - phi[1]

    # V(φ) hardcoded from pass-1 winner.
    @inline V_user(phi) = {V_julia}
    V_vec = V_user.(phi)

    # V positivity: graded penalty.
    n_bad = 0
    @inbounds for i in 1:n
        if !isfinite(V_vec[i]) || V_vec[i] <= 0
            n_bad += 1
        end
    end
    if n_bad > 0
        frac = T(n_bad) / T(n)
        return L(1.0e3 * (1.0 + 10.0 * frac))
    end

    # Evaluate ξ(φ) on the same grid via PySR's tree evaluator.
    xi_vec, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(1e6)
    end

    # Central differences: V', V'', ξ'.
    Vp = similar(V_vec)
    Vpp = similar(V_vec)
    xip = similar(V_vec)
    @inbounds Vp[1] = (V_vec[2] - V_vec[1]) / dphi
    @inbounds Vp[end] = (V_vec[end] - V_vec[end-1]) / dphi
    @inbounds Vpp[1] = (V_vec[3] - 2*V_vec[2] + V_vec[1]) / (dphi*dphi)
    @inbounds Vpp[end] = (V_vec[end] - 2*V_vec[end-1] + V_vec[end-2]) / (dphi*dphi)
    @inbounds xip[1] = (xi_vec[2] - xi_vec[1]) / dphi
    @inbounds xip[end] = (xi_vec[end] - xi_vec[end-1]) / dphi
    @inbounds for i in 2:n-1
        Vp[i] = (V_vec[i+1] - V_vec[i-1]) / (2*dphi)
        Vpp[i] = (V_vec[i+1] - 2*V_vec[i] + V_vec[i-1]) / (dphi*dphi)
        xip[i] = (xi_vec[i+1] - xi_vec[i-1]) / (2*dphi)
    end

    # Q(φ) = V' + (4/3) V² ξ'   (slow-roll force in EGB)
    Q = similar(V_vec)
    @inbounds for i in 1:n
        Q[i] = Vp[i] + T(4/3) * V_vec[i]^2 * xip[i]
    end

    # ε(φ) = (1/2) Q V' / V²
    eps_arr = similar(V_vec)
    @inbounds for i in 1:n
        eps_arr[i] = T(0.5) * Q[i] * Vp[i] / (V_vec[i]^2)
    end

    # φ_end via first sign-change of (ε − 1).
    end_idx = 0
    phi_end = T(0)
    @inbounds for i in 1:n-1
        a = eps_arr[i] - T(1)
        b = eps_arr[i+1] - T(1)
        if isfinite(a) && isfinite(b) && a*b < 0
            phi_end = phi[i] - a*dphi/(b - a)
            end_idx = i
            break
        end
    end
    if end_idx == 0
        eps_found = false
        eps_min = T(0)
        @inbounds for i in 1:n
            val = eps_arr[i]
            if isfinite(val)
                if !eps_found || val < eps_min
                    eps_min = val
                    eps_found = true
                end
            end
        end
        if !eps_found
            return L(2.0e3)
        end
        return L(2.0e3 * (1.0 + abs(log10(max(abs(eps_min - T(1)), T(1e-30))))))
    end

    # N(φ) = integral from φ_end to φ of V/Q dφ via cumulative trapezoid.
    integrand = V_vec ./ Q
    cum_int = zeros(T, n)
    @inbounds for i in 2:n
        cum_int[i] = cum_int[i-1] + T(0.5)*(integrand[i] + integrand[i-1])*dphi
    end
    cum_at_end = cum_int[end_idx] +
        (cum_int[end_idx+1] - cum_int[end_idx]) * (phi_end - phi[end_idx]) / dphi
    N_grid = cum_int .- cum_at_end

    # φ_pivot on the slow-roll side (ε < 1, N = N_pivot).
    pivot_idx = 0
    @inbounds for i in 1:n-1
        a = N_grid[i] - N_pivot
        b = N_grid[i+1] - N_pivot
        if a*b < 0 && eps_arr[i] < T(1) && eps_arr[i+1] < T(1)
            pivot_idx = i
            break
        end
    end
    if pivot_idx == 0
        finite_found = false
        N_max = T(0)
        @inbounds for i in 1:n
            val = N_grid[i]
            if isfinite(val)
                if !finite_found || val > N_max
                    N_max = val
                    finite_found = true
                end
            end
        end
        if !finite_found
            return L(1.5e3 * 2.0)
        end
        deficit = max(T(0), N_pivot - N_max)
        return L(1.5e3 * (1.0 + deficit / N_pivot))
    end
    if pivot_idx == 1 || pivot_idx == n
        return L(1.5e3)
    end

    eps_pivot = eps_arr[pivot_idx]
    Q_pivot   = Q[pivot_idx]
    V_pivot   = V_vec[pivot_idx]

    # Production-formula n_s: 1 - 2 eps + (Q/(eps*V)) * d_eps/d_phi at pivot.
    deps_dphi = (eps_arr[pivot_idx+1] - eps_arr[pivot_idx-1]) / (2*dphi)
    n_s_pred = T(1) - T(2)*eps_pivot + (Q_pivot / (eps_pivot*V_pivot)) * deps_dphi
    r_pred   = T(16) * eps_pivot

    chi2 = ((n_s_pred - target_ns) / sigma_ns)^2 +
           ((r_pred   - target_r ) / sigma_r )^2

    # GR-limit penalty: δ_1 = -(4/3) ξ' Q at the pivot.
    if enforce_egb
        delta1 = -T(4/3) * xip[pivot_idx] * Q_pivot
        d1abs  = abs(delta1)
        chi2 += T(2.0e3) * exp(-d1abs / max(T(0.5) * egb_min, T(1e-30)))
    end

    return L(chi2)
end
"""


# ---------------------------------------------------------------------------
# Config + result types
# ---------------------------------------------------------------------------
@dataclass
class SearchConfig:
    # Mandatory targets
    target_ns: float = 0.974
    sigma_ns: float = 0.003
    target_r: float = 0.0
    sigma_r: float = 0.018

    # Optional production-grade targets (None ⇒ excluded from χ²)
    target_lnAs: float | None = None       # ln(10¹⁰ A_s); Planck ≈ 3.044
    sigma_lnAs: float = 0.014
    target_alphas: float | None = None     # running of n_s
    sigma_alphas: float = 0.013
    target_nT: float | None = None         # tensor spectral index
    sigma_nT: float = 0.1
    target_cT2: float | None = None        # tensor sound-speed squared
    sigma_cT2: float = 0.05

    N_pivot: float = 55.0

    # Loss kernel:
    #   "production"     — slow-roll closed-form (background EOMs + full
    #                      perturbation kernel with c_T², c_S²). ~10 ms / call,
    #                      default for n_s/r searches.
    #   "production_gw"  — full background ODE + Mukhanov-Sasaki for the
    #                      target k modes + relic-GW transfer.
    #                      ~0.5–2 s / call; required when omega_gw_targets
    #                      or omega_gw_band_min are set.
    loss_kind: str = "production"

    # Relic-GW targets (only used when loss_kind == "production_gw").
    # Each entry: (frequency_in_Hz, target_Omega_GW_h2, sigma_Omega_GW_h2).
    # Interpreted as a target on log10(Ω_GW h²); sigma is taken in log10 space
    # using the relative formula sigma_log = (sigma/target)/ln(10).
    omega_gw_targets: tuple[tuple[float, float, float], ...] = ()
    # (f_lo_Hz, f_hi_Hz, target_min_Omega_GW_h2): adds a band-floor penalty
    # that increases when the spectrum drops below target_min anywhere in
    # [f_lo, f_hi]. Useful for "make this loud enough to detect at LISA".
    omega_gw_band_min: tuple[float, float, float] | None = None
    T_reh_GeV: float | None = 1.0e15

    # EGB-sector enforcement: ξ ≡ 0 reduces our action to plain GR, which
    # is not what we're searching for. When True (default) we add a soft
    # penalty for |δ₁(φ_pivot)| < `egb_min_delta1` to push the search away
    # from the GR-limit basin, and we drop ξ=0 from the candidate list in
    # the second SR pass. Pass `enforce_egb=False` to allow the GR
    # baseline as a member of the search (e.g. for controlled comparison).
    enforce_egb: bool = True
    egb_min_delta1: float = 1.0e-4

    # Julia physics-χ² loss for V and ξ searches.
    #   "auto" — try the Julia loss; on any failure, fall back silently to
    #            the multi-family MSE seed sweep.
    #   True   — require the Julia loss; raise if it fails to install.
    #   False  — skip Julia, use MSE seed sweep only.
    use_julia_loss: str | bool = "auto"

    # Coordinate-descent joint search:
    # When > 0, after pass 2 we re-run V (with best ξ hardcoded) and ξ
    # (with new best V hardcoded) for `joint_rounds` extra cycles. Each
    # round refines (V, ξ) toward a self-consistent fixed point. Set 0
    # to disable; 1–2 is usually enough.
    joint_rounds: int = 0

    # Seed-family library — see search/seed_families.py.
    # PySR's evolutionary search uses MSE against y_seed for fitness; if
    # we use one fixed seed shape, the hall-of-fame is biased toward that
    # shape's basin. We sweep multiple seed families per pass and merge
    # the hall-of-fames before re-ranking by physics-χ². Empty string in
    # either tuple means "use the default Starobinsky-like seed only".
    v_seed_families: tuple[str, ...] = (
        "starobinsky", "hilltop", "pole", "exp_plateau"
    )
    xi_seed_families: tuple[str, ...] = (
        "exp_decay", "pole", "power_law", "tanh"
    )
    # Iterations per family: total wall time scales as
    #   len(v_families) * niterations + top_k_V * len(xi_families) * niterations
    # so cut niterations when sweeping many families.
    niterations_per_family: int | None = None

    # PySR hyperparameters
    niterations: int = 40
    populations: int = 35
    population_size: int = 33
    maxsize: int = 25
    parsimony: float = 1.0e-3

    binary_operators: tuple[str, ...] = ("+", "-", "*", "/")
    unary_operators: tuple[str, ...] = ("exp", "log", "sqrt", "tanh")

    # Search strategy
    mode: str = "two_pass"   # "two_pass" or "joint"
    top_k_V: int = 5         # how many V candidates to retain in two-pass mode

    # Sampling grid for X data fed to PySR (PySR fits an objective via
    # `loss_function` so X values are largely irrelevant; we still need them).
    n_samples: int = 96
    phi_sample_range: tuple[float, float] = (-10.0, 10.0)

    phi_search_range: tuple[float, float] = (-15.0, 15.0)
    phi_search_grid: int = 4001

    runs_dir: str | Path = "runs"

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class SearchResult:
    V_expr: str
    xi_expr: str
    chi2: float
    # Core
    n_s: float
    r: float
    epsilon: float
    phi_N: float
    phi_end: float
    elapsed_s: float
    # Production-grade extras (NaN if computed via the leading-order kernel)
    n_T: float = float("nan")
    alpha_s: float = float("nan")
    P_S: float = float("nan")
    P_T: float = float("nan")
    c_T2: float = float("nan")
    delta1: float = float("nan")
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {**self.__dict__}


# ---------------------------------------------------------------------------
# Sympy → callable
# ---------------------------------------------------------------------------
def _sympy_to_callable(expr_str: str) -> Callable[[np.ndarray], np.ndarray]:
    """Parse a string expression in φ to a vectorised callable.

    Accepts both `phi` and `x0` as the field variable — PySR returns
    hall-of-fame expressions using `x0` (its internal feature name).
    """
    phi = sp.Symbol("phi", real=True)
    try:
        expr = sp.sympify(expr_str, locals={"phi": phi, "x0": phi})
    except Exception as e:
        raise ValueError(f"Could not parse expression {expr_str!r}: {e}")
    return sp.lambdify(phi, expr, modules=["numpy"])


def expressions_to_model(V_expr: str, xi_expr: str = "0", *, name: str = "model") -> EGBModel:
    return EGBModel(
        V=_sympy_to_callable(V_expr),
        xi=_sympy_to_callable(xi_expr),
        name=name,
        description=f"V={V_expr};  ξ={xi_expr}",
    )


# ---------------------------------------------------------------------------
# χ² loss as a function of expression strings
# ---------------------------------------------------------------------------
def chi2_for_expressions(
    V_expr: str,
    xi_expr: str,
    cfg: SearchConfig,
) -> float:
    """Evaluate χ² for a (V, ξ) expression pair using the production kernel."""
    try:
        with np.errstate(divide="ignore", invalid="ignore",
                         over="ignore", under="ignore"):
            model = expressions_to_model(V_expr, xi_expr)
            if cfg.loss_kind == "production_gw":
                return chi2_relic_gw(
                    model,
                    target_ns=cfg.target_ns, sigma_ns=cfg.sigma_ns,
                    target_r=cfg.target_r, sigma_r=cfg.sigma_r,
                    target_lnAs=cfg.target_lnAs, sigma_lnAs=cfg.sigma_lnAs,
                    omega_gw_targets=list(cfg.omega_gw_targets) or None,
                    omega_gw_band_min=cfg.omega_gw_band_min,
                    N_pivot=cfg.N_pivot,
                    T_reh_GeV=cfg.T_reh_GeV,
                    enforce_egb=cfg.enforce_egb,
                    egb_min_delta1=cfg.egb_min_delta1,
                )
            # production (default): slow-roll closed-form with full perturbations
            obs_full = compute_observables_full(
                model,
                N_pivot=cfg.N_pivot,
                phi_range=cfg.phi_search_range,
                n_grid=cfg.phi_search_grid,
            )
            return chi2_full(
                obs_full,
                target_ns=cfg.target_ns, sigma_ns=cfg.sigma_ns,
                target_r=cfg.target_r, sigma_r=cfg.sigma_r,
                target_lnAs=cfg.target_lnAs, sigma_lnAs=cfg.sigma_lnAs,
                target_alphas=cfg.target_alphas, sigma_alphas=cfg.sigma_alphas,
                target_nT=cfg.target_nT, sigma_nT=cfg.sigma_nT,
                target_cT2=cfg.target_cT2, sigma_cT2=cfg.sigma_cT2,
                model=model,    # enables soft-invalid penalty for NaN cases
                enforce_egb=cfg.enforce_egb,
                egb_min_delta1=cfg.egb_min_delta1,
            )
    except Exception:
        return 1.0e6


def observables_for_result(V_expr: str, xi_expr: str, cfg: SearchConfig) -> dict:
    """Production-grade observables for a (V, ξ) pair (slow-roll closed-form
    is fine here — used only for ranking, not for the loss)."""
    try:
        with np.errstate(divide="ignore", invalid="ignore",
                         over="ignore", under="ignore"):
            model = expressions_to_model(V_expr, xi_expr)
            o = compute_observables_full(
                model, N_pivot=cfg.N_pivot,
                phi_range=cfg.phi_search_range, n_grid=cfg.phi_search_grid,
            )
            return o.as_dict()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# PySR custom loss: PySR sees y_i = f(x_i) regression, but we override the
# loss function so the y values are unused. We still pass dummy X, y so the
# scikit-style API is happy.
# ---------------------------------------------------------------------------
def _make_pysr(cfg: SearchConfig, kind: str, fixed_other: Optional[str] = None,
               *, julia_loss_source: Optional[str] = None) -> "PySRRegressor":
    if PySRRegressor is None:
        raise RuntimeError("PySR is not installed. `pip install pysr` and run `pysr.install()`.")

    # We embed the *current* candidate (V or ξ) into a Julia closure. PySR
    # gets passed a `loss_function` (Julia source string) that calls back into
    # our Python χ² evaluator via JuliaCall would be ideal — but for the MVP
    # we use PySR's `extra_sympy_mappings` and `loss_function` is approximated
    # in Julia as MSE on a dummy target. The actual χ² is then re-evaluated
    # in Python on the hall_of_fame after the run finishes. This means PySR
    # explores by raw fitting of a "dummy" curve but we re-rank using physics.
    #
    # For thesis-grade work, replace this with a true Julia loss kernel
    # (see `physics/kernel.jl` and the `custom_loss` example in PySR docs).

    # PySR's keyword set has drifted across versions:
    #   * < 0.19  : `equation_file=...`
    #   * 0.19+   : the equivalent kwarg is `output_directory=...`
    #               (or `temp_equation_file=False` plus `tempdir=...`)
    #   * 1.x     : `equation_file` removed, `output_directory` is the way.
    # We try a sequence of (kwargs, value) pairs and accept the first that
    # the installed version is happy with. The hall-of-fame CSV is also
    # accessible via `regressor.equations_` after fit, so we don't strictly
    # need to direct it ourselves; saving to runs/ is just a convenience.
    base_kwargs: dict = dict(
        niterations=cfg.niterations,
        populations=cfg.populations,
        population_size=cfg.population_size,
        maxsize=cfg.maxsize,
        parsimony=cfg.parsimony,
        binary_operators=list(cfg.binary_operators),
        unary_operators=list(cfg.unary_operators),
        model_selection="best",
        progress=False,
        deterministic=False,
        verbosity=1,
    )
    if julia_loss_source is not None:
        # Bypass MSE: PySR will compile this Julia source as the loss.
        # The (X, y) we still pass to .fit() is irrelevant for the LOSS,
        # but X is used as the φ-grid via dataset.X inside the loss.
        base_kwargs["loss_function"] = julia_loss_source
    runs_subdir = Path(cfg.runs_dir) / f"hall_of_fame_{kind}"
    # Try each output-routing kwarg in order; drop the kwarg entirely if
    # nothing works (PySR will use a tempdir).
    for output_kwarg, output_value in (
        ("output_directory", str(runs_subdir)),
        ("equation_file", str(runs_subdir.with_suffix(".csv"))),
        (None, None),
    ):
        kwargs = dict(base_kwargs)
        if output_kwarg:
            kwargs[output_kwarg] = output_value
        try:
            return PySRRegressor(**kwargs)
        except TypeError:
            continue
    # Should be unreachable — at least the (None, None) path is keyword-free.
    return PySRRegressor(**base_kwargs)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_joint_search(
    cfg: SearchConfig,
    *,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> list[SearchResult]:
    """Run a joint search for (V, ξ) and return ranked candidates by χ²."""
    import sys
    if PySRRegressor is None:
        raise RuntimeError("PySR is not installed.")

    runs_dir = Path(cfg.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    log = progress_cb or (lambda s: None)
    t0 = time.time()

    # Pass 1 has TWO routes for the V search, in priority order:
    #
    #   (A) Julia physics-χ² loss — PySR's evolutionary search optimises
    #       the actual EGB χ² via `loss_function=` (no y_seed bias). This
    #       is the right answer; multi-family fallback below is a fallback.
    #   (B) Multi-family MSE seed sweep — run PySR several times against
    #       different inflation-family targets and merge the hall-of-fames.
    #       Cross-family diversity but each sub-search is still MSE-biased.
    #
    # `cfg.use_julia_loss` selects:
    #   "auto" → try (A); on any failure fall back to (B).
    #   True   → require (A); raise on failure.
    #   False  → skip (A); use (B).
    from .seed_families import V_FAMILIES, XI_FAMILIES, get_v_family, get_xi_family

    phi = np.linspace(*cfg.phi_sample_range, cfg.n_samples)
    phi[np.isclose(phi, 0.0, atol=1e-12)] = 1.0e-6
    X = phi.reshape(-1, 1)

    V_candidate_pool: list[str] = []
    iters_per_family = (cfg.niterations_per_family
                        if cfg.niterations_per_family is not None
                        else max(8, cfg.niterations
                                 // max(1, len(cfg.v_seed_families) or 1)))
    cfg_per_family = replace(cfg, niterations=iters_per_family)

    julia_loss_used = False
    if cfg.use_julia_loss:
        log(f"[1/2] PySR V(φ) search via Julia physics-χ² loss "
            f"(targets: n_s={cfg.target_ns}±{cfg.sigma_ns}, "
            f"r={cfg.target_r}±{cfg.sigma_r}, N={cfg.N_pivot}) …")
        loss_src = _build_julia_v_loss_source(cfg)
        # Pass our φ-grid as X. y is irrelevant when loss_function is set.
        X_julia = phi.reshape(1, -1).astype(np.float32) \
            if False else phi.reshape(-1, 1).astype(np.float32)
        # PySR expects (n_samples, n_features); inside the loss we view
        # dataset.X as (n_features, n_samples), which is the same array
        # transposed via Julia's row-major / column-major view.
        y_dummy = np.zeros(len(phi), dtype=np.float32)
        try:
            log("      building PySR regressor for V (Julia loss) …")
            pysr_V = _make_pysr(cfg_per_family, kind="V_julia",
                                julia_loss_source=loss_src)
            log("      calling pysr_V.fit() — Julia JIT may take 2–5 min on first run …")
            print(f"[DeepEGB] PySR V-fit starting | stdout={type(sys.stdout).__name__} stderr={type(sys.stderr).__name__}",
                  file=sys.stderr, flush=True)
            import os as _os
            print(f"[DeepEGB] stdout fd={sys.stdout.fileno() if hasattr(sys.stdout,'fileno') else 'N/A'} "
                  f"stderr fd={sys.stderr.fileno() if hasattr(sys.stderr,'fileno') else 'N/A'}",
                  file=sys.stderr, flush=True)
            pysr_V.fit(X_julia, y_dummy)
            print("[DeepEGB] PySR V-fit done", file=sys.stderr, flush=True)
            family_candidates = _hall_of_fame_strings(pysr_V, top_k=cfg.top_k_V * 2)
            log(f"      Julia-loss search: {len(family_candidates)} candidates "
                f"(top: {family_candidates[0] if family_candidates else '—'})")
            V_candidate_pool.extend(family_candidates)
            julia_loss_used = True
        except Exception as exc:        # noqa: BLE001
            if cfg.use_julia_loss is True:
                # Hard mode: fail loudly.
                raise RuntimeError(
                    f"Julia physics-χ² loss failed and use_julia_loss=True: {exc}"
                ) from exc
            log(f"      Julia-loss path failed ({exc}); "
                "falling back to multi-family MSE sweep.")

    if not V_candidate_pool:
        # Fallback (B): multi-family MSE seed sweep.
        v_family_list = (
            [get_v_family(name) for name in cfg.v_seed_families]
            if cfg.v_seed_families else [V_FAMILIES[0]]
        )
        log(f"[1/2] PySR V(φ) MSE seed sweep across {len(v_family_list)} "
            f"families ({', '.join(f.name for f in v_family_list)}) …")
        for fam in v_family_list:
            try:
                y_seed = np.asarray(fam.fn(phi), dtype=float)
            except Exception as exc:        # noqa: BLE001
                log(f"      seed {fam.name!r} failed to evaluate: {exc}; skipping")
                continue
            pysr_V = _make_pysr(cfg_per_family, kind=f"V_{fam.name}")
            try:
                pysr_V.fit(X, y_seed)
            except Exception as exc:        # noqa: BLE001
                log(f"      PySR fit failed for V seed {fam.name!r}: {exc}")
                continue
            family_candidates = _hall_of_fame_strings(pysr_V, top_k=cfg.top_k_V)
            log(f"      family {fam.name:14}: {len(family_candidates)} candidates "
                f"(top: {family_candidates[0] if family_candidates else '—'})")
            V_candidate_pool.extend(family_candidates)

    V_candidates = list(dict.fromkeys(V_candidate_pool))      # dedupe, preserve order
    if not V_candidates:
        raise RuntimeError("No V candidates from either Julia loss or "
                           "multi-family sweep. Check PySR/Julia install.")
    log(f"      [V search] Julia loss used: {julia_loss_used}; "
        f"{len(V_candidates)} unique candidates")

    # Re-rank V candidates by EGB χ² (with ξ = 0).
    V_ranked = sorted(
        ((vstr, chi2_for_expressions(vstr, "0", cfg)) for vstr in V_candidates),
        key=lambda it: it[1],
    )
    log(f"      Top V (across families, with ξ=0): " + ", ".join(
        f"{v[:30]}…  χ²={c:.3g}" for v, c in V_ranked[:3]))

    # ---- Pass 2: for each top V, search ξ ----
    #   Path A (Julia): use the ξ Julia loss with V hardcoded — full EGB
    #   physics-χ² (incl. GR-limit penalty when enforce_egb=True), no MSE.
    #   Path B (multi-family MSE): sweep seed families and merge hall-of-
    #   fames as a fallback / complement.
    # When use_julia_loss is on, we run BOTH paths and merge candidates;
    # the cross-family seeds give exploration breadth, the Julia loss
    # gives unbiased exploitation.
    xi_family_list = (
        [get_xi_family(name) for name in cfg.xi_seed_families]
        if cfg.xi_seed_families else [XI_FAMILIES[0]]
    )
    results: list[SearchResult] = []
    for v_idx, (V_str, _) in enumerate(V_ranked[: cfg.top_k_V]):
        log(f"[2/2] ξ(φ) search given V = {V_str[:40]!r} "
            f"({v_idx+1}/{cfg.top_k_V}) …")
        xi_candidate_pool: list[str] = []

        # Path A: Julia loss with V hardcoded.
        julia_xi_used = False
        if cfg.use_julia_loss:
            try:
                xi_loss_src = _build_julia_xi_loss_source(cfg, V_str)
                X_julia = phi.reshape(-1, 1).astype(np.float32)
                y_dummy = np.zeros(len(phi), dtype=np.float32)
                pysr_xi = _make_pysr(cfg_per_family,
                                     kind=f"xi_v{v_idx}_julia",
                                     julia_loss_source=xi_loss_src)
                print(f"[DeepEGB] PySR ξ-fit starting (V{v_idx})",
                      file=sys.stderr, flush=True)
                pysr_xi.fit(X_julia, y_dummy)
                print(f"[DeepEGB] PySR ξ-fit done (V{v_idx})",
                      file=sys.stderr, flush=True)
                julia_xi = _hall_of_fame_strings(pysr_xi, top_k=cfg.top_k_V * 2)
                log(f"      Julia ξ-loss: {len(julia_xi)} candidates "
                    f"(top: {julia_xi[0] if julia_xi else '—'})")
                xi_candidate_pool.extend(julia_xi)
                julia_xi_used = True
            except Exception as exc:        # noqa: BLE001
                if cfg.use_julia_loss is True:
                    raise RuntimeError(
                        f"Julia ξ-loss failed and use_julia_loss=True: {exc}"
                    ) from exc
                log(f"      Julia ξ-loss path failed ({exc}); "
                    "using multi-family MSE only.")

        # Path B: multi-family MSE seeds — also do this when Julia is on,
        # to broaden the candidate pool and mitigate any single-loss bias.
        for fam in xi_family_list:
            try:
                y_seed = np.asarray(fam.fn(phi), dtype=float)
            except Exception as exc:        # noqa: BLE001
                log(f"      seed {fam.name!r} failed: {exc}; skipping")
                continue
            pysr_xi = _make_pysr(cfg_per_family, kind=f"xi_v{v_idx}_{fam.name}")
            try:
                pysr_xi.fit(X, y_seed)
            except Exception as exc:        # noqa: BLE001
                log(f"      PySR fit failed for ξ seed {fam.name!r}: {exc}")
                continue
            xi_candidate_pool.extend(_hall_of_fame_strings(pysr_xi, top_k=cfg.top_k_V))
        xi_candidates = list(dict.fromkeys(xi_candidate_pool))
        if not cfg.enforce_egb:
            xi_candidates = ["0", *xi_candidates]
        log(f"      [ξ search] Julia: {julia_xi_used}; "
            f"{len(xi_candidates)} unique candidates")

        for xi_str in xi_candidates:
            chi2 = chi2_for_expressions(V_str, xi_str, cfg)
            obs_dict = observables_for_result(V_str, xi_str, cfg)
            if not obs_dict:
                continue

            def _get(k, default=float("nan")):
                v = obs_dict.get(k, default)
                return float(v) if v is not None else default

            results.append(SearchResult(
                V_expr=V_str,
                xi_expr=xi_str,
                chi2=chi2,
                n_s=_get("n_s"),
                r=_get("r"),
                epsilon=_get("epsilon"),
                phi_N=_get("phi_N"),
                phi_end=_get("phi_end"),
                elapsed_s=time.time() - t0,
                n_T=_get("n_T"),
                alpha_s=_get("alpha_s"),
                P_S=_get("P_S"),
                P_T=_get("P_T"),
                c_T2=_get("c_T2"),
                delta1=_get("delta1"),
                extra={"V_index": v_idx, "loss_kind": cfg.loss_kind},
            ))

    results.sort(key=lambda r: (math.inf if not math.isfinite(r.chi2) else r.chi2))
    log(f"Found {len(results)} candidates after passes 1+2; "
        f"best χ² = {results[0].chi2 if results else float('nan'):.3g}")

    # ---- Coordinate-descent joint refinement rounds ----
    if cfg.use_julia_loss and cfg.joint_rounds > 0 and results:
        for round_idx in range(cfg.joint_rounds):
            log(f"[joint round {round_idx + 1}/{cfg.joint_rounds}] "
                "refining (V, ξ) jointly via Julia loss …")
            # Take the top candidates and refine each
            refinement_seed = results[: max(1, cfg.top_k_V // 2)]
            new_results: list[SearchResult] = []
            for seed_result in refinement_seed:
                V_str_in = seed_result.V_expr
                xi_str_in = seed_result.xi_expr
                # (a) Re-search V with current ξ hardcoded
                try:
                    src = _build_julia_v_loss_with_xi_source(cfg, xi_str_in)
                    pysr_V = _make_pysr(cfg_per_family,
                                        kind=f"V_round{round_idx}_xi{abs(hash(xi_str_in)) % 10000}",
                                        julia_loss_source=src)
                    X_julia = phi.reshape(-1, 1).astype(np.float32)
                    y_dummy = np.zeros(len(phi), dtype=np.float32)
                    pysr_V.fit(X_julia, y_dummy)
                    new_V_candidates = _hall_of_fame_strings(pysr_V, top_k=3)
                except Exception as exc:        # noqa: BLE001
                    log(f"      V-with-fixed-ξ refinement failed: {exc}; "
                        "keeping previous V")
                    new_V_candidates = [V_str_in]

                # (b) For each refined V, re-search ξ
                for V_new in new_V_candidates:
                    try:
                        src = _build_julia_xi_loss_source(cfg, V_new)
                        pysr_xi = _make_pysr(cfg_per_family,
                                             kind=f"xi_round{round_idx}_V{abs(hash(V_new)) % 10000}",
                                             julia_loss_source=src)
                        pysr_xi.fit(X_julia, y_dummy)
                        xi_refined = _hall_of_fame_strings(pysr_xi, top_k=3)
                    except Exception as exc:        # noqa: BLE001
                        log(f"      ξ-with-fixed-V refinement failed: {exc}; "
                            f"keeping previous ξ")
                        xi_refined = [xi_str_in]

                    for xi_new in xi_refined:
                        chi2 = chi2_for_expressions(V_new, xi_new, cfg)
                        obs_dict = observables_for_result(V_new, xi_new, cfg)
                        if not obs_dict:
                            continue

                        def _g(k, default=float("nan")):
                            v = obs_dict.get(k, default)
                            return float(v) if v is not None else default

                        new_results.append(SearchResult(
                            V_expr=V_new, xi_expr=xi_new, chi2=chi2,
                            n_s=_g("n_s"), r=_g("r"), epsilon=_g("epsilon"),
                            phi_N=_g("phi_N"), phi_end=_g("phi_end"),
                            elapsed_s=time.time() - t0,
                            n_T=_g("n_T"), alpha_s=_g("alpha_s"),
                            P_S=_g("P_S"), P_T=_g("P_T"),
                            c_T2=_g("c_T2"), delta1=_g("delta1"),
                            extra={"loss_kind": cfg.loss_kind,
                                   "joint_round": round_idx + 1},
                        ))
            # Merge into the global pool, dedupe, re-sort
            seen = set()
            merged: list[SearchResult] = []
            for r in [*new_results, *results]:
                key = (r.V_expr, r.xi_expr)
                if key not in seen:
                    seen.add(key)
                    merged.append(r)
            merged.sort(key=lambda r: (math.inf if not math.isfinite(r.chi2)
                                        else r.chi2))
            results = merged
            log(f"      after round {round_idx + 1}: {len(results)} unique "
                f"candidates; best χ² = "
                f"{results[0].chi2 if results else float('nan'):.3g}")

    log(f"Final pool: {len(results)} candidates; "
        f"best χ² = {results[0].chi2 if results else float('nan'):.3g}")
    return results


def _subprocess_worker(cfg_dict: dict, result_q, log_q) -> None:
    """Entry point for the spawned subprocess that runs PySR search.

    juliacall deadlocks when called from asyncio.to_thread() because the
    asyncio event loop occupies the main thread and juliacall cannot complete
    its Python-callback handshake. Running in a spawned subprocess gives PySR
    a clean main thread with no asyncio state.
    """
    try:
        from deepegb.search.pysr_search import SearchConfig, run_joint_search

        cfg = SearchConfig(**{
            k: v for k, v in cfg_dict.items()
            if k in SearchConfig.__dataclass_fields__
        })

        def log(msg: str) -> None:
            try:
                log_q.put_nowait(msg)
            except Exception:
                pass

        results = run_joint_search(cfg, progress_cb=log)
        result_q.put(("ok", results))
    except Exception as exc:
        import traceback
        result_q.put(("error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))


def run_joint_search_subprocess(
    cfg: SearchConfig,
    *,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> list[SearchResult]:
    """Run `run_joint_search` in a spawned subprocess.

    Use this instead of `run_joint_search` when calling from within an
    asyncio.to_thread() context (e.g. Agno tool execution), where juliacall
    deadlocks because the asyncio event loop owns the main thread.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    result_q: mp.Queue = ctx.Queue()
    log_q: mp.Queue = ctx.Queue()

    proc = ctx.Process(
        target=_subprocess_worker,
        args=(cfg.to_dict(), result_q, log_q),
        daemon=True,
    )
    proc.start()

    log = progress_cb or (lambda s: None)

    while proc.is_alive():
        # Drain log messages while the subprocess is running.
        while True:
            try:
                log(log_q.get(timeout=0.2))
            except Exception:
                break

    # Drain any remaining log messages after process exits.
    while True:
        try:
            log(log_q.get_nowait())
        except Exception:
            break

    proc.join()

    try:
        status, data = result_q.get_nowait()
    except Exception:
        code = proc.exitcode
        raise RuntimeError(
            f"Search subprocess exited (code={code}) without returning results. "
            f"Check stderr for errors."
        )

    if status == "error":
        raise RuntimeError(data)
    return data


def _hall_of_fame_strings(reg: "PySRRegressor", top_k: int = 5) -> list[str]:
    try:
        eqs = reg.equations_  # PySR exposes a DataFrame
    except Exception:
        return []
    if eqs is None or len(eqs) == 0:
        return []
    # Prefer simpler equations: PySR sorts by complexity in `equations_`.
    out: list[str] = []
    for _, row in eqs.iterrows():
        s = str(row.get("equation", "")).strip().replace("x0", "phi")
        if s and s not in out:
            out.append(s)
        if len(out) >= top_k:
            break
    return out
