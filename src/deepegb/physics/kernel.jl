# EGB slow-roll χ² loss for use directly inside PySR's evolutionary search.
#
# This file is the documented reference implementation of the loss that the
# Python side (`_build_julia_v_loss_source` in `search/pysr_search.py`)
# generates dynamically with user targets substituted in. PySR's
# `loss_function=...` argument expects a Julia source string, and we need
# to inject runtime targets (n_s, r, N_pivot, …) into it, so the actual
# string passed to PySR is built in Python — but the LOGIC matches this
# file line-for-line.
#
# WHY THIS EXISTS
# ---------------
# PySR's default fitness is MSE between the candidate tree and a (X, y)
# regression target. We don't want MSE to a fixed shape; we want the
# symbolic search to optimise the inflation physics directly. By passing
# our own loss as Julia source, PySR's evolutionary search no longer
# climbs Starobinsky-shaped MSE — it climbs EGB physics-χ².
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
# This is the textbook GR baseline. Pass 2 of the SR search adds a
# non-trivial ξ(φ) afterwards via a separate Julia loss (TODO).
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

"""
    egb_chi2_v_only(V_grid, Vp_grid, Vpp_grid, phi_grid; targets...)

Compute the slow-roll EGB χ² for a candidate V(φ) sampled on `phi_grid`,
in the GR limit ξ ≡ 0. Returns a Float32 with a soft graded penalty for
partly-broken candidates so PySR retains a usable gradient.
"""
function egb_chi2_v_only(V::AbstractVector, Vp::AbstractVector,
                         Vpp::AbstractVector, phi::AbstractVector;
                         target_ns::Real = 0.965,
                         sigma_ns::Real = 0.005,
                         target_r::Real = 0.0,
                         sigma_r::Real = 0.05,
                         N_pivot::Real = 55.0)
    n = length(phi)
    dphi = phi[2] - phi[1]

    # 1. V positivity and finiteness — soft penalty proportional to badness.
    n_bad = 0
    @inbounds for i in 1:n
        if !isfinite(V[i]) || V[i] <= 0
            n_bad += 1
        end
    end
    if n_bad > 0
        frac = n_bad / n
        return 1.0e3 * (1.0 + 10.0 * frac)
    end

    # 2. Slow-roll ε(φ) = (1/2)(V'/V)².
    eps = similar(V)
    @inbounds for i in 1:n
        eps[i] = 0.5 * (Vp[i] / V[i])^2
    end

    # 3. First sign change of (ε − 1) on the φ-grid → φ_end.
    phi_end = NaN
    end_idx = -1
    @inbounds for i in 1:(n-1)
        a = eps[i] - 1.0
        b = eps[i+1] - 1.0
        if isfinite(a) && isfinite(b) && a * b < 0
            phi_end = phi[i] - a * dphi / (b - a)
            end_idx = i
            break
        end
    end
    if end_idx < 0
        eps_min = minimum(eps)
        return 2.0e3 * (1.0 + abs(log10(max(abs(eps_min - 1.0), 1e-30))))
    end

    # 4. N(φ) via cumulative trapezoid of V/Q (= V/V' in the ξ=0 limit).
    integrand = V ./ Vp
    cum = zeros(n)
    @inbounds for i in 2:n
        cum[i] = cum[i-1] + 0.5 * (integrand[i] + integrand[i-1]) * dphi
    end
    cum_at_end = cum[end_idx] +
                 (cum[end_idx+1] - cum[end_idx]) * (phi_end - phi[end_idx]) / dphi
    N_grid = cum .- cum_at_end

    # 5. φ_pivot on the slow-roll side (ε < 1).
    pivot_idx = -1
    @inbounds for i in 1:(n-1)
        if (N_grid[i] - N_pivot) * (N_grid[i+1] - N_pivot) < 0 &&
           eps[i] < 1.0 && eps[i+1] < 1.0
            pivot_idx = i
            break
        end
    end
    if pivot_idx < 0
        N_max = maximum(filter(isfinite, N_grid))
        deficit = max(0.0, N_pivot - N_max)
        return 1.5e3 * (1.0 + deficit / N_pivot)
    end

    eps_pivot = eps[pivot_idx]
    eta_pivot = Vpp[pivot_idx] / V[pivot_idx]

    # 6. Observables and χ².
    n_s_pred = 1.0 - 6.0 * eps_pivot + 2.0 * eta_pivot
    r_pred   = 16.0 * eps_pivot

    chi2 = ((n_s_pred - target_ns) / sigma_ns)^2 +
           ((r_pred   - target_r ) / sigma_r )^2

    return Float32(chi2)
end
