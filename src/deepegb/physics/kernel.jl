# EGB slow-roll loss kernel (Julia mirror of egb_slow_roll.py).
#
# This is the hot inner loop intended to be compiled by PySR's Julia backend
# for ~100× speedup over the Python χ² in `chi2_for_expressions`. Wire it
# into PySR via the `loss_function` argument.
#
# Conventions match egb_slow_roll.py:
#     Q(φ) = V'(φ) + (4/3) V(φ)² ξ'(φ)
#     ε(φ) = (1/2) Q(φ) V'(φ) / V(φ)²
#     End of inflation: ε(φ_end) = 1
#     N(φ)  = ∫_φ^{φ_end} V/Q dφ
#     n_s − 1 = −2ε + (Q/(εV)) dε/dφ |_{φ_N}
#     r       = 16 ε|_{φ_N}
#
# Status: stub. Activate by replacing chi2_for_expressions in pysr_search.py
# with a Julia call once you have JuliaCall configured; SymbolicRegression.jl
# will inline this for free.

module DeepEGBKernel

using ForwardDiff

"""Build a callable expression tree → χ² loss for one (V, ξ) candidate."""
function chi2_loss(V::Function, xi::Function;
                   target_ns::Float64, sigma_ns::Float64,
                   target_r::Float64, sigma_r::Float64,
                   N_pivot::Float64 = 55.0,
                   phi_range::Tuple{Float64,Float64} = (-15.0, 15.0),
                   n_grid::Int = 4001)

    Vprime(φ)   = ForwardDiff.derivative(V, φ)
    ξprime(φ)   = ForwardDiff.derivative(xi, φ)
    Q(φ)        = Vprime(φ) + (4.0/3.0) * V(φ)^2 * ξprime(φ)
    ε(φ)        = 0.5 * Q(φ) * Vprime(φ) / V(φ)^2

    # End of inflation
    φs = range(phi_range[1], phi_range[2]; length=n_grid)
    εs = [ε(φ) for φ in φs]
    φ_end = NaN
    for i in 1:n_grid-1
        a, b = εs[i] - 1.0, εs[i+1] - 1.0
        if isfinite(a) && isfinite(b) && a*b < 0
            φ_end = φs[i] - a*(φs[i+1] - φs[i])/(b - a)
            break
        end
    end
    isnan(φ_end) && return 1.0e6

    # e-fold counting
    integrand(φ) = V(φ)/Q(φ)
    cum = zeros(n_grid)
    for i in 2:n_grid
        cum[i] = cum[i-1] + 0.5*(integrand(φs[i]) + integrand(φs[i-1]))*(φs[i]-φs[i-1])
    end
    cum_end = cum[argmin(abs.(φs .- φ_end))]
    Ns = cum_end .- cum

    # horizon crossing N e-folds before end
    φ_N = NaN
    for i in 1:n_grid-1
        if (Ns[i]-N_pivot)*(Ns[i+1]-N_pivot) < 0
            φ_N = φs[i] - (Ns[i]-N_pivot)*(φs[i+1]-φs[i])/((Ns[i+1]-N_pivot)-(Ns[i]-N_pivot))
            break
        end
    end
    isnan(φ_N) && return 1.0e6

    eps_N = ε(φ_N)
    Q_N   = Q(φ_N)
    V_N   = V(φ_N)
    deps  = ForwardDiff.derivative(ε, φ_N)

    n_s = 1.0 - 2.0*eps_N + (Q_N / (eps_N*V_N))*deps
    r   = 16.0 * eps_N
    return ((n_s - target_ns)/sigma_ns)^2 + ((r - target_r)/sigma_r)^2
end

end # module
