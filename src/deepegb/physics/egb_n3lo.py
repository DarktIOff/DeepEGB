"""
Exact analytic (N3LO Green's-function) observables for EGB inflation —
the production analytic path.

Goal
----
Primordial observables with *exact* slow-roll coefficients: power spectra
through third order (N3LO) and spectral indices through fourth order
(N4LO) in the slow-roll/flow expansion, with the exact Green's-function
constants C = γ_E + ln 2 − 2, π², ζ(3).  No WKB / uniform-asymptotic
residual: unlike the UAA expansion (see `egb_uaa.py`, kept only as a
cross-check), these series converge onto the true Mukhanov–Sasaki result
order by order.

Method (exact sector reduction)
-------------------------------
Both EGB perturbation sectors obey a canonical mode equation
(Hwang–Noh 2005; Wu–Zhu–Wang 2017, arXiv:1707.08020, Eqs. 2.8–2.12)

    μ''(η) + ( c² k² − z''/z ) μ = 0 ,      μ = z ·(R or h_λ),

with sector-specific (z, c) given EXACTLY by background quantities:

  scalar:  z_R² = a² (φ̇² + 6 δ̄ ξ̇ H³) / [ (1 − δ̄/2)² H² ],
           c_R² = 1 + [8 δ̄ ξ̇ H Ḣ + 2 δ̄² H² (ξ̈ − ξ̇H)] / (φ̇² + 6 δ̄ ξ̇ H³),
           δ̄ ≡ δ₁ / (1 − δ₁),    δ₁ ≡ 4 ξ̇ H  (M_pl = 1);
  tensor:  z_h² = a² (1 − δ₁),
           c_h² = 1 − 4 (ξ̈ − ξ̇ H) / (1 − δ₁).

The transformation to "sound time"  dς = c dη,  ṽ = √c μ,  z̃ = z √c
brings each sector EXACTLY to the minimal form

    d²ṽ/dς² + ( k² − z̃_{,ςς}/z̃ ) ṽ = 0 ,        P_sector = (k³/2π²)|ṽ/z̃|².

This is isomorphic to the GR tensor problem with a(η) → z̃(ς).  We
therefore apply the N3LO master formulas of Auclair & Ringeval
(arXiv:2205.12608; vendored ancillary `_n3lo_master.py`) with the
substitutions

    H  → H̃ ≡ z̃_{,ς}/z̃²,         (effective Hubble rate)
    ε₁ → ε̃₁ = 1 − ℋ̃_{,ς}/ℋ̃²,    ℋ̃ ≡ z̃_{,ς}/z̃,
    ε̃_{i+1} = d ln ε̃_i / d ln z̃,  (effective flow hierarchy)
    η  → ς,   pivot at  −k ς_dia = 1.

The effective flow functions are computed exactly from the full
Friedmann–Klein-Gordon background trajectory — *no slow-roll truncation
of the background dynamics enters anywhere*.  The only expansion is the
controlled flow expansion of the master formula itself.

Numerical strategy (important for the higher flow functions)
------------------------------------------------------------
Repeatedly applying finite differences to trajectory-derived grids
amplifies the O(rtol) interpolation noise of the ODE dense output and
destroys ε̃₂..ε̃₄ (the noise grows by ~1/ΔN per derivative).  We instead

  1. compute every first-level time derivative *analytically* along the
     trajectory: dπ/dN and ε₁ from the exact per-step linear system
     (`egb_background._step_rhs`), hence φ̈ = H²(dπ/dN − π ε₁),
     ξ̇ = ξ_{,φ} φ̇, ξ̈ = ξ_{,φφ} φ̇² + ξ_{,φ} φ̈, and
     dδ₁/dN = 4 ξ̈ − δ₁ ε₁ in closed form;
  2. build d ln z̃/dN analytically (only the small d ln c²/dN residual,
     itself O(δ₁), uses one numerical gradient of an analytic grid);
  3. take ε̃₁ = 1 − (1 − ε₁ + d ln X/dN)/X with X ≡ (d ln z̃/dN)/c —
     a single numerical gradient of a smooth analytic grid;
  4. obtain ε̃₂, ε̃₃, ε̃₄ from a local polynomial fit of ln ε̃₁ against
     Ñ = ln z̃ in a window around the evaluation point:
       ε̃₂ = p′,  ε̃₃ = p″/p′,  ε̃₄ = p‴/p″ − p″/p′,
     which averages the residual noise instead of differentiating it.

Normalisation: the AR tensor master equals 8·(k³/2π²)|ṽ/z̃|² for z̃ = a
(GR limit P_T = 2H²/π²).  Hence

    P_S = master(H̃_S, ε̃_S)/8,        P_T = master(H̃_T, ε̃_T),

fixed by the GR limit and independent of the GB coupling (the coupling
enters only through z̃, c).  Validated against the Mukhanov–Sasaki
integrator in tests/test_egb_n3lo.py.

References
----------
* Auclair P., Ringeval C., *Slow-Roll Inflation at N3LO*,
  arXiv:2205.12608, PRD 106, 063512 (2022).        — master formulas
* Beltrán Jiménez J., Musso M., Ringeval C., arXiv:1303.2788
  — exact tensor↔general-scalar mapping underlying the reduction.
* Martin J., Ringeval C., Vennin V., arXiv:1303.2120 — N2LO with
  varying sound speed (consistency check of the method).
* Hwang J., Noh H., gr-qc/0507025 — EGB perturbation equations.
* Wu Q., Zhu T., Wang A., arXiv:1707.08020 — exact (z, c) for both EGB
  sectors (Eqs. 2.8–2.12); their UAA spectra are the cross-check path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _n3lo_master as _master
from .egb_background import BackgroundTrajectory
from .egb_slow_roll import EGBModel

# Step (in ln k) used to differentiate the *analytic* master polynomial
# for the runnings α_s, α_t.  This differentiates a closed-form cubic-in-
# ln k expression, not numerical data: the only error is O(h²·∂⁴ln P),
# far below the O(ε⁴) truncation of the series itself.
_DLNK = 0.1

# Half-width (e-folds) and degree of the local fit of ln ε̃₁(Ñ).
_FIT_HALF_WIDTH = 2.5
_FIT_DEGREE = 5

_TINY = 1.0e-300


# ---------------------------------------------------------------------------
# Analytic first-level derivatives along the trajectory
# ---------------------------------------------------------------------------
def _analytic_grids(model: EGBModel, traj: BackgroundTrajectory) -> dict[str, np.ndarray]:
    """Closed-form background derivatives on the N grid.

    Returns φ̇, φ̈, ξ̇, ξ̈, δ₁, dδ₁/dN, dπ/dN, ε₁, H — all evaluated
    analytically per point through the exact per-step algebra (no
    numerical differentiation of trajectory data).  Vectorised via
    `step_rhs_grid` (with a transparent scalar fallback inside it).
    """
    from .egb_background import step_rhs_grid

    g = step_rhs_grid(model, traj.phi, traj.pi)
    pi = traj.pi
    H = g["H"]
    upi, eps1 = g["upi"], g["eps1"]
    xip, xipp = g["xip"], g["xipp"]
    phidot = pi * H
    phiddot = H * H * (upi - pi * eps1)
    xidot = xip * phidot
    xiddot = xipp * phidot * phidot + xip * phiddot
    delta1 = 4.0 * xidot * H
    # dδ₁/dN = (1/H) d(4ξ̇H)/dt = 4ξ̈ − δ₁ε₁
    ddelta1 = 4.0 * xiddot - delta1 * eps1
    return dict(upi=upi, eps1=eps1, H=H, phidot=phidot, phiddot=phiddot,
                xidot=xidot, xiddot=xiddot, delta1=delta1,
                ddelta1_dN=ddelta1)


# ---------------------------------------------------------------------------
# Sector definitions: exact (z, c, d ln z̃/dN) on the trajectory grid
# ---------------------------------------------------------------------------
def sector_grids(model: EGBModel, traj: BackgroundTrajectory, sector: str,
                 ag: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray] | None:
    """Exact z², c² and the analytic part of d ln z̃/dN for one sector.

    Scalar (WZW Eqs. 2.8–2.9): z_R²/a² = (π_N² + (3/2) δ̄ δ₁)/(1 − δ̄/2)²,
    tensor (WZW Eqs. 2.11–2.12): z_h²/a² = 1 − δ₁ — see module docstring.
    """
    if ag is None:
        ag = _analytic_grids(model, traj)
    a, H = traj.a, ag["H"]
    eps1, d1, dd1 = ag["eps1"], ag["delta1"], ag["ddelta1_dN"]
    xidot, xiddot = ag["xidot"], ag["xiddot"]
    one_m_d1 = 1.0 - d1
    if np.any(np.abs(one_m_d1) < 1.0e-12):
        return None

    N = traj.N
    dN = float(N[1] - N[0])

    with np.errstate(divide="ignore", invalid="ignore"):
        if sector == "tensor":
            z2_over_a2 = one_m_d1
            c2 = 1.0 - 4.0 * (xiddot - xidot * H) / one_m_d1
            # d ln z/dN = 1 − (dδ₁/dN)/(2(1−δ₁))
            dlnz_dN = 1.0 - 0.5 * dd1 / one_m_d1
        elif sector == "scalar":
            dbar = d1 / one_m_d1
            ddbar = dd1 / (one_m_d1 * one_m_d1)         # dδ̄/dN
            pi_N = traj.pi
            num = pi_N * pi_N + 1.5 * dbar * d1          # (φ̇²+6δ̄ξ̇H³)/H²
            if np.any(num <= 0):
                return None
            z2_over_a2 = num / (1.0 - 0.5 * dbar) ** 2
            phidot = ag["phidot"]
            Hdot = -eps1 * H * H
            den_c = phidot * phidot + 6.0 * dbar * xidot * H ** 3
            c2 = 1.0 + (8.0 * dbar * xidot * H * Hdot
                        + 2.0 * dbar * dbar * H * H * (xiddot - xidot * H)
                        ) / den_c
            # d ln z/dN analytically:
            dnum_dN = (2.0 * pi_N * ag["upi"]
                       + 1.5 * (ddbar * d1 + dbar * dd1))
            dlnz_dN = (1.0 + 0.5 * dnum_dN / num
                       + 0.5 * ddbar / (1.0 - 0.5 * dbar))
        else:
            raise ValueError(f"unknown sector {sector!r}")

    if np.any(z2_over_a2 <= 0) or np.any(c2 <= 0) or not np.all(np.isfinite(c2)):
        return None

    # d ln z̃/dN = d ln z/dN + (1/4) d ln c²/dN.  The c² grid is analytic
    # and its log-derivative is O(δ-flow); one numerical gradient here
    # contributes negligible noise.
    dlnc2_dN = np.gradient(np.log(c2), dN, edge_order=2)
    dlnztilde_dN = dlnz_dN + 0.25 * dlnc2_dN

    return dict(z=np.sqrt(z2_over_a2) * a, c2=c2, c=np.sqrt(c2),
                dlnztilde_dN=dlnztilde_dN)


# ---------------------------------------------------------------------------
# Reduction to the canonical system and effective flow hierarchy
# ---------------------------------------------------------------------------
@dataclass
class SectorFlow:
    """Effective (sound-time) quantities for one sector on the N grid."""

    N: np.ndarray
    Ntilde: np.ndarray       # ln z̃ (effective e-folds)
    sigma: np.ndarray        # sound time ς(N) < 0
    H_eff: np.ndarray        # H̃ = z̃_{,ς}/z̃²
    eps1: np.ndarray         # ε̃₁ grid
    c2: np.ndarray           # sector sound speed squared (exact)

    def flow_at_sigma(self, sigma_target: float) -> dict[str, float] | None:
        """Effective flow ε̃₁..ε̃₄, H̃, c² at sound time ς_target.

        ε̃₂..ε̃₄ come from a local polynomial fit of ln ε̃₁ against Ñ
        (see module docstring), evaluated at the target point.
        """
        if not (self.sigma[0] < sigma_target < self.sigma[-1]):
            return None
        N_t = float(np.interp(sigma_target, self.sigma, self.N))
        mask = np.abs(self.N - N_t) <= _FIT_HALF_WIDTH
        if mask.sum() < 8 * (_FIT_DEGREE + 1):
            return None
        e1_win = self.eps1[mask]
        if np.any(e1_win <= 0) or not np.all(np.isfinite(e1_win)):
            return None
        u = self.Ntilde[mask] - float(np.interp(N_t, self.N, self.Ntilde))
        coef = np.polynomial.polynomial.polyfit(u, np.log(e1_win), _FIT_DEGREE)
        p = np.polynomial.polynomial.Polynomial(coef)
        dp = p.deriv()
        d2p = dp.deriv()
        d3p = d2p.deriv()
        p0, p1, p2, p3 = (float(p(0.0)), float(dp(0.0)),
                          float(d2p(0.0)), float(d3p(0.0)))
        eps1 = float(np.exp(p0))
        eps2 = p1
        eps3 = p2 / p1 if p1 != 0.0 else 0.0
        eps4 = (p3 / p2 - p2 / p1) if (p1 != 0.0 and p2 != 0.0) else 0.0
        H_eff = float(np.interp(N_t, self.N, self.H_eff))
        c2 = float(np.interp(N_t, self.N, self.c2))
        return dict(N=N_t, eps1=eps1, eps2=eps2, eps3=eps3, eps4=eps4,
                    H_eff=H_eff, c2=c2)


def reduce_sector(model: EGBModel, traj: BackgroundTrajectory, sector: str,
                  ag: dict[str, np.ndarray] | None = None) -> SectorFlow | None:
    """Build the exact effective flow quantities for one sector."""
    if ag is None:
        ag = _analytic_grids(model, traj)
    g = sector_grids(model, traj, sector, ag=ag)
    if g is None:
        return None
    z, c, c2 = g["z"], g["c"], g["c2"]
    dlnzt = g["dlnztilde_dN"]

    N = traj.N
    dN = float(N[1] - N[0])
    aH = traj.a * ag["H"]
    eps1_bg = ag["eps1"]

    # Sound time ς(N) = −∫_N^{N_end} c/(aH) dN'  (post-inflation tail is
    # O(e^{-(N_end−N)}) relative — negligible at the pivot).
    integrand = c / np.maximum(aH, _TINY)
    I = np.concatenate([[0.0], np.cumsum(
        0.5 * (integrand[1:] + integrand[:-1]) * np.diff(N))])
    sigma = -(I[-1] - I)

    # ℋ̃ = (aH/c)·dlnz̃/dN = aH·X with X ≡ dlnz̃/dN / c.
    X = dlnzt / c
    if np.any(X <= 0):
        return None
    # ε̃₁ = 1 − ℋ̃_{,ς}/ℋ̃² with
    #   ℋ̃_{,ς}/ℋ̃² = [(1 − ε₁) + d ln X/dN] / (c X)     and  c X = dlnz̃/dN
    # (uses d ln(aH)/dN = 1 − ε₁ exactly; only ln X is differentiated
    # numerically and it is an analytic, slowly-varying grid).
    dlnX_dN = np.gradient(np.log(X), dN, edge_order=2)
    eps1_eff = 1.0 - (1.0 - eps1_bg + dlnX_dN) / dlnzt

    Ntilde = np.log(z) + 0.5 * np.log(c)
    H_eff = aH * X / (z * np.sqrt(c))                 # H̃ = ℋ̃/z̃

    return SectorFlow(N=N, Ntilde=Ntilde, sigma=sigma,
                      H_eff=H_eff, eps1=eps1_eff, c2=c2)


# ---------------------------------------------------------------------------
# Master-formula evaluation per sector
# ---------------------------------------------------------------------------
def _sector_spectrum(flow: SectorFlow, k: float) -> dict[str, float] | None:
    """𝒫(k) = (k³/2π²)|ṽ/z̃|², its index and running for one sector.

    The dia point is the sector's own −kς = 1; the master formula is the
    analytic cubic in ln(k/k_dia): the index comes from the N4LO closed
    form and the running from differentiating the closed-form polynomial.
    """
    if not (np.isfinite(k) and k > 0):
        return None
    d = flow.flow_at_sigma(-1.0 / k)
    if d is None or not all(np.isfinite(v) for v in d.values()):
        return None

    def P_at(kk: float) -> float:
        return float(_master.tensor_power_spectrum(
            kk, k, d["H_eff"], d["eps1"], d["eps2"], d["eps3"])) / 8.0

    P0 = P_at(k)
    n = float(_master.tensor_spectral_index(
        d["eps1"], d["eps2"], d["eps3"], d["eps4"]))
    h = _DLNK
    lnP = [np.log(P_at(k * np.exp(s))) for s in (-h, 0.0, h)]
    alpha = float((lnP[2] - 2.0 * lnP[1] + lnP[0]) / (h * h))
    return dict(P=P0, n=n, alpha=alpha, c2=d["c2"], eps1=d["eps1"],
                N_dia=d["N"])


# ---------------------------------------------------------------------------
# Production entry point
# ---------------------------------------------------------------------------
def compute_observables_n3lo(
    model: EGBModel,
    N_pivot: float | None = None,
    phi_range: tuple[float, float] | None = None,
    *,
    traj: BackgroundTrajectory | None = None,
):
    """Exact-coefficient analytic observables for an EGB model.

    1. Integrate the full Friedmann–Klein-Gordon background.
    2. Reduce both perturbation sectors to canonical form (sound time);
       build the effective flow hierarchies exactly from the trajectory.
    3. Evaluate the AR N3LO/N4LO master formulas at the pivot scale
       k_* = a H |_{N_end − N_pivot}.

    Returns a `FullObservables` or None (caller falls back to the
    Mukhanov–Sasaki and slow-roll paths).
    """
    from ..config.defaults import DEFAULTS
    from .egb_background import integrate_with_pivot
    from .egb_perturbations import FullObservables

    if N_pivot is None:
        N_pivot = DEFAULTS.N_pivot
    if phi_range is None:
        phi_range = DEFAULTS.phi_range

    if traj is None:
        traj = integrate_with_pivot(model, N_pivot=N_pivot, phi_range=phi_range)
    if traj is None or traj.N_end <= N_pivot:
        return None

    N_star = traj.N_end - N_pivot
    pivot = traj.at_N(N_star)
    k_pivot = pivot["a"] * pivot["H"]
    if not (np.isfinite(k_pivot) and k_pivot > 0):
        return None

    ag = _analytic_grids(model, traj)
    flow_S = reduce_sector(model, traj, "scalar", ag=ag)
    flow_T = reduce_sector(model, traj, "tensor", ag=ag)
    if flow_S is None or flow_T is None:
        return None

    S = _sector_spectrum(flow_S, k_pivot)
    T = _sector_spectrum(flow_T, k_pivot)
    if S is None or T is None:
        return None

    P_S = S["P"]
    P_T = 8.0 * T["P"]
    if not (np.isfinite(P_S) and np.isfinite(P_T) and P_S > 0 and P_T > 0):
        return None

    n_s = 1.0 + S["n"]
    n_T = T["n"]
    alpha_s = S["alpha"]
    r = P_T / P_S

    egb_cons = (r / (-8.0 * n_T)
                if (np.isfinite(n_T) and n_T != 0.0) else float("nan"))

    return FullObservables(
        n_s=float(n_s), n_T=float(n_T), r=float(r), alpha_s=float(alpha_s),
        P_S=float(P_S), P_T=float(P_T),
        c_T2=float(T["c2"]), c_S2=float(S["c2"]),
        epsilon=float(pivot["eps1"]), delta1=float(pivot["delta1"]),
        H_pivot=float(pivot["H"]), phi_N=float(pivot["phi"]),
        phi_end=float(traj.phi_end),
        N_pivot=float(N_pivot), egb_consistency=float(egb_cons),
    )
