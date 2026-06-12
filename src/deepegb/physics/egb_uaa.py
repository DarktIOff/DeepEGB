"""
Third-order uniform-asymptotic-approximation (UAA) observables for EGB
inflation — CROSS-CHECK PATH ONLY.

The production analytic path is `egb_n3lo.py` (Green's-function N3LO with
exact slow-roll coefficients).  The UAA expansion implemented here carries
an irreducible method residual — the overall 181/(36e³) ≈ 0.25039
normalisation (≈0.15% high) and ln3-type constants in place of the exact
C = γ_E + ln2 − 2 — which does NOT shrink as more slow-roll orders are
added.  Use it to cross-validate the WZW literature results, not for
parameter inference.

This module evaluates the primordial observables with closed-form
expressions that are exact through **second order in the slow-roll
expansion of the spectra** (third order for the spectral indices and
runnings), derived with the third-order uniform asymptotic approximation
by Wu, Zhu & Wang, arXiv:1707.08020 (Phys. Rev. D 96, 103515), for the
action (M_pl = 1)

    S = ∫ d⁴x √(-g) [ R/2 − ½(∂φ)² − V(φ) − ½ ξ(φ) 𝒢 ].

Unlike the leading-order slow-roll fallback in `egb_perturbations.py`,
the slow-roll hierarchies here are computed **exactly from the full
numerical background trajectory** (Friedmann + Klein-Gordon, see
`egb_background.py`), not from the V/Q slow-roll truncation:

    ε₁ = −Ḣ/H²,        ε_{n+1} = d ln ε_n / dN,
    δ₁ = 4 ξ̇ H,        δ_{n+1} = d ln δ_n / dN.

Sound speeds are the exact background expressions (WZW Eqs. 2.9–2.12):

    c_S² = 1 + [8 δ ξ̇ H Ḣ + 2 δ² H² (ξ̈ − ξ̇ H)] / (φ̇² + 6 δ ξ̇ H³),
           δ ≡ δ₁ / (1 − δ₁),
    c_T² = 1 − 4 (ξ̈ − ξ̇ H) / (1 − δ₁).

Observables (WZW §III.4, "Expressions at Horizon Crossing"). All
quantities are evaluated at the *later* of the scalar/tensor sound-
horizon crossings; the two branches (c_S > c_T and c_S < c_T) differ in
a handful of second-order coefficients and are both implemented.
Through second order both spectra carry the overall UAA constant
181/(36 e³) ≈ 0.25039 (0.15 % above the exact de Sitter normalisation —
the known, k-independent residual of the third-order UAA; it cancels
identically in n_s, n_T, α_s, α_t and r).

Conventions: P_S ≡ Δ²_R and P_T ≡ 8 Δ²_h (so that P_T → 2H²/π² and
r = P_T/P_S → 16 ε₁ in the GR limit), matching `egb_perturbations.py`.

Note: every appearance of ε₃ (δ₃) in the formulas is in the product
ε₂ε₃ = dε₂/dN (δ₂δ₃ = dδ₂/dN), which we compute directly — these
products stay finite where ε₂ or δ₂ cross zero, while ε₃, δ₃ alone
diverge harmlessly.

References
----------
* Wu Q., Zhu T., Wang A., *Primordial Spectra of slow-roll inflation at
  second-order with the Gauss-Bonnet correction*, arXiv:1707.08020. — WZW
* Zhu T. et al., arXiv:1308.5708, 1407.8011 — third-order UAA method.
* Li T.-C., Zhu T., Wang A., arXiv:2212.08253 — same method in 4D-EGB.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .egb_background import BackgroundTrajectory
from .egb_slow_roll import EGBModel

LN3 = float(np.log(3.0))
PI2 = float(np.pi ** 2)
# Third-order UAA normalisation constant 181/(36 e³).
UAA_AMP = 181.0 / (36.0 * float(np.e) ** 3)

# δ₁ below this is treated as the exact GR limit (δ-hierarchy ≡ 0).
_GR_DELTA1_FLOOR = 1.0e-14


# ---------------------------------------------------------------------------
# Slow-roll hierarchy, exact from the background trajectory
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SlowRollHierarchy:
    """ε/δ hierarchies + exact sound speeds at one trajectory point.

    e2e3 ≡ ε₂ε₃ = dε₂/dN and d2d3 ≡ δ₂δ₃ = dδ₂/dN are stored as
    products (see module docstring).
    """

    N: float
    phi: float
    H: float
    eps1: float
    eps2: float
    e2e3: float
    delta1: float
    delta2: float
    d2d3: float
    c_S2: float
    c_T2: float

    @property
    def is_valid(self) -> bool:
        return bool(np.all(np.isfinite([
            self.H, self.eps1, self.eps2, self.e2e3,
            self.delta1, self.delta2, self.d2d3,
        ])) and self.H > 0 and self.eps1 > 0)


def hierarchy_grids(traj: BackgroundTrajectory) -> dict[str, np.ndarray]:
    """Compute the slow-roll hierarchies and exact sound speeds on the
    uniform N grid of a full background trajectory.

    Everything is derived from (N, φ, π, H, ε₁, δ₁) — the trajectory of
    the *full* Friedmann–Klein-Gordon system — via d/dN derivatives, so
    no slow-roll truncation of the background enters.
    """
    N = traj.N
    dN = float(N[1] - N[0])
    H = traj.H
    eps1 = traj.eps1
    delta1 = traj.delta1

    ln_e1 = np.log(np.maximum(np.abs(eps1), 1.0e-300))
    eps2 = np.gradient(ln_e1, dN, edge_order=2)
    e2e3 = np.gradient(eps2, dN, edge_order=2)

    if np.nanmax(np.abs(delta1)) < _GR_DELTA1_FLOOR:
        delta2 = np.zeros_like(N)
        d2d3 = np.zeros_like(N)
    else:
        ln_d1 = np.log(np.maximum(np.abs(delta1), 1.0e-300))
        delta2 = np.gradient(ln_d1, dN, edge_order=2)
        d2d3 = np.gradient(delta2, dN, edge_order=2)

    # Exact sound speeds (WZW Eqs. 2.9, 2.12), all from trajectory data:
    #   ξ̇ = δ₁/(4H),  ξ̈ = H dξ̇/dN,  φ̇ = πH,  Ḣ = −ε₁H².
    xidot = delta1 / (4.0 * np.maximum(H, 1.0e-300))
    xiddot = H * np.gradient(xidot, dN, edge_order=2)
    phidot = traj.pi * H
    Hdot = -eps1 * H * H

    one_m_d1 = 1.0 - delta1
    with np.errstate(divide="ignore", invalid="ignore"):
        dl = np.where(np.abs(one_m_d1) > 1.0e-12, delta1 / one_m_d1, np.nan)
        num_s = 8.0 * dl * xidot * H * Hdot \
            + 2.0 * dl * dl * H * H * (xiddot - xidot * H)
        den_s = phidot * phidot + 6.0 * dl * xidot * H ** 3
        c_S2 = np.where(np.abs(den_s) > 0, 1.0 + num_s / den_s, np.nan)
        c_T2 = np.where(np.abs(one_m_d1) > 1.0e-12,
                        1.0 - 4.0 * (xiddot - xidot * H) / one_m_d1, np.nan)

    return dict(N=N, phi=traj.phi, H=H,
                eps1=eps1, eps2=eps2, e2e3=e2e3,
                delta1=delta1, delta2=delta2, d2d3=d2d3,
                c_S2=c_S2, c_T2=c_T2)


def hierarchy_at(traj: BackgroundTrajectory, N_eval: float,
                 grids: dict[str, np.ndarray] | None = None) -> SlowRollHierarchy:
    """Interpolate the slow-roll hierarchy at absolute e-fold N_eval."""
    g = grids if grids is not None else hierarchy_grids(traj)
    vals = {k: float(np.interp(N_eval, g["N"], g[k]))
            for k in ("phi", "H", "eps1", "eps2", "e2e3",
                      "delta1", "delta2", "d2d3", "c_S2", "c_T2")}
    return SlowRollHierarchy(N=float(N_eval), **vals)


# ---------------------------------------------------------------------------
# Horizon-crossing observables (WZW §III.4)
# ---------------------------------------------------------------------------
def uaa_observables(sr: SlowRollHierarchy) -> dict[str, float]:
    """Evaluate the third-order UAA horizon-crossing formulas.

    Returns dict with keys P_S, P_T, n_s, n_T, alpha_s, alpha_t, r,
    c_S2, c_T2.  Spectra are exact through O(ε²); indices and runnings
    through O(ε³).  Branches on sign(c_S − c_T) per WZW §III.4.
    """
    nan = float("nan")
    out = dict(P_S=nan, P_T=nan, n_s=nan, n_T=nan,
               alpha_s=nan, alpha_t=nan, r=nan,
               c_S2=sr.c_S2, c_T2=sr.c_T2)
    if not sr.is_valid:
        return out

    e1, e2, e23 = sr.eps1, sr.eps2, sr.e2e3
    d1, d2, d23 = sr.delta1, sr.delta2, sr.d2d3
    H2 = sr.H * sr.H
    L = LN3
    L2 = L * L

    x = 2.0 * e1 - d1
    if abs(x) < 1.0e-300:
        return out
    # Scalar mode crosses later (case A) or earlier (case B)?
    case_A = True
    if np.isfinite(sr.c_S2) and np.isfinite(sr.c_T2) \
            and sr.c_S2 > 0 and sr.c_T2 > 0:
        case_A = np.sqrt(sr.c_S2) >= np.sqrt(sr.c_T2)

    # ---- scalar spectrum  Δ²_R = UAA_AMP H²/(π² x) {1 + 2ln3 ε₁
    #                              + B1/x + B2/x²} ----
    B1 = (-0.5 * d1 * d1
          - (114.0 / 181.0 + L) * d1 * d2
          + (315.0 / 181.0) * d1 * e1
          - (992.0 / 181.0) * e1 * e1
          - (134.0 / 181.0 - 2.0 * L) * e1 * e2)

    # Branch-dependent second-order coefficients.
    if case_A:
        c_d1sq_e1sq = 1603.0 / 543.0 + 2.0 * L2 - 630.0 * L / 181.0
        c_d2_d1sq_e1 = -1775.0 / 543.0 + 2.0 * L2 + 456.0 * L / 181.0
        c_d1sq_e1_e2 = PI2 / 12.0 - 2827.0 / 1629.0 - L2 - 47.0 * L / 181.0
        c_d1_e1cu = 908.0 / 543.0 - 8.0 * L2 + 1796.0 * L / 181.0
        c_d1cu_e1 = L - 677.0 / 362.0
        c_d1_e1sq_e2 = -PI2 / 3.0 + 17854.0 / 1629.0 - 8.0 * L
        c_d2_d1cu = 2.0
    else:
        c_d1sq_e1sq = -569.0 / 543.0 + 2.0 * L2 - 630.0 * L / 181.0
        c_d2_d1sq_e1 = -2318.0 / 543.0 + 2.0 * L2 + 456.0 * L / 181.0
        c_d1sq_e1_e2 = -4456.0 / 1629.0 + PI2 / 12.0 - L2 - 47.0 * L / 181.0
        c_d1_e1cu = 3080.0 / 543.0 - 8.0 * L2 + 1796.0 * L / 181.0
        c_d1cu_e1 = L - 315.0 / 362.0
        c_d1_e1sq_e2 = 21112.0 / 1629.0 - PI2 / 3.0 - 8.0 * L
        c_d2_d1cu = 2.5

    B2 = (
        d1 * d1 * e1 * e1 * c_d1sq_e1sq
        + d2 * d1 * d1 * e1 * c_d2_d1sq_e1
        + d1 * d1 * e1 * e2 * c_d1sq_e1_e2
        + d1 * e1 ** 3 * c_d1_e1cu
        + d2 * d1 * e1 * e1 * (1942.0 / 543.0 - 4.0 * L2 + 536.0 * L / 181.0)
        + d1 * e1 * e2 * e2 * (-PI2 / 12.0 + 172.0 / 1629.0 + L2
                               - 134.0 * L / 181.0)
        + d2 * d2 * d1 * e1 * (-PI2 / 12.0 - 1034.0 / 1629.0 + L2
                               + 228.0 * L / 181.0)
        + d23 * d1 * e1 * (-PI2 / 12.0 - 1034.0 / 1629.0 + L2
                           + 228.0 * L / 181.0)
        + d2 * d1 * e1 * e2 * (PI2 / 6.0 + 94.0 / 1629.0 - 4.0 * L2
                               - 188.0 * L / 181.0)
        + d1 * e1 * e23 * (-PI2 / 12.0 + 172.0 / 1629.0 + L2
                           - 134.0 * L / 181.0)
        + d1 ** 3 * e1 * c_d1cu_e1
        + d1 * e1 * e1 * e2 * c_d1_e1sq_e2
        + 0.75 * d1 ** 4
        + c_d2_d1cu * d2 * d1 ** 3
        + d2 * d2 * d1 * d1 * (1013.0 / 1086.0 + 0.5 * L2
                               + 114.0 * L / 181.0)
        + d23 * d1 * d1 * (PI2 / 24.0 + 517.0 / 1629.0 - 0.5 * L2
                           - 114.0 * L / 181.0)
        + e1 ** 4 * (2068.0 / 543.0 + 8.0 * L2 - 2520.0 * L / 181.0)
        + e1 * e1 * e2 * e2 * (658.0 / 543.0 + 2.0 * L2
                               - 268.0 * L / 181.0)
        + e1 ** 3 * e2 * (PI2 / 3.0 - 14752.0 / 1629.0 + 4.0 * L2
                          + 188.0 * L / 181.0)
        + e1 * e1 * e23 * (PI2 / 6.0 - 344.0 / 1629.0 - 2.0 * L2
                           + 268.0 * L / 181.0)
    )

    # Δ²_R; |x| keeps P_S > 0 in the strongly-GB-dominated regime δ₁ > 2ε₁.
    P_S = UAA_AMP * H2 / (np.pi ** 2 * abs(x)) * (
        1.0 + 2.0 * L * e1 + B1 / x + B2 / (x * x))

    # ---- tensor spectrum  P_T = 8 Δ²_h ----
    c_d1_e1_T = (677.0 / 362.0 - L) if case_A else (1039.0 / 362.0 - L)
    P_T = 8.0 * UAA_AMP * H2 / np.pi ** 2 * (
        1.0 - 0.5 * d1
        + (-496.0 / 181.0 + 2.0 * L) * e1
        - d1 * d1 / 8.0
        + (67.0 / 181.0 + 0.5 * L) * d1 * d2
        + c_d1_e1_T * d1 * e1
        + (517.0 / 543.0 - 630.0 * L / 181.0 + 2.0 * L2) * e1 * e1
        + (-4636.0 / 1629.0 + PI2 / 12.0 + 496.0 * L / 181.0 - L2) * e1 * e2)

    # ---- spectral indices and runnings (same in both branches) ----
    ns_b2 = (
        8.0 * d1 * e1 ** 3
        - 2.0 * d1 * d1 * e1 * e1
        - 3.0 * d1 * d1 * d2 * e1
        + d1 * e2 * e1 * e1 * (350.0 / 27.0 - 8.0 * L)
        + d1 * d2 * d2 * e1 * (-34.0 / 27.0 - 2.0 * L)
        + d1 * e2 * e2 * e1 * (20.0 / 27.0 - 2.0 * L)
        + d1 * d23 * e1 * (-34.0 / 27.0 - 2.0 * L)
        + d1 * d1 * e2 * e1 * (2.0 * L - 20.0 / 27.0)
        + d1 * d2 * e2 * e1 * (14.0 / 27.0 + 4.0 * L)
        + d1 * e23 * e1 * (20.0 / 27.0 - 2.0 * L)
        + 0.5 * d1 ** 3 * d2
        + d1 * d1 * d23 * (17.0 / 27.0 + L)
        - 8.0 * e1 ** 4
        + e2 * e1 ** 3 * (8.0 * L - 404.0 / 27.0)
        + e23 * e1 * e1 * (4.0 * L - 40.0 / 27.0)
    )
    n_s = float(1.0 - 2.0 * e1
                + (d1 * d2 - 2.0 * e2 * e1) / x
                + ns_b2 / (x * x))

    alpha_s = float((
        8.0 * d1 * e2 * e1 * e1
        + 2.0 * d1 * d2 * d2 * e1
        + 2.0 * d1 * e2 * e2 * e1
        + 2.0 * d1 * d23 * e1
        - 2.0 * d1 * d1 * e2 * e1
        - 4.0 * d1 * d2 * e2 * e1
        + 2.0 * d1 * e23 * e1
        - d1 * d1 * d23
        - 8.0 * e2 * e1 ** 3
        - 4.0 * e23 * e1 * e1
    ) / (x * x))

    n_T = float(-2.0 * e1 - 2.0 * e1 * e1 - 0.5 * d1 * d2
                + (-74.0 / 27.0 + 2.0 * L) * e1 * e2)
    alpha_t = float(-2.0 * e1 * e2)

    # r from the truncated spectra (matches the repo definition r = P_T/P_S
    # and agrees with WZW Eq. 3.45 through second order).
    r = float(P_T / P_S) if (np.isfinite(P_S) and P_S > 0) else nan

    out.update(P_S=float(P_S), P_T=float(P_T), n_s=n_s, n_T=n_T,
               alpha_s=alpha_s, alpha_t=alpha_t, r=r)
    return out


# ---------------------------------------------------------------------------
# Production entry point
# ---------------------------------------------------------------------------
def compute_observables_uaa(
    model: EGBModel,
    N_pivot: float | None = None,
    phi_range: tuple[float, float] | None = None,
    *,
    traj: BackgroundTrajectory | None = None,
):
    """Analytic third-order-UAA observables on the full background.

    1. Integrate the full Friedmann–Klein-Gordon background
       (`integrate_with_pivot`) unless a trajectory is supplied.
    2. Build the ε/δ hierarchies and exact c_S², c_T² from the
       trajectory (no slow-roll truncation of the background).
    3. Evaluate the WZW horizon-crossing formulas at
       N* = N_end − N_pivot.

    Returns a `FullObservables`, or None if the background integration
    fails or the hierarchy is invalid at the pivot (caller falls back
    to Mukhanov–Sasaki / slow-roll paths).
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
    sr = hierarchy_at(traj, N_star)
    if not sr.is_valid:
        return None

    obs = uaa_observables(sr)
    if not all(np.isfinite(obs[k]) for k in ("P_S", "P_T", "n_s", "n_T", "r")):
        return None
    if obs["P_S"] <= 0 or obs["P_T"] <= 0:
        return None

    egb_cons = (obs["r"] / (-8.0 * obs["n_T"])
                if (np.isfinite(obs["n_T"]) and obs["n_T"] != 0.0)
                else float("nan"))

    return FullObservables(
        n_s=obs["n_s"], n_T=obs["n_T"], r=obs["r"], alpha_s=obs["alpha_s"],
        P_S=obs["P_S"], P_T=obs["P_T"],
        c_T2=obs["c_T2"], c_S2=obs["c_S2"],
        epsilon=sr.eps1, delta1=sr.delta1,
        H_pivot=sr.H, phi_N=sr.phi, phi_end=float(traj.phi_end),
        N_pivot=float(N_pivot), egb_consistency=float(egb_cons),
    )
