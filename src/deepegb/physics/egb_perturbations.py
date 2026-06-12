"""
Production-grade EGB inflation observables.

This module provides the main observable-computation pipeline for EGB
inflation.  Starting with the v2 refactor, **the primary (exact) path**
solves the full Friedmann–Klein–Gordon ODE system to obtain the background
trajectory, then computes tensor and scalar power spectra via Mukhanov–
Sasaki mode integration.  Spectral indices and running are derived from
finite differences of the mode-computed P(k) in ln k.

If the full-background integration fails (stiff ODE, no trajectory, etc.),
a **fallback (slow-roll)** path computes observables from the slow-roll
truncation of the background using V/Q quadrature for the pivot location
and numerical N-derivatives for spectral indices.

Primary (exact) path
--------------------
  1. `integrate_with_pivot` solves the full ODE for the background
     trajectory (Friedmann constraint + Klein–Gordon equation).
  2. `k_pivot_from_traj` locates the comoving pivot wavenumber.
  3. `tensor_power_spectrum` and `scalar_power_spectrum` integrate the
     canonical Mukhanov–Sasaki mode equation for each k.
  4. n_s, n_T, α_s are computed from finite differences of ln P in ln k
     at k = k_* · exp(±dlnk).

Fallback (slow-roll) path
-------------------------
  * The GB-corrected slow-roll parameters ε₁, η, δ_1, δ_2.
  * The tensor sound speed  c_T²(φ) = F_T/G_T  with the explicit forms
    F_T = M_pl² (1 − 4 ξ̈/M_pl²),   G_T = M_pl² (1 − 4 ξ̇H/M_pl²)
    (Hwang & Noh 2005; Kawai & Soda 1999).
  * The tensor power spectrum  P_T = 2 H² / [π² M_pl² (1−δ_1) c_T³]
    evaluated at sound-horizon crossing (k* = a*H*/c_T*).
  * The scalar power spectrum  P_S = H²/(8 π² M_pl² ε)  with the GB-
    corrected ε (Koh-Lee-Tumurtushaa 2014; Yi-Gong-Sabir 2018).
  * Spectral indices n_s, n_T computed as **numerical N-derivatives** of
    ln P_S, ln P_T along the inflationary trajectory.
  * The running α_s = dn_s/dlnk.
  * The tensor-to-scalar ratio r = P_T/P_S (no longer simply 16ε).

Conventions
-----------
Action (M_pl = 1 throughout):

    S = ∫ d⁴x √(-g) [ R/2 − ½ (∂φ)² − V(φ) − ½ ξ(φ) 𝒢 ]

with 𝒢 = R² − 4 R_{μν}R^{μν} + R_{μνρσ}R^{μνρσ}.

Slow-roll truncation of the background:
    H² ≈ V/3,   3 H φ̇ ≈ −Q,   Q ≡ V_,φ + (4/3) V² ξ_,φ
    ε(φ) ≡ −Ḣ/H² ≈ Q V_,φ /(2 V²)
    δ₁(φ) ≡ 4 ξ̇ H = −(4/3) ξ_,φ Q                (KLT 2014, Eq. 9)
    ε₂ ≡ d ln ε / dN,   δ₂ ≡ d ln |δ₁| / dN       (running parameters)

Tensor and scalar perturbation forms:
    G_T = 1 − δ₁,                 (Hwang-Noh 2005, Eq. 95)
    F_T = 1 − δ₁(δ₂ + ε),         (using ξ̈ = (H δ₁ /4)(δ₂ + ε))
    c_T² = F_T / G_T

The leading-order scalar sound speed in single-field EGB is
c_S² = 1 + O(slow-roll²); we therefore set c_S² = 1 for the MVP and
flag this at the function level. Replace `_compute_c_S2` with the full
Hwang-Noh expression when targeting precision below 10⁻³ on n_s.

References
----------
* Hwang J., Noh H., *Cosmological perturbations in a generalized gravity
  including tachyonic condensation*, gr-qc/0507025  (2005).
* Kawai S., Soda J., *Evolution of fluctuations during graceful exit in
  string cosmology*, gr-qc/9901002 (1999).
* Koh S., Lee B.-H., Tumurtushaa G., *Reconstruction of the Scalar Field
  Potential in Inflationary Models with a Gauss-Bonnet term*,
  arXiv:1404.0027 (2014).        — denoted KLT
* Yi Z., Gong Y., Sabir M., *Inflation with Gauss-Bonnet coupling*,
  arXiv:1811.01580 (2018).        — denoted YGS
* Odintsov S.D., Oikonomou V.K., *Viable Inflation in Scalar-Gauss-Bonnet
  Gravity and Reconstruction from Observational Indices*,
  arXiv:1810.04645 (2018).        — denoted OO
* Mukhanov V.F., Feldman H.A., Brandenberger R.H., *Theory of cosmological
  perturbations*, Phys. Rep. 215 (1992) 203.   — MS mode integration
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Tuple

import numpy as np

from .egb_slow_roll import EGBModel, end_of_inflation, _N_to_phi_table

ArrayLike = np.ndarray | float


# ---------------------------------------------------------------------------
# Background quantities along the trajectory (slow-roll truncation).
# ---------------------------------------------------------------------------
def _Vp(model: EGBModel, phi: float, h: float = 1.0e-4) -> float:
    return float((model.V(phi + h) - model.V(phi - h)) / (2 * h))


def _Vpp(model: EGBModel, phi: float, h: float = 1.0e-4) -> float:
    return float((model.V(phi + h) - 2 * model.V(phi) + model.V(phi - h)) / (h * h))


def _xip(model: EGBModel, phi: float, h: float = 1.0e-4) -> float:
    return float((model.xi(phi + h) - model.xi(phi - h)) / (2 * h))


def _xipp(model: EGBModel, phi: float, h: float = 1.0e-4) -> float:
    return float((model.xi(phi + h) - 2 * model.xi(phi) + model.xi(phi - h)) / (h * h))


def background_at(model: EGBModel, phi: float) -> dict[str, float]:
    """Return slow-roll background quantities at φ (M_pl=1).

    Keys: V, Vp, Vpp, xi, xip, xipp, H, phidot, xidot, Q, eps, delta1.
    """
    V = float(model.V(phi))
    Vp = _Vp(model, phi)
    Vpp = _Vpp(model, phi)
    xip = _xip(model, phi)
    xipp = _xipp(model, phi)
    if V <= 0 or not np.isfinite(V):
        return dict(V=V, Vp=Vp, Vpp=Vpp, xi=float(model.xi(phi)),
                    xip=xip, xipp=xipp,
                    H=np.nan, phidot=np.nan, xidot=np.nan,
                    Q=np.nan, eps=np.nan, delta1=np.nan)

    H = np.sqrt(V / 3.0)
    Q = Vp + (4.0 / 3.0) * V * V * xip                    # Q = V_,φ + (4/3) V² ξ_,φ
    phidot = -Q / (3.0 * H)                                # KG, slow-roll
    xidot = xip * phidot                                   # ξ̇ = ξ_,φ · φ̇
    eps = 0.5 * Q * Vp / (V * V)                          # KLT Eq. 12
    delta1 = 4.0 * xidot * H                              # δ₁ = 4 ξ̇ H, KLT Eq. 9
    return dict(V=V, Vp=Vp, Vpp=Vpp, xi=float(model.xi(phi)),
                xip=xip, xipp=xipp,
                H=H, phidot=phidot, xidot=xidot,
                Q=Q, eps=eps, delta1=delta1)


def background_along(
    model: EGBModel,
    phi_grid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Vectorised background quantities along a grid of φ values."""
    out = {k: np.full_like(phi_grid, np.nan, dtype=float)
           for k in ("V", "Vp", "Vpp", "xi", "xip", "xipp",
                     "H", "phidot", "xidot", "Q", "eps", "delta1")}
    for i, p in enumerate(phi_grid):
        bg = background_at(model, float(p))
        for k in out:
            out[k][i] = bg[k]
    return out


# ---------------------------------------------------------------------------
# Sound speeds
# ---------------------------------------------------------------------------
def compute_c_T2(
    model: EGBModel,
    phi: float,
    eps: float | None = None,
    delta1: float | None = None,
) -> float:
    """Tensor sound speed squared.

    From the EGB tensor action [Hwang-Noh 2005, Eq. 95]:

        G_T = M_pl²(1 − δ₁)
        F_T = M_pl²(1 − 4 ξ̈ / M_pl²)
        c_T² = F_T / G_T

    Using ξ̈ = (H δ₁ / 4)(δ₂ + ε) so that 4 ξ̈/M_pl² = δ₁(δ₂ + ε) [Eq. (B.5)
    of YGS 2018, equivalently differentiating ξ̇ = δ₁/(4H) by t]:

        c_T² = (1 − δ₁(δ₂ + ε)) / (1 − δ₁)

    To compute δ₂ we differentiate δ₁(φ) along the slow-roll trajectory.
    """
    bg = background_at(model, phi)
    if not np.isfinite(bg["delta1"]):
        return np.nan
    eps = bg["eps"] if eps is None else eps
    d1 = bg["delta1"] if delta1 is None else delta1

    # In the GR limit δ₁ → 0 (i.e. ξ' = 0), c_T² → 1 exactly because both
    # F_T and G_T reduce to M_pl². Handle separately to avoid 0·∞ from δ₂.
    if abs(d1) < 1.0e-14:
        return 1.0

    # δ₂ = d ln|δ₁|/dN  with dN = -V/Q dφ
    delta2 = _running_dlnX_dN(model, phi, "delta1")
    if not np.isfinite(delta2):
        return np.nan

    F_T = 1.0 - d1 * (delta2 + eps)
    G_T = 1.0 - d1
    if abs(G_T) < 1e-12:
        return np.nan
    return F_T / G_T


def compute_c_S2(model: EGBModel, phi: float, *,
                  eps: float | None = None,
                  delta1: float | None = None,
                  delta2: float | None = None,
                  H: float | None = None,
                  xip: float | None = None) -> float:
    """Scalar sound speed squared — EXACT expression in flow variables.

    From the exact EGB scalar action (Hwang-Noh 2005; Wu-Zhu-Wang 2017,
    arXiv:1707.08020 Eq. 2.9), with δ̄ ≡ δ₁/(1−δ₁) and M_pl = 1:

        c_S² = 1 + [8 δ̄ ξ̇ H Ḣ + 2 δ̄² H² (ξ̈ − ξ̇H)] / (φ̇² + 6 δ̄ ξ̇ H³).

    Using ξ̇H = δ₁/4, ξ̈ = (δ₁/4)(δ₂+ε₁)·H⁰, Ḣ = −ε₁H² and the exact
    background identity φ̇²/H² = 2ε₁ − δ₁ − δ₁ε₁ + δ₁δ₂ this becomes

        c_S² = 1 + δ₁·[ −2 δ̄ ε₁ + (δ̄²/2)(δ₂ + ε₁ − 1) ]
                   / [ 2ε₁ − δ₁ − δ₁ε₁ + δ₁δ₂ + (3/2) δ̄ δ₁ ].

    The deviation from 1 is O(δ₁²/ε₁, δ₁ ε₁) — the previous approximate
    form (∝ ξ_,φ² H²/ε₁) overestimated it by O(1/ε₁) for steep ξ(φ).

    (H and xip are accepted for backwards compatibility and unused.)
    """
    bg = None
    if eps is None or delta1 is None:
        bg = background_at(model, phi)
        if not (np.isfinite(bg["eps"]) and bg["eps"] > 0
                and np.isfinite(bg["delta1"])):
            return np.nan
        eps = bg["eps"] if eps is None else eps
        delta1 = bg["delta1"] if delta1 is None else delta1
    if delta2 is None:
        if abs(delta1) < 1.0e-14:
            return 1.0
        delta2 = _running_dlnX_dN(model, phi, "delta1")
        if not np.isfinite(delta2):
            return np.nan
    one_minus_d1 = 1.0 - delta1
    if abs(one_minus_d1) < 1.0e-12:
        return np.nan
    dbar = delta1 / one_minus_d1
    den = (2.0 * eps - delta1 - delta1 * eps + delta1 * delta2
           + 1.5 * dbar * delta1)
    if not np.isfinite(den) or den <= 0:
        return np.nan
    num = delta1 * (-2.0 * dbar * eps
                    + 0.5 * dbar * dbar * (delta2 + eps - 1.0))
    return float(1.0 + num / den)


# ---------------------------------------------------------------------------
# Power spectra at sound-horizon crossing
# ---------------------------------------------------------------------------
def power_spectra_at(model: EGBModel, phi: float) -> dict[str, float]:
    """Return P_S, P_T, c_T², c_S², n_s, n_T at field value φ.

    Amplitudes (M_pl = 1, evaluated at sound-horizon crossing):
        P_S = H² / (8 π² ε c_S)                          [KLT 2014, generalised]
        P_T = 2 H² / [π² (1 − δ₁) c_T³]                  [HN 2005]

    The factor of 1/c_S in P_S accounts for the scalar sound horizon,
    k_*^scalar = a H / c_S; in the GR limit c_S → 1 and we recover the
    textbook KLT formula (PS).
    """
    bg = background_at(model, phi)
    if not (np.isfinite(bg["H"]) and np.isfinite(bg["eps"]) and bg["eps"] > 0):
        return dict(P_S=np.nan, P_T=np.nan, c_T2=np.nan, c_S2=np.nan,
                    n_s=np.nan, n_T=np.nan)
    cT2 = compute_c_T2(model, phi, eps=bg["eps"], delta1=bg["delta1"])
    cS2 = compute_c_S2(model, phi)
    H = bg["H"]
    eps = bg["eps"]
    delta1 = bg["delta1"]

    if np.isfinite(cS2) and cS2 > 0:
        cS = np.sqrt(cS2)
        P_S = (H * H) / (8.0 * np.pi**2 * eps * cS)
    else:
        P_S = np.nan
    if np.isfinite(cT2) and cT2 > 0 and (1.0 - delta1) > 0:
        cT = np.sqrt(cT2)
        P_T = 2.0 * H * H / (np.pi**2 * (1.0 - delta1) * cT**3)
    else:
        P_T = np.nan

    return dict(P_S=P_S, P_T=P_T, c_T2=cT2, c_S2=cS2, n_s=np.nan, n_T=np.nan)


# ---------------------------------------------------------------------------
# Numerical helpers: dlnX/dN along the slow-roll trajectory
# ---------------------------------------------------------------------------
def _running_dlnX_dN(model: EGBModel, phi: float, key: str,
                      dphi: float = 5.0e-3) -> float:
    """Compute d ln|X|/dN at φ, where X is one of the keys returned by
    background_at, and dN = −V/Q dφ along the slow-roll trajectory."""
    bg0 = background_at(model, phi)
    if not np.isfinite(bg0["Q"]) or bg0["Q"] == 0:
        return np.nan
    bg_p = background_at(model, phi + dphi)
    bg_m = background_at(model, phi - dphi)
    X0 = bg0[key]
    Xp = bg_p[key]
    Xm = bg_m[key]
    if not (np.isfinite(X0) and np.isfinite(Xp) and np.isfinite(Xm)):
        return np.nan
    if X0 == 0:
        return np.nan
    dX_dphi = (Xp - Xm) / (2.0 * dphi)
    # dN/dφ = -V/Q   ⇒  dφ/dN = -Q/V
    dphi_dN = -bg0["Q"] / bg0["V"]
    return (dX_dphi * dphi_dN) / X0


def _running_dphi(model: EGBModel, phi: float, key: str,
                   dphi: float = 5.0e-3) -> float:
    """Compute dX/dφ at φ via central differences, where X is a key
    returned by background_at."""
    bg_p = background_at(model, phi + dphi)
    bg_m = background_at(model, phi - dphi)
    Xp = bg_p[key]
    Xm = bg_m[key]
    if not (np.isfinite(Xp) and np.isfinite(Xm)):
        return np.nan
    return (Xp - Xm) / (2.0 * dphi)


# ---------------------------------------------------------------------------
# Full observables: solve trajectory, evaluate at horizon crossing,
# differentiate ln P_S, ln P_T numerically to get n_s, n_T, α_s.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FullObservables:
    """Production-grade observables for an EGB inflation model."""

    n_s: float
    n_T: float
    r: float
    alpha_s: float                   # dn_s / dlnk
    P_S: float
    P_T: float
    c_T2: float
    c_S2: float
    epsilon: float
    delta1: float
    H_pivot: float
    phi_N: float
    phi_end: float
    N_pivot: float
    egb_consistency: float = float("nan")  # r/(-8nT); EGB-aware metric

    @property
    def is_valid(self) -> bool:
        return all(np.isfinite([self.n_s, self.n_T, self.r,
                                self.P_S, self.P_T, self.c_T2,
                                self.epsilon, self.phi_N, self.phi_end]))

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}


def _bracket_phi_for_N(
    model: EGBModel,
    phi_end: float,
    N_target: float,
    phi_range: tuple[float, float],
    n_grid: int,
) -> float | None:
    """Find φ_N s.t. ∫_{φ_end}^{φ_N} V/Q dφ = N_target on the slow-roll side."""
    phi_grid, N_grid = _N_to_phi_table(model, phi_end, phi_range, n_grid)
    bg = background_along(model, phi_grid)
    side_mask = (bg["eps"] > 0) & (bg["eps"] < 1.0)

    diff = N_grid - N_target
    candidates: list[float] = []
    for i in range(len(phi_grid) - 1):
        a, b = diff[i], diff[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if not (side_mask[i] and side_mask[i + 1]):
            continue
        if a * b < 0:
            phi_root = phi_grid[i] - a * (phi_grid[i + 1] - phi_grid[i]) / (b - a)
            candidates.append(phi_root)
    if not candidates:
        return None
    candidates.sort(key=lambda p: abs(p - phi_end))
    return float(candidates[0])


# ---------------------------------------------------------------------------
# Primary path: full-background trajectory + Mukhanov–Sasaki mode integration
# ---------------------------------------------------------------------------
def _observables_from_trajectory(
    model: EGBModel,
    N_pivot: float,
    phi_range: tuple[float, float],
    dlnk: float,
) -> FullObservables | None:
    """Attempt exact observables from full-background + mode integration.

    This is the **primary (exact) path** for ``compute_observables_full``.
    It solves the full Friedmann–Klein–Gordon ODE to obtain the background
    trajectory via ``integrate_with_pivot``, then computes tensor and scalar
    power spectra using Mukhanov–Sasaki mode integration at the pivot scale
    and at k = k_* · exp(±dlnk).  Spectral indices n_s, n_T and running α_s
    are derived from finite differences of ln P in ln k.

    Returns ``None`` if the trajectory integration fails, the pivot point is
    not contained in the trajectory, or the mode spectra contain non-finite /
    non-positive values.  The caller should fall back to the slow-roll path.

    Parameters
    ----------
    model : EGBModel
        The inflation model (V, ξ, derivatives).
    N_pivot : float
        Number of e-folds before end of inflation at the pivot scale.
    phi_range : tuple[float, float]
        Field-value range for the background integration.
    dlnk : float
        Step size in ln k for finite-difference spectral indices.

    References
    ----------
    * Mukhanov, Feldman & Brandenberger, Phys. Rep. 215 (1992) 203.
    * Hwang & Noh, gr-qc/0507025 (2005).
    """
    # Local imports to avoid circular dependency:
    # egb_perturbations ← egb_modes ← egb_perturbations
    from .egb_background import integrate_with_pivot
    from .egb_modes import (k_pivot_from_traj, scalar_power_spectrum,
                             tensor_power_spectrum)

    traj = integrate_with_pivot(model, N_pivot=N_pivot, phi_range=phi_range)
    if traj is None or traj.N_end <= N_pivot:
        return None

    k_pivot = k_pivot_from_traj(traj, N_pivot=N_pivot)
    if not (np.isfinite(k_pivot) and k_pivot > 0):
        return None

    # k values for finite-difference spectral indices
    k_lo = k_pivot * np.exp(-dlnk)
    k_hi = k_pivot * np.exp(dlnk)
    k_arr = np.array([k_lo, k_pivot, k_hi])

    # Power spectra via Mukhanov–Sasaki mode integration
    P_T_arr, _ = tensor_power_spectrum(model, k_arr, traj=traj,
                                        N_pivot=N_pivot)
    P_S_arr, _ = scalar_power_spectrum(model, k_arr, traj=traj,
                                        N_pivot=N_pivot)

    if not (np.all(np.isfinite(P_T_arr)) and np.all(np.isfinite(P_S_arr))):
        return None
    if not (np.all(P_T_arr > 0) and np.all(P_S_arr > 0)):
        return None

    # Spectral indices from finite differences in ln k
    ln_PT = np.log(P_T_arr)
    ln_PS = np.log(P_S_arr)
    n_T = float((ln_PT[2] - ln_PT[0]) / (2.0 * dlnk))
    n_s = float(1.0 + (ln_PS[2] - ln_PS[0]) / (2.0 * dlnk))
    alpha_s = float(
        (ln_PS[2] - 2.0 * ln_PS[1] + ln_PS[0]) / (dlnk ** 2)
    )

    r = float(P_T_arr[1] / P_S_arr[1])

    # Background quantities from trajectory interpolation at pivot
    pivot_idx = int(np.argmin(np.abs(traj.N - (traj.N_end - N_pivot))))
    H_pivot = float(traj.H[pivot_idx])
    eps1_pivot = float(traj.eps1[pivot_idx])
    delta1_pivot = float(traj.delta1[pivot_idx])
    phi_pivot = float(traj.phi[pivot_idx])
    phi_end = float(traj.phi_end)

    # Sound speeds (slow-roll approximation for c_T², c_S² fields;
    # the exact physics is captured by the mode integrators)
    cT2 = compute_c_T2(model, phi_pivot, eps=eps1_pivot, delta1=delta1_pivot)
    cS2 = compute_c_S2(model, phi_pivot)

    # EGB consistency metric: r / (-8 n_T); deviates from 1 in EGB
    if np.isfinite(n_T) and n_T != 0.0:
        egb_cons = r / (-8.0 * n_T)
    else:
        egb_cons = float("nan")

    return FullObservables(
        n_s=n_s, n_T=n_T, r=r, alpha_s=alpha_s,
        P_S=float(P_S_arr[1]), P_T=float(P_T_arr[1]),
        c_T2=float(cT2) if np.isfinite(cT2) else float("nan"),
        c_S2=float(cS2) if np.isfinite(cS2) else float("nan"),
        epsilon=eps1_pivot, delta1=delta1_pivot,
        H_pivot=H_pivot, phi_N=phi_pivot, phi_end=phi_end,
        N_pivot=N_pivot, egb_consistency=egb_cons,
    )


def compute_observables_full(
    model: EGBModel,
    N_pivot: float | None = None,
    phi_range: tuple[float, float] | None = None,
    n_grid: int = 4001,
    dN_for_running: float = 0.5,
    dlnk: float = 0.5,
    method: str = "n3lo",
) -> FullObservables:
    """Compute production-grade observables for an EGB inflation model.

    Defaults for N_pivot and phi_range come from the centralized config
    (deepegb.config.defaults.DEFAULTS).

    Primary (analytic N3LO) path — method="n3lo" (default)
    ------------------------------------------------------
    Solve the full Friedmann–Klein–Gordon ODE for the background, reduce
    both perturbation sectors to canonical Mukhanov–Sasaki form exactly,
    and evaluate the Green's-function N3LO/N4LO closed-form spectra and
    indices — exact slow-roll coefficients (C = γ_E + ln2 − 2, π², ζ(3);
    Auclair & Ringeval 2022 master formulas + exact EGB sector mapping).
    See ``egb_n3lo.compute_observables_n3lo``.  Analytic n_s, n_T, α_s
    are accurate through third order in the flow parameters; c_S², c_T²
    are exact.  ~5× faster than mode integration.

    Secondary (Mukhanov–Sasaki) path — method="ms"
    ----------------------------------------------
    Solve the full Friedmann–Klein–Gordon ODE via ``integrate_with_pivot``
    to obtain the background trajectory.  Then compute P_T(k) and P_S(k)
    using Mukhanov–Sasaki mode integration at the pivot scale k_* and at
    k_* · exp(±dlnk).  Spectral indices n_s, n_T and running α_s are
    derived from finite differences of ln P in ln k.  Used as a
    cross-check of the analytic path and as its fallback.

    Fallback (slow-roll) path
    -------------------------
    If the full-background integration fails — stiff ODE, trajectory too
    short, mode spectra non-finite — the function falls back to the slow-
    roll closed-form computation:

      1. Find φ_end from ε(φ_end) = 1.
      2. Find φ_pivot N e-folds before end (from V/Q quadrature).
      3. Compute background quantities, c_T², P_S, P_T at φ_pivot.
      4. Compute n_s, n_T as numerical N-derivatives of ln P_S, ln P_T
         using a small ΔN step on either side of φ_pivot.
      5. Compute α_s = dn_s/dlnk likewise (second derivative of ln P_S).

    Parameters
    ----------
    dlnk : float
        Step size in ln k for finite-difference spectral indices in the
        mode-integration path.  Default 0.5.
    method : str
        "n3lo" (default): analytic N3LO first, then MS, then slow-roll.
        "ms": skip the analytic path and use mode integration directly.
    """
    from ..config.defaults import DEFAULTS
    if N_pivot is None:
        N_pivot = DEFAULTS.N_pivot
    if phi_range is None:
        phi_range = DEFAULTS.phi_range

    # ── Primary (analytic N3LO) path ──
    if method == "n3lo":
        try:
            from .egb_n3lo import compute_observables_n3lo
            primary = compute_observables_n3lo(
                model, N_pivot=N_pivot, phi_range=phi_range)
            if primary is not None:
                return primary
        except Exception:
            pass  # fall through to the MS path

    # ── Secondary path: full-background ODE + Mukhanov–Sasaki ──
    try:
        primary = _observables_from_trajectory(
            model, N_pivot, phi_range, dlnk)
        if primary is not None:
            return primary
    except Exception:
        pass  # fall through to slow-roll fallback

    # ── Fallback: slow-roll closed-form ──
    phi_end = end_of_inflation(model, phi_range=phi_range, n_grid=n_grid)
    if phi_end is None:
        return FullObservables(
            n_s=np.nan, n_T=np.nan, r=np.nan, alpha_s=np.nan,
            P_S=np.nan, P_T=np.nan, c_T2=np.nan, c_S2=np.nan,
            epsilon=np.nan, delta1=np.nan, H_pivot=np.nan,
            phi_N=np.nan, phi_end=np.nan, N_pivot=N_pivot,
            egb_consistency=np.nan,
        )

    phi_N = _bracket_phi_for_N(model, phi_end, N_pivot, phi_range, n_grid)
    if phi_N is None:
        return FullObservables(
            n_s=np.nan, n_T=np.nan, r=np.nan, alpha_s=np.nan,
            P_S=np.nan, P_T=np.nan, c_T2=np.nan, c_S2=np.nan,
            epsilon=np.nan, delta1=np.nan, H_pivot=np.nan,
            phi_N=np.nan, phi_end=phi_end, N_pivot=N_pivot,
            egb_consistency=np.nan,
        )

    # Background at pivot
    bg = background_at(model, phi_N)
    spec = power_spectra_at(model, phi_N)
    if not np.isfinite(spec["P_S"]) or not np.isfinite(spec["P_T"]):
        return FullObservables(np.nan, np.nan, np.nan, np.nan,
                               spec["P_S"], spec["P_T"],
                               spec["c_T2"], spec["c_S2"],
                               bg["eps"], bg["delta1"], bg["H"],
                               phi_N, phi_end, N_pivot,
                               egb_consistency=float("nan"))

    # Find φ values dN_for_running e-folds either side of pivot
    phi_lo = _bracket_phi_for_N(model, phi_end,
                                 N_pivot - dN_for_running, phi_range, n_grid)
    phi_hi = _bracket_phi_for_N(model, phi_end,
                                 N_pivot + dN_for_running, phi_range, n_grid)
    if phi_lo is None or phi_hi is None:
        eps = bg["eps"]
        d1 = bg["delta1"]
        deps_dphi = _running_dphi(model, phi_N, "eps")
        if np.isfinite(deps_dphi) and np.isfinite(bg["Q"]) and bg["Q"] != 0:
            n_s_lo = 1.0 - 2.0 * eps + (bg["Q"] / (eps * bg["V"])) * deps_dphi
        else:
            n_s_lo = np.nan
        dPT_dphi = _running_dphi(model, phi_N, "delta1")
        if np.isfinite(dPT_dphi) and np.isfinite(bg["Q"]) and bg["Q"] != 0:
            n_T_lo = -2.0 * eps + (bg["Q"] / (eps * bg["V"])) * dPT_dphi * eps
        else:
            n_T_lo = -2.0 * eps - d1
        r_val = spec["P_T"] / spec["P_S"]

        # EGB consistency in fallback branch
        if np.isfinite(n_T_lo) and n_T_lo != 0.0:
            egb_cons_lo = r_val / (-8.0 * n_T_lo)
        else:
            egb_cons_lo = float("nan")

        return FullObservables(
            n_s=n_s_lo, n_T=n_T_lo, r=r_val, alpha_s=np.nan,
            P_S=spec["P_S"], P_T=spec["P_T"],
            c_T2=spec["c_T2"], c_S2=spec["c_S2"],
            epsilon=eps, delta1=d1, H_pivot=bg["H"],
            phi_N=phi_N, phi_end=phi_end, N_pivot=N_pivot,
            egb_consistency=egb_cons_lo,
        )

    spec_lo = power_spectra_at(model, phi_lo)
    spec_hi = power_spectra_at(model, phi_hi)

    # Sign convention: N_pivot counts e-folds *backwards* from the end of
    # inflation, so larger N_pivot ⇒ EARLIER in inflation ⇒ SMALLER k at
    # horizon crossing. Hence  d ln k / d N_pivot = −1  at leading order,
    # and  d/d ln k = −d/d N_pivot.
    if (spec_lo["P_S"] > 0 and spec_hi["P_S"] > 0):
        d_logPS_dN = (np.log(spec_hi["P_S"]) - np.log(spec_lo["P_S"])) / (2 * dN_for_running)
        n_s_minus_1 = -d_logPS_dN
        n_s = 1.0 + n_s_minus_1
    else:
        n_s = np.nan

    if (spec_lo["P_T"] > 0 and spec_hi["P_T"] > 0):
        d_logPT_dN = (np.log(spec_hi["P_T"]) - np.log(spec_lo["P_T"])) / (2 * dN_for_running)
        n_T = -d_logPT_dN
    else:
        n_T = np.nan

    # α_s = d n_s / d ln k = +d²lnP_S / dN² (the sign flip squares to +1).
    if (spec_lo["P_S"] > 0 and spec_hi["P_S"] > 0):
        log_PS_pivot = np.log(spec["P_S"])
        log_PS_lo = np.log(spec_lo["P_S"])
        log_PS_hi = np.log(spec_hi["P_S"])
        alpha_s = (log_PS_hi - 2 * log_PS_pivot + log_PS_lo) / (dN_for_running**2)
    else:
        alpha_s = np.nan

    r_val = spec["P_T"] / spec["P_S"]

    # EGB-aware consistency metric: in GR, r = -8 n_T; in EGB this
    # is broken by δ₁ and c_T² corrections. We report the deviation
    # from unity as the EGB consistency ratio.
    if np.isfinite(n_T) and n_T != 0.0:
        egb_cons = r_val / (-8.0 * n_T)
    else:
        egb_cons = float("nan")

    return FullObservables(
        n_s=n_s, n_T=n_T, r=r_val, alpha_s=alpha_s,
        P_S=spec["P_S"], P_T=spec["P_T"],
        c_T2=spec["c_T2"], c_S2=spec["c_S2"],
        epsilon=bg["eps"], delta1=bg["delta1"], H_pivot=bg["H"],
        phi_N=phi_N, phi_end=phi_end, N_pivot=N_pivot,
        egb_consistency=egb_cons,
    )


# ---------------------------------------------------------------------------
# Loss for symbolic regression — extended for production observables
# ---------------------------------------------------------------------------
def chi2_full(
    obs: FullObservables,
    *,
    target_ns: float,
    sigma_ns: float = 0.003,
    target_r: float = 0.0,
    sigma_r: float = 0.019,
    target_lnAs: float | None = None,    # ln(10¹⁰ A_s) ≈ 3.044 (Planck)
    sigma_lnAs: float = 0.014,
    target_alphas: float | None = None,
    sigma_alphas: float = 0.0052,
    target_nT: float | None = None,
    sigma_nT: float = 0.1,
    target_cT2: float | None = None,     # GW170817 forces c_T² ≈ 1 today,
    sigma_cT2: float = 0.05,             # not during inflation; usually skip.
    invalid_penalty: float = 1.0e6,      # legacy fallback for is_valid=False
    model: "EGBModel | None" = None,     # NEW: enables soft penalty
    enforce_egb: bool = True,            # NEW: reject ξ → 0 GR limit
    egb_min_delta1: float = 1.0e-4,
) -> float:
    """Generalised χ² in the (n_s, r, [optional A_s, α_s, n_T, c_T²]) plane.

    Returns the scalar PySR consumes. For the per-component breakdown +
    failure reasons, use `chi2_full_with_breakdown` instead, which is
    what the agent tools call.
    """
    bd = chi2_full_with_breakdown(
        obs, target_ns=target_ns, sigma_ns=sigma_ns,
        target_r=target_r, sigma_r=sigma_r,
        target_lnAs=target_lnAs, sigma_lnAs=sigma_lnAs,
        target_alphas=target_alphas, sigma_alphas=sigma_alphas,
        target_nT=target_nT, sigma_nT=sigma_nT,
        target_cT2=target_cT2, sigma_cT2=sigma_cT2,
        invalid_penalty=invalid_penalty, model=model,
        enforce_egb=enforce_egb, egb_min_delta1=egb_min_delta1,
    )
    return bd.total


def chi2_full_with_breakdown(
    obs: FullObservables,
    *,
    target_ns: float,
    sigma_ns: float = 0.003,
    target_r: float = 0.0,
    sigma_r: float = 0.019,
    target_lnAs: float | None = None,
    sigma_lnAs: float = 0.014,
    target_alphas: float | None = None,
    sigma_alphas: float = 0.0052,
    target_nT: float | None = None,
    sigma_nT: float = 0.1,
    target_cT2: float | None = None,
    sigma_cT2: float = 0.05,
    invalid_penalty: float = 1.0e6,
    model: "EGBModel | None" = None,
    enforce_egb: bool = True,
    egb_min_delta1: float = 1.0e-4,
) -> "Chi2Breakdown":
    """Atomized χ² with reasons. The agent tools surface this dict so the
    LLM can see exactly which component is dominating the loss.
    """
    from .diagnostics import (
        Chi2Breakdown,
        chi2_full_breakdown,
        soft_invalid_penalty,
    )

    if obs is None or not obs.is_valid:
        if model is not None:
            soft, reasons = soft_invalid_penalty(model)
        else:
            soft = invalid_penalty
            reasons = ["observables NaN; no model object passed for soft penalty"]
        return Chi2Breakdown(
            total=soft, components={}, reasons=reasons,
            is_valid=False, soft_penalty=soft,
        )

    return chi2_full_breakdown(
        obs,
        target_ns=target_ns, sigma_ns=sigma_ns,
        target_r=target_r, sigma_r=sigma_r,
        target_lnAs=target_lnAs, sigma_lnAs=sigma_lnAs,
        target_alphas=target_alphas, sigma_alphas=sigma_alphas,
        target_nT=target_nT, sigma_nT=sigma_nT,
        target_cT2=target_cT2, sigma_cT2=sigma_cT2,
        enforce_egb=enforce_egb, egb_min_delta1=egb_min_delta1,
    )


def integrate_background_robust(
    model: "EGBModel",
    *,
    N_pivot: float | None = None,
    phi_range: tuple[float, float] | None = None,
    obs: "FullObservables | None" = None,
):
    """Try a ladder of φ-ranges / tolerances until the full background ODE
    converges. Returns (traj, log_lines) where traj is `None` only if
    every ladder step failed.

    Defaults from centralized config.
    """
    from ..config.defaults import DEFAULTS
    from .egb_background import integrate_background, integrate_with_pivot

    if N_pivot is None:
        N_pivot = DEFAULTS.N_pivot
    if phi_range is None:
        phi_range = DEFAULTS.phi_range
    log: list[str] = []
    # Step 1: pivot-aware default
    traj = integrate_with_pivot(model, N_pivot=N_pivot, phi_range=phi_range)
    if traj is not None:
        log.append("background OK with default pivot heuristic")
        return traj, log
    log.append("default integrate_with_pivot failed")

    # Step 2: a φ-range derived from the slow-roll observables (if we have them)
    if obs is not None and np.isfinite(obs.phi_end):
        delta = max(20.0, abs(obs.phi_N - obs.phi_end) * 5.0
                    if np.isfinite(obs.phi_N) else 20.0)
        if (np.isfinite(obs.phi_N) and obs.phi_N > obs.phi_end):
            traj_range = (obs.phi_end - 1.0, obs.phi_end + delta)
        else:
            traj_range = (obs.phi_end - delta, obs.phi_end + 1.0)
        traj = integrate_with_pivot(model, N_pivot=N_pivot, phi_range=traj_range)
        if traj is not None:
            log.append(f"background OK with slow-roll-derived range {traj_range}")
            return traj, log
        log.append(f"slow-roll-derived range {traj_range} failed too")

    # Step 3: explicit φ_init at +/-N_pivot+5 e-folds away from phi_end
    if obs is not None and np.isfinite(obs.phi_end):
        for sign in (+1, -1):
            phi_init = float(obs.phi_end + sign *
                             max(2.0, abs(obs.phi_N - obs.phi_end)))
            for buf in (5.0, 10.0, 20.0):
                traj = integrate_background(
                    model, phi_init,
                    N_max=N_pivot + buf + 20.0,
                )
                if traj is not None and traj.N_end > N_pivot:
                    log.append(f"background OK with phi_init={phi_init:.3f}, "
                               f"buffer={buf}")
                    return traj, log
        log.append("explicit-phi_init ladder also failed")

    # Step 4: looser tolerances (handles stiff EGB)
    if obs is not None and np.isfinite(obs.phi_end):
        for rtol in (1e-7, 1e-5):
            phi_init = float(obs.phi_end +
                             (1 if obs.phi_N > obs.phi_end else -1) *
                             max(2.0, abs(obs.phi_N - obs.phi_end)))
            traj = integrate_background(
                model, phi_init,
                N_max=N_pivot + 25.0, rtol=rtol, atol=rtol * 1e-2,
            )
            if traj is not None and traj.N_end > N_pivot:
                log.append(f"background OK with looser tol rtol={rtol}")
                return traj, log
    log.append("looser-tolerance ladder also failed")
    return None, log


def chi2_relic_gw(
    model: "EGBModel",
    *,
    target_ns: float,
    sigma_ns: float = 0.003,
    target_r: float = 0.0,
    sigma_r: float = 0.019,
    omega_gw_targets: list[tuple[float, float, float]] | None = None,
    omega_gw_band_min: tuple[float, float, float] | None = None,
    N_pivot: float = 55.0,
    T_reh_GeV: float | None = 1.0e15,
    target_lnAs: float | None = None,
    sigma_lnAs: float = 0.014,
    phi_range: tuple[float, float] = (-15.0, 15.0),
    invalid_penalty: float = 1.0e8,
    enforce_egb: bool = True,
    egb_min_delta1: float = 1.0e-4,
) -> float:
    """Scalar χ² with full Mukhanov-Sasaki + relic-GW pipeline. Use
    `chi2_relic_gw_with_breakdown` to get the per-component dict."""
    bd = chi2_relic_gw_with_breakdown(
        model, target_ns=target_ns, sigma_ns=sigma_ns,
        target_r=target_r, sigma_r=sigma_r,
        omega_gw_targets=omega_gw_targets,
        omega_gw_band_min=omega_gw_band_min,
        N_pivot=N_pivot, T_reh_GeV=T_reh_GeV,
        target_lnAs=target_lnAs, sigma_lnAs=sigma_lnAs,
        phi_range=phi_range, invalid_penalty=invalid_penalty,
        enforce_egb=enforce_egb, egb_min_delta1=egb_min_delta1,
    )
    return bd.total


def chi2_relic_gw_with_breakdown(
    model: "EGBModel",
    *,
    target_ns: float,
    sigma_ns: float = 0.003,
    target_r: float = 0.0,
    sigma_r: float = 0.019,
    omega_gw_targets: list[tuple[float, float, float]] | None = None,
    omega_gw_band_min: tuple[float, float, float] | None = None,
    N_pivot: float = 55.0,
    T_reh_GeV: float | None = 1.0e15,
    target_lnAs: float | None = None,
    sigma_lnAs: float = 0.014,
    phi_range: tuple[float, float] = (-15.0, 15.0),
    invalid_penalty: float = 1.0e8,
    enforce_egb: bool = True,
    egb_min_delta1: float = 1.0e-4,
) -> "Chi2Breakdown":
    """Atomized χ² for full MS + relic-GW pipeline.

    Returns Chi2Breakdown with per-component contributions:

      * "n_s", "r", "lnAs"               — slow-roll closed-form
      * "omega_gw@<freq>Hz"              — per pointwise target
      * "omega_gw_band_<f_lo>_<f_hi>"    — band-floor deficit
      * "background_failure"              — soft penalty if MS pipeline
                                            couldn't run

    `reasons` carries human-readable diagnoses, e.g. "Ω_GW @ 1mHz is
    4 decades above target — model is too loud, weaken ξ".
    """
    from .diagnostics import (
        Chi2Breakdown,
        chi2_full_breakdown,
        chi2_omega_gw_breakdown,
        soft_invalid_penalty,
    )
    from .egb_modes import tensor_power_spectrum
    from .relic_gw import (
        OMEGA_R_H2,
        k_inflation_to_today_Mpc_inv,
        transfer_function_sq,
    )

    # Step 1: closed-form (n_s, r, A_s) breakdown
    obs = compute_observables_full(model, N_pivot=N_pivot, phi_range=phi_range)
    if not obs.is_valid:
        soft, reasons = soft_invalid_penalty(model, phi_range=phi_range)
        return Chi2Breakdown(total=soft, components={}, reasons=reasons,
                             is_valid=False, soft_penalty=soft)
    sr_bd = chi2_full_breakdown(
        obs,
        target_ns=target_ns, sigma_ns=sigma_ns,
        target_r=target_r, sigma_r=sigma_r,
        target_lnAs=target_lnAs, sigma_lnAs=sigma_lnAs,
        enforce_egb=enforce_egb, egb_min_delta1=egb_min_delta1,
    )
    components: dict[str, float] = dict(sr_bd.components)
    reasons: list[str] = list(sr_bd.reasons)

    if not omega_gw_targets and not omega_gw_band_min:
        total = float(sum(components.values()))
        return Chi2Breakdown(total=total, components=components,
                             reasons=reasons, is_valid=True)

    # Step 2: robust background integration
    traj, log = integrate_background_robust(model, N_pivot=N_pivot,
                                             phi_range=phi_range, obs=obs)
    if traj is None:
        components["background_failure"] = invalid_penalty * 1.0e-3
        reasons.append(
            "Full background ODE failed across the entire fallback ladder. "
            "Diagnosis steps: " + " → ".join(log) + ". "
            "Common causes: (i) ξ(φ) makes Q change sign across the "
            "trajectory, (ii) the GB term makes the Friedmann constraint "
            "have no real positive H², (iii) ε develops a sharp feature "
            "(e.g. inflection point) that solve_ivp can't resolve at the "
            "default tolerances."
        )
        total = float(sum(components.values()))
        return Chi2Breakdown(total=total, components=components,
                             reasons=reasons, is_valid=True)

    # Step 3: MS for each target frequency
    pivot_idx = int(np.argmin(np.abs(traj.N - (traj.N_end - N_pivot))))
    H_pivot = float(traj.H[pivot_idx])
    a_pivot = float(traj.a[pivot_idx])
    k_pivot_inflation = float(a_pivot * H_pivot)
    inflation_per_today = k_pivot_inflation / 0.05

    if omega_gw_targets:
        ks_inflation = [(f / 0.65e-15) * inflation_per_today
                        for f, _, _ in omega_gw_targets]
        k_arr = np.array(ks_inflation)
        P_T_arr, mode_results = tensor_power_spectrum(
            model, k_arr, traj=traj, N_pivot=N_pivot)
        k_today_arr = k_inflation_to_today_Mpc_inv(
            k_arr, H_pivot=H_pivot, a_pivot=a_pivot)
        Tin_arr = (np.full_like(k_today_arr, T_reh_GeV)
                   if T_reh_GeV is not None else None)
        T_sq = transfer_function_sq(k_today_arr, T_in_GeV=Tin_arr)
        Omega_arr = (OMEGA_R_H2 / 24.0) * P_T_arr * T_sq

        omega_at_f = {tgt[0]: float(Om) for tgt, Om in zip(
            omega_gw_targets, Omega_arr)}
        gw_bd = chi2_omega_gw_breakdown(
            omega_gw_targets, omega_gw_values=omega_at_f)
        components.update(gw_bd.components)
        reasons.extend(gw_bd.reasons)

    if omega_gw_band_min is not None:
        f_lo, f_hi, target_min = omega_gw_band_min
        f_pts = np.logspace(np.log10(f_lo), np.log10(f_hi), 16)
        ks_inflation = [(f / 0.65e-15) * inflation_per_today for f in f_pts]
        k_arr = np.array(ks_inflation)
        P_T_arr, _ = tensor_power_spectrum(model, k_arr, traj=traj,
                                            N_pivot=N_pivot)
        k_today_arr = k_inflation_to_today_Mpc_inv(
            k_arr, H_pivot=H_pivot, a_pivot=a_pivot)
        Tin_arr = (np.full_like(k_today_arr, T_reh_GeV)
                   if T_reh_GeV is not None else None)
        T_sq = transfer_function_sq(k_today_arr, T_in_GeV=Tin_arr)
        Omega_arr = (OMEGA_R_H2 / 24.0) * P_T_arr * T_sq
        deficits = np.maximum(0.0, np.log10(max(target_min, 1e-30))
                              - np.log10(np.maximum(Omega_arr, 1e-30)))
        band_chi2 = float(np.sum(deficits ** 2))
        key = f"omega_gw_band_{f_lo:.0e}_{f_hi:.0e}"
        components[key] = band_chi2
        if band_chi2 > 100:
            min_omega = float(np.nanmin(Omega_arr))
            reasons.append(f"Band-floor deficit: minimum Ω_GW in [{f_lo:.0e},"
                           f" {f_hi:.0e}] Hz is {min_omega:.2e}, target≥"
                           f"{target_min:.2e} ⇒ raise V_pivot or sharpen "
                           "ξ(φ) near horizon exit.")

    total = float(sum(components.values()))
    return Chi2Breakdown(total=total, components=components,
                         reasons=reasons, is_valid=True)


# ---------------------------------------------------------------------------
# EGB-aware consistency metric
# ---------------------------------------------------------------------------
def egb_consistency_metric(obs: FullObservables) -> dict[str, float]:
    """Compute the EGB-aware consistency metric replacing the GR-only r/(-8nT).

    In pure GR, the single-field consistency relation is r = -8 n_T.
    In EGB inflation, this is modified by the Gauss-Bonnet coupling:
    the tensor sound speed c_T² ≠ 1 and δ₁ ≠ 0 break the simple relation.

    We report:
      * ``egb_consistency``: r / (-8 n_T) — equals 1 in GR, deviates in EGB.
      * ``c_T2_deviation``: 1 - c_T² — measures how far tensor propagation
        deviates from luminal.
      * ``delta1_magnitude``: |δ₁| — measures the strength of the GB coupling
        at horizon crossing.

    References
    ----------
    * GR limit: Liddle & Lyth, *The Primordial Density Perturbation*
      (Cambridge, 2009), Eq. (7.37).
    * EGB breaking: Kawai & Soda 1999 (gr-qc/9901002);
      Hwang & Noh 2005 (gr-qc/0507025).
    """
    out: dict[str, float] = {
        "c_T2_deviation": float("nan"),
        "delta1_magnitude": float("nan"),
        "egb_consistency": float("nan"),
    }
    if obs.is_valid:
        out["c_T2_deviation"] = 1.0 - obs.c_T2
        out["delta1_magnitude"] = abs(obs.delta1)
        if np.isfinite(obs.n_T) and obs.n_T != 0.0:
            out["egb_consistency"] = obs.r / (-8.0 * obs.n_T)
    return out
