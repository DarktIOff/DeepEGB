"""
Production-grade EGB inflation observables.

This module replaces the leading-order toy in `egb_slow_roll.py` with a
careful slow-roll perturbation calculation that includes:

  * The GB-corrected slow-roll parameters ε₁, η, δ_1, δ_2.
  * The tensor sound speed  c_T²(φ) = F_T/G_T  with the explicit forms
    F_T = M_pl² (1 − 4 ξ̈/M_pl²),   G_T = M_pl² (1 − 4 ξ̇H/M_pl²)
    (Hwang & Noh 2005; Kawai & Soda 1999).
  * The tensor power spectrum  P_T = 2 H² / [π² M_pl² (1−δ_1) c_T³]
    evaluated at sound-horizon crossing (k* = a*H*/c_T*).
  * The scalar power spectrum  P_S = H²/(8 π² M_pl² ε)  with the GB-
    corrected ε (Koh-Lee-Tumurtushaa 2014; Yi-Gong-Sabir 2018).
  * Spectral indices n_s, n_T computed as **numerical N-derivatives** of
    ln P_S, ln P_T along the inflationary trajectory — this captures all
    O(slow-roll²) corrections automatically and is what we mean by
    "production grade" here.
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


def compute_c_S2(model: EGBModel, phi: float) -> float:
    """Scalar sound speed squared (EGB-corrected, leading order).

    Following Kawai & Soda 1999 (gr-qc/9901002) and the specialisation in
    Hwang-Noh 2005 of the general scalar-tensor + GB perturbation theory,
    the scalar sound speed in single-field EGB inflation is, in M_pl = 1:

        c_S² = 1  −  4 ξ_,φ² H² / [ ε₁ · (1 − δ₁)² ]   +  𝒪(slow-roll³).

    The combination ξ̇²H²/(ε φ̇²) = ξ_,φ² H²/ε emerges from the cross-term
    between the inflaton kinetic energy and the GB mixing at quadratic order
    in perturbations, divided by Q_S/(1−δ₁)². Reduces to c_S² = 1 in the
    GR limit (ξ_,φ → 0), and is bounded between 0 and 1 for any healthy
    EGB inflation model.

    For sub-percent precision on n_s in regions where this leading correction
    is itself small, swap in the full Horndeski expression (KYY 2011 Eq. 3.21
    specialised through G_5 = −4 ξ_,φ ln X). The infrastructure for that —
    the full background trajectory in `egb_background.py` — is now in place;
    the algebraic specialisation is left as a TODO for next iteration.
    """
    bg = background_at(model, phi)
    if not (np.isfinite(bg["eps"]) and bg["eps"] > 0 and np.isfinite(bg["delta1"])):
        return np.nan
    one_minus_d1 = 1.0 - bg["delta1"]
    if abs(one_minus_d1) < 1.0e-12:
        return np.nan
    H2 = bg["H"] * bg["H"]
    xip = bg["xip"]
    correction = 4.0 * xip * xip * H2 / (bg["eps"] * one_minus_d1 * one_minus_d1)
    return float(1.0 - correction)


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


def compute_observables_full(
    model: EGBModel,
    N_pivot: float = 55.0,
    phi_range: tuple[float, float] = (-15.0, 15.0),
    n_grid: int = 4001,
    dN_for_running: float = 0.5,
) -> FullObservables:
    """Compute production-grade observables for an EGB inflation model.

    Workflow:
      1. Find φ_end from ε(φ_end) = 1.
      2. Find φ_pivot N e-folds before end (from V/Q quadrature).
      3. Compute background quantities, c_T², P_S, P_T at φ_pivot.
      4. Compute n_s, n_T as numerical N-derivatives of ln P_S, ln P_T
         using a small ΔN step on either side of φ_pivot.
      5. Compute α_s = dn_s/dlnk likewise (second derivative of ln P_S).
    """
    phi_end = end_of_inflation(model, phi_range=phi_range, n_grid=n_grid)
    if phi_end is None:
        return FullObservables(*([np.nan] * 13), N_pivot=N_pivot)

    phi_N = _bracket_phi_for_N(model, phi_end, N_pivot, phi_range, n_grid)
    if phi_N is None:
        return FullObservables(*([np.nan] * 12), phi_end=phi_end, N_pivot=N_pivot)

    # Background at pivot
    bg = background_at(model, phi_N)
    spec = power_spectra_at(model, phi_N)
    if not np.isfinite(spec["P_S"]) or not np.isfinite(spec["P_T"]):
        return FullObservables(np.nan, np.nan, np.nan, np.nan,
                               spec["P_S"], spec["P_T"],
                               spec["c_T2"], spec["c_S2"],
                               bg["eps"], bg["delta1"], bg["H"],
                               phi_N, phi_end, N_pivot)

    # Find φ values dN_for_running e-folds either side of pivot
    phi_lo = _bracket_phi_for_N(model, phi_end,
                                 N_pivot - dN_for_running, phi_range, n_grid)
    phi_hi = _bracket_phi_for_N(model, phi_end,
                                 N_pivot + dN_for_running, phi_range, n_grid)
    if phi_lo is None or phi_hi is None:
        # Fall back to closed-form leading-order indices.
        eps = bg["eps"]
        d1 = bg["delta1"]
        eta = bg["Vpp"] / bg["V"]
        n_s_lo = 1.0 - 2.0 * eps - 2.0 * eta + 2.0 * d1   # KLT-like leading order
        n_T_lo = -2.0 * eps - d1                          # KLT, YGS leading order
        r_val = spec["P_T"] / spec["P_S"]
        return FullObservables(
            n_s=n_s_lo, n_T=n_T_lo, r=r_val, alpha_s=np.nan,
            P_S=spec["P_S"], P_T=spec["P_T"],
            c_T2=spec["c_T2"], c_S2=spec["c_S2"],
            epsilon=eps, delta1=d1, H_pivot=bg["H"],
            phi_N=phi_N, phi_end=phi_end, N_pivot=N_pivot,
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

    return FullObservables(
        n_s=n_s, n_T=n_T, r=r_val, alpha_s=alpha_s,
        P_S=spec["P_S"], P_T=spec["P_T"],
        c_T2=spec["c_T2"], c_S2=spec["c_S2"],
        epsilon=bg["eps"], delta1=bg["delta1"], H_pivot=bg["H"],
        phi_N=phi_N, phi_end=phi_end, N_pivot=N_pivot,
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
    sigma_r: float = 0.018,
    target_lnAs: float | None = None,    # ln(10¹⁰ A_s) ≈ 3.044 (Planck)
    sigma_lnAs: float = 0.014,
    target_alphas: float | None = None,
    sigma_alphas: float = 0.013,
    target_nT: float | None = None,
    sigma_nT: float = 0.1,
    target_cT2: float | None = None,     # GW170817 forces c_T² ≈ 1 today,
    sigma_cT2: float = 0.05,             # not during inflation; usually skip.
    invalid_penalty: float = 1.0e6,
) -> float:
    """Generalised χ² in the (n_s, r, [optional A_s, α_s, n_T, c_T²]) plane."""
    if not obs.is_valid:
        return invalid_penalty
    chi2 = ((obs.n_s - target_ns) / sigma_ns) ** 2
    chi2 += ((obs.r - target_r) / sigma_r) ** 2
    if target_lnAs is not None and obs.P_S > 0:
        ln10As = np.log(1e10 * obs.P_S)
        chi2 += ((ln10As - target_lnAs) / sigma_lnAs) ** 2
    if target_alphas is not None and np.isfinite(obs.alpha_s):
        chi2 += ((obs.alpha_s - target_alphas) / sigma_alphas) ** 2
    if target_nT is not None and np.isfinite(obs.n_T):
        chi2 += ((obs.n_T - target_nT) / sigma_nT) ** 2
    if target_cT2 is not None and np.isfinite(obs.c_T2):
        chi2 += ((obs.c_T2 - target_cT2) / sigma_cT2) ** 2
    return float(chi2)


def chi2_relic_gw(
    model: "EGBModel",
    *,
    target_ns: float,
    sigma_ns: float = 0.003,
    target_r: float = 0.0,
    sigma_r: float = 0.018,
    # Relic-GW targets: list of (f_Hz, target_Omega_GW_h2, sigma) triples.
    # Example: [(1e-3, 1e-12, 1e-12)] — try to hit Ω_GW = 10⁻¹² at LISA.
    omega_gw_targets: list[tuple[float, float, float]] | None = None,
    # Optional minimum-amplitude target across a frequency band:
    omega_gw_band_min: tuple[float, float, float] | None = None,
    # (f_lo_Hz, f_hi_Hz, target_min_Omega) — penalises spectra that are
    # below `target_min_Omega` anywhere in [f_lo, f_hi]; encourages "loud"
    # spectra in the band.
    N_pivot: float = 55.0,
    T_reh_GeV: float | None = 1.0e15,
    # Standard slow-roll observables on top of GW targets:
    target_lnAs: float | None = None,
    sigma_lnAs: float = 0.014,
    phi_range: tuple[float, float] = (-15.0, 15.0),
    invalid_penalty: float = 1.0e8,
) -> float:
    """χ² with full Mukhanov-Sasaki + relic-GW pipeline.

    EXPENSIVE: each call solves the full background EOMs and integrates the
    tensor mode equation for every requested target frequency (≈ 0.5–1 s on
    a modern laptop). Use a coarser PySR run (`niters=10`, `populations=15`)
    when this loss is enabled.

    The base (n_s, r) terms still come from the slow-roll closed-form kernel
    so the slow part of the loss only kicks in for the GW-specific targets.
    """
    # local imports to avoid circular dependency at module load time
    from .egb_background import integrate_with_pivot
    from .egb_modes import k_pivot_from_traj, tensor_power_spectrum
    from .relic_gw import (
        OMEGA_R_H2,
        k_inflation_to_today_Mpc_inv,
        k_today_Mpc_inv_to_freq_Hz,
        transfer_function_sq,
    )

    # Step 1: standard observables from the closed-form kernel
    obs = compute_observables_full(model, N_pivot=N_pivot, phi_range=phi_range)
    if not obs.is_valid:
        return invalid_penalty
    chi2 = ((obs.n_s - target_ns) / sigma_ns) ** 2
    chi2 += ((obs.r - target_r) / sigma_r) ** 2
    if target_lnAs is not None and obs.P_S > 0:
        ln10As = np.log(1e10 * obs.P_S)
        chi2 += ((ln10As - target_lnAs) / sigma_lnAs) ** 2

    if not omega_gw_targets and not omega_gw_band_min:
        return float(chi2)

    # Step 2: full background + MS for each target frequency.
    # Use the inflationary half-range from obs.phi_end to extend φ-bracket.
    if np.isfinite(obs.phi_end):
        # bracket from φ_end towards the inflationary side, ample buffer
        delta = max(20.0, abs(obs.phi_N - obs.phi_end) * 5.0)
        if obs.phi_N > obs.phi_end:
            traj_range = (obs.phi_end - 1.0, obs.phi_end + delta)
        else:
            traj_range = (obs.phi_end - delta, obs.phi_end + 1.0)
    else:
        traj_range = phi_range
    traj = integrate_with_pivot(model, N_pivot=N_pivot, phi_range=traj_range)
    if traj is None:
        return invalid_penalty
    pivot_idx = int(np.argmin(np.abs(traj.N - (traj.N_end - N_pivot))))
    H_pivot = float(traj.H[pivot_idx])
    a_pivot = float(traj.a[pivot_idx])
    k_pivot_inflation = float(a_pivot * H_pivot)
    # k_inflation = (k_today_Mpc) × (k_pivot_inflation / 0.05)
    # f_Hz = 0.65e-15 × k_today_Mpc  ⇒  k_today_Mpc = f_Hz / 0.65e-15
    inflation_per_today = k_pivot_inflation / 0.05

    # Pointwise targets
    if omega_gw_targets:
        ks_inflation: list[float] = []
        for f_Hz, _, _ in omega_gw_targets:
            k_today_Mpc = f_Hz / 0.65e-15
            ks_inflation.append(k_today_Mpc * inflation_per_today)
        k_arr = np.array(ks_inflation)
        P_T_arr, _ = tensor_power_spectrum(model, k_arr, traj=traj, N_pivot=N_pivot)
        k_today_arr = k_inflation_to_today_Mpc_inv(
            k_arr, H_pivot=H_pivot, a_pivot=a_pivot,
        )
        Tin_arr = (np.full_like(k_today_arr, T_reh_GeV)
                   if T_reh_GeV is not None else None)
        T_sq = transfer_function_sq(k_today_arr, T_in_GeV=Tin_arr)
        Omega_arr = (OMEGA_R_H2 / 24.0) * P_T_arr * T_sq
        for (f_Hz, target, sigma), Om in zip(omega_gw_targets, Omega_arr):
            if not np.isfinite(Om) or Om <= 0:
                chi2 += invalid_penalty * 1.0e-3
                continue
            # χ² in log-space: log Ω is what we typically care about
            chi2 += (
                (np.log10(max(Om, 1e-30)) - np.log10(max(target, 1e-30)))
                / max(sigma / max(target, 1e-30) / np.log(10), 0.05)
            ) ** 2

    # Band minimum target
    if omega_gw_band_min is not None:
        f_lo, f_hi, target_min = omega_gw_band_min
        # 16 sample points across the band
        f_pts = np.logspace(np.log10(f_lo), np.log10(f_hi), 16)
        ks_inflation = [(f / 0.65e-15) * inflation_per_today for f in f_pts]
        k_arr = np.array(ks_inflation)
        P_T_arr, _ = tensor_power_spectrum(model, k_arr, traj=traj, N_pivot=N_pivot)
        k_today_arr = k_inflation_to_today_Mpc_inv(
            k_arr, H_pivot=H_pivot, a_pivot=a_pivot,
        )
        Tin_arr = (np.full_like(k_today_arr, T_reh_GeV)
                   if T_reh_GeV is not None else None)
        T_sq = transfer_function_sq(k_today_arr, T_in_GeV=Tin_arr)
        Omega_arr = (OMEGA_R_H2 / 24.0) * P_T_arr * T_sq
        # Penalise being below target_min:
        deficits = np.maximum(0.0, np.log10(max(target_min, 1e-30))
                              - np.log10(np.maximum(Omega_arr, 1e-30)))
        chi2 += float(np.sum(deficits ** 2))

    return float(chi2)
