"""
EGB slow-roll inflation — observables in leading slow-roll.

Conventions
-----------
Action (reduced Planck mass M_pl = 1 throughout):

    S = ∫ d⁴x √(-g) [ R/2 − (1/2)(∂φ)² − V(φ) − (1/2) ξ(φ) 𝒢 ]

where 𝒢 = R² − 4 R_{μν} R^{μν} + R_{μνρσ} R^{μνρσ}.

In FLRW with single field φ(t) the slow-roll regime gives, to leading order in
the slow-roll hierarchy (see e.g. Koh, Lee & Tumurtushaa 1404.0027 and
Yi, Gong & Sabir 1811.01580):

    H² ≈ V / 3
    3 H φ̇ ≈ −V_,φ − (4/3) V² ξ_,φ          (because 12 H⁴ ξ_,φ = (4/3) V² ξ_,φ)

We define the GB-corrected slow-roll "force" Q(φ):

    Q(φ) ≡ V_,φ(φ) + (4/3) V(φ)² ξ_,φ(φ)

and the leading-order slow-roll parameter

    ε(φ) ≡ −Ḣ/H² ≈ (1/2) Q(φ) V_,φ(φ) / V(φ)²

End of inflation is defined by ε(φ_end) = 1. The number of e-folds counted
backwards from the end of inflation is

    N(φ) = ∫_{φ}^{φ_end} (V(ψ) / Q(ψ)) dψ           (sign chosen so N > 0)

Observables at horizon crossing N e-folds before end:

    n_s − 1 ≈ −2 ε(φ_N) + (Q/(ε V))|_{φ_N} · dε/dφ|_{φ_N}
    r       ≈ 16 ε(φ_N)

The n_s expression follows from n_s − 1 = −2ε − ε₂ with
ε₂ = dlnε/dN = (dε/dφ)(dφ/dN) and dφ/dN = −Q/V in slow-roll.

Caveats
-------
* The above is the **leading-order** EGB slow-roll. Sub-leading corrections to
  r from the modified tensor speed c_T² are NOT included; for thesis-grade
  numbers swap in `egb_full_eom.py` (TODO).
* We assume single-field, slow-roll, no entropic perturbations.
* Sign conventions match Yi-Gong-Sabir 2018; if your reference differs by an
  overall sign on ξ or a factor of 4/3, adjust `Q` accordingly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

ArrayLike = np.ndarray | float


# ---------------------------------------------------------------------------
# Model representation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EGBModel:
    """A single-field EGB inflation model: V(φ) and ξ(φ) as Python callables."""

    V: Callable[[ArrayLike], ArrayLike]
    xi: Callable[[ArrayLike], ArrayLike]
    name: str = "model"
    description: str = ""

    # Numerical-derivative step.
    h: float = 1.0e-4

    def V_phi(self, phi: ArrayLike) -> ArrayLike:
        return (self.V(phi + self.h) - self.V(phi - self.h)) / (2 * self.h)

    def V_phiphi(self, phi: ArrayLike) -> ArrayLike:
        return (self.V(phi + self.h) - 2 * self.V(phi) + self.V(phi - self.h)) / (self.h**2)

    def xi_phi(self, phi: ArrayLike) -> ArrayLike:
        return (self.xi(phi + self.h) - self.xi(phi - self.h)) / (2 * self.h)

    # ------------------------------------------------------------------ Q, ε
    def Q(self, phi: ArrayLike) -> ArrayLike:
        """Q(φ) = V_,φ + (4/3) V² ξ_,φ — the GB-corrected slow-roll force."""
        Vp = self.V_phi(phi)
        Xp = self.xi_phi(phi)
        return Vp + (4.0 / 3.0) * self.V(phi) ** 2 * Xp

    def epsilon(self, phi: ArrayLike) -> ArrayLike:
        """Leading-order slow-roll parameter ε(φ) = (1/2) Q V_,φ / V²."""
        V = self.V(phi)
        return 0.5 * self.Q(phi) * self.V_phi(phi) / (V * V)


# ---------------------------------------------------------------------------
# End of inflation, e-fold counting, horizon crossing
# ---------------------------------------------------------------------------
def _scan_epsilon(
    model: EGBModel,
    phi_grid: np.ndarray,
) -> np.ndarray:
    """Return ε(φ) on a grid; safe against V=0 singularities."""
    eps = np.zeros_like(phi_grid, dtype=float)
    for i, p in enumerate(phi_grid):
        try:
            v = float(model.V(p))
            if not np.isfinite(v) or v <= 0:
                eps[i] = np.inf
                continue
            eps[i] = float(model.epsilon(p))
        except Exception:
            eps[i] = np.inf
    return eps


def end_of_inflation(
    model: EGBModel,
    phi_range: tuple[float, float] = (-15.0, 15.0),
    n_grid: int = 4001,
) -> float | None:
    """Find φ_end such that ε(φ_end) = 1, choosing the crossing nearest to a
    valid slow-roll region.

    Returns None if no end-of-inflation can be found in the given range.
    """
    phi_grid = np.linspace(*phi_range, n_grid)
    eps = _scan_epsilon(model, phi_grid)

    # Look for sign changes of (ε − 1).
    diff = eps - 1.0
    crossings: list[float] = []
    for i in range(len(phi_grid) - 1):
        a, b = diff[i], diff[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a * b < 0:
            # Linear interpolation for the root.
            phi_root = phi_grid[i] - a * (phi_grid[i + 1] - phi_grid[i]) / (b - a)
            crossings.append(phi_root)
    if not crossings:
        return None

    # Prefer crossings adjacent to a region where ε < 1 over many points
    # (a real slow-roll plateau).
    def score(phi_c: float) -> float:
        # number of grid points on the slow-roll side that are < 1
        below = (eps < 1.0).astype(int)
        idx = int(np.argmin(np.abs(phi_grid - phi_c)))
        # count contiguous run of "below" on either side of idx
        run = 0
        i = idx
        while i >= 0 and below[i]:
            run += 1
            i -= 1
        i = idx + 1
        while i < len(phi_grid) and below[i]:
            run += 1
            i += 1
        return run

    crossings.sort(key=score, reverse=True)
    return float(crossings[0])


def _N_to_phi_table(
    model: EGBModel,
    phi_end: float,
    phi_range: tuple[float, float],
    n_grid: int = 4001,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (phi_grid, N_grid) where N is e-folds before end of inflation.

    In slow-roll  dφ/dN_cosmic = −Q/V, so e-folds counted *backwards from
    end of inflation* are
        N(φ) = ∫_{φ_end}^{φ}  V(ψ)/Q(ψ)  dψ
    The sign comes out positive on the inflationary trajectory regardless of
    which way φ rolls, because V/Q changes sign with Q.
    """
    phi_grid = np.linspace(*phi_range, n_grid)
    integrand = np.full_like(phi_grid, np.nan)
    for i, p in enumerate(phi_grid):
        try:
            V = float(model.V(p))
            Q = float(model.Q(p))
            if Q == 0 or not np.isfinite(V) or not np.isfinite(Q):
                continue
            integrand[i] = V / Q
        except Exception:
            continue

    # cumulative integral ∫_{phi_min}^{φ} V/Q dφ
    valid = np.isfinite(integrand)
    integrand_safe = np.where(valid, integrand, 0.0)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (integrand_safe[1:] + integrand_safe[:-1])
                                            * np.diff(phi_grid))])

    # N(φ) = ∫_{φ_end}^{φ} V/Q dφ = cum[φ] − cum[φ_end]
    cum_at_end = float(np.interp(phi_end, phi_grid, cum))
    N_grid = cum - cum_at_end
    N_grid = np.where(valid, N_grid, np.nan)
    return phi_grid, N_grid


def horizon_crossing(
    model: EGBModel,
    phi_end: float,
    N_target: float = 55.0,
    phi_range: tuple[float, float] = (-15.0, 15.0),
    n_grid: int = 4001,
) -> float | None:
    """Find φ_N such that ∫_{φ_N}^{φ_end} V/Q dφ = N_target."""
    phi_grid, N_grid = _N_to_phi_table(model, phi_end, phi_range, n_grid)
    # Pick the inflationary side: ε < 1 around the trajectory.
    eps = _scan_epsilon(model, phi_grid)
    side_mask = eps < 1.0

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
    # Pick candidate nearest to phi_end on the slow-roll side.
    candidates.sort(key=lambda p: abs(p - phi_end))
    return float(candidates[0])


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Observables:
    n_s: float
    r: float
    epsilon: float
    phi_N: float
    phi_end: float
    N: float

    @property
    def is_valid(self) -> bool:
        return all(np.isfinite([self.n_s, self.r, self.epsilon, self.phi_N, self.phi_end]))

    def as_dict(self) -> dict:
        return dict(n_s=self.n_s, r=self.r, epsilon=self.epsilon,
                    phi_N=self.phi_N, phi_end=self.phi_end, N=self.N)


def _eps_phi_derivative(model: EGBModel, phi: float, h: float = 1e-3) -> float:
    return (float(model.epsilon(phi + h)) - float(model.epsilon(phi - h))) / (2 * h)


def analyze_model(
    model: EGBModel,
    N_target: float = 55.0,
    phi_range: tuple[float, float] = (-15.0, 15.0),
    n_grid: int = 4001,
) -> Observables:
    """Compute (n_s, r) for an EGB model at N e-folds before end of inflation."""
    phi_end = end_of_inflation(model, phi_range=phi_range, n_grid=n_grid)
    if phi_end is None:
        return Observables(np.nan, np.nan, np.nan, np.nan, np.nan, N_target)

    phi_N = horizon_crossing(model, phi_end, N_target, phi_range=phi_range, n_grid=n_grid)
    if phi_N is None:
        return Observables(np.nan, np.nan, np.nan, np.nan, phi_end, N_target)

    eps = float(model.epsilon(phi_N))
    Q = float(model.Q(phi_N))
    V = float(model.V(phi_N))
    deps = _eps_phi_derivative(model, phi_N)

    # n_s − 1 = −2ε − ε₂, with ε₂ = (dε/dφ)(dφ/dN) and dφ/dN = −Q/V
    # ⇒ ε₂ = −(Q/V) · dε/dφ ⇒ −ε₂ = (Q/V) · dε/dφ.
    # Divide by ε to get n_s − 1: n_s − 1 = −2ε + (Q/(εV)) dε/dφ.
    n_s = 1.0 - 2.0 * eps + (Q / (eps * V)) * deps
    r = 16.0 * eps  # leading order; TODO: tensor-speed correction

    return Observables(n_s=n_s, r=r, epsilon=eps,
                       phi_N=phi_N, phi_end=phi_end, N=N_target)


# ---------------------------------------------------------------------------
# Loss for symbolic regression
# ---------------------------------------------------------------------------
def chi2_loss(
    obs: Observables,
    target_ns: float,
    sigma_ns: float,
    target_r: float = 0.0,
    sigma_r: float = 0.05,
    invalid_penalty: float = 1.0e6,
) -> float:
    """χ² distance in (n_s, r) plane to the target."""
    if not obs.is_valid:
        return invalid_penalty
    return (
        ((obs.n_s - target_ns) / sigma_ns) ** 2
        + ((obs.r - target_r) / sigma_r) ** 2
    )
