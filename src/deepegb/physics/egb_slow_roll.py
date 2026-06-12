"""
Core EGB-inflation data types and trajectory utilities.

This module provides

* `EGBModel` — the (V, ξ) data class used by every production kernel.
* `end_of_inflation` — robust φ-bracket finder for ε(φ_end) = 1, used to
  seed both the slow-roll closed-form kernel (`egb_perturbations.py`)
  and the full numerical background integrator (`egb_background.py`).
* `_N_to_phi_table` — slow-roll quadrature N(φ) = ∫_{φ_end}^{φ} V/Q dφ.
  Used by the closed-form kernel and by the φ-init heuristic in
  `egb_background.integrate_with_pivot`.

This file used to contain a leading-order toy kernel (`analyze_model`,
`chi2_loss`, `Observables`) — it has been **removed**. All observables
should be computed via `compute_observables_full` (slow-roll closed-form)
or `tensor_power_spectrum` / `scalar_power_spectrum` (full Mukhanov-Sasaki).

Conventions
-----------
Action (M_pl = 1):

    S = ∫ d⁴x √(-g) [ R/2 − (1/2)(∂φ)² − V(φ) − (1/2) ξ(φ) 𝒢 ]

The slow-roll-truncated background EOMs give

    H² ≈ V/3,   3 H φ̇ ≈ −Q,    Q ≡ V_,φ + (4/3) V² ξ_,φ
    ε(φ) ≈ Q V_,φ / (2 V²),     N(φ) = ∫_{φ_end}^{φ} V/Q dψ

These are used as a *seed* for the full numerical integration — the
production observables come from the full perturbation theory in
`egb_perturbations.py` and the Mukhanov-Sasaki kernels in `egb_modes.py`.

References for the conventions: Hwang-Noh 2005 (gr-qc/0507025),
Koh-Lee-Tumurtushaa 2014 (arXiv:1404.0027), Yi-Gong-Sabir 2018
(arXiv:1811.01580).
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

    # Numerical-derivative step (M_pl units).
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
        """Slow-roll-truncated ε(φ) = Q V_,φ / (2 V²). Used to find φ_end."""
        V = self.V(phi)
        return 0.5 * self.Q(phi) * self.V_phi(phi) / (V * V)


# ---------------------------------------------------------------------------
# End of inflation, e-fold counting
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


def end_of_inflation_all(
    model: EGBModel,
    phi_range: tuple[float, float] = (-15.0, 15.0),
    n_grid: int = 4001,
) -> list[float]:
    """All φ_end candidates with ε(φ_end) = 1, sorted so that crossings
    adjacent to the longest contiguous slow-roll plateaus come first.
    Models can have several inflationary basins (e.g. ±φ for even V);
    callers should try each.
    """
    phi_grid = np.linspace(*phi_range, n_grid)
    eps = _scan_epsilon(model, phi_grid)

    # Sign changes of (ε − 1)
    diff = eps - 1.0
    crossings: list[float] = []
    for i in range(len(phi_grid) - 1):
        a, b = diff[i], diff[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a * b < 0:
            phi_root = phi_grid[i] - a * (phi_grid[i + 1] - phi_grid[i]) / (b - a)
            crossings.append(phi_root)
    if not crossings:
        return []

    # Rank by the length of the adjacent contiguous slow-roll plateau
    def score(phi_c: float) -> float:
        below = (eps < 1.0).astype(int)
        idx = int(np.argmin(np.abs(phi_grid - phi_c)))
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
    return [float(c) for c in crossings]


def end_of_inflation(
    model: EGBModel,
    phi_range: tuple[float, float] = (-15.0, 15.0),
    n_grid: int = 4001,
) -> float | None:
    """Find φ_end such that ε(φ_end) = 1, choosing the crossing nearest to a
    valid slow-roll region.  Returns None if no end-of-inflation exists in
    the given range.
    """
    cands = end_of_inflation_all(model, phi_range=phi_range, n_grid=n_grid)
    return cands[0] if cands else None


def _N_to_phi_table(
    model: EGBModel,
    phi_end: float,
    phi_range: tuple[float, float],
    n_grid: int = 4001,
) -> tuple[np.ndarray, np.ndarray]:
    """Slow-roll quadrature for e-folds counted backwards from end of inflation:

        N(φ) = ∫_{φ_end}^{φ} V(ψ) / Q(ψ) dψ

    Comes out positive on the inflationary trajectory irrespective of the
    rolling direction (V/Q changes sign with Q).  Used to seed the full
    background integrator with a sensible φ_init.
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

    valid = np.isfinite(integrand)
    integrand_safe = np.where(valid, integrand, 0.0)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (integrand_safe[1:] + integrand_safe[:-1])
                                            * np.diff(phi_grid))])
    cum_at_end = float(np.interp(phi_end, phi_grid, cum))
    N_grid = cum - cum_at_end
    N_grid = np.where(valid, N_grid, np.nan)
    return phi_grid, N_grid
