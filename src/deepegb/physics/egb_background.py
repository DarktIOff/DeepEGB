"""
Full numerical background integration for EGB inflation.

We solve the Einstein–Klein–Gordon system

    3 H² = (1/2)φ̇² + V + 12 H³ ξ̇                                 (F1, Friedmann)
    M_pl²(3H² + 2Ḣ) = −(1/2)φ̇² + V + 4 H² ξ̈ + 8 H Ḣ ξ̇         (F2)
    φ̈ + 3 H φ̇ + V_,φ + 12 H²(Ḣ + H²) ξ_,φ = 0                    (KG)

(YGS 2018 Eqs. 2.4–2.6; HN 2005 Eqs. 4–5) on a flat FRW background with
M_pl = 1.

Integration variable
--------------------
We use the number of e-folds N = ln(a/a_init) as the integration variable.
The state vector is (φ, π) where π ≡ dφ/dN = φ̇/H.  We reconstruct H from
the Friedmann constraint at each step, solve a 2×2 linear system for
(d²φ/dN², ε₁ ≡ −Ḣ/H²), and step forward.

Algebra at every step
---------------------
Set u ≡ d/dN.  Then φ̇ = H π, φ̈ = H²(uπ + π·(uH)/H) and we use uH/H = −ε₁.

KG (rewritten in N):     uπ + (3 − ε₁) π + V_,φ/H² + 12 ξ_,φ (1 − ε₁) H² = 0
F2 − F1 simplified:      Ḣ = (1/2)·[ −φ̇² + 4 H² ξ̈ + 8 H Ḣ ξ̇ ]
                              with ξ̈ = ξ_,φφ φ̇² + ξ_,φ φ̈

Combining gives a 2×2 linear system for (uπ, ε₁) at each grid point. We
solve it with `numpy.linalg.solve` for numerical robustness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .egb_slow_roll import EGBModel, end_of_inflation


# ---------------------------------------------------------------------------
# Friedmann constraint solver. Using π ≡ dφ/dN ⇒ φ̇ = π H, the constraint
#
#     3 H² = (1/2) π² H² + V + 12 H³ · ξ_,φ · (π H)
#
# becomes a *quadratic* in x ≡ H²:
#
#     [12 ξ_,φ π] x² + [(π²/2) − 3] x + V = 0.
#
# In the GR limit (ξ_,φ → 0), x = V / (3 − π²/2).  For nonzero coupling we
# pick the positive real root closest to the GR seed.
# ---------------------------------------------------------------------------
def hubble_from_constraint(V: float, xi_p: float, pi: float,
                           H_seed: float | None = None) -> float:
    """Return H ≥ 0 satisfying the Friedmann constraint with π = dφ/dN."""
    if not np.isfinite(V) or V <= 0:
        return float("nan")
    denom = 3.0 - 0.5 * pi * pi
    if denom <= 0:
        return float("nan")  # kinetic energy too large for inflation
    H2_GR = V / denom
    H_GR = np.sqrt(H2_GR)
    A = 12.0 * xi_p * pi
    if abs(A) < 1.0e-14:
        return H_GR
    # Quadratic in H²: A x² + B x + V = 0,  B = (π²/2 − 3)
    B = 0.5 * pi * pi - 3.0
    disc = B * B - 4.0 * A * V
    if disc < 0:
        return float("nan")
    sq = np.sqrt(disc)
    cands_x: list[float] = []
    for x in ((-B + sq) / (2.0 * A), (-B - sq) / (2.0 * A)):
        if x > 0 and np.isfinite(x):
            cands_x.append(x)
    if not cands_x:
        return float("nan")
    seed = H_seed if (H_seed is not None and H_seed > 0) else H_GR
    cands_x.sort(key=lambda x: abs(np.sqrt(x) - seed))
    return float(np.sqrt(cands_x[0]))


# ---------------------------------------------------------------------------
# Per-step linear system for (uπ, ε₁) where u ≡ d/dN.
# ---------------------------------------------------------------------------
def _step_rhs(model: EGBModel, phi: float, pi: float) -> tuple[float, float, dict]:
    """Return (dπ/dN, ε₁) and a dict of auxiliaries given (φ, π=dφ/dN)."""
    V = float(model.V(phi))
    Vp = float(model.V_phi(phi))
    Vpp = float(model.V_phiphi(phi))
    xip = float(model.xi_phi(phi))
    h = model.h
    xipp = float((model.xi(phi + h) - 2.0 * model.xi(phi) + model.xi(phi - h)) / (h * h))
    H = hubble_from_constraint(V, xip, pi)
    if not np.isfinite(H) or H <= 0:
        return (float("nan"), float("nan"),
                {"H": H, "V": V, "Vp": Vp, "xip": xip, "xipp": xipp})
    phidot = pi * H

    # Linear 2×2 in (uπ, ε₁):
    #   uπ · 1                             + ε₁ · ( π )           = − [(3) π + V_,φ/H² + 12 ξ_,φ H²]
    #   uπ · ( 4 H² ξ_,φ )                 + ε₁ · (2 − 8 H ξ_,φ π H ) = − [φ̇² − 4 H² ξ_,φφ π² H² ]
    # (the second eq comes from rearranging F2−F1 with ξ̈ = ξ_,φφ φ̇² + ξ_,φ φ̈
    #  and Ḣ = −ε₁ H²; we divide by H² to eliminate units.)

    # Derivation (M_pl=1, π = dφ/dN, u = d/dN):
    #   φ̇ = π H,   φ̈ = H²(uπ − π ε₁),   Ḣ = −ε₁ H²,
    #   ξ̇ = ξ_,φ π H,   ξ̈ = H²[ξ_,φφ π² + ξ_,φ(uπ − π ε₁)].
    # KG (÷H²):    uπ + (−π − 12 H² ξ_,φ) ε₁ = −3π − V_,φ/H² − 12 H² ξ_,φ
    # F2−F1 (÷H²):  4 H² ξ_,φ · uπ + (2 − 12 H² ξ_,φ π) ε₁
    #                              = π²(1 − 4 H² ξ_,φφ) + 12 H² ξ_,φ π
    # GR limit (ξ_,φ → 0, ξ_,φφ → 0):
    #   ε₁ = π²/2,   uπ + (3 − ε₁) π + V_,φ/H² = 0.
    H2 = H * H
    a11 = 1.0
    a12 = -pi - 12.0 * H2 * xip
    b1 = -3.0 * pi - Vp / H2 - 12.0 * H2 * xip

    a21 = 4.0 * H2 * xip
    a22 = 2.0 - 12.0 * H2 * xip * pi
    b2 = pi * pi * (1.0 - 4.0 * H2 * xipp) + 12.0 * H2 * xip * pi

    M = np.array([[a11, a12], [a21, a22]], dtype=float)
    rhs = np.array([b1, b2], dtype=float)
    try:
        sol = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return (float("nan"), float("nan"),
                {"H": H, "V": V, "Vp": Vp, "xip": xip, "xipp": xipp})
    upi, eps1 = float(sol[0]), float(sol[1])
    aux = {"H": H, "V": V, "Vp": Vp, "Vpp": Vpp, "xip": xip, "xipp": xipp,
           "phidot": phidot}
    return upi, eps1, aux


# ---------------------------------------------------------------------------
# Background trajectory dataclass
# ---------------------------------------------------------------------------
@dataclass
class BackgroundTrajectory:
    """Full background trajectory in N for an EGB inflation model."""

    N: np.ndarray             # e-folds, monotonically increasing
    phi: np.ndarray
    pi: np.ndarray            # dφ/dN
    H: np.ndarray
    eps1: np.ndarray          # −Ḣ/H²
    delta1: np.ndarray        # 4 ξ̇ H
    a: np.ndarray             # scale factor (a/a_init)
    tau: np.ndarray           # conformal time τ s.t. dτ = dt/a
    t: np.ndarray             # cosmic time
    phi_end: float
    N_end: float

    def at_N(self, N_target: float) -> dict[str, float]:
        """Interpolate trajectory quantities at a chosen N (counting from
        start of integration, NOT 'before end')."""
        out = {}
        for k in ("phi", "pi", "H", "eps1", "delta1", "a", "tau", "t"):
            out[k] = float(np.interp(N_target, self.N, getattr(self, k)))
        return out

    def N_before_end(self, n: float) -> float:
        """Convert 'n e-folds before end of inflation' to absolute N."""
        return self.N_end - n


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def integrate_background(
    model: EGBModel,
    phi_init: float,
    *,
    pi_init: float | None = None,
    N_max: float = 80.0,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-11,
    n_grid_out: int = 8001,
) -> BackgroundTrajectory | None:
    """Integrate the background from φ_init forward in N until ε₁ = 1.

    Parameters
    ----------
    phi_init : starting field value (deep inflationary regime).
    pi_init  : starting dφ/dN. If None, use slow-roll predictor π ≈ −Q/V.
    N_max    : safety upper limit on number of e-folds to integrate.
    rtol/atol: solver tolerances.
    n_grid_out: number of evenly-spaced N points in the returned trajectory.
    """
    # Slow-roll predictor for π_init
    V0 = float(model.V(phi_init))
    Vp0 = float(model.V_phi(phi_init))
    xip0 = float(model.xi_phi(phi_init))
    if V0 <= 0 or not np.isfinite(V0):
        return None
    Q0 = Vp0 + (4.0 / 3.0) * V0 * V0 * xip0
    if pi_init is None:
        pi_init = -Q0 / V0       # = φ̇/H = (−Q/(3H))/H · 3 = −Q/(3 H²) · H ⇒ −Q/(V) using H²=V/3
        # i.e. dφ/dN = (dφ/dt)/(dN/dt) = φ̇/H ≈ −Q/(3H²) = −Q/V

    # End-of-inflation event: ε₁ − 1 = 0
    def event(N, y):
        phi, pi = y
        _, eps1, _ = _step_rhs(model, phi, pi)
        if not np.isfinite(eps1):
            return -1.0
        return eps1 - 1.0
    event.terminal = True
    event.direction = +1.0

    def rhs(N, y):
        phi, pi = y
        upi, eps1, _ = _step_rhs(model, phi, pi)
        if not np.isfinite(upi):
            return [0.0, 0.0]
        return [pi, upi]

    sol = solve_ivp(
        rhs, (0.0, N_max), [phi_init, pi_init],
        events=event, dense_output=True, rtol=rtol, atol=atol,
        method="DOP853",
    )
    if not sol.success or len(sol.t) < 5:
        return None

    # Use either the event time or the final integration time
    N_end = float(sol.t_events[0][0]) if len(sol.t_events[0]) > 0 else float(sol.t[-1])
    N_grid = np.linspace(0.0, N_end, n_grid_out)
    Y = sol.sol(N_grid)
    phi_grid = Y[0]
    pi_grid = Y[1]

    H_grid = np.empty_like(N_grid)
    eps1_grid = np.empty_like(N_grid)
    delta1_grid = np.empty_like(N_grid)
    for i in range(N_grid.size):
        upi, eps1, aux = _step_rhs(model, float(phi_grid[i]), float(pi_grid[i]))
        H_grid[i] = aux.get("H", np.nan)
        eps1_grid[i] = eps1
        # δ₁ = 4 ξ̇ H = 4 ξ_,φ φ̇ H = 4 ξ_,φ π H²
        delta1_grid[i] = 4.0 * aux.get("xip", np.nan) * pi_grid[i] * (H_grid[i] ** 2)

    a_grid = np.exp(N_grid)             # a/a_init
    # Conformal time τ: dτ = dt/a, dt = dN/H ⇒ dτ = dN/(a H)
    integrand_tau = 1.0 / np.maximum(a_grid * H_grid, 1.0e-30)
    tau_grid = np.concatenate(
        [[0.0], np.cumsum(0.5 * (integrand_tau[1:] + integrand_tau[:-1]) * np.diff(N_grid))]
    )
    # Cosmic time t: dt = dN/H
    integrand_t = 1.0 / np.maximum(H_grid, 1.0e-30)
    t_grid = np.concatenate(
        [[0.0], np.cumsum(0.5 * (integrand_t[1:] + integrand_t[:-1]) * np.diff(N_grid))]
    )

    phi_end = float(phi_grid[-1])

    return BackgroundTrajectory(
        N=N_grid, phi=phi_grid, pi=pi_grid, H=H_grid,
        eps1=eps1_grid, delta1=delta1_grid,
        a=a_grid, tau=tau_grid, t=t_grid,
        phi_end=phi_end, N_end=N_end,
    )


# ---------------------------------------------------------------------------
# Convenience: pick a φ_init that yields ≥ N_pivot+buffer e-folds of inflation
# ---------------------------------------------------------------------------
def _slow_roll_phi_for_N(model: EGBModel, N_target: float,
                         phi_range: tuple[float, float]) -> float | None:
    """Use the slow-roll quadrature to pick a starting φ that should produce
    ~N_target e-folds before end of inflation."""
    phi_end = end_of_inflation(model, phi_range=phi_range, n_grid=4001)
    if phi_end is None:
        return None
    # Walk outward from φ_end until ∫ V/Q ≈ N_target
    sign = 1.0 if (phi_range[1] - phi_end) > (phi_end - phi_range[0]) else -1.0
    phi = phi_end
    accum = 0.0
    n = 4001
    grid = np.linspace(phi_end, phi_end + sign * (max(phi_range) - min(phi_range)), n)
    for i in range(1, n):
        a, b = grid[i - 1], grid[i]
        try:
            VQa = float(model.V(a)) / (float(model.V_phi(a))
                                       + (4.0 / 3.0) * float(model.V(a)) ** 2 * float(model.xi_phi(a)))
            VQb = float(model.V(b)) / (float(model.V_phi(b))
                                       + (4.0 / 3.0) * float(model.V(b)) ** 2 * float(model.xi_phi(b)))
        except Exception:
            return None
        accum += 0.5 * (VQa + VQb) * (b - a) * sign     # makes accum > 0
        if accum >= N_target:
            return float(b)
    return None


def integrate_with_pivot(
    model: EGBModel,
    N_pivot: float | None = None,
    *,
    buffer: float = 5.0,
    phi_range: tuple[float, float] | None = None,
) -> BackgroundTrajectory | None:
    """Integrate so that the trajectory contains at least N_pivot+buffer
    e-folds of inflation, with the pivot crossing at N = N_end − N_pivot.

    N_pivot and phi_range default to the centralized config values.
    """
    from ..config.defaults import DEFAULTS
    if N_pivot is None:
        N_pivot = DEFAULTS.N_pivot
    if phi_range is None:
        phi_range = DEFAULTS.phi_range
    phi_init = _slow_roll_phi_for_N(model, N_pivot + buffer, phi_range)
    if phi_init is None:
        return None
    traj = integrate_background(model, phi_init, N_max=N_pivot + 2 * buffer + 20.0)
    return traj
