"""
Mukhanov–Sasaki mode integration for EGB inflation.

We integrate the canonical mode equation for each comoving wavenumber k
from deep sub-horizon (Bunch–Davies vacuum) to the end of inflation, then
read off the power spectrum amplitude.

Tensor sector
-------------
For each polarisation, h_k satisfies (in EGB; KLT 2014 + HN 2005):

    d²h/dN² + (3 − ε₁ + d ln G_T/dN) dh/dN + (c_T² k² /(a H)²) h  =  0,

with G_T = M_pl² (1 − δ₁) and c_T² = F_T/G_T given in `egb_perturbations`.

We integrate the canonical Mukhanov variable v_T = a√(G_T) h. Substituting
h = v_T / z_T with z_T = a √G_T and using
    d/dτ = a H d/dN,
    d²/dτ² = (a H)² [d²/dN² + (1 − ε₁) d/dN]
gives

    d²v_T/dN² + (1 − ε₁) dv_T/dN + [c_T² k²/(a H)² − M_T(N)] v_T  =  0,

where the *effective mass* in N-variables is

    M_T(N) ≡  z_T''(τ) / [z_T (a H)²]
            =  (1 − ε₁) g_N + g_NN + g_N²,
    g(N)   ≡  ln z_T  =  N + (1/2) ln G_T(N).

The same form holds for the scalar sector with z_S² = 2 a² ε₁/c_S²,
g_S(N) ≡ ln z_S = N + (1/2) ln(2 ε₁/c_S²).

Initial condition
-----------------
For each k, start at N_init where c k/(a H) ≥ sub_horizon_factor so the
mode is well inside its sound horizon. Bunch–Davies vacuum gives, in
canonical units,

    v(τ_init) = e^{−i c k τ_init} / √(2 c k),
    v'(τ_init) = −i c k v(τ_init).

Mapping to N and absorbing the irrelevant overall phase:

    v(N_init) = 1/√(2 c k),
    dv/dN|_init = −i (c k)/(a H)|_init · v(N_init).

References
----------
* Mukhanov V.F., Feldman H.A., Brandenberger R.H., Phys. Rep. 215 (1992) 203.
* Hwang J., Noh H., gr-qc/0507025.
* Kobayashi T., arXiv:1901.07183 (Horndeski review).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from .egb_background import BackgroundTrajectory, integrate_with_pivot
from .egb_perturbations import compute_c_S2, compute_c_T2
from .egb_slow_roll import EGBModel


# ---------------------------------------------------------------------------
# Per-trajectory helpers
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryInterpolants:
    """Cached arrays on the uniform-N grid of a BackgroundTrajectory."""

    traj: BackgroundTrajectory
    G_T: np.ndarray         # 1 − δ₁
    c_T2: np.ndarray
    c_S2: np.ndarray
    g_T: np.ndarray         # ln z_T = N + 1/2 ln G_T
    g_T_N: np.ndarray       # d g_T / dN
    g_T_NN: np.ndarray      # d² g_T / dN²
    g_S: np.ndarray         # ln z_S = N + 1/2 ln(2 ε₁ / c_S²)
    g_S_N: np.ndarray
    g_S_NN: np.ndarray
    M_T: np.ndarray         # effective mass for v_T equation
    M_S: np.ndarray         # effective mass for v_S equation


def _smooth_ln(arr: np.ndarray, floor: float = 1.0e-30) -> np.ndarray:
    """Take ln of an array, replacing non-positive entries by interpolation."""
    safe = np.where((arr > 0) & np.isfinite(arr), arr, np.nan)
    out = np.log(np.clip(safe, floor, None))
    bad = ~np.isfinite(out)
    if bad.any() and (~bad).sum() > 1:
        idx = np.arange(out.size)
        out[bad] = np.interp(idx[bad], idx[~bad], out[~bad])
    return out


def _N_derivatives(N: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (dy/dN, d²y/dN²) on a uniform N grid using central differences."""
    dN = N[1] - N[0]
    y_N = np.gradient(y, dN, edge_order=2)
    y_NN = np.gradient(y_N, dN, edge_order=2)
    return y_N, y_NN


def precompute_mode_inputs(model: EGBModel,
                           traj: BackgroundTrajectory) -> TrajectoryInterpolants:
    """Precompute G_T, c_T², c_S², z, M(N) on the trajectory grid.

    Sound speeds are evaluated with **trajectory-exact** H, ε₁, δ₁, ξ_,φ
    (from the full ODE integration) rather than the slow-roll H²=V/3 seed.
    This eliminates the leading source of c_S² error without changing the
    formula itself.
    """
    n = traj.N.size
    G_T = 1.0 - traj.delta1
    c_T2 = np.empty(n)
    c_S2 = np.empty(n)
    for i, p in enumerate(traj.phi):
        c_T2[i] = compute_c_T2(model, float(p),
                                eps=float(traj.eps1[i]),
                                delta1=float(traj.delta1[i]))
        xip_i = float(model.xi_phi(float(p)))
        c_S2[i] = compute_c_S2(model, float(p),
                                eps=float(traj.eps1[i]),
                                delta1=float(traj.delta1[i]),
                                H=float(traj.H[i]),
                                xip=xip_i)
    # Replace bad entries with linear interpolation from neighbours.
    for arr in (G_T, c_T2, c_S2):
        bad = ~np.isfinite(arr) | (arr <= 0)
        if bad.any() and (~bad).sum() > 1:
            arr[bad] = np.interp(traj.N[bad], traj.N[~bad], arr[~bad])
        else:
            arr[bad] = 1.0

    # Tensor: g_T = N + (1/2) ln G_T  (since a/a_init = e^N ⇒ ln a = N).
    g_T = traj.N + 0.5 * _smooth_ln(G_T)
    g_T_N, g_T_NN = _N_derivatives(traj.N, g_T)

    # Scalar: g_S = N + (1/2) ln (2 ε₁ / c_S²)
    eps = np.maximum(traj.eps1, 1.0e-30)
    g_S_arg = 2.0 * eps / np.maximum(c_S2, 1.0e-30)
    g_S = traj.N + 0.5 * _smooth_ln(g_S_arg)
    g_S_N, g_S_NN = _N_derivatives(traj.N, g_S)

    one_minus_eps = 1.0 - traj.eps1
    M_T = one_minus_eps * g_T_N + g_T_NN + g_T_N * g_T_N
    M_S = one_minus_eps * g_S_N + g_S_NN + g_S_N * g_S_N

    return TrajectoryInterpolants(
        traj=traj, G_T=G_T, c_T2=c_T2, c_S2=c_S2,
        g_T=g_T, g_T_N=g_T_N, g_T_NN=g_T_NN,
        g_S=g_S, g_S_N=g_S_N, g_S_NN=g_S_NN,
        M_T=M_T, M_S=M_S,
    )


# ---------------------------------------------------------------------------
# Single-mode integrator (in N variable)
# ---------------------------------------------------------------------------
@dataclass
class ModeResult:
    k: float
    P: float
    is_valid: bool
    N_horizon: float
    N_init: float
    N_final: float


def _interp_factory(N_grid: np.ndarray, y: np.ndarray):
    return lambda N: float(np.interp(N, N_grid, y))


def integrate_mode_in_N(
    k: float,
    interp: TrajectoryInterpolants,
    *,
    sector: str = "tensor",
    sub_horizon_factor: float = 50.0,
    rtol: float = 1.0e-8,
    atol: float = 1.0e-12,
) -> tuple[float, ModeResult]:
    """Integrate one canonical mode v(N) and return (|v|² at end, ModeResult)."""
    traj = interp.traj
    N_grid = traj.N
    aH = traj.a * traj.H
    if sector == "tensor":
        c2 = interp.c_T2
        M = interp.M_T
    elif sector == "scalar":
        c2 = interp.c_S2
        M = interp.M_S
    else:
        raise ValueError(f"unknown sector {sector!r}")

    # k/aH along the trajectory (in units of c·k):
    ratio = (np.sqrt(np.maximum(c2, 1.0e-30)) * k) / np.maximum(aH, 1.0e-30)
    # Horizon crossing: ratio = 1
    crossing_idx = int(np.argmin(np.abs(ratio - 1.0)))
    # Initial idx: walk back until ratio ≥ sub_horizon_factor
    init_idx = 0
    for i in range(crossing_idx, -1, -1):
        if ratio[i] >= sub_horizon_factor:
            init_idx = i
            break
    if init_idx >= crossing_idx:
        # mode is never deeply sub-horizon ⇒ trajectory too short for this k
        return float("nan"), ModeResult(k=k, P=float("nan"), is_valid=False,
                                        N_horizon=float(N_grid[crossing_idx]),
                                        N_init=float(N_grid[init_idx]),
                                        N_final=float(N_grid[-1]))

    # Integration interval
    N_init = float(N_grid[init_idx])
    N_final = float(N_grid[-1])

    # Bunch-Davies in N (drop irrelevant overall phase, keep complex amplitude):
    cT_init = float(np.sqrt(max(c2[init_idx], 1.0e-30)))
    aH_init = float(aH[init_idx])
    norm = 1.0 / np.sqrt(2.0 * cT_init * k)
    Re_v0 = norm
    Im_v0 = 0.0
    # dv/dN|_init = -i c k/(aH) v_init  ⇒ Re(v') = 0, Im(v') = -ck/(aH) · norm
    Re_vp0 = 0.0
    Im_vp0 = -cT_init * k / aH_init * norm

    c2_of = _interp_factory(N_grid, c2)
    M_of = _interp_factory(N_grid, M)
    eps_of = _interp_factory(N_grid, traj.eps1)
    aH_of = _interp_factory(N_grid, aH)

    def rhs(N, y):
        c2N = c2_of(N)
        MN = M_of(N)
        epsN = eps_of(N)
        aHN = aH_of(N)
        # v_NN + (1-ε) v_N + (c² k²/(aH)² - M) v = 0
        coef_v = c2N * k * k / (aHN * aHN) - MN
        coef_vN = 1.0 - epsN
        # y = [Re v, Re v_N, Im v, Im v_N]
        return [
            y[1],
            -coef_vN * y[1] - coef_v * y[0],
            y[3],
            -coef_vN * y[3] - coef_v * y[2],
        ]

    sol = solve_ivp(
        rhs, (N_init, N_final),
        [Re_v0, Re_vp0, Im_v0, Im_vp0],
        method="DOP853", rtol=rtol, atol=atol,
    )
    if not sol.success:
        return float("nan"), ModeResult(k=k, P=float("nan"), is_valid=False,
                                        N_horizon=float(N_grid[crossing_idx]),
                                        N_init=N_init, N_final=N_final)
    Re_v, _, Im_v, _ = sol.y[:, -1]
    v_sq = Re_v * Re_v + Im_v * Im_v
    return v_sq, ModeResult(k=k, P=float("nan"), is_valid=True,
                            N_horizon=float(N_grid[crossing_idx]),
                            N_init=N_init, N_final=N_final)


# ---------------------------------------------------------------------------
# Power spectra over a range of k
# ---------------------------------------------------------------------------
def tensor_power_spectrum(
    model: EGBModel,
    k_array: np.ndarray,
    *,
    traj: BackgroundTrajectory | None = None,
    N_pivot: float = 55.0,
) -> tuple[np.ndarray, list[ModeResult]]:
    """P_T(k) for an array of comoving wavenumbers k.

    P_T(k) = 2 (k³/2π²) |h_k|² with |h_k|² = 2 |v_T|²/(a² G_T) (the factor
    of 2 accounts for the two polarisations).
    """
    if traj is None:
        traj = integrate_with_pivot(model, N_pivot=N_pivot)
        if traj is None:
            return np.full_like(k_array, np.nan, dtype=float), []
    interp = precompute_mode_inputs(model, traj)
    a_end = float(traj.a[-1])
    G_T_end = float(interp.G_T[-1])

    P_T = np.empty(k_array.size, dtype=float)
    results: list[ModeResult] = []
    for i, k in enumerate(k_array):
        v_sq, mr = integrate_mode_in_N(float(k), interp, sector="tensor")
        if not (np.isfinite(v_sq) and v_sq > 0):
            P_T[i] = float("nan")
            results.append(mr)
            continue
        # Per-polarisation Mukhanov variable v = (a √G_T / √2) h ⇒
        # |h|² = 2|v|²/(a² · G_T/2) = 4|v|²/(a² G_T)  for one polarisation.
        # Total tensor power = 2 × (k³/2π²) × |h|².
        h_sq = 4.0 * v_sq / (a_end * a_end * G_T_end)
        P_T[i] = 2.0 * (k ** 3) / (2.0 * np.pi ** 2) * h_sq
        mr.P = P_T[i]
        results.append(mr)
    return P_T, results


def scalar_power_spectrum(
    model: EGBModel,
    k_array: np.ndarray,
    *,
    traj: BackgroundTrajectory | None = None,
    N_pivot: float = 55.0,
) -> tuple[np.ndarray, list[ModeResult]]:
    """P_S(k) for an array of comoving wavenumbers k.

    P_S(k) = (k³/2π²) |R_k|² with |R_k|² = |v_S|²/z_S² and z_S² = 2 a² ε₁/c_S².
    """
    if traj is None:
        traj = integrate_with_pivot(model, N_pivot=N_pivot)
        if traj is None:
            return np.full_like(k_array, np.nan, dtype=float), []
    interp = precompute_mode_inputs(model, traj)
    a_end = float(traj.a[-1])
    eps_end = float(traj.eps1[-1])
    cS2_end = float(interp.c_S2[-1])
    zS2_end = 2.0 * a_end * a_end * eps_end / max(cS2_end, 1.0e-30)

    P_S = np.empty(k_array.size, dtype=float)
    results: list[ModeResult] = []
    for i, k in enumerate(k_array):
        v_sq, mr = integrate_mode_in_N(float(k), interp, sector="scalar")
        if not (np.isfinite(v_sq) and v_sq > 0):
            P_S[i] = float("nan")
            results.append(mr)
            continue
        R_sq = v_sq / zS2_end
        P_S[i] = (k ** 3) / (2.0 * np.pi ** 2) * R_sq
        mr.P = P_S[i]
        results.append(mr)
    return P_S, results


def k_pivot_from_traj(traj: BackgroundTrajectory, N_pivot: float = 55.0) -> float:
    """Comoving wavenumber that crossed the Hubble horizon N_pivot e-folds
    before end of inflation."""
    idx = int(np.argmin(np.abs(traj.N - (traj.N_end - N_pivot))))
    return float(traj.a[idx] * traj.H[idx])
