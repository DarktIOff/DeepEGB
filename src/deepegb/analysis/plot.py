"""Diagnostic plot for an EGB inflation model — production-grade.

A 6-panel figure:

    ┌──────────────┬──────────────┬─────────────────┐
    │ V(φ)         │ ξ(φ)         │ ε(φ), |δ₁(φ)|   │
    ├──────────────┼──────────────┼─────────────────┤
    │ c_T²(φ)      │ ln P_S(N)    │ (n_s, r) plane  │
    └──────────────┴──────────────┴─────────────────┘
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..physics import (
    EXPERIMENT_BANDS,
    background_along,
    compute_c_T2,
    compute_observables_full,
    integrate_with_pivot,
    k_inflation_to_today_Mpc_inv,
    k_pivot_from_traj,
    power_spectra_at,
    relic_gw_spectrum,
)
from ..search.pysr_search import expressions_to_model


def plot_egb_model(
    V_expr: str,
    xi_expr: str = "0",
    *,
    N: float = 55.0,
    out_path: str | Path = "egb_diagnostic.png",
    phi_range: tuple[float, float] = (-10.0, 10.0),
    title: str | None = None,
) -> str:
    """Render a 6-panel diagnostic plot. Returns the saved path as a string."""
    model = expressions_to_model(V_expr, xi_expr)
    obs = compute_observables_full(model, N_pivot=N, phi_range=phi_range, n_grid=4001)

    phi = np.linspace(*phi_range, 600)
    bg = background_along(model, phi)
    cT2 = np.array([compute_c_T2(model, p) for p in phi])

    # P_S along trajectory (parametrized by φ; we'll convert to N for display)
    PS_arr = np.array([power_spectra_at(model, p)["P_S"] for p in phi])

    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.suptitle(title or f"EGB diagnostic: V={V_expr}    ξ={xi_expr}",
                 fontsize=11)

    # 1: V(φ)
    ax[0, 0].plot(phi, bg["V"], lw=1.6)
    ax[0, 0].set_xlabel(r"$\phi/M_{\rm Pl}$"); ax[0, 0].set_ylabel(r"$V(\phi)$")
    ax[0, 0].set_title("Potential"); ax[0, 0].grid(alpha=0.25)

    # 2: ξ(φ)
    ax[0, 1].plot(phi, bg["xi"], lw=1.6, color="C2")
    ax[0, 1].set_xlabel(r"$\phi/M_{\rm Pl}$"); ax[0, 1].set_ylabel(r"$\xi(\phi)$")
    ax[0, 1].set_title("GB coupling"); ax[0, 1].grid(alpha=0.25)

    # 3: ε(φ) and δ₁(φ)
    ax[0, 2].semilogy(phi, np.abs(bg["eps"]), lw=1.6, label=r"$\varepsilon$")
    ax[0, 2].semilogy(phi, np.abs(bg["delta1"]) + 1e-30, lw=1.6,
                      ls="--", label=r"$|\delta_1|$")
    ax[0, 2].axhline(1.0, color="k", lw=0.7, ls=":", alpha=0.6)
    if np.isfinite(obs.phi_end):
        ax[0, 2].axvline(obs.phi_end, color="grey", lw=0.6, alpha=0.6)
    if np.isfinite(obs.phi_N):
        ax[0, 2].axvline(obs.phi_N, color="C0", lw=0.6, alpha=0.6)
    ax[0, 2].set_xlabel(r"$\phi/M_{\rm Pl}$"); ax[0, 2].set_ylabel("slow-roll")
    ax[0, 2].set_title(r"$\varepsilon(\phi),\ |\delta_1(\phi)|$")
    ax[0, 2].legend(fontsize=8); ax[0, 2].grid(alpha=0.25, which="both")

    # 4: c_T²(φ)
    ax[1, 0].plot(phi, cT2, lw=1.6, color="C3")
    ax[1, 0].axhline(1.0, color="k", lw=0.6, ls="--", alpha=0.6)
    if np.isfinite(obs.phi_N):
        ax[1, 0].axvline(obs.phi_N, color="C0", lw=0.6, alpha=0.6,
                         label=fr"$\phi_N$  ($c_T^2={obs.c_T2:.4f}$)")
    ax[1, 0].set_xlabel(r"$\phi/M_{\rm Pl}$"); ax[1, 0].set_ylabel(r"$c_T^2$")
    ax[1, 0].set_title(r"Tensor sound speed")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.25)

    # 5: ln P_S(φ)
    valid = np.isfinite(PS_arr) & (PS_arr > 0)
    if valid.any():
        ax[1, 1].plot(phi[valid], np.log(PS_arr[valid]), lw=1.6, color="C4")
    ax[1, 1].set_xlabel(r"$\phi/M_{\rm Pl}$"); ax[1, 1].set_ylabel(r"$\ln P_{\mathcal{R}}$")
    if np.isfinite(obs.phi_N):
        ax[1, 1].axvline(obs.phi_N, color="C0", lw=0.6, alpha=0.6)
    ax[1, 1].set_title("Scalar power along trajectory")
    ax[1, 1].grid(alpha=0.25)

    # 6: (n_s, r) plane
    p = ax[1, 2]
    p.set_xlim(0.93, 1.00); p.set_ylim(1e-4, 0.3); p.set_yscale("log")
    p.set_xlabel(r"$n_s$"); p.set_ylabel(r"$r$")
    p.set_title(r"$(n_s, r)$ prediction")
    p.axvspan(0.957, 0.975, color="C0", alpha=0.15, label="Planck 2018 1σ-ish")
    p.axvspan(0.971, 0.977, color="C1", alpha=0.15, label="ACT DR6 1σ-ish")
    p.axhline(0.036, color="k", ls=":", lw=0.8, alpha=0.6, label=r"$r<0.036$ (BK18 95%)")
    if obs.is_valid:
        p.plot([obs.n_s], [max(obs.r, 1e-5)], marker="*", markersize=15,
               color="red")
        cons = obs.r / (-8 * obs.n_T) if obs.n_T not in (0.0,) and obs.n_T == obs.n_T else float('nan')
        p.annotate(
            (f"$n_s={obs.n_s:.4f}$\n$r={obs.r:.4f}$\n"
             f"$n_T={obs.n_T:.4f}$\n$\\alpha_s={obs.alpha_s:.4g}$\n"
             f"$c_T^2={obs.c_T2:.4f}$\n"
             f"$r/(-8n_T)={cons:.3f}$"),
            xy=(obs.n_s, max(obs.r, 1e-5)),
            xytext=(8, 8), textcoords="offset points", fontsize=8,
        )
    else:
        p.text(0.5, 0.5, "no finite observables",
               transform=p.transAxes, ha="center", va="center", color="red")
    p.legend(fontsize=7, loc="lower right"); p.grid(alpha=0.25, which="both")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return str(out_path)


def plot_relic_gw_spectrum(
    V_expr: str,
    xi_expr: str = "0",
    *,
    N: float = 55.0,
    n_decades: float = 8.0,
    n_k: int = 30,
    T_reh_GeV: float | None = 1.0e15,
    out_path: str | Path = "relic_gw.png",
    title: str | None = None,
    phi_range: tuple[float, float] = (-15.0, 15.0),
) -> str:
    """Plot Ω_GW(f) h² across the relic GW frequency band, with shaded
    overlays for PTA, LISA, DECIGO, ET/LIGO sensitivity ranges."""
    model = expressions_to_model(V_expr, xi_expr)
    traj = integrate_with_pivot(model, N_pivot=N, phi_range=phi_range)
    if traj is None:
        raise RuntimeError("background integration failed for relic-GW plot")

    k_pivot = k_pivot_from_traj(traj, N_pivot=N)
    k_arr = k_pivot * np.logspace(-n_decades / 2, n_decades / 2, n_k)
    spec = relic_gw_spectrum(model, k_arr, traj=traj, N_pivot=N, T_reh_GeV=T_reh_GeV)

    fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    if spec.f_today is not None:
        valid = np.isfinite(spec.Omega_GW_h2) & (spec.Omega_GW_h2 > 0)
        ax.loglog(spec.f_today[valid], spec.Omega_GW_h2[valid],
                  "o-", lw=2, ms=5, color="C3", label=r"$\Omega_{\rm GW} h^2$")

        # Experimental band overlays
        colors = {"PTA": "C0", "LISA": "C1", "DECIGO": "C4", "ET": "C2",
                  "LIGO": "C7", "CMB-pol": "C5"}
        for band, (lo, hi) in EXPERIMENT_BANDS.items():
            ax.axvspan(lo, hi, alpha=0.10, color=colors.get(band, "grey"),
                       label=band)

    ax.set_xlabel(r"$f$ today  [Hz]")
    ax.set_ylabel(r"$\Omega_{\rm GW}\,h^2$")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.set_title(title or
                 f"Relic GW spectrum:  V={V_expr},  ξ={xi_expr},  T_reh={T_reh_GeV} GeV")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return str(out_path)
