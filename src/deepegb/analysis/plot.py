"""Diagnostic plot for an EGB inflation model — production-grade.

A 7-panel figure:

    ┌──────────────┬──────────────┬─────────────────┐
    │ V(φ)         │ ξ(φ)         │ ε(φ), |δ₁(φ)|   │
    ├──────────────┼──────────────┼─────────────────┤
    │ c_T²(φ)      │ ln P_S(N)    │ (n_s, r) plane  │
    ├──────────────┴──────────────┴─────────────────┤
    │ Relic GW spectrum Ω_GW(f) h² + detectors      │
    └────────────────────────────────────────────────┘
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..config.defaults import DEFAULTS
from ..physics import (
    DETECTORS,
    background_along,
    compute_N_pivot_from_model,
    compute_c_T2,
    compute_observables_full,
    egb_consistency_metric,
    integrate_with_pivot,
    k_inflation_to_today_Mpc_inv,
    k_pivot_from_traj,
    power_spectra_at,
    relic_gw_spectrum,
)
from ..search.pysr_search import expressions_to_model


def _resolve_N_auto(model, N, phi_range, T_reh_GeV=None):
    N_was_auto = N is None
    if N is None:
        kw = dict(
            phi_range=phi_range,
            N_min=DEFAULTS.physics.N_pivot_min,
            N_max=DEFAULTS.physics.N_pivot_max,
        )
        if T_reh_GeV is not None:
            kw["T_reh_GeV"] = T_reh_GeV
        N = compute_N_pivot_from_model(model, **kw)
    return float(N), N_was_auto


def plot_egb_model(
    V_expr: str,
    xi_expr: str | None = None,
    *,
    N: float | None = None,
    out_path: str | Path = "egb_diagnostic.png",
    phi_range: tuple[float, float] | None = None,
    title: str | None = None,
    n_decades_gw: float | None = None,
    n_k_gw: int = 20,
    T_reh_GeV: float | None = None,
) -> str:
    """Render a 7-panel diagnostic plot (6 physics + 1 GW spectrum).
    Returns the saved path as a string."""
    if xi_expr is None:
        xi_expr = DEFAULTS.default_xi_expr
    if phi_range is None:
        phi_range = DEFAULTS.phi_range
    if n_decades_gw is None:
        n_decades_gw = DEFAULTS.n_decades
    if T_reh_GeV is None:
        T_reh_GeV = DEFAULTS.T_reh_GeV

    model = expressions_to_model(V_expr, xi_expr)
    N, N_was_auto = _resolve_N_auto(model, N, phi_range, T_reh_GeV=T_reh_GeV)
    obs = compute_observables_full(model, N_pivot=N, phi_range=phi_range, n_grid=4001)

    phi = np.linspace(*phi_range, 600)
    bg = background_along(model, phi)
    cT2 = np.array([compute_c_T2(model, p) for p in phi])

    # P_S along trajectory (parametrized by φ; we'll convert to N for display)
    PS_arr = np.array([power_spectra_at(model, p)["P_S"] for p in phi])

    fig, ax = plt.subplots(3, 3, figsize=(16, 14),
                           gridspec_kw={"height_ratios": [1, 1, 1.2]})
    # Remove bottom-right and bottom-middle axes; span bottom row for GW
    ax_gw = fig.add_subplot(3, 1, 3)
    ax[2, 0].remove()
    ax[2, 1].remove()
    ax[2, 2].remove()

    _title = title or f"EGB diagnostic: V={V_expr}    ξ={xi_expr}"
    if N_was_auto:
        _title += f"    [auto N_pivot={N:.2f}]"
    fig.suptitle(_title, fontsize=11)

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

    # 6: (n_s, r) plane with EGB-aware annotation
    p = ax[1, 2]
    p.set_xlim(0.93, 1.00); p.set_ylim(1e-4, 0.3); p.set_yscale("log")
    p.set_xlabel(r"$n_s$"); p.set_ylabel(r"$r$")
    p.set_title(r"$(n_s, r)$ prediction")
    p.axvspan(0.957, 0.975, color="C0", alpha=0.15, label="Planck 2018 1σ-ish")
    p.axvspan(0.971, 0.977, color="C1", alpha=0.15, label="ACT DR6 1σ-ish")
    p.axhline(0.036, color="k", ls=":", lw=0.8, alpha=0.6, label=r"$r<0.036$ (BK18 95%)")
    if obs.is_valid:
        cons = egb_consistency_metric(obs)
        egb_cons = cons["egb_consistency"]
        p.plot([obs.n_s], [max(obs.r, 1e-5)], marker="*", markersize=15,
               color="red")
        p.annotate(
            (f"$n_s={obs.n_s:.4f}$\n$r={obs.r:.4f}$\n"
             f"$n_T={obs.n_T:.4f}$\n$\\alpha_s={obs.alpha_s:.4g}$\n"
             f"$c_T^2={obs.c_T2:.4f}$\n"
             f"EGB cons.={egb_cons:.3f}"),
            xy=(obs.n_s, max(obs.r, 1e-5)),
            xytext=(8, 8), textcoords="offset points", fontsize=8,
        )
    else:
        p.text(0.5, 0.5, "no finite observables",
               transform=p.transAxes, ha="center", va="center", color="red")
    p.legend(fontsize=7, loc="lower right"); p.grid(alpha=0.25, which="both")

    # 7: Relic GW spectrum panel
    traj = integrate_with_pivot(model, N_pivot=N, phi_range=phi_range)
    if traj is not None:
        k_pivot = k_pivot_from_traj(traj, N_pivot=N)
        k_arr_gw = k_pivot * np.logspace(-n_decades_gw / 2, n_decades_gw / 2, n_k_gw)
        spec = relic_gw_spectrum(model, k_arr_gw, traj=traj, N_pivot=N,
                                 T_reh_GeV=T_reh_GeV)
        if spec.f_today is not None:
            gw_valid = np.isfinite(spec.Omega_GW_h2) & (spec.Omega_GW_h2 > 0)
            if gw_valid.any():
                ax_gw.loglog(spec.f_today[gw_valid], spec.Omega_GW_h2[gw_valid],
                             "o-", lw=2.4, ms=5, color="#d62728",
                             zorder=10, label=r"$\Omega_{\rm GW}\,h^2$  (this model)")
        # Detector overlay
        if spec.f_today is not None:
            f_min_gw = max(min(spec.f_today.min() * 0.1, 1e-19), 1e-22)
            f_max_gw = max(spec.f_today.max() * 10, 1e4)
        else:
            f_min_gw, f_max_gw = 1e-19, 1e4
        f_grid = np.logspace(np.log10(f_min_gw), np.log10(f_max_gw), 400)
        for d in DETECTORS[:8]:  # top detectors for readability
            sens = d.sensitivity(f_grid)
            finite = np.isfinite(sens) & (sens < 1e-3)
            if not finite.any():
                continue
            ax_gw.plot(f_grid[finite], sens[finite],
                       color=PROBE_COLORS.get(d.probe, "grey"),
                       ls=ERA_LINESTYLES.get(d.era, "-"),
                       lw=1.0, alpha=0.6, label=d.name)
    else:
        ax_gw.text(0.5, 0.5, "background integration failed — no GW panel",
                   transform=ax_gw.transAxes, ha="center", va="center", color="red")
    ax_gw.set_xlabel(r"$f$ today  [Hz]")
    ax_gw.set_ylabel(r"$\Omega_{\rm GW}\,h^2$")
    ax_gw.grid(alpha=0.25, which="both")
    ax_gw.set_ylim(1e-22, 1e-3)
    ax_gw.legend(loc="upper right", fontsize=6, ncol=3, framealpha=0.85)
    ax_gw.set_title("Relic GW spectrum + detector sensitivities")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return str(out_path)


PROBE_COLORS = {
    "PTA": "#1f77b4",
    "space": "#ff7f0e",
    "ground": "#2ca02c",
    "CMB": "#9467bd",
}
ERA_LINESTYLES = {
    "current":  "-",
    "planned":  "--",
    "proposed": ":",
}


def plot_relic_gw_spectrum(
    V_expr: str,
    xi_expr: str | None = None,
    *,
    N: float | None = None,
    n_decades: float | None = None,
    n_k: int | None = None,
    T_reh_GeV: float | None = None,
    out_path: str | Path = "relic_gw.png",
    title: str | None = None,
    phi_range: tuple[float, float] | None = None,
    detectors: tuple[str, ...] | None = None,
    show_eras: tuple[str, ...] = ("current", "planned", "proposed"),
) -> str:
    """Plot Ω_GW(f) h² across the relic-GW frequency band with overlay of
    detector sensitivity curves from the catalogue in `physics.detectors`.

    Parameters
    ----------
    detectors  : explicit list of detector names to overlay; default = all.
    show_eras  : restrict overlays to detectors in these eras.
    n_decades  : log-decades of k around pivot. Default from config (8).
    n_k        : number of k samples. Default from config (30).
    T_reh_GeV  : reheating temperature (GeV). Default from config.
    """
    if xi_expr is None:
        xi_expr = DEFAULTS.default_xi_expr
    if phi_range is None:
        phi_range = DEFAULTS.phi_range
    if n_decades is None:
        n_decades = DEFAULTS.n_decades
    if n_k is None:
        n_k = DEFAULTS.n_k
    if T_reh_GeV is None:
        T_reh_GeV = DEFAULTS.T_reh_GeV

    model = expressions_to_model(V_expr, xi_expr)
    N, N_was_auto = _resolve_N_auto(model, N, phi_range, T_reh_GeV=T_reh_GeV)
    traj = integrate_with_pivot(model, N_pivot=N, phi_range=phi_range)
    if traj is None:
        raise RuntimeError("background integration failed for relic-GW plot")

    k_pivot = k_pivot_from_traj(traj, N_pivot=N)
    k_arr = k_pivot * np.logspace(-n_decades / 2, n_decades / 2, n_k)
    spec = relic_gw_spectrum(model, k_arr, traj=traj, N_pivot=N, T_reh_GeV=T_reh_GeV)

    fig, ax = plt.subplots(1, 1, figsize=(13, 7))

    # 1. The model's predicted Ω_GW
    if spec.f_today is not None:
        valid = np.isfinite(spec.Omega_GW_h2) & (spec.Omega_GW_h2 > 0)
        ax.loglog(spec.f_today[valid], spec.Omega_GW_h2[valid],
                  "o-", lw=2.4, ms=5, color="#d62728",
                  zorder=10, label=r"$\Omega_{\rm GW}\,h^2$  (this model)")

    # 2. Detector sensitivity curves
    if spec.f_today is not None:
        f_min = max(min(spec.f_today.min() * 0.1, 1e-19), 1e-22)
        f_max = max(spec.f_today.max() * 10, 1e4)
    else:
        f_min, f_max = 1e-19, 1e4
    f_grid = np.logspace(np.log10(f_min), np.log10(f_max), 400)

    detector_set = set(detectors) if detectors else None
    for d in DETECTORS:
        if d.era not in show_eras:
            continue
        if detector_set is not None and d.name not in detector_set:
            continue
        sens = d.sensitivity(f_grid)
        finite = np.isfinite(sens)
        if not finite.any():
            continue
        ax.plot(
            f_grid[finite], sens[finite],
            color=PROBE_COLORS.get(d.probe, "grey"),
            ls=ERA_LINESTYLES.get(d.era, "-"),
            lw=1.4, alpha=0.85, label=f"{d.name} ({d.probe}, {d.era})",
        )

    ax.set_xlabel(r"$f$ today  [Hz]")
    ax.set_ylabel(r"$\Omega_{\rm GW}\,h^2$")
    ax.grid(alpha=0.25, which="both")
    ax.set_xlim(f_min, f_max)
    ax.set_ylim(1e-22, 1e-3)
    ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.85)
    _title = title or f"Relic GW spectrum:  V={V_expr},  ξ={xi_expr},  T_reh={T_reh_GeV} GeV"
    if N_was_auto:
        _title += f"    [auto N_pivot={N:.2f}]"
    ax.set_title(_title)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return str(out_path)
