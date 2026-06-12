"""Engine test: Starobinsky potential with linear Gauss-Bonnet coupling.

    V(φ) = V₀ (1 − e^{−√(2/3) φ})²,        ξ(φ) = ξ₀ φ

Finds the optimal constants (V₀, ξ₀) against the ACT DR6 targets
(n_s = 0.9752 ± 0.0030 P-ACT-LBDR2; r < 0.038 ⇒ 0 ± 0.019;
dn_s/dlnk = 0.0062 ± 0.0052; ln10¹⁰A_s = 3.044 ± 0.014) using the
analytic N3LO engine, and produces diagnostic plots.

Physics used to reduce the search:
  * The background EOMs are exactly invariant under V → λV, ξ → ξ/λ
    (the rescaling behind `normalize.py`).  Shape observables
    (n_s, r, α_s, n_T, δ₁, c²) therefore depend on (V₀, ξ₀) only through
    the product  g ≡ ξ₀ V₀ ;  V₀ is afterwards fixed *exactly* by A_s.
  * The scan is over the signed effective coupling g, then refined by
    golden-section on the winning branch.

Run:  python scripts/test_staro_linear_xi.py
Outputs: outputs/staro_linear_xi/*.png + results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deepegb.config.defaults import DEFAULTS
from deepegb.physics import EGBModel
from deepegb.physics.egb_background import integrate_with_pivot
from deepegb.physics.egb_n3lo import compute_observables_n3lo, _analytic_grids
from deepegb.physics.egb_perturbations import _observables_from_trajectory

T = DEFAULTS.targets
A_S_TARGET = float(np.exp(T.lnAs) * 1.0e-10)        # ln(10^10 A_s) → A_s
N_PIVOT = DEFAULTS.N_pivot
PHI_RANGE = (0.05, 10.0)
V0_REF = 1.0e-10            # reference amplitude for the scan (rescaled later)

OUT = Path(__file__).resolve().parents[1] / "outputs" / "staro_linear_xi"
OUT.mkdir(parents=True, exist_ok=True)


def make_model(V0: float, xi0: float) -> EGBModel:
    return EGBModel(
        V=lambda p: V0 * (1.0 - np.exp(-np.sqrt(2.0 / 3.0) * p)) ** 2,
        xi=lambda p: xi0 * p,
        name=f"staro+lin-xi (V0={V0:.3e}, xi0={xi0:.3e})",
    )


def observables_for_g(g: float):
    """Observables for effective coupling g = ξ₀V₀ (at V₀ = V0_REF)."""
    obs = compute_observables_n3lo(make_model(V0_REF, g / V0_REF),
                                   N_pivot=N_PIVOT, phi_range=PHI_RANGE)
    return obs


def shape_chi2(obs) -> dict[str, float]:
    """χ² of the V₀-independent shape observables (A_s is normalised
    exactly afterwards, so its term is zero by construction)."""
    if obs is None:
        return dict(total=1.0e6, ns=np.nan, r=np.nan, alphas=np.nan)
    c_ns = ((obs.n_s - T.ns) / T.ns_sigma) ** 2
    c_r = ((obs.r - T.r) / T.r_sigma) ** 2
    c_a = (((obs.alpha_s - T.alphas) / T.alphas_sigma) ** 2
           if T.alphas is not None else 0.0)
    return dict(total=c_ns + c_r + c_a, ns=c_ns, r=c_r, alphas=c_a)


def main() -> None:
    t_start = time.time()

    # ---------------- 1. scan the signed effective coupling ----------------
    # δ₁ ≈ −0.028 g for this family ⇒ |g| ∈ [1e-4, ~6] covers
    # |δ₁| ∈ [3e-6, 0.17].
    g_mag = np.logspace(-4, 0.8, 34)               # |ξ₀V₀|, both signs
    g_scan = np.concatenate([-g_mag[::-1], [0.0], g_mag])

    rows = []
    for g in g_scan:
        obs = observables_for_g(float(g))
        chi = shape_chi2(obs)
        rows.append((float(g), obs, chi))
        if obs is not None:
            print(f"  g={g:+.3e}  n_s={obs.n_s:.5f} r={obs.r:.4f} "
                  f"alpha_s={obs.alpha_s:+.2e} d1={obs.delta1:+.2e} "
                  f"chi2={chi['total']:.1f}")
        else:
            print(f"  g={g:+.3e}  [background/observables failed]")

    valid = [(g, o, c) for g, o, c in rows if o is not None]
    if not valid:
        raise SystemExit("No valid points in the scan — widen the range.")

    # ---------------- 2. golden-section refine on the winning branch -------
    g_best0, _, c_best0 = min(valid, key=lambda r: r[2]["total"])
    print(f"\nScan winner: g = {g_best0:+.3e}, chi2 = {c_best0['total']:.2f}")

    def f(g: float) -> float:
        return shape_chi2(observables_for_g(g))["total"]

    lo, hi = g_best0 * 0.4, g_best0 * 2.5
    if lo > hi:
        lo, hi = hi, lo
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c_, d_ = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c_), f(d_)
    for _ in range(28):
        if fc < fd:
            b, d_, fd = d_, c_, fc
            c_ = b - gr * (b - a)
            fc = f(c_)
        else:
            a, c_, fc = c_, d_, fd
            d_ = a + gr * (b - a)
            fd = f(d_)
    g_opt = c_ if fc < fd else d_

    obs_opt = observables_for_g(g_opt)
    chi_opt = shape_chi2(obs_opt)

    # ---------------- 3. exact A_s normalisation ⇒ (V₀, ξ₀) ----------------
    lam = A_S_TARGET / obs_opt.P_S
    V0_opt = lam * V0_REF
    xi0_opt = (g_opt / V0_REF) / lam            # ξ → ξ/λ
    model_opt = make_model(V0_opt, xi0_opt)
    obs_final = compute_observables_n3lo(model_opt, N_pivot=N_PIVOT,
                                         phi_range=PHI_RANGE)
    # cross-check with full Mukhanov–Sasaki
    obs_ms = _observables_from_trajectory(model_opt, N_PIVOT, PHI_RANGE, 0.5)

    print("\n================ optimal constants ================")
    print(f"  g = xi0*V0 = {g_opt:+.6e}   (rescaling invariant)")
    print(f"  V0  = {V0_opt:.6e}  M_pl^4")
    print(f"  xi0 = {xi0_opt:+.6e}  M_pl^-3")
    print(f"  n_s     = {obs_final.n_s:.5f}   (target {T.ns} ± {T.ns_sigma})")
    print(f"  r       = {obs_final.r:.5f}   (limit r<0.038, 95%)")
    print(f"  alpha_s = {obs_final.alpha_s:+.5e} (target {T.alphas} ± {T.alphas_sigma})")
    print(f"  ln10^10 A_s = {np.log(obs_final.P_S*1e10):.4f} (target {T.lnAs})")
    print(f"  n_T     = {obs_final.n_T:+.5e}")
    print(f"  delta1  = {obs_final.delta1:+.4e}   eps1 = {obs_final.epsilon:.4e}")
    print(f"  c_S^2   = {obs_final.c_S2:.8f}   c_T^2 = {obs_final.c_T2:.8f}")
    print(f"  r/(-8 n_T) = {obs_final.egb_consistency:.4f}")
    print(f"  shape chi2 = {chi_opt['total']:.3f} "
          f"(ns {chi_opt['ns']:.2f}, r {chi_opt['r']:.2f}, "
          f"alpha {chi_opt['alphas']:.2f})")
    if obs_ms is not None:
        print(f"  [MS cross-check] n_s={obs_ms.n_s:.5f} r={obs_ms.r:.5f} "
              f"P_S={obs_ms.P_S:.4e} (n3lo {obs_final.P_S:.4e})")

    # ---------------- 4. plots ----------------
    gs = np.array([g for g, o, c in valid])
    ns_arr = np.array([o.n_s for _, o, _ in valid])
    r_arr = np.array([o.r for _, o, _ in valid])
    al_arr = np.array([o.alpha_s for _, o, _ in valid])
    d1_arr = np.array([o.delta1 for _, o, _ in valid])
    ct2_arr = np.array([o.c_T2 for _, o, _ in valid])
    chi_arr = np.array([c["total"] for _, _, c in valid])

    # (a) n_s – r plane
    fig, ax = plt.subplots(figsize=(7, 5.2))
    for sgn, color, lbl in ((1, "tab:red", r"$\xi_0>0$"),
                            (-1, "tab:blue", r"$\xi_0<0$")):
        m = (np.sign(gs) == sgn)
        order = np.argsort(np.abs(gs[m]))
        ax.plot(ns_arr[m][order], r_arr[m][order], "-o", ms=3,
                color=color, label=lbl)
    gr_pt = [(o.n_s, o.r) for g, o, _ in valid if g == 0.0]
    if gr_pt:
        ax.plot(*gr_pt[0], "ks", ms=9, label=r"GR ($\xi_0=0$)")
    ax.plot(obs_final.n_s, obs_final.r, "g*", ms=18, label="best fit")
    ax.axvspan(T.ns - T.ns_sigma, T.ns + T.ns_sigma, alpha=0.25,
               color="gold", label=r"ACT DR6 $n_s$ (1$\sigma$)")
    ax.axvspan(T.ns - 2 * T.ns_sigma, T.ns + 2 * T.ns_sigma, alpha=0.12,
               color="gold")
    ax.axhline(0.038, ls="--", color="gray",
               label=r"$r<0.038$ (95%, P-ACT-LB+BK18)")
    ax.set_xlabel(r"$n_s$")
    ax.set_ylabel(r"$r$")
    ax.set_title(r"Starobinsky + $\xi_0\phi$ Gauss–Bonnet coupling "
                 f"(N3LO engine, N={N_PIVOT:.0f})")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "ns_r_plane.png", dpi=160)

    # (b) observables vs signed coupling
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax_, y, lbl, tgt, sig in (
        (axes[0, 0], ns_arr, r"$n_s$", T.ns, T.ns_sigma),
        (axes[0, 1], r_arr, r"$r$", 0.0, T.r_sigma),
        (axes[1, 0], al_arr, r"$\alpha_s$", T.alphas, T.alphas_sigma),
        (axes[1, 1], d1_arr, r"$\delta_1$", None, None),
    ):
        ax_.plot(gs, y, ".-", ms=3)
        if tgt is not None:
            ax_.axhline(tgt, color="gold")
            ax_.axhspan(tgt - sig, tgt + sig, alpha=0.25, color="gold")
        ax_.axvline(g_opt, color="green", ls=":", lw=1)
        ax_.set_ylabel(lbl)
        ax_.set_xscale("symlog", linthresh=1e-4)
        ax_.grid(alpha=0.3)
    for ax_ in axes[1]:
        ax_.set_xlabel(r"$g \equiv \xi_0 V_0$  [$M_{\rm pl}$ units]")
    fig.suptitle("Observables vs effective GB coupling (green: best fit)")
    fig.tight_layout()
    fig.savefig(OUT / "observables_vs_coupling.png", dpi=160)

    # (c) chi2 vs coupling
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(gs, chi_arr, ".-")
    ax.axvline(g_opt, color="green", ls=":",
               label=f"best fit g={g_opt:+.2e}")
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_yscale("log")
    ax.set_xlabel(r"$g \equiv \xi_0 V_0$")
    ax.set_ylabel(r"shape $\chi^2$ ($n_s$, $r$, $\alpha_s$)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "chi2_vs_coupling.png", dpi=160)

    # (d) best-fit background evolution
    traj = integrate_with_pivot(model_opt, N_pivot=N_PIVOT,
                                phi_range=PHI_RANGE)
    ag = _analytic_grids(model_opt, traj)
    Nbe = traj.N_end - traj.N
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.semilogy(Nbe, np.abs(ag["eps1"]), label=r"$\varepsilon_1$")
    ax.semilogy(Nbe, np.abs(ag["delta1"]), label=r"$|\delta_1|$")
    ax.axvline(N_PIVOT, color="gray", ls=":", label=r"pivot $N_*$")
    ax.set_xlabel(r"e-folds before end of inflation")
    ax.set_ylabel("slow-roll parameters")
    ax.invert_xaxis()
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Best-fit background evolution")
    fig.tight_layout()
    fig.savefig(OUT / "background_best_fit.png", dpi=160)

    # ---------------- 5. results.json ----------------
    res = dict(
        model="V0*(1-exp(-sqrt(2/3)*phi))**2, xi0*phi",
        N_pivot=N_PIVOT,
        targets=dict(ns=T.ns, ns_sigma=T.ns_sigma, r_limit95=0.038,
                     alphas=T.alphas, alphas_sigma=T.alphas_sigma,
                     lnAs=T.lnAs),
        g_opt=float(g_opt), V0=float(V0_opt), xi0=float(xi0_opt),
        observables=obs_final.as_dict(),
        shape_chi2=chi_opt,
        ms_crosscheck=(obs_ms.as_dict() if obs_ms is not None else None),
        elapsed_s=time.time() - t_start,
    )
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    print(f"\nPlots + results.json written to {OUT}  "
          f"({time.time()-t_start:.1f} s total)")


if __name__ == "__main__":
    main()
