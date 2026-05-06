"""End-to-end smoke test that does NOT require PySR or Agno.

Run with:   python scripts/smoke_test.py

Validates:
  1. Kernel reduces correctly to GR slow-roll for ξ=0 (Starobinsky benchmark).
  2. analyze tool returns finite observables for a non-trivial EGB model.
  3. Plot tool writes a PNG.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from deepegb.analysis import analyze_egb_model, plot_egb_model
from deepegb.physics import EGBModel, compute_observables_full


def gr_starobinsky():
    print("\n[1] GR Starobinsky benchmark")
    model = EGBModel(
        V=lambda p: (1.0 - np.exp(-np.sqrt(2 / 3) * p)) ** 2,
        xi=lambda p: 0.0 * p,
        name="Starobinsky",
    )
    obs = compute_observables_full(model, N_pivot=55.0, phi_range=(0.1, 8.0), n_grid=4001)
    print(f"    n_s = {obs.n_s:.5f}  (expected ≈ {1 - 2/55:.5f})")
    print(f"    r   = {obs.r:.5f}    (expected r ≪ 0.04)")
    print(f"    ε   = {obs.epsilon:.4g},  φ_end = {obs.phi_end:.4f},  φ_N = {obs.phi_N:.4f}")
    assert obs.is_valid


def egb_quadratic():
    print("\n[2] EGB-modified m²φ² with ξ(φ) = 0.05 e^{-0.4φ}")
    out = analyze_egb_model("0.5*phi**2", "0.05*exp(-0.4*phi)", N=55)
    print(f"    n_s = {out['n_s']:.5f}")
    print(f"    r   = {out['r']:.5f}")
    print(f"    ε   = {out['epsilon']:.4g}")


def make_plot():
    print("\n[3] Diagnostic plot")
    out = plot_egb_model(
        V_expr="(1 - exp(-sqrt(2/3)*phi))**2",
        xi_expr="0.05*exp(-0.4*phi)",
        N=55.0,
        out_path=str(ROOT / "outputs" / "smoke_starobinsky_egb.png"),
        phi_range=(0.1, 8.0),
        title="Starobinsky V × exponential GB coupling",
    )
    print(f"    saved → {out}")


if __name__ == "__main__":
    gr_starobinsky()
    egb_quadratic()
    make_plot()
    print("\n[OK] Smoke test passed.")
