# Physics conventions — leading-order kernel

> **Read [PHYSICS_PRODUCTION.md](PHYSICS_PRODUCTION.md) instead if you are
> using the production kernel** (`compute_observables_full`, the χ² used by
> `deepegb search`, and everything in `egb_perturbations.py`).
> This file describes the simpler `egb_slow_roll.py` kernel, which is kept
> only as a backward-compatible toy and a sanity-check oracle.


## Action

In reduced-Planck-mass units (`M_pl = 1`):

```
S = ∫ d⁴x √(-g) [ R/2 − (1/2)(∂φ)² − V(φ) − (1/2) ξ(φ) 𝒢 ]
```

with the Gauss–Bonnet density `𝒢 = R² − 4 R_μν R^μν + R_μνρσ R^μνρσ`.

## Slow-roll formulas implemented

For the FLRW background, retaining only leading terms in the slow-roll
hierarchy (`ε, |η|, |δ_1| ≪ 1`):

```
H²       ≈ V/3
3 H φ̇    ≈ −V_,φ − (4/3) V² ξ_,φ                ⟹  φ̇ = −Q / √(3V)
Q(φ)     ≡ V_,φ + (4/3) V² ξ_,φ
ε(φ)     ≡ −Ḣ/H² ≈ (1/2) Q V_,φ / V²
ε(φ_end) = 1
N(φ)     = ∫_{φ}^{φ_end} (V/Q) dψ
n_s − 1  = −2 ε(φ_N) + (Q/(εV))|_{φ_N} · (dε/dφ)|_{φ_N}
r        = 16 ε(φ_N)              (leading order, ignoring c_T correction)
```

The `n_s` formula uses `n_s − 1 = −2ε − ε₂` with
`ε₂ = (dε/dφ)·(dφ/dN)` and `dφ/dN = −Q/V`.

References this matches:
- *Koh, Lee, Tumurtushaa* — [arXiv:1404.0027](https://arxiv.org/abs/1404.0027)
- *Yi, Gong, Sabir* — [arXiv:1811.01580](https://arxiv.org/abs/1811.01580)
- *Hwang, Noh* — [arXiv:0507025](https://arxiv.org/abs/gr-qc/0507025)

## What is **not** yet included

| feature | impact | where to fix |
|---|---|---|
| Modified tensor speed `c_T²` and corresponding modification of `r` | sub-leading shift to `r` | replace `r = 16 ε` with `r = 16 ε · F_T` in `analyze_model` |
| Sub-leading slow-roll corrections (δ-tower) to `n_s` | a few × 10⁻³ on `n_s` | promote to numerical solution of the full background EOMs |
| Tensor power spectrum across the relic-GW frequency band | needed for thesis-grade GW predictions | add `tensor_spectrum.py` solving the mode equation for `h_k` from horizon exit through reheating |
| Multi-field / hybrid exit | extends model space | upgrade the `EGBModel` dataclass; rewrite the kernel |

## How to swap in a more accurate kernel

1. Implement a numerical solver for the background EOMs (we wrote the JL
   skeleton in `physics/kernel.jl`).
2. Replace `analyze_model` with a function that solves the EOMs and computes
   `n_s, r, n_T, c_T` from them at horizon crossing.
3. Re-run the test suite (`pytest tests/`) — the tests pin only the
   leading-order GR limit, so a more accurate kernel should still pass them
   within tolerance.
