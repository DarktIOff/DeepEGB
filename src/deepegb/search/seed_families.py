"""
Seed-shape library for the PySR symbolic search.

Why this module exists
----------------------
PySR is a free-form symbolic regressor — it can build any expression from
the operator set {+, -, *, /, exp, log, sqrt, tanh, ...}. But its fitness
function is MSE against a fixed `y_seed`. If `y_seed` is fixed (e.g. a
Starobinsky shape), the evolutionary search converges toward Starobinsky
look-alikes — even though the *operators* could build any function.

Physics re-ranking after the fact only picks the best of an already-biased
set. So if every member of every hall-of-fame looks Starobinsky-ish, no
amount of physics re-ranking surfaces a genuinely different family.

The fix here is to run PySR multiple times, each with a different seed
shape, and merge the hall-of-fames before re-ranking. The seed library
spans the textbook inflation families:

    starobinsky      — plateau via exp(-c φ)
    quadratic        — m²φ² chaotic
    quartic          — λφ⁴ chaotic
    hilltop          — (1 - (φ/μ)²)²
    natural          — 1 + cos(φ/f)  (axion-monodromy seed)
    pole             — 1 - μᵏ/φᵏ      (brane / Kallosh family)
    monodromy        — |φ| + small osc
    exp_plateau      — 1 - exp(-|φ|/μ)

For ξ(φ) we similarly span:

    exp_decay        — string-dilaton-like
    power_law        — analytic at origin
    pole             — Kanti-style "natural" coupling
    tanh             — bounded, smooth
    linear           — minimal, bare
    cosine           — periodic (axionic GB)

The TRUE fix — a physics-only loss bypassing y_seed entirely — is to write
the EGB χ² in Julia and pass it as PySR's `loss_function=`. We have
`physics/kernel.jl` stubbed for this; the multi-seed approach is the
intermediate fix that doesn't require Julia surgery.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class SeedFamily:
    name: str
    fn: Callable[[np.ndarray], np.ndarray]
    description: str


# ---------------------------------------------------------------------------
# Inflaton-potential V(φ) families
# ---------------------------------------------------------------------------
V_FAMILIES: tuple[SeedFamily, ...] = (
    SeedFamily(
        name="starobinsky",
        fn=lambda phi: (1.0 - np.exp(-np.sqrt(2.0 / 3.0) * np.abs(phi))) ** 2 + 1e-3,
        description="Plateau model; the ACT-DR6 reference baseline.",
    ),
    SeedFamily(
        name="quadratic",
        fn=lambda phi: 0.5 * phi ** 2 + 1e-3,
        description="m²φ²; ruled out by Planck but a useful diversity seed.",
    ),
    SeedFamily(
        name="quartic",
        fn=lambda phi: 0.05 * phi ** 4 + 1e-3,
        description="λφ⁴; also ruled out, included for shape diversity.",
    ),
    SeedFamily(
        name="hilltop",
        fn=lambda phi: (1.0 - (phi / 5.0) ** 2) ** 2 + 1e-3,
        description="Quartic hilltop; classic ε-runaway alternative.",
    ),
    SeedFamily(
        name="natural",
        fn=lambda phi: 1.0 + np.cos(phi / 3.0) + 1e-3,
        description="Natural / axionic; pseudo-Nambu-Goldstone inflaton.",
    ),
    SeedFamily(
        name="pole",
        fn=lambda phi: 1.0 - 8.0 / (phi ** 2 + 1.0),
        description="Brane / pole-inflation Kallosh family (ACT-favoured).",
    ),
    SeedFamily(
        name="monodromy",
        fn=lambda phi: np.abs(phi) + 0.3 * np.cos(phi / 2.0) + 1e-3,
        description="Axion-monodromy with oscillatory feature.",
    ),
    SeedFamily(
        name="exp_plateau",
        fn=lambda phi: 1.0 - np.exp(-np.abs(phi) / 3.0) + 1e-3,
        description="Smooth plateau, intermediate between Starobinsky and brane.",
    ),
)


# ---------------------------------------------------------------------------
# Gauss-Bonnet coupling ξ(φ) families
# ---------------------------------------------------------------------------
XI_FAMILIES: tuple[SeedFamily, ...] = (
    SeedFamily(
        name="exp_decay",
        fn=lambda phi: 0.1 * np.exp(-0.5 * np.abs(phi)) + 1e-4,
        description="String-dilaton-like exponential coupling.",
    ),
    SeedFamily(
        name="power_law",
        fn=lambda phi: 0.01 * np.abs(phi) ** 2 + 1e-4,
        description="Analytic-at-origin power-law; common phenomenological choice.",
    ),
    SeedFamily(
        name="pole",
        fn=lambda phi: 0.1 / (phi ** 2 + 1.0),
        description="Kanti-style natural coupling; localised at small φ.",
    ),
    SeedFamily(
        name="tanh",
        fn=lambda phi: 0.1 * np.tanh(phi) + 0.05,
        description="Bounded smooth coupling; popular in EGB-inflation papers.",
    ),
    SeedFamily(
        name="linear",
        fn=lambda phi: 0.05 * phi + 0.1,
        description="Minimal linear coupling; bare-bones EGB.",
    ),
    SeedFamily(
        name="cosine",
        fn=lambda phi: 0.1 * np.cos(phi) + 0.15,
        description="Periodic / axionic GB coupling (Satoh-Soda-style).",
    ),
)


def get_v_family(name: str) -> SeedFamily:
    for fam in V_FAMILIES:
        if fam.name == name:
            return fam
    raise KeyError(f"unknown V family {name!r}; available: "
                   + ", ".join(f.name for f in V_FAMILIES))


def get_xi_family(name: str) -> SeedFamily:
    for fam in XI_FAMILIES:
        if fam.name == name:
            return fam
    raise KeyError(f"unknown ξ family {name!r}; available: "
                   + ", ".join(f.name for f in XI_FAMILIES))
