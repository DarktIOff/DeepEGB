"""
COBE/Planck amplitude normalisation for EGB inflation models.

A symbolic-regression search returns (V, ξ) with arbitrary overall scale —
the SR optimiser doesn't care what V₀ is, since (n_s, r, c_T², δ_1) are
invariant under a particular rescaling.  But to compare against
observation you DO need an absolute amplitude: Planck 2018 measures

    ln(10¹⁰ A_s) = 3.044 ± 0.014   ⇒   A_s = P_𝓡(k_pivot) ≈ 2.10 × 10⁻⁹.

This module rescales (V, ξ) to match that.

The rescaling
-------------
For our action S = ∫√−g [R/2 − ½(∂φ)² − V − ½ξ𝒢], the transformation

    V(φ) → λ · V(φ),     ξ(φ) → ξ(φ) / λ

is an EXACT slow-roll invariance of the observable combinations
{ε, η, δ_1, c_T², c_S², n_s, n_T, r}, while H² → λH² and consequently
P_𝓡 = H²/(8π² ε) → λ · P_𝓡.

Derivation sketch.  In slow-roll H² ≈ V/3 so H_new = √λ H_old. The KG
equation gives  3H φ̇ = −V_,φ − 12H⁴ξ_,φ; substituting V → λV, ξ → ξ/λ
and using H⁴_new = λ² H⁴_old yields φ̇_new = √λ φ̇_old. Then

    Ḣ_new/H²_new = λ^{3/2} (H Ḣ)_old / (√λ · λ · H²_old) = Ḣ_old/H²_old,

so ε is invariant; δ_1 = 4ξ̇H follows from ξ̇_new = ξ̇_old/√λ and
H_new = √λ H_old, giving δ_1_new = δ_1_old; c_T² depends only on
combinations of δ_1, ε, δ_2 so it is invariant too. The single quantity
that scales is the amplitude: P_𝓡_new = λ P_𝓡_old.

Therefore, given any candidate model with arbitrary V scale, we can
compute its current P_𝓡 from the production kernel and choose

    λ = P_𝓡_target / P_𝓡_candidate

to land exactly on the Planck amplitude — without disturbing the
spectral indices, tensor-to-scalar ratio, or any GB observable.

Caveats
-------
* Slow-roll truncation: the invariance is exact only at leading order in
  slow-roll. Sub-leading corrections (≤ 1 % on n_s) can in principle
  shift slightly under rescaling; we verify the invariants numerically
  after normalisation and report any residual drift.
* P_𝓡 depends on the e-fold convention N_pivot: normalising at N=55 vs
  N=60 gives a different λ. We expose N as a parameter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Planck 2018 default: ln(10¹⁰ A_s) = 3.044  ⇒  A_s = 2.0989...e-9
PLANCK_A_S: float = 2.1e-9
PLANCK_LN_10_10_A_S: float = 3.044


@dataclass(frozen=True)
class NormalizationResult:
    """Result of a P_𝓡 rescaling for (V, ξ)."""

    valid: bool
    V_expr_normalized: str
    xi_expr_normalized: str
    lambda_factor: float
    P_S_before: float
    P_S_target: float
    P_S_after: float
    # Drift in invariants (should be ~0 by design; flag if larger).
    n_s_before: float
    n_s_after: float
    r_before: float
    r_after: float
    epsilon_before: float
    epsilon_after: float
    delta1_before: float
    delta1_after: float
    c_T2_before: float
    c_T2_after: float
    # Convenience
    H_pivot_before: float
    H_pivot_after: float
    ln_10_10_A_s_before: float
    ln_10_10_A_s_after: float
    notes: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name)
                for f in self.__dataclass_fields__.values()}


def normalize_egb_model(
    V_expr: str,
    xi_expr: str = "0",
    *,
    N: float = 55.0,
    P_S_target: float = PLANCK_A_S,
    phi_range: tuple[float, float] = (-15.0, 15.0),
    n_grid: int = 4001,
) -> NormalizationResult:
    """Rescale (V, ξ) so P_𝓡(φ_pivot) matches `P_S_target`.

    Applies V → λV, ξ → ξ/λ with λ = P_S_target / P_S_current, then
    verifies by re-evaluating the production kernel on the normalised
    model and checking the slow-roll invariants are preserved.
    """
    from .egb_perturbations import compute_observables_full
    from ..search.pysr_search import expressions_to_model

    notes: list[str] = []

    # Step 1: evaluate the candidate as-is
    model_before = expressions_to_model(V_expr, xi_expr)
    obs_before = compute_observables_full(
        model_before, N_pivot=N, phi_range=phi_range, n_grid=n_grid,
    )
    if not obs_before.is_valid:
        return _failed(V_expr, xi_expr, P_S_target,
                       "compute_observables_full returned invalid observables")
    if not (np.isfinite(obs_before.P_S) and obs_before.P_S > 0):
        return _failed(V_expr, xi_expr, P_S_target,
                       f"P_S not positive ({obs_before.P_S}); cannot rescale")

    # Step 2: compute λ
    lam = float(P_S_target) / float(obs_before.P_S)
    if not np.isfinite(lam) or lam <= 0:
        return _failed(V_expr, xi_expr, P_S_target,
                       f"non-physical rescale factor λ={lam}")

    # Step 3: build the normalised expressions
    V_norm = f"({lam}) * ({V_expr})"
    xi_str = xi_expr.strip()
    if xi_str in ("0", "0.0", "0.0*phi") or _is_zero_expression(xi_str):
        xi_norm = "0"
    else:
        xi_norm = f"({xi_str}) / ({lam})"

    # Step 4: verify invariants
    model_after = expressions_to_model(V_norm, xi_norm)
    obs_after = compute_observables_full(
        model_after, N_pivot=N, phi_range=phi_range, n_grid=n_grid,
    )
    if not obs_after.is_valid:
        notes.append("Normalised model failed to produce finite observables — "
                     "slow-roll invariance is only leading-order; consider a "
                     "different N_pivot or wider phi_range.")
        return NormalizationResult(
            valid=False, V_expr_normalized=V_norm, xi_expr_normalized=xi_norm,
            lambda_factor=lam,
            P_S_before=obs_before.P_S, P_S_target=P_S_target,
            P_S_after=float("nan"),
            n_s_before=obs_before.n_s, n_s_after=float("nan"),
            r_before=obs_before.r, r_after=float("nan"),
            epsilon_before=obs_before.epsilon, epsilon_after=float("nan"),
            delta1_before=obs_before.delta1, delta1_after=float("nan"),
            c_T2_before=obs_before.c_T2, c_T2_after=float("nan"),
            H_pivot_before=obs_before.H_pivot, H_pivot_after=float("nan"),
            ln_10_10_A_s_before=float(np.log(1e10 * obs_before.P_S)),
            ln_10_10_A_s_after=float("nan"),
            notes=notes,
        )

    # Sanity checks on invariants
    def _drift(a: float, b: float, name: str, tol: float) -> None:
        if np.isfinite(a) and np.isfinite(b):
            denom = max(abs(a), 1e-30)
            if abs(a - b) / denom > tol:
                notes.append(
                    f"{name} drift {abs(a-b)/denom:.2e} > {tol:.0e} after "
                    "rescale — slow-roll truncation residual; observable is "
                    "still close to the invariant value but verify if "
                    "precision matters."
                )

    _drift(obs_before.n_s,     obs_after.n_s,     "n_s",     1e-3)
    _drift(obs_before.r,       obs_after.r,       "r",       2e-2)
    _drift(obs_before.epsilon, obs_after.epsilon, "ε",       2e-2)
    _drift(obs_before.c_T2,    obs_after.c_T2,    "c_T²",    1e-3)

    amp_drift = abs(obs_after.P_S - P_S_target) / P_S_target
    if amp_drift > 1e-3:
        notes.append(
            f"Amplitude target not perfectly met: P_S_after = "
            f"{obs_after.P_S:.4e}, target = {P_S_target:.4e}, "
            f"drift = {amp_drift:.2e}. This usually means the model "
            "responds non-linearly to V scaling — examine the input "
            "expression for explicit V₀ factors that should also be "
            "rescaled, or pre-strip any leading constant."
        )

    return NormalizationResult(
        valid=True,
        V_expr_normalized=V_norm,
        xi_expr_normalized=xi_norm,
        lambda_factor=lam,
        P_S_before=obs_before.P_S, P_S_target=P_S_target,
        P_S_after=obs_after.P_S,
        n_s_before=obs_before.n_s, n_s_after=obs_after.n_s,
        r_before=obs_before.r, r_after=obs_after.r,
        epsilon_before=obs_before.epsilon, epsilon_after=obs_after.epsilon,
        delta1_before=obs_before.delta1, delta1_after=obs_after.delta1,
        c_T2_before=obs_before.c_T2, c_T2_after=obs_after.c_T2,
        H_pivot_before=obs_before.H_pivot, H_pivot_after=obs_after.H_pivot,
        ln_10_10_A_s_before=float(np.log(1e10 * obs_before.P_S)),
        ln_10_10_A_s_after=float(np.log(1e10 * obs_after.P_S)),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_zero_expression(s: str) -> bool:
    """Cheap check for whether a Sympy string is identically zero."""
    try:
        import sympy as sp
        phi = sp.Symbol("phi", real=True)
        expr = sp.sympify(s, locals={"phi": phi})
        return bool(sp.simplify(expr) == 0)
    except Exception:
        return False


def _failed(V_expr: str, xi_expr: str, P_S_target: float, msg: str) -> NormalizationResult:
    return NormalizationResult(
        valid=False,
        V_expr_normalized=V_expr,
        xi_expr_normalized=xi_expr,
        lambda_factor=float("nan"),
        P_S_before=float("nan"),
        P_S_target=P_S_target,
        P_S_after=float("nan"),
        n_s_before=float("nan"), n_s_after=float("nan"),
        r_before=float("nan"), r_after=float("nan"),
        epsilon_before=float("nan"), epsilon_after=float("nan"),
        delta1_before=float("nan"), delta1_after=float("nan"),
        c_T2_before=float("nan"), c_T2_after=float("nan"),
        H_pivot_before=float("nan"), H_pivot_after=float("nan"),
        ln_10_10_A_s_before=float("nan"),
        ln_10_10_A_s_after=float("nan"),
        notes=[msg],
    )
