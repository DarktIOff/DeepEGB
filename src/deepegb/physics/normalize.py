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
* The invariance is exact only at leading order in slow-roll. Sub-leading
  corrections (≤ 1 % on n_s) can shift slightly under rescaling.
* When λ is very far from 1 (e.g. λ ~ 10⁻¹⁰), the ξ → ξ/λ rescaling
  makes ξ explode by orders of magnitude, driving |δ₁| >> 1 and breaking
  the perturbative EGB assumption. We guard against this.
* P_𝓡 depends on the e-fold convention N_pivot: normalising at N=55 vs
  N=60 gives a different λ. We expose N as a parameter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

PLANCK_A_S: float = 2.1e-9
PLANCK_LN_10_10_A_S: float = 3.044

_DELTA1_MAX_FOR_RESCALING = 0.5
_LAM_SAFETY_FLOOR = 1.0e-6
_LAM_SAFETY_CEIL = 1.0e6


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
    H_pivot_before: float
    H_pivot_after: float
    ln_10_10_A_s_before: float
    ln_10_10_A_s_after: float
    notes: list[str]
    method: str

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

    Strategy (in priority order):

    1. **Exact rescaling**: V → λV, ξ → ξ/λ.  Exact invariance of all
       slow-roll observables at leading order. Applied only when λ is
       within a safe range and the post-rescaling |δ₁| stays bounded.

    2. **V-only rescaling**: V → λV, ξ unchanged. Not an invariance —
       n_s and r shift slightly — but keeps δ₁ and ξ stable. Used as a
       fallback when the exact rescaling would violate the perturbative
       EGB assumption (|δ₁| too large after ξ/λ).

    3. **No rescaling**: When even V-only would break the model. Returns
       `valid=False` with diagnostics explaining why and suggesting a
       re-run with `target_lnAs` in the search loss.
    """
    from .egb_perturbations import compute_observables_full
    from ..search.pysr_search import expressions_to_model

    notes: list[str] = []

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

    lam = float(P_S_target) / float(obs_before.P_S)
    if not np.isfinite(lam) or lam <= 0:
        return _failed(V_expr, xi_expr, P_S_target,
                        f"non-physical rescale factor λ={lam}")

    lnAs_before = float(np.log(1e10 * obs_before.P_S))
    d1_before = abs(float(obs_before.delta1)) if np.isfinite(obs_before.delta1) else 0.0

    xi_is_trivial = xi_expr.strip() in ("0", "0.0") or _is_zero_expression(xi_expr)

    # ── Strategy 1: exact rescaling V→λV, ξ→ξ/λ ──────────────────────
    if xi_is_trivial:
        xi_norm = "0"
    else:
        xi_norm = f"({xi_expr.strip()}) / ({lam})"

    if _LAM_SAFETY_FLOOR < lam < _LAM_SAFETY_CEIL:
        result = _try_rescaling(
            V_expr, xi_expr, lam, xi_norm, P_S_target, N, phi_range, n_grid,
            obs_before, notes, method="exact",
        )
        if result is not None:
            return result
        notes.append(
            "Exact rescaling V→λV, ξ→ξ/λ failed verification (δ₁ exploded "
            "or observables went NaN). The λ factor is too far from 1 for "
            "the perturbative invariance to hold. Falling back to V-only."
        )

    # ── Strategy 2: V-only rescaling (ξ unchanged) ────────────────────
    if not xi_is_trivial:
        V_norm_vonly = f"({lam}) * ({V_expr})"
        result = _try_v_only_rescaling(
            V_expr, xi_expr, lam, V_norm_vonly, P_S_target, N, phi_range, n_grid,
            obs_before, notes,
        )
        if result is not None:
            return result
        notes.append(
            "V-only rescaling also produced invalid observables. The model's "
            "background is sensitive to V scale in a way that breaks the "
            "trajectory integration."
        )
    else:
        V_norm_exact = f"({lam}) * ({V_expr})"
        model_after = expressions_to_model(V_norm_exact, "0")
        obs_after = compute_observables_full(
            model_after, N_pivot=N, phi_range=phi_range, n_grid=n_grid,
        )
        if obs_after.is_valid:
            return _build_result(
                V_norm_exact, "0", lam, P_S_target, obs_before, obs_after,
                notes, method="exact_trivial_xi",
            )

    # ── Strategy 3: cannot rescale ────────────────────────────────────
    notes.append(
        f"λ = P_S_target / P_S = {lam:.4e} is outside the safe range "
        f"[{_LAM_SAFETY_FLOOR:.0e}, {_LAM_SAFETY_CEIL:.0e}]. "
        "The V→λV, ξ→ξ/λ invariance only holds at leading slow-roll order; "
        "rescaling by this factor would violate the perturbative EGB "
        f"assumption (pre-rescaling |δ₁|={d1_before:.2e}, post-rescaling "
        "|δ₁| would be ~" +
        (f"{d1_before / lam:.2e}" if lam > 0 and np.isfinite(d1_before) else "NaN") +
        "). SUGGESTION: re-run the search with target_lnAs=3.044 in the "
        "loss function so PySR discovers amplitude-correct models directly."
    )
    return NormalizationResult(
        valid=False,
        V_expr_normalized=V_expr,
        xi_expr_normalized=xi_expr,
        lambda_factor=lam,
        P_S_before=obs_before.P_S, P_S_target=P_S_target,
        P_S_after=float("nan"),
        n_s_before=obs_before.n_s, n_s_after=float("nan"),
        r_before=obs_before.r, r_after=float("nan"),
        epsilon_before=obs_before.epsilon, epsilon_after=float("nan"),
        delta1_before=obs_before.delta1, delta1_after=float("nan"),
        c_T2_before=obs_before.c_T2, c_T2_after=float("nan"),
        H_pivot_before=obs_before.H_pivot, H_pivot_after=float("nan"),
        ln_10_10_A_s_before=lnAs_before,
        ln_10_10_A_s_after=float("nan"),
        notes=notes,
        method="failed",
    )


def _try_rescaling(
    V_expr: str, xi_expr: str, lam: float, xi_norm: str,
    P_S_target: float, N: float,
    phi_range: tuple[float, float], n_grid: int,
    obs_before, notes: list[str], *, method: str,
) -> NormalizationResult | None:
    """Try exact V→λV, ξ→ξ/λ rescaling. Return None if it fails."""
    from ..search.pysr_search import expressions_to_model
    from .egb_perturbations import compute_observables_full

    V_norm = f"({lam}) * ({V_expr})"
    model_after = expressions_to_model(V_norm, xi_norm)
    obs_after = compute_observables_full(
        model_after, N_pivot=N, phi_range=phi_range, n_grid=n_grid,
    )
    if not obs_after.is_valid:
        return None

    d1_after = abs(float(obs_after.delta1)) if np.isfinite(obs_after.delta1) else float("inf")
    if d1_after > _DELTA1_MAX_FOR_RESCALING:
        return None

    return _build_result(
        V_norm, xi_norm, lam, P_S_target, obs_before, obs_after,
        notes, method=method,
    )


def _try_v_only_rescaling(
    V_expr: str, xi_expr: str, lam: float, V_norm: str,
    P_S_target: float, N: float,
    phi_range: tuple[float, float], n_grid: int,
    obs_before, notes: list[str],
) -> NormalizationResult | None:
    """Try V→λV only (ξ unchanged). Return None if it fails."""
    from ..search.pysr_search import expressions_to_model
    from .egb_perturbations import compute_observables_full

    model_after = expressions_to_model(V_norm, xi_expr)
    obs_after = compute_observables_full(
        model_after, N_pivot=N, phi_range=phi_range, n_grid=n_grid,
    )
    if not obs_after.is_valid:
        return None

    new_notes = list(notes)
    new_notes.append(
        "V-only rescaling applied: V→λV, ξ unchanged. "
        "This is NOT an exact slow-roll invariance — n_s and r may have "
        "shifted slightly. Verify the post-rescaling values are acceptable."
    )
    d_ns = abs(obs_after.n_s - obs_before.n_s)
    d_r = abs(obs_after.r - obs_before.r)
    if d_ns > 0.005:
        new_notes.append(
            f"n_s drifted by {d_ns:.4f} under V-only rescaling — "
            "consider re-running search with target_lnAs=3.044 instead."
        )
    if d_r > 0.005:
        new_notes.append(
            f"r drifted by {d_r:.4f} under V-only rescaling."
        )

    return _build_result(
        V_norm, xi_expr, lam, P_S_target, obs_before, obs_after,
        new_notes, method="v_only",
    )


def _build_result(
    V_norm: str, xi_norm: str, lam: float, P_S_target: float,
    obs_before, obs_after, notes: list[str], *, method: str,
) -> NormalizationResult:
    from ..search.pysr_search import expressions_to_model
    from .egb_perturbations import compute_observables_full

    result_notes = list(notes)

    def _drift(a: float, b: float, name: str, tol: float) -> None:
        if np.isfinite(a) and np.isfinite(b):
            denom = max(abs(a), 1e-30)
            if abs(a - b) / denom > tol:
                result_notes.append(
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
        result_notes.append(
            f"Amplitude target not perfectly met: P_S_after = "
            f"{obs_after.P_S:.4e}, target = {P_S_target:.4e}, "
            f"drift = {amp_drift:.2e}."
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
        notes=result_notes,
        method=method,
    )


def _is_zero_expression(s: str) -> bool:
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
        method="failed",
    )
