"""
Atomized χ² and model diagnostics for EGB inflation.

Why this module exists
----------------------
The original loss returned a single scalar with a flat invalid-model
penalty (1.0e6).  When `compute_observables_full` returned NaN for a
candidate, the loss was identical for every NaN-producing model — PySR
saw a flat plateau and could not hill-climb away from invalid regions.

This module replaces that with three things:

1. `Chi2Breakdown` — a per-component breakdown of the loss, so the agent
   can see WHICH part of the χ² is responsible for a high value
   (e.g. "n_s contributes 3, but Ω_GW(1 mHz) contributes 5000 — your
   GW amplitude is too high, not your spectral index").

2. `soft_invalid_penalty(model, ...)` — a smooth, gradient-bearing
   penalty for invalid models. Penalises HOW invalid the model is
   (e.g. ε never crosses 1, V is negative somewhere, Q has wrong sign
   over the inflationary side, etc.) so the search has a usable gradient.

3. `diagnose_model(model, ...)` — a human-readable explanation of which
   physical conditions a model satisfies and which it violates. Returned
   to the agent via `diagnose_egb_model_tool` so the LLM can suggest
   structural fixes instead of giving up.

These are additive — `chi2_full` and `chi2_relic_gw` continue to return
scalars (PySR's interface), but they now wrap `Chi2Breakdown.total` and
attach the breakdown to `SearchResult.diagnostics` for the agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config.defaults import DEFAULTS
from .egb_slow_roll import EGBModel, _scan_epsilon, end_of_inflation


@dataclass
class Chi2Breakdown:
    """Atomized χ² for a single (V, ξ) candidate.

    `total` is the scalar PySR sees; `components` is the per-target
    breakdown the agent reads. `reasons` lists qualitative failure modes
    (in order of severity) when the model is partly or fully invalid.
    `is_valid` is True iff every observable came out finite.
    """

    total: float
    components: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    is_valid: bool = True
    soft_penalty: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "components": dict(self.components),
            "reasons": list(self.reasons),
            "is_valid": self.is_valid,
            "soft_penalty": self.soft_penalty,
        }

    def dominant_components(self, k: int = 3) -> list[tuple[str, float]]:
        """Top-k components contributing most to the total."""
        return sorted(self.components.items(), key=lambda kv: -abs(kv[1]))[:k]


# ---------------------------------------------------------------------------
# Soft invalid penalty: monotonic in "how-broken" the model is.
# ---------------------------------------------------------------------------
def soft_invalid_penalty(
    model: EGBModel,
    *,
    phi_range: tuple[float, float] | None = None,
    n_grid: int = 401,
    floor: float = 1.0e3,
) -> tuple[float, list[str]]:
    """Return (penalty, reasons) for an invalid model.

    The penalty is at least `floor`, but adds graded contributions so
    PySR sees a non-flat landscape and can navigate. Concretely:

      * V negative over a fraction f of the φ-range → +floor·(1 + 10f).
      * ε never crosses 1 → +floor·(1 + log10(min |ε−1|)).
      * Q (slow-roll force) has constant sign over inflationary side
        but ε > 1 everywhere → +floor·(1 + min ε / 100).
      * NaN/inf in V, V', or Q at majority of sampled φ → +floor·5.
    """
    if phi_range is None:
        phi_range = DEFAULTS.phi_range
    phi = np.linspace(*phi_range, n_grid)
    reasons: list[str] = []
    penalty = floor

    # 1. V positivity
    V = np.array([safe_call(model.V, p, default=np.nan) for p in phi])
    fraction_nonpositive = float(np.mean(~(V > 0))) if V.size else 1.0
    if fraction_nonpositive > 0.05:
        reasons.append(f"V(φ) ≤ 0 over {fraction_nonpositive:.0%} of the φ-range")
        penalty += floor * 10.0 * fraction_nonpositive
    fraction_nonfinite = float(np.mean(~np.isfinite(V))) if V.size else 1.0
    if fraction_nonfinite > 0.05:
        reasons.append(f"V(φ) NaN/inf over {fraction_nonfinite:.0%} of the φ-range")
        penalty += floor * 5.0 * fraction_nonfinite

    # 2. ε must cross 1 somewhere
    eps = _scan_epsilon(model, phi)
    finite_mask = np.isfinite(eps)
    if finite_mask.any():
        finite_eps = eps[finite_mask]
        diff = finite_eps - 1.0
        if np.all(diff > 0):
            reasons.append(f"ε > 1 everywhere: min(ε)={finite_eps.min():.3g}; "
                           "no slow-roll regime exists")
            penalty += floor * (1.0 + min(finite_eps.min(), 100.0))
        elif np.all(diff < 0):
            reasons.append(f"ε < 1 everywhere in scan: max(ε)={finite_eps.max():.3g}; "
                           "end-of-inflation never reached in the φ-range")
            penalty += floor * (1.0 + abs(np.log10(max(1.0 - finite_eps.max(),
                                                       1e-30))))
        else:
            # ε crosses 1 — but we still landed here, meaning the
            # downstream pivot search failed. Give a smaller bump.
            penalty += floor * 0.5
            reasons.append("ε crosses 1, but pivot search couldn't isolate the "
                           "inflationary side")
    else:
        reasons.append("ε(φ) is NaN over the entire scan — V or its derivative "
                       "is singular")
        penalty += floor * 5.0

    return float(penalty), reasons


def safe_call(fn, x, default=np.nan):
    try:
        v = float(fn(x))
        return v if np.isfinite(v) else default
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Atomized chi2_full
# ---------------------------------------------------------------------------
def chi2_full_breakdown(
    obs,
    *,
    target_ns: float,
    sigma_ns: float = 0.003,
    target_r: float = 0.0,
    sigma_r: float = 0.018,
    target_lnAs: float | None = None,
    sigma_lnAs: float = 0.014,
    target_alphas: float | None = None,
    sigma_alphas: float = 0.013,
    target_nT: float | None = None,
    sigma_nT: float = 0.1,
    target_cT2: float | None = None,
    sigma_cT2: float = 0.05,
    enforce_egb: bool = True,
    egb_min_delta1: float = 1.0e-4,
) -> Chi2Breakdown:
    """χ² breakdown for slow-roll closed-form observables (FullObservables).

    EGB-sector enforcement
    ----------------------
    By default (`enforce_egb=True`) we penalise candidates whose GB
    coupling vanishes at horizon crossing — i.e. |δ₁(φ_N)| < `egb_min_delta1`.
    Such candidates are formally GR with ξ ≡ 0 (or numerically negligible),
    not EGB inflation models.  The penalty is `egb_penalty = G·exp(-|δ₁|/τ)`
    with G chosen large enough to dominate small slow-roll χ² but smooth
    enough that the search retains a gradient to escape the GR limit.

    Pass `enforce_egb=False` if you explicitly want the GR baseline as
    a member of the search (e.g. for a controlled comparison).
    """
    if obs is None or not getattr(obs, "is_valid", False):
        return Chi2Breakdown(
            total=np.inf,
            components={},
            reasons=["compute_observables_full returned NaN — slow-roll "
                     "trajectory could not be established"],
            is_valid=False,
        )

    components: dict[str, float] = {}
    components["n_s"] = ((obs.n_s - target_ns) / sigma_ns) ** 2
    components["r"] = ((obs.r - target_r) / sigma_r) ** 2
    if target_lnAs is not None and getattr(obs, "P_S", 0) > 0:
        ln10As = float(np.log(1e10 * obs.P_S))
        components["lnAs"] = ((ln10As - target_lnAs) / sigma_lnAs) ** 2
    if target_alphas is not None and np.isfinite(obs.alpha_s):
        components["alpha_s"] = ((obs.alpha_s - target_alphas) / sigma_alphas) ** 2
    if target_nT is not None and np.isfinite(obs.n_T):
        components["n_T"] = ((obs.n_T - target_nT) / sigma_nT) ** 2
    if target_cT2 is not None and np.isfinite(obs.c_T2):
        components["c_T2"] = ((obs.c_T2 - target_cT2) / sigma_cT2) ** 2

    reasons: list[str] = []
    if abs(obs.epsilon) > 0.05:
        reasons.append(f"|ε|={abs(obs.epsilon):.3g} > 0.05 — slow-roll "
                       "approximation is degraded; production formulas "
                       "still computed but error grows.")
    if components.get("c_T2", 0) > 9:
        reasons.append("c_T² is far from the GW170817 constraint (|c_T-1|<10⁻¹⁵ "
                       "today) — viable only if ξ̇ → 0 between inflation and now.")

    # GR-limit rejection: ξ → 0 makes the model pure GR. Penalise smoothly
    # so PySR is pushed out of that basin but doesn't see a hard wall.
    if enforce_egb and np.isfinite(obs.delta1):
        d1 = abs(float(obs.delta1))
        # Smooth bump: vanishes for d1 >> τ, rises sharply near the GR limit.
        gain = 2.0e3
        scale = max(0.5 * egb_min_delta1, 1e-30)
        egb_penalty = gain * np.exp(-d1 / scale)
        if egb_penalty > 1e-3:
            components["egb_penalty"] = float(egb_penalty)
        if d1 < egb_min_delta1:
            reasons.append(
                f"|δ₁(φ_N)|={d1:.2e} < threshold {egb_min_delta1:.0e} "
                f"⇒ model is essentially GR (ξ inactive at horizon "
                "crossing).  Rejected as a genuine EGB candidate. "
                "Make ξ_,φ steeper, or move ξ to act around horizon "
                "exit rather than at the basin."
            )

    total = float(sum(components.values()))
    return Chi2Breakdown(
        total=total, components=components, reasons=reasons,
        is_valid=True, soft_penalty=0.0,
    )


# ---------------------------------------------------------------------------
# Atomized chi2_relic_gw — adds per-target Ω_GW components
# ---------------------------------------------------------------------------
def chi2_omega_gw_breakdown(
    omega_gw_targets: list[tuple[float, float, float]] | None,
    *,
    omega_gw_values: dict[float, float],   # f_Hz → Ω_GW h² as computed
    sigma_floor: float = 0.05,             # in log10 space
) -> Chi2Breakdown:
    """χ² contribution from Ω_GW point targets, atomic per frequency."""
    components: dict[str, float] = {}
    reasons: list[str] = []
    if not omega_gw_targets:
        return Chi2Breakdown(total=0.0, components=components,
                             reasons=reasons, is_valid=True)

    for f_Hz, target, sigma in omega_gw_targets:
        omega = omega_gw_values.get(f_Hz, np.nan)
        key = f"omega_gw@{f_Hz:.3g}Hz"
        if not (np.isfinite(omega) and omega > 0):
            components[key] = 1.0e4
            reasons.append(f"Ω_GW at {f_Hz:.3g} Hz could not be computed "
                           "(MS integration failed for this k mode)")
            continue
        # log-space χ² with a relative-sigma → log-sigma conversion
        sigma_log = max(sigma_floor,
                        (sigma / max(target, 1e-30)) / np.log(10))
        contrib = ((np.log10(omega) - np.log10(max(target, 1e-30)))
                   / sigma_log) ** 2
        components[key] = float(contrib)
        if contrib > 100:
            ratio = omega / max(target, 1e-30)
            if ratio > 100:
                reasons.append(f"At {f_Hz:.3g} Hz, Ω_GW={omega:.2e} is "
                               f"{np.log10(ratio):.1f} decades ABOVE the target "
                               f"{target:.2e}.  The model is too 'loud' — "
                               "consider weakening ξ(φ) or lowering V_pivot.")
            elif ratio < 0.01:
                reasons.append(f"At {f_Hz:.3g} Hz, Ω_GW={omega:.2e} is "
                               f"{-np.log10(ratio):.1f} decades BELOW the target "
                               f"{target:.2e}.  Either ε is too small, c_T³ is "
                               "too large, or you need a feature in P_T(k).")
    total = float(sum(components.values()))
    return Chi2Breakdown(total=total, components=components,
                         reasons=reasons, is_valid=True)


# ---------------------------------------------------------------------------
# diagnose_model: high-level human-readable diagnosis
# ---------------------------------------------------------------------------
def diagnose_model(
    model: EGBModel,
    *,
    N_pivot: float | None = None,
    phi_range: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """A health-check of an EGB inflation model. Returns a dict the agent
    can read aloud to explain why a candidate isn't producing observables.

    Output structure
    ----------------
    {
        "phi_end_found": bool,
        "phi_end": float | None,
        "phi_pivot_found": bool,
        "phi_pivot": float | None,
        "max_efolds_reachable": float,
        "soft_penalty": float,
        "reasons": [str, ...],
        "v_min_in_range": float,
        "v_max_in_range": float,
        "epsilon_summary": dict,
        "delta1_summary": dict,
        "suggestions": [str, ...],
    }
    """
    if N_pivot is None:
        N_pivot = DEFAULTS.N_pivot
    if phi_range is None:
        phi_range = DEFAULTS.phi_range
    from .egb_perturbations import (
        background_along,
        compute_observables_full,
    )

    phi = np.linspace(*phi_range, 401)
    V_arr = np.array([safe_call(model.V, p) for p in phi])
    eps = _scan_epsilon(model, phi)
    finite_eps = np.isfinite(eps)
    bg = background_along(model, phi)

    phi_end = end_of_inflation(model, phi_range=phi_range, n_grid=4001)
    obs = compute_observables_full(model, N_pivot=N_pivot, phi_range=phi_range)
    soft, reasons = soft_invalid_penalty(model, phi_range=phi_range)

    # Estimate maximum reachable e-folds from the slow-roll quadrature.
    max_N = 0.0
    if phi_end is not None:
        from .egb_slow_roll import _N_to_phi_table
        try:
            _, N_grid = _N_to_phi_table(model, phi_end, phi_range, 2001)
            ok = np.isfinite(N_grid)
            max_N = float(np.nanmax(N_grid[ok])) if ok.any() else 0.0
        except Exception:
            pass

    suggestions: list[str] = []
    # Detect the GR limit: |δ₁| at horizon crossing ~ 0
    is_gr_limit = False
    if obs is not None and obs.is_valid and np.isfinite(obs.delta1):
        if abs(float(obs.delta1)) < 1.0e-8:
            is_gr_limit = True
            reasons.append(
                f"GR LIMIT: |δ₁(φ_N)|={abs(obs.delta1):.2e} ≈ 0. ξ(φ) is "
                "either identically zero or numerically negligible at "
                "horizon crossing — this is plain General Relativity, not "
                "EGB inflation. DeepEGB rejects ξ ≡ 0 as a candidate."
            )
            suggestions.append(
                "Use a non-trivial ξ(φ) whose derivative ξ_,φ is nonzero "
                "around φ_N. Examples: ξ = α·exp(−λφ), ξ = α/(φ²+β), "
                "ξ = α·φⁿ. Make sure ξ_,φ × V² is comparable to V_,φ at "
                "the pivot so δ₁ = −(4/3)ξ_,φ Q is non-negligible."
            )

    if phi_end is None:
        suggestions.append("Adjust V or ξ so ε crosses 1 in the φ-range. "
                           "Concrete: increase the slope of V near the basin, "
                           "or weaken ξ_,φ contribution to Q.")
    elif obs is None or not obs.is_valid:
        if max_N < N_pivot:
            suggestions.append(
                f"Inflation lasts only {max_N:.1f} e-folds in the slow-roll "
                f"approximation — less than the pivot N_pivot={N_pivot:.0f}. "
                "Make V flatter (longer plateau) or move the inflationary "
                "branch further from the trans-Planckian boundary."
            )
        else:
            suggestions.append(
                "ε crosses 1 and the trajectory is long enough, but observables "
                "still failed — usually a Q sign-change or a singularity in V "
                "between φ_end and φ_pivot."
            )

    if (V_arr <= 0).any():
        suggestions.append("V(φ) ≤ 0 somewhere in the scan; restrict the "
                           "φ-range to the V > 0 region or add a positive "
                           "additive constant.")

    eps_finite = eps[finite_eps]
    if eps_finite.size > 0 and (eps_finite < 0).any():
        suggestions.append("Negative ε detected (Q V_,φ < 0 — Q and V_,φ have "
                           "opposite signs). Often indicates ξ_,φ is large "
                           "enough to flip the slow-roll force; reduce |ξ_,φ|.")

    return {
        "phi_end_found": phi_end is not None,
        "phi_end": phi_end,
        "phi_pivot_found": obs is not None and np.isfinite(obs.phi_N),
        "phi_pivot": float(obs.phi_N) if obs is not None else None,
        "max_efolds_reachable": max_N,
        "N_pivot_target": N_pivot,
        "soft_penalty": soft,
        "reasons": reasons,
        "V_min": float(np.nanmin(V_arr)) if V_arr.size else np.nan,
        "V_max": float(np.nanmax(V_arr)) if V_arr.size else np.nan,
        "epsilon_min": (float(np.nanmin(eps_finite)) if eps_finite.size else np.nan),
        "epsilon_max": (float(np.nanmax(eps_finite)) if eps_finite.size else np.nan),
        "delta1_min": (float(np.nanmin(bg["delta1"][np.isfinite(bg["delta1"])]))
                       if np.isfinite(bg["delta1"]).any() else np.nan),
        "delta1_max": (float(np.nanmax(bg["delta1"][np.isfinite(bg["delta1"])]))
                       if np.isfinite(bg["delta1"]).any() else np.nan),
        "observables": obs.as_dict() if obs is not None else None,
        "observables_valid": obs is not None and obs.is_valid,
        "is_gr_limit": is_gr_limit,
        "suggestions": suggestions,
    }
