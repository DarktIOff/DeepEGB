"""Single source of truth for DeepEGB defaults loaded from configs/default.yaml.

All numeric defaults, policy constants, and configuration used across the
codebase should be read from the `DEFAULTS` singleton exposed by this module.

Usage
-----
    from deepegb.config import DEFAULTS

    N_pivot = DEFAULTS.N_pivot
    phi_range = DEFAULTS.phi_range
    xi_default = DEFAULTS.default_xi_expr

Implementation
--------------
On first import, we locate ``configs/default.yaml`` (relative to the package
root), parse it with PyYAML, and expose the resulting dict as a frozen
``Defaults`` dataclass.  If the YAML file is missing, hardcoded fallback
values matching the shipped YAML are used so the package still works.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _find_yaml() -> Path | None:
    """Locate configs/default.yaml relative to the package install."""
    # Try relative to this file (editable install)
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "configs" / "default.yaml"
        if candidate.exists():
            return candidate
    # Try CWD
    candidate = Path("configs/default.yaml")
    if candidate.exists():
        return candidate
    return None


def _load_yaml() -> dict[str, Any]:
    p = _find_yaml()
    if p is not None:
        with open(p) as f:
            return yaml.safe_load(f) or {}
    return {}


_RAW = _load_yaml()


def _get(d: dict, *keys, default=None):
    """Nested dict get."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
        if d is None:
            return default
    return d


@dataclass(frozen=True)
class _PhysicsDefaults:
    N_pivot: float
    N_pivot_min: float
    N_pivot_max: float
    phi_range: tuple[float, float]
    d_phi: float
    M_pl: float


@dataclass(frozen=True)
class _SearchDefaults:
    niterations: int
    populations: int
    population_size: int
    maxsize: int
    binary_operators: tuple[str, ...]
    unary_operators: tuple[str, ...]
    outputs: tuple[str, ...]
    parsimony: float
    seed_expressions_V: tuple[str, ...]
    seed_expressions_xi: tuple[str, ...]


@dataclass(frozen=True)
class _TargetDefaults:
    ns: float
    ns_sigma: float
    r: float
    r_sigma: float


@dataclass(frozen=True)
class _GWDefaults:
    n_decades: float
    n_k: int
    T_reh_GeV: float
    sub_horizon_factor: float


@dataclass(frozen=True)
class _AgentDefaults:
    sr_timeout_s: int
    runs_dir: str
    outputs_dir: str


@dataclass(frozen=True)
class Defaults:
    """Frozen container for all DeepEGB defaults. Single source of truth."""

    physics: _PhysicsDefaults
    search: _SearchDefaults
    targets: _TargetDefaults
    gw: _GWDefaults
    agent: _AgentDefaults
    default_xi_expr: str

    # Convenience accessors — these are the most-used fields
    @property
    def N_pivot(self) -> float:
        """Pivot e-folds before end of inflation (clamped to [N_pivot_min, N_pivot_max])."""
        return self.physics.N_pivot

    @property
    def phi_range(self) -> tuple[float, float]:
        return self.physics.phi_range

    @property
    def d_phi(self) -> float:
        return self.physics.d_phi

    @property
    def T_reh_GeV(self) -> float:
        return self.gw.T_reh_GeV

    @property
    def n_decades(self) -> float:
        return self.gw.n_decades

    @property
    def n_k(self) -> int:
        return self.gw.n_k


def _build_defaults(raw: dict[str, Any]) -> Defaults:
    ph = raw.get("physics", {})
    se = raw.get("search", {})
    tg = raw.get("targets", {})
    gw = raw.get("gw", {})
    ag = raw.get("agent", {})

    seed_V = se.get("seed_expressions", {}).get("V", [])
    seed_xi = se.get("seed_expressions", {}).get("xi", [])

    return Defaults(
        physics=_PhysicsDefaults(
            N_pivot=float(ph.get("N_pivot", 55)),
            N_pivot_min=float(ph.get("N_pivot_min", 50)),
            N_pivot_max=float(ph.get("N_pivot_max", 60)),
            phi_range=tuple(ph.get("phi_range", [-15.0, 15.0])),
            d_phi=float(ph.get("d_phi", 1e-4)),
            M_pl=float(ph.get("M_pl", 1.0)),
        ),
        search=_SearchDefaults(
            niterations=int(se.get("niterations", 40)),
            populations=int(se.get("populations", 35)),
            population_size=int(se.get("population_size", 33)),
            maxsize=int(se.get("maxsize", 25)),
            binary_operators=tuple(se.get("binary_operators", ["+", "-", "*", "/", "^"])),
            unary_operators=tuple(se.get("unary_operators", ["exp", "log", "sqrt", "tanh"])),
            outputs=tuple(se.get("outputs", ["V", "xi"])),
            parsimony=float(se.get("parsimony", 1e-3)),
            seed_expressions_V=tuple(seed_V) if isinstance(seed_V, list) else (),
            seed_expressions_xi=tuple(seed_xi) if isinstance(seed_xi, list) else (),
        ),
        targets=_TargetDefaults(
            ns=float(tg.get("ns", 0.9752)),
            ns_sigma=float(tg.get("ns_sigma", 0.003)),
            r=float(tg.get("r", 0.025)),
            r_sigma=float(tg.get("r_sigma", 0.013)),
        ),
        gw=_GWDefaults(
            n_decades=float(gw.get("n_decades", 8.0)),
            n_k=int(gw.get("n_k", 30)),
            T_reh_GeV=float(gw.get("T_reh_GeV", 1e15)),
            sub_horizon_factor=float(gw.get("sub_horizon_factor", 50.0)),
        ),
        agent=_AgentDefaults(
            sr_timeout_s=int(ag.get("sr_timeout_s", 3600)),
            runs_dir=str(ag.get("runs_dir", "runs")),
            outputs_dir=str(ag.get("outputs_dir", "outputs")),
        ),
        default_xi_expr=str(raw.get("default_xi_expr", "xi0/(phi^2 + 1)")),
    )


DEFAULTS: Defaults = _build_defaults(_RAW)
"""Module-level singleton — import and use directly."""


def get_defaults() -> Defaults:
    """Return the module-level Defaults singleton."""
    return DEFAULTS
