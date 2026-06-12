"""Symbolic derivation of closed-form EGB inflation observables in terms of
the slow-roll hierarchy (ε_i, δ_i) with EXACT Green's-function constants.

Produces the "paper formulas": P_S, P_T, n_s, n_T, α_s, α_t, r through
third order in the flow parameters, at the common Hubble-crossing pivot
k = aH, for the action (M_pl = 1)

    S = ∫ d⁴x √(-g) [ R/2 − ½(∂φ)² − V(φ) − ½ ξ(φ) 𝒢 ].

Derivation chain (every step exact up to the final flow truncation):

 1. Sector data exactly in flow variables (WZW 1707.08020 Eqs. 2.8-2.12 +
    the exact background identity φ̇²/H² = 2ε₁ − δ₁ − δ₁ε₁ + δ₁δ₂):
        scalar: q_S ≡ z_R²/a², c_S²;   tensor: q_T = 1−δ₁, c_T².
 2. Exact sound-time reduction  dς = c dη, z̃ = z√c: effective flow ε̃_i,
    H̃² = H²W²/(q c³) with W ≡ dln z̃/dN, and ς = −(c/aH)·F.
 3. The N3LO master bracket of Auclair & Ringeval (arXiv:2205.12608),
    extracted SYMBOLICALLY from the vendored ancillary code by shimming
    numpy/scipy with sympy.  Constants exact: C = γ_E+ln2−2, π², ζ(3).
 4. Indices via n = F·D(ln 𝒫) along the moving pivot (dln k/dN = 1/F on
    −kς = 1); runnings via a second F·D.
 5. Lie-series shift from each sector's −kς = 1 point to the common
    Hubble-crossing point k = a*H*; truncation at total flow order 3.

Implementation: explicit graded flow series (dict grade → coefficient,
coefficients are expanded sympy expressions that may carry powers of
1/(2ε₁−δ₁) etc.); all arithmetic is grade convolution — no slow
multivariate gcd anywhere.

Outputs:  docs/PAPER_FORMULAS.md
Run:      python scripts/derive_paper_formulas.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


# ===========================================================================
# 0.  Graded flow series
# ===========================================================================
NSYM = 6
GMAX = 4          # internal max grade carried (final slices below)

e = sp.symbols(f"e1:{NSYM+1}", real=True)
d = sp.symbols(f"d1:{NSYM+1}", real=True)
C = sp.Symbol("C", real=True)
Z3 = sp.Symbol("zeta3", positive=True)
Hs = sp.Symbol("H", positive=True)


def _D_expr(expr):
    out = sp.Integer(0)
    for i in range(NSYM - 1):
        out += sp.diff(expr, e[i]) * e[i] * e[i + 1]
        out += sp.diff(expr, d[i]) * d[i] * d[i + 1]
    return out


class FS:
    """Flow series: {grade: coefficient}, coefficients expanded sympy."""

    __slots__ = ("c",)

    def __init__(self, c=None):
        self.c = {g: v for g, v in (c or {}).items() if v != 0}

    # -------- constructors
    @staticmethod
    def const(v):
        return FS({0: sp.sympify(v)})

    @staticmethod
    def sym(s, grade=1):
        return FS({grade: s})

    # -------- basics
    def __add__(self, o):
        o = _fs(o)
        c = dict(self.c)
        for g, v in o.c.items():
            c[g] = sp.expand(c.get(g, 0) + v)
        return FS(c)

    __radd__ = __add__

    def __neg__(self):
        return FS({g: -v for g, v in self.c.items()})

    def __sub__(self, o):
        return self + (-_fs(o))

    def __rsub__(self, o):
        return _fs(o) + (-self)

    def __mul__(self, o):
        o = _fs(o)
        c = {}
        for g1, v1 in self.c.items():
            for g2, v2 in o.c.items():
                g = g1 + g2
                if g > GMAX:
                    continue
                c[g] = sp.expand(c.get(g, 0) + v1 * v2)
        return FS(c)

    __rmul__ = __mul__

    def inv(self):
        if not self.c:
            raise ZeroDivisionError
        g0 = min(self.c)
        c0 = self.c[g0]
        ic0 = sp.together(1 / c0)
        # t = (self/lead) − 1, grades ≥ 1
        t = {g - g0: sp.expand(v * ic0) for g, v in self.c.items() if g > g0}
        inv = {0: sp.Integer(1)}
        for k in range(1, GMAX - (-g0) + 1 + abs(g0)):
            if k > GMAX + abs(g0):
                break
            s = -sum(t.get(j, 0) * inv.get(k - j, 0)
                     for j in range(1, k + 1))
            s = sp.expand(s)
            if s != 0:
                inv[k] = s
        return FS({k - g0: sp.expand(v * ic0)
                   for k, v in inv.items() if k - g0 <= GMAX})

    def __truediv__(self, o):
        return self * _fs(o).inv()

    def __rtruediv__(self, o):
        return _fs(o) * self.inv()

    # -------- calculus / functions
    def D(self):
        return FS({g + 1: sp.expand(_D_expr(v))
                   for g, v in self.c.items() if g + 1 <= GMAX})

    def log1(self):
        """ln(series) for series with c[0] == 1."""
        assert self.c.get(0, 0) == 1 and min(self.c) == 0
        t = FS({g: v for g, v in self.c.items() if g > 0})
        out, tp = FS(), FS.const(1)
        for k in range(1, GMAX + 1):
            tp = tp * t
            out = out + sp.Rational((-1) ** (k + 1), k) * tp
        return out

    def exp0(self):
        """exp(series) for series with no grade-0 part."""
        assert 0 not in self.c and (not self.c or min(self.c) >= 1)
        out, tp = FS.const(1), FS.const(1)
        for k in range(1, GMAX + 1):
            tp = tp * self
            out = out + tp / sp.factorial(k)
        return out

    def powr(self, p):
        """series**p (p rational) for series with c[0] == 1."""
        assert self.c.get(0, 0) == 1 and min(self.c) == 0
        t = FS({g: v for g, v in self.c.items() if g > 0})
        out, tp, coef = FS.const(1), FS.const(1), sp.Integer(1)
        for k in range(1, GMAX + 1):
            coef = coef * (sp.Rational(p) - (k - 1)) / k
            tp = tp * t
            out = out + coef * tp
        return out

    def slice(self, gmax):
        return FS({g: v for g, v in self.c.items() if g <= gmax})

    def expr(self):
        return sp.Add(*self.c.values()) if self.c else sp.Integer(0)


def _fs(x):
    return x if isinstance(x, FS) else FS.const(x)


E1f, E2f = FS.sym(e[0]), FS.sym(e[1])
D1f, D2f = FS.sym(d[0]), FS.sym(d[1])
ONE = FS.const(1)

# ===========================================================================
# 1.  Exact sector data
# ===========================================================================
log("building exact sector quantities …")

phidot2 = 2 * E1f - D1f - D1f * E1f + D1f * D2f          # φ̇²/H², exact
dbar = D1f / (ONE - D1f)

den_S = phidot2 + sp.Rational(3, 2) * dbar * D1f
Q_S = den_S / ((ONE - dbar / 2) * (ONE - dbar / 2))
C2_S = ONE + D1f * (-2 * dbar * E1f
                    + dbar * dbar / 2 * (D2f + E1f - 1)) / den_S
Q_T = ONE - D1f
C2_T = ONE - D1f * (D2f + E1f - 1) / (ONE - D1f)

# ===========================================================================
# 2.  Master bracket (symbolic, from the vendored AR code)
# ===========================================================================
log("extracting AR master bracket symbolically …")


class _NpShim:
    pi = sp.pi
    euler_gamma = sp.EulerGamma

    @staticmethod
    def log(x):
        return sp.log(x)


class _SfShim:
    @staticmethod
    def zeta(n):
        return Z3 if n == 3 else sp.zeta(n)


import deepegb.physics._n3lo_master as _m       # noqa: E402

_saved = (_m.np, _m.sf, _m.CONSTANT_C)
_m.np, _m.sf, _m.CONSTANT_C = _NpShim, _SfShim, C
E1s, E2s, E3s = sp.symbols("E1 E2 E3", real=True)
_P = _m.tensor_power_spectrum(sp.Integer(1), sp.Integer(1), sp.Integer(1),
                              E1s, E2s, E3s)
# AR's Python source carries float fractions (e.g. 1/12*np.pi**2);
# nsimplify restores the exact rationals.
_S = sp.nsimplify(sp.expand(_P / (2 / sp.pi ** 2)), rational=True)
S_BRACKET = sp.Poly(_S, E1s, E2s, E3s)
_m.np, _m.sf, _m.CONSTANT_C = _saved
assert sp.simplify(S_BRACKET.eval((0, 0, 0)) - 1) == 0
log(f"   bracket: {len(S_BRACKET.terms())} monomials; dS limit OK")


# The bracket depends on the background only through H, H', H'', H''' —
# i.e. through u₁ ≡ ε̃₁, u₂ ≡ D(ε̃₁) = ε̃₁ε̃₂, u₃ ≡ D²(ε̃₁) =
# ε̃₁ε̃₂(ε̃₂+ε̃₃).  Structurally this requires the coefficients of
# E1E2² and E1E2E3 to coincide; we verify and re-express the bracket as
# a polynomial in (u₁,u₂,u₃), which removes ALL 1/ε̃₁ ratios from the
# composition (only benign 1/(2ε₁−δ₁)-type denominators remain).
_terms = {m_: c_ for m_, c_ in S_BRACKET.terms()}
assert sp.simplify(_terms[(1, 2, 0)] - _terms[(1, 1, 1)]) == 0, \
    "E1E2² / E1E2E3 coefficient pairing violated"
_U_COEFFS = {
    (0, 0, 0): _terms.get((0, 0, 0), 0),       # 1
    (1, 0, 0): _terms.get((1, 0, 0), 0),       # u1
    (2, 0, 0): _terms.get((2, 0, 0), 0),       # u1²
    (0, 1, 0): _terms.get((1, 1, 0), 0),       # u2   (= E1E2)
    (3, 0, 0): _terms.get((3, 0, 0), 0),       # u1³
    (1, 1, 0): _terms.get((2, 1, 0), 0),       # u1u2 (= E1²E2)
    (0, 0, 1): _terms.get((1, 2, 0), 0),       # u3   (= E1E2²+E1E2E3)
}


def bracket_series(u1: FS, u2: FS, u3: FS) -> FS:
    """Evaluate the master bracket as a polynomial in (u₁,u₂,u₃)."""
    out = FS()
    u1sq = u1 * u1
    pieces = {
        (0, 0, 0): FS.const(1),
        (1, 0, 0): u1,
        (2, 0, 0): u1sq,
        (0, 1, 0): u2,
        (3, 0, 0): u1sq * u1,
        (1, 1, 0): u1 * u2,
        (0, 0, 1): u3,
    }
    for key, coef in _U_COEFFS.items():
        if coef != 0:
            out = out + coef * pieces[key]
    return out


# ===========================================================================
# 3.  Per-sector reduction at the dia point −kς = 1
# ===========================================================================
def reduce_sector(q: FS, c2: FS, tag: str) -> dict:
    log(f"   [{tag}] flow geometry …")
    Dlnq = q.D() / q
    s1c = c2.D() / (2 * c2)
    W = ONE + sp.Rational(1, 2) * Dlnq + sp.Rational(1, 2) * s1c

    F = FS.const(1)
    geom = (ONE - E1f - s1c).inv()
    for _ in range(GMAX + 1):
        F = (ONE + F.D()) * geom

    DlnWc = W.D() / W - s1c
    eps1t = ONE - (ONE - E1f + DlnWc) / W
    # u₂ = dε̃₁/dÑ = D(ε̃₁)/W,  u₃ = d²ε̃₁/dÑ² = D(u₂)/W  (no 1/ε̃₁!)
    u2t = eps1t.D() / W
    u3t = u2t.D() / W
    log(f"   [{tag}] effective flow done; master bracket …")

    Sb = bracket_series(eps1t, u2t, u3t)
    log(f"   [{tag}] bracket composed; index/running …")

    P_red = (W * W / q) * c2.powr(sp.Rational(-3, 2)) * Sb

    DlnP = (FS.const(-2) * E1f + 2 * (W.D() / W) - Dlnq - 3 * s1c
            + Sb.D() / Sb)
    n = F * DlnP
    alpha = F * n.D()

    lnx = sp.Rational(1, 2) * c2.log1() + F.log1()
    psi = E1f + s1c + F.D() / F
    log(f"   [{tag}] done")
    return dict(P_red=P_red, n=n, alpha=alpha, lnx=lnx, psi=psi)


SEC = {"S": reduce_sector(Q_S, C2_S, "scalar"),
       "T": reduce_sector(Q_T, C2_T, "tensor")}

# ===========================================================================
# 4.  Lie-series shift dia → common pivot k = a*H*
# ===========================================================================
log("computing dia→pivot shifts …")


def solve_shift(lnx: FS, psi: FS) -> FS:
    Dpsi, DDpsi = psi.D(), psi.D().D()
    inv1mpsi = (ONE - psi).inv()
    dN = FS()
    for _ in range(GMAX + 1):
        dN = (lnx + dN * dN * Dpsi / 2 + dN * dN * dN * DDpsi / 6) * inv1mpsi
    return dN


def lie_shift(Q: FS, dN: FS) -> FS:
    """Q(N_dia) re-expressed in flow at N*: Σ_k δN^k/k! D^k(Q)."""
    out, term, Dk = Q, Q, Q
    dNp = FS.const(1)
    for k in range(1, GMAX + 1):
        Dk = Dk.D()
        dNp = dNp * dN
        out = out + dNp * Dk / sp.factorial(k)
    return out


RESULT = {}
for X in ("S", "T"):
    sec = SEC[X]
    dN = solve_shift(sec["lnx"], sec["psi"])
    n_at = lie_shift(sec["n"], dN)
    a_at = lie_shift(sec["alpha"], dN)
    P_at = lie_shift(sec["P_red"], dN)
    # H²(N_dia)/H*² = exp(∫ −2ε₁) = exp(−2ε₁δN − δN²Dε₁ − δN³D²ε₁/3)
    lnH2 = (FS.const(-2) * E1f * dN - dN * dN * E1f.D()
            - dN * dN * dN * E1f.D().D() / 3)
    P_at = P_at * lnH2.exp0()
    RESULT[X] = dict(n=n_at.slice(3), alpha=a_at.slice(3),
                     P=P_at, dN=dN)
    log(f"   sector {X} shifted")

n_s_expr = 1 + RESULT["S"]["n"].expr()
n_T_expr = RESULT["T"]["n"].expr()
alpha_s_expr = RESULT["S"]["alpha"].expr()
alpha_t_expr = RESULT["T"]["alpha"].expr()
P_S_red = RESULT["S"]["P"].slice(2)          # LO grade −1 + 3 orders
P_T_red = RESULT["T"]["P"].slice(3)          # LO grade 0 + 3 orders
r_fs = (8 * RESULT["T"]["P"] / RESULT["S"]["P"]).slice(4)   # LO grade 1
r_expr = r_fs.expr()
P_S_expr = Hs ** 2 / (4 * sp.pi ** 2) * P_S_red.expr()
P_T_expr = 8 * Hs ** 2 / (4 * sp.pi ** 2) * P_T_red.expr()

# ===========================================================================
# 5.  GR-limit check (Stewart–Lyth / AR second order)
# ===========================================================================
log("GR-limit check …")
gr = {di: 0 for di in d}


def upto(expr, n):
    out = sp.Integer(0)
    for t in sp.Add.make_args(sp.expand(expr)):
        deg = sum(sp.degree(t, s) for s in (*e, *d) if t.has(s))
        if deg <= n:
            out += t
    return out


ns_gr = upto(sp.expand((n_s_expr - 1).subs(gr)), 2)
nt_gr = upto(sp.expand(n_T_expr.subs(gr)), 2)
ns_expected = (-2 * e[0] - e[1] - 2 * e[0] ** 2 - (2 * C + 3) * e[0] * e[1]
               - C * e[1] * e[2])
nt_expected = -2 * e[0] - 2 * e[0] ** 2 - 2 * (C + 1) * e[0] * e[1]
print("  n_s−1 (GR, ≤2nd):", sp.expand(ns_gr))
print("  diff vs Stewart-Lyth:", sp.simplify(sp.expand(ns_gr - ns_expected)))
print("  n_T   (GR, ≤2nd):", sp.expand(nt_gr))
print("  diff vs Stewart-Lyth:", sp.simplify(sp.expand(nt_gr - nt_expected)))

# ===========================================================================
# 6.  Numerical validation against the production engine
# ===========================================================================
log("numerical validation vs engine …")
from deepegb.physics import EGBModel                                  # noqa
from deepegb.physics.egb_background import integrate_with_pivot      # noqa
from deepegb.physics.egb_n3lo import (_analytic_grids,               # noqa
                                      compute_observables_n3lo)

C_NUM = float(np.euler_gamma + np.log(2) - 2)
Z3_NUM = 1.2020569031595943
ARGS = (*e[:5], *d[:5], Hs)
fns = {}
for key, expr in dict(n_s=n_s_expr, n_T=n_T_expr, r=r_expr,
                      alpha_s=alpha_s_expr, P_S=P_S_expr,
                      P_T=P_T_expr).items():
    extra = expr.free_symbols - set(ARGS) - {C, Z3}
    if extra:
        log(f"   WARNING: {key} has unexpected symbols {extra}")
    fns[key] = sp.lambdify(ARGS, expr.subs({C: C_NUM, Z3: Z3_NUM}), "numpy")


def flow_at_pivot(model, N_pivot, phi_range):
    traj = integrate_with_pivot(model, N_pivot=N_pivot, phi_range=phi_range)
    ag = _analytic_grids(model, traj)
    N = traj.N
    Nstar = traj.N_end - N_pivot
    mask = np.abs(N - Nstar) <= 2.5
    u = N[mask] - Nstar

    def hier(grid):
        w = grid[mask]
        cs = np.polynomial.polynomial.polyfit(u, np.log(np.abs(w)), 6)
        p = np.polynomial.polynomial.Polynomial(cs)
        d1p, d2p, d3p, d4p = (p.deriv(k) for k in (1, 2, 3, 4))
        f1 = float(np.exp(p(0.0))) * float(np.sign(w[len(w) // 2]))
        f2 = float(d1p(0.0))
        f3 = float(d2p(0.0)) / f2 if f2 else 0.0
        f4 = (float(d3p(0.0)) / float(d2p(0.0)) - float(d2p(0.0)) / f2
              if (f2 and float(d2p(0.0))) else 0.0)
        # f5 = dln|f4|/dN — rarely needed (grade-4 sliced); set 0
        return f1, f2, f3, f4, 0.0

    e_h = hier(ag["eps1"])
    d_h = ((0.0,) * 5 if np.max(np.abs(ag["delta1"])) < 1e-14
           else hier(ag["delta1"]))
    Hv = float(np.interp(Nstar, N, ag["H"]))
    return (*e_h, *d_h, Hv)


MODELS = {
    "GR-Starobinsky": (EGBModel(
        V=lambda p: 1e-10 * (1 - np.exp(-np.sqrt(2 / 3) * p)) ** 2,
        xi=lambda p: 0.0 * p), (0.05, 10.0)),
    "Staro+xi0*phi (ACT best fit)": (EGBModel(
        V=lambda p: 2.101e-10 * (1 - np.exp(-np.sqrt(2 / 3) * p)) ** 2,
        xi=lambda p: 4.973e7 * p), (0.05, 10.0)),
    "EGB-quad strong": (EGBModel(
        V=lambda p: 0.5e-10 * p ** 2,
        xi=lambda p: 1.0e9 / (p ** 2 + 1.0)), (-15.0, 15.0)),
}

report = []
for name, (model, rng) in MODELS.items():
    obs = compute_observables_n3lo(model, N_pivot=55.0, phi_range=rng)
    flow = flow_at_pivot(model, 55.0, rng)
    print(f"\n  == {name}  (e1={flow[0]:.3e}, d1={flow[5]:+.3e})")
    rows = []
    for key, eng in (("n_s", obs.n_s), ("n_T", obs.n_T), ("r", obs.r),
                     ("alpha_s", obs.alpha_s), ("P_S", obs.P_S),
                     ("P_T", obs.P_T)):
        val = float(fns[key](*flow))
        drel = (val - eng) / eng if eng != 0 else float("nan")
        rows.append((key, val, eng, drel))
        print(f"     {key:8s} formula={val: .8e}  engine={eng: .8e}  "
              f"rel diff={drel:+.2e}")
    report.append((name, flow, rows))

# ===========================================================================
# 7.  Cache + Markdown/LaTeX output
# ===========================================================================
log("caching expressions …")
_CACHE = Path(__file__).resolve().parents[1] / "outputs" / \
    "paper_formulas_srepr.py"
_CACHE.parent.mkdir(parents=True, exist_ok=True)
_FINALS = dict(n_s=n_s_expr, n_T=n_T_expr, alpha_s=alpha_s_expr,
               alpha_t=alpha_t_expr, P_S_red=P_S_red.expr(),
               P_T_red=P_T_red.expr(), r=r_expr)
_CACHE.write_text("EXPRS = {\n" + "".join(
    f"  {k!r}: {sp.srepr(v)!r},\n" for k, v in _FINALS.items()) + "}\n")

log("writing docs/PAPER_FORMULAS.md …")

_FLOWSET = set(e) | set(d)


def flow_degree(t) -> int:
    """Total flow degree of a term, counting 1/(2ε₁−δ₁)-type powers."""
    deg = sp.Integer(0)
    for base, ex in t.as_powers_dict().items():
        if base in _FLOWSET:
            deg += ex
        elif base.free_symbols & _FLOWSET:
            sub = sp.Add.make_args(sp.expand(base))[0]
            deg += ex * flow_degree(sub)
    return int(deg)


def flow_latex(s: str) -> str:
    import re
    s = re.sub(r"\be(\d)\b", r"\\varepsilon_{\1}", s)
    s = re.sub(r"\bd(\d)\b", r"\\delta_{\1}", s)
    s = s.replace("zeta3", r"\\zeta(3)")
    return s


def pretty(expr, by_order=True):
    """Group by total flow order for readable display."""
    expr = sp.expand(expr)
    if not by_order:
        return flow_latex(sp.latex(expr))
    groups: dict[int, list] = {}
    for t in sp.Add.make_args(expr):
        groups.setdefault(flow_degree(t), []).append(t)
    lines = []
    for deg in sorted(groups):
        part = sp.together(sp.Add(*groups[deg]))
        lines.append(flow_latex(sp.latex(part)))
    return ("\n\\\\&\\quad+ ".join(lines))


doc = []
doc.append("# EGB inflation observables at third order in the slow-roll "
           "hierarchy\n\n")
doc.append("Exact-coefficient closed forms with the Green's-function "
           "constants $C=\\gamma_E+\\ln 2-2\\approx-0.7296$, $\\pi^2$, "
           "$\\zeta(3)$.  Generated by `scripts/derive_paper_formulas.py` "
           "— do not edit by hand.\n\n")
doc.append("**Action** ($M_{\\rm pl}=1$): "
           "$S=\\int d^4x\\sqrt{-g}\\,[R/2-\\tfrac12(\\partial\\phi)^2"
           "-V(\\phi)-\\tfrac12\\xi(\\phi)\\mathcal G]$.\n\n")
doc.append("**Flow variables** at the Hubble-crossing pivot $k=aH$: "
           "$\\varepsilon_1=-\\dot H/H^2$, $\\delta_1=4\\dot\\xi H$, "
           "$\\varepsilon_{i+1}=d\\ln\\varepsilon_i/dN$, "
           "$\\delta_{i+1}=d\\ln\\delta_i/dN$.\n\n")
doc.append("**Method**: exact sound-time reduction of both EGB sectors "
           "(Wu–Zhu–Wang arXiv:1707.08020 Eqs. 2.8–2.12; exact background "
           "identity $\\dot\\phi^2/H^2=2\\varepsilon_1-\\delta_1-"
           "\\delta_1\\varepsilon_1+\\delta_1\\delta_2$) composed with the "
           "Green's-function N3LO master of Auclair & Ringeval "
           "(arXiv:2205.12608); each sector evaluated at its own sound-"
           "horizon point $-k\\varsigma=1$ and Lie-shifted to the common "
           "pivot $k=aH$; truncated at total flow order 3.  Terms are "
           "grouped by ascending total flow order.\n\n")

for title, expr in (
    (r"n_s-1", sp.expand(n_s_expr - 1)),
    (r"n_T", sp.expand(n_T_expr)),
    (r"\alpha_s \equiv dn_s/d\ln k", sp.expand(alpha_s_expr)),
    (r"\alpha_T \equiv dn_T/d\ln k", sp.expand(alpha_t_expr)),
    (r"P_S \cdot 4\pi^2/H_*^2", P_S_red.expr()),
    (r"P_T \cdot \pi^2/(2 H_*^2)", sp.expand(P_T_red.expr())),
    (r"r \equiv P_T/P_S", r_expr),
):
    doc.append(f"\n## ${title}$\n\n")
    doc.append("$$\n\\begin{aligned}\n&" + pretty(expr)
               + "\n\\end{aligned}\n$$\n")

doc.append("\n## Numerical validation against the production engine\n\n")
doc.append("| model | quantity | closed form | engine | rel. diff |\n")
doc.append("|---|---|---|---|---|\n")
for name, flow, rows in report:
    for key, val, eng, drel in rows:
        doc.append(f"| {name} | {key} | {val:.8e} | {eng:.8e} "
                   f"| {drel:+.1e} |\n")
doc.append("\nResidual differences are the genuine $O(\\epsilon^4)$ "
           "truncation of the closed forms (the engine keeps the exact "
           "effective-flow mapping numerically).\n")

out = Path(__file__).resolve().parents[1] / "docs" / "PAPER_FORMULAS.md"
out.write_text("".join(doc))
log(f"written {out}")
