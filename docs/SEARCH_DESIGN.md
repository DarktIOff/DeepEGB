# How the symbolic search actually works

This document is the honest answer to the question
*"how does the search tool define what to look for? It seems to dance
around the same models."*

You were right — it does dance around a fixed shape. Here's why, what
we did about it, and what the proper fix looks like.

## The mismatch between PySR's fitness and ours

PySR is a free-form symbolic regressor. Given operators
`{+, -, *, /, exp, log, sqrt, tanh, ...}` and a target dataset `(X, y)`,
its evolutionary search generates expression trees that minimise

$$
\text{MSE} = \frac{1}{n}\sum_i \bigl(\hat f(x_i) - y_i\bigr)^2.
$$

We want the search to minimise the **EGB physics χ²** instead — i.e.
the distance in the (n_s, r, c_T², Ω_GW(f), …) space to user-specified
targets. PySR doesn't natively know about that loss; in our pipeline,
the χ² is only evaluated *after* PySR finishes, as a re-ranking step on
the hall-of-fame.

This means:

* The **exploration** is driven by MSE against `y_seed`.
* The **selection** afterwards is by physics-χ².

If `y_seed` is a single fixed shape, every member of every hall-of-fame
gets pushed toward that basin. Re-ranking by physics afterwards picks the
best of an already-biased set — so no matter how many iterations PySR
runs, the candidates all rhyme with the seed.

The previous version of DeepEGB hard-coded
`y_seed = (1 - exp(-√(2/3)|φ|))²` (a Starobinsky shape). Hence the
"every model is Starobinsky-flavoured" pattern you noticed.

## What we do now: multi-family seed sweep

Instead of one fixed seed, we run PySR multiple times — once per
inflation-potential family from a curated library
([`src/deepegb/search/seed_families.py`](../src/deepegb/search/seed_families.py)).
For each family we:

1. Build `y_seed = family.fn(phi)` (e.g. Starobinsky, hilltop, pole,
   natural-inflation, monodromy, exponential-plateau).
2. Run PySR with that seed for `niterations_per_family` iterations.
3. Collect the family's hall-of-fame (a Pareto frontier of
   complexity-vs-MSE).

We then merge all hall-of-fames, deduplicate, and re-rank by
physics-χ². The same scheme applies to ξ(φ) in pass 2 with a separate
ξ-family library (exp_decay, power_law, pole, tanh, linear, cosine).

This gives genuine cross-family diversity in the candidate pool. The
basins each PySR run explores are different starting from the very first
generation, so the re-ranking step has real choices to make.

**Caveat.** This is still not a true physics loss. Within each family's
sub-search, the bias toward that family's shape is still there. We
mitigate by sweeping many families; we *do not* eliminate the bias.

## Default seed families

V(φ) — eight inflation potential families:

| family       | shape                          | reference |
|---|---|---|
| starobinsky  | $(1 - e^{-\sqrt{2/3}\phi})^2$ | Starobinsky 1980; ACT-DR6 baseline |
| quadratic    | $\tfrac12 m^2 \phi^2$         | chaotic inflation, Linde 1983 |
| quartic      | $\lambda\phi^4$               | chaotic, ruled out by Planck |
| hilltop      | $(1 - (\phi/\mu)^2)^2$       | Boubekeur-Lyth 2005 |
| natural      | $1 + \cos(\phi/f)$           | axion, Freese-Frieman-Olinto 1990 |
| pole         | $1 - 8/\phi^2$               | brane / Kallosh family, ACT-favoured |
| monodromy    | $\|\phi\| + \cos(\phi/f)$    | Silverstein-Westphal 2008 |
| exp_plateau  | $1 - e^{-\|\phi\|/\mu}$       | smooth plateau |

ξ(φ) — six GB-coupling families:

| family    | shape              | reference |
|---|---|---|
| exp_decay | $\xi_0 e^{-\lambda\phi}$ | string-dilaton-like |
| power_law | $\xi_0 \phi^n$     | analytic at origin |
| pole      | $\xi_0/(\phi^2+\beta)$ | Kanti-style natural coupling |
| tanh      | $\xi_0 \tanh(\lambda\phi)$ | bounded smooth |
| linear    | $\xi_0 \phi$       | bare-bones |
| cosine    | $\xi_0 \cos(\phi/f)$ | axionic GB / Satoh-Soda |

Default subset (from `SearchConfig.v_seed_families` /
`xi_seed_families`):

```
v_seed_families  = ("starobinsky", "hilltop", "pole", "exp_plateau")
xi_seed_families = ("exp_decay", "pole", "power_law", "tanh")
```

Override via:

```bash
deepegb search --ns 0.974 --r 0.0 --N 55 \
   --v-families "starobinsky,natural,monodromy" \
   --xi-families "exp_decay,cosine" \
   --iters-per-family 30
```

Or in the agent: pass `enforce_egb=True` (default) and the rest is
config — for now, family choice is CLI-only.

## The proper fix (now wired up): Julia physics-χ² loss for V

Pass 1 of the search now optimises the actual EGB physics-χ² as PySR's
fitness function — no `y_seed` MSE bias at all. PySR's
`loss_function=...` parameter accepts a Julia source string; we
generate that string at runtime in
[`pysr_search.py::_build_julia_v_loss_source`](../src/deepegb/search/pysr_search.py)
with the user's targets substituted in.

The Julia loss does, per fitness call:

1. `eval_tree_array(tree, dataset.X, options)` — evaluate V(φ) at all
   grid points in one shot.
2. Central-difference V', V'' on the grid.
3. ε(φ) = ½(V'/V)², find first ε=1 crossing → φ_end.
4. Cumulative trapezoid of V/V' from φ_end → N(φ).
5. Find φ_pivot where N = N_pivot on the slow-roll side.
6. n_s = 1 − 6ε + 2η, r = 16ε at the pivot.
7. Return χ² with soft graded penalties (1e3..2e3) for partly-broken
   candidates so the genetic search still has a gradient.

The reference Julia implementation lives in
[`physics/kernel.jl`](../src/deepegb/physics/kernel.jl) for documentation
and easy review; the actual string passed to PySR is generated at
runtime so we can substitute (n_s, r, σ, N_pivot) per call.

**Selecting the loss path.** `SearchConfig.use_julia_loss` controls it:

* `"auto"` (default) — try the Julia loss; on any failure (no Julia, kernel
  compile error, etc.) fall back silently to the multi-family MSE sweep.
* `True` — require the Julia loss; raise on failure.
* `False` — skip Julia; use the multi-family MSE sweep only.

CLI: `--julia-loss auto|on|off`.

**What's still on MSE.** Pass 2 (the ξ search for each fixed top V) is
still the multi-family MSE sweep. The next step is to extend the Julia
loss to take a fixed V expression as a captured constant and search ξ
against the same physics-χ² — same shape as the V loss, two more lines
of substitution.

**Joint multi-output** (one PySR run that fits V and ξ together) is the
*final* step: PySR's `nout=2` mode plus a Julia loss that receives both
trees. That's a meaningful refactor of `_make_pysr` and the result
plumbing; not a one-day job.

## TL;DR

* PySR explores expressions freely from the operator set.
* Pass 1 (V search): now uses **Julia physics-χ² loss** by default. PySR's
  evolutionary search optimises EGB observables directly — no Starobinsky
  MSE bias. Auto-fallback to the multi-family MSE sweep if Julia fails.
* Pass 2 (ξ search): still **multi-family MSE seed sweep**. Cross-family
  diversity is the mechanism. Julia-loss extension for ξ is next.
* The result re-ranking by physics-χ² in Python is unchanged — but it
  now operates on a candidate pool that came out of the right basin to
  begin with.
