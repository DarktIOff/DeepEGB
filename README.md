# DeepEGB

An AI agent for research and model discovery of inflation in Einstein-Gauss-Bonnet
gravity. Inspired by *DeepInflation* ([arXiv:2601.14288](https://arxiv.org/abs/2601.14288))
by Peng et al., extended to discover both the scalar potential `V(phi)` and the
Gauss-Bonnet coupling `xi(phi)` jointly via symbolic regression, with a physics
kernel built on exact Green's-function perturbation theory rather than asymptotic
approximations.

DeepEGB is built to run primarily against **local LLMs** served by `llama.cpp`
(OpenAI-compatible `/v1` endpoint), with optional fallback to Anthropic Claude,
OpenAI, or Z.AI GLM for harder reasoning steps.

## What it does

1. **Joint symbolic search** — uses PySR (genetic programming) to discover
   pairs `(V(phi), xi(phi))` whose predictions for the inflationary observables
   `(n_s, r, alpha_s, ...)` match a user-specified target. Current baseline:
   ACT DR6 + Planck + DESI DR2 BAO + BK18 (Calabrese et al. 2025, arXiv:2503.14454).

2. **Exact N3LO perturbation engine** — all observables (`n_s`, `n_T`, `r`,
   `alpha_s`, `P_S`, `P_T`, `c_S^2`, `c_T^2`) computed from the Auclair-Ringeval
   Green's-function master formulas (arXiv:2205.12608) with exact constants
   `C = gamma_E + ln2 - 2`, `pi^2`, `zeta(3)`. No UAA/WKB approximations.

3. **Full background and mode integration** — `egb_background.py` solves the
   Friedmann + Klein-Gordon equations without slow-roll truncation; `egb_modes.py`
   integrates the tensor and scalar Mukhanov-Sasaki equations from sub-horizon
   Bunch-Davies initial conditions.

4. **Relic GW spectrum** — integrates the tensor power spectrum and applies the
   radiation/matter-domination transfer function to produce `Omega_GW(f) h^2`
   today, covering the PTA, LISA, and ground-based detector bands. The k-array
   is anchored 2 decades below the CMB pivot and extends 18 decades upward.

5. **Agent-driven exploration** — an Agno main agent orchestrates a
   symbolic-regression sub-agent and a set of analysis tools, taking
   natural-language requests and translating them into search jobs.

6. **CosmoRAG + arXiv MCP** — a curated EGB inflation literature index served
   via MCP for RAG retrieval; the arXiv MCP server is available for fetching
   new papers during a session.

## Physics

The action is `S = integral d^4x sqrt(-g) [R/2 - (1/2)(d phi)^2 - V(phi) - (1/2) xi(phi) G]`
with `M_pl = 1`, where `G` is the Gauss-Bonnet scalar.

Slow-roll hierarchy: `eps_1 = -dH/H^2`, `delta_1 = 4 xi_dot H`, with higher
orders defined by `eps_{i+1} = d ln eps_i / dN`, `delta_{i+1} = d ln delta_i / dN`.

The EGB reheating history roughly doubles the e-fold budget relative to GR; the
pivot window is therefore `N_pivot in [50, 110]` rather than the GR `[50, 60]`.

Scalar sound speed `c_S^2` is computed from the exact WZW expression (Eq. 2.9,
arXiv:1707.08020); tensor speed `c_T^2` from Eq. 2.12. The sector flow is
reduced to an effective slow-roll hierarchy via the Auclair-Ringeval sound-time
mapping, and the N3LO master formulas are evaluated at the sector pivot.

The normalize-then-score strategy (`V -> lambda V`, `xi -> xi/lambda` with
`lambda` fixed by `ln(10^10 A_s) = 3.044`) means the symbolic search operates
entirely in shape space; amplitude calibration is an exact post-hoc rescaling,
not a free parameter in the loss.

## Architecture

```
User (CLI / chat)
      |
      v
Main Agent (Agno)
  provider:  local llama.cpp /v1   (default)
             Anthropic / OpenAI / GLM (fallback)
  tools:
    search_egb_potentials       -- PySR joint search (subprocess, non-daemonic)
    analyze_egb_model           -- single-model observables
    diagnose_egb_model          -- background integration diagnostics
    normalize_egb_model         -- exact V->lambda V amplitude calibration
    plot_egb_model              -- 6-panel diagnostic plot
    relic_gw_spectrum_tool      -- Omega_GW(f) h^2 + per-detector detectability
    retrieve_literature_tool    -- local RAG fallback (requires faiss index)
    cosmorag_search_tool        -- MCP RAG over curated EGB corpus (primary)
    arxiv MCP tools             -- search_papers, download_paper, read_paper
      |
      v
EGB Physics Kernel  (src/deepegb/physics/)
  egb_background.py    -- exact Friedmann + KG integration; multi-basin init
  egb_n3lo.py          -- sector grids, flow reduction, N3LO master evaluation
  _n3lo_master.py      -- Auclair-Ringeval ancillary (arXiv:2205.12608)
  egb_perturbations.py -- observable dispatch (n3lo / MS / slow-roll)
  egb_modes.py         -- Mukhanov-Sasaki tensor/scalar mode integration
  egb_slow_roll.py     -- slow-roll parameter grid; all ε=1 crossings
  relic_gw.py          -- transfer function, Omega_GW, detector catalogue
      |
      v
PySR symbolic regression  (src/deepegb/search/pysr_search.py)
  pass-1: V ranking via fiducial xi ladder (g in ±1e-3...3e-2)
  pass-2: joint (V, xi) search seeded with best fiducial xi
  refinement: N-scan over [50, 110] + constant polishing on top-k candidates
  loss: exact r = 16 eps_1 - 8 delta_1; alpha_s Q-form stencil
```

## Installation

```bash
git clone <repo> && cd DeepEGB
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# Julia + PySR (one-time):
python -c "import pysr; pysr.install()"
```

Run a local LLM with llama.cpp (example, Qwen2.5-Coder 32B):

```bash
llama-server -m ~/models/qwen2.5-coder-32b-q4_k_m.gguf \
             -c 32768 --host 127.0.0.1 --port 8080
```

Point DeepEGB at it via `.env`:

```
DEEPEGB_LLM_BASE_URL=http://127.0.0.1:8080/v1
DEEPEGB_LLM_API_KEY=sk-no-key-needed
DEEPEGB_LLM_MODEL=local
# Optional:
ANTHROPIC_API_KEY=...
DEEPEGB_PROVIDER=anthropic
OPENAI_API_KEY=...
ZAI_API_KEY=...
```

## Quick start

```bash
# Pure SR run, no agent:
deepegb search --ns 0.9752 --ns-sigma 0.0030 \
               --r 0.0    --r-sigma  0.019 \
               --alphas 0.0062 --alphas-sigma 0.0052 \
               --N-max 110 --niters 40

# Agent chat (local LLM):
deepegb chat

# Agent with Anthropic Claude:
deepegb chat --provider anthropic

# Single-model analysis:
deepegb analyze "V0*(1 - exp(-sqrt(2/3)*phi))^2" "xi0*phi"
```

## Observational targets (ACT DR6 baseline)

| Observable        | Target     | Sigma  | Source                                |
|-------------------|------------|--------|---------------------------------------|
| n_s               | 0.9752     | 0.0030 | P-ACT-LBDR2, arXiv:2503.14454 §8.1   |
| r (95% UL)        | < 0.038    | 0.019  | P-ACT-LB + BK18, arXiv:2503.14454 §4.4 |
| ln(10^10 A_s)     | 3.044      | 0.014  | Planck 2018 (ACT DR6 consistent)      |
| dn_s/dlnk         | +0.0062    | 0.0052 | P-ACT-LB, arXiv:2503.14454 Eq. 2     |

The positive running `dn_s/dlnk ~ +0.006` is unreachable in smooth GR slow roll
(alpha_s ~ -2 eps_1 eps_2 ~ -6e-4 for Starobinsky). EGB models with steep
`xi(phi)` generate a large `delta_2 = d ln delta_1 / dN` that drives alpha_s
positive without appreciably worsening n_s or r. The best-fit chi^2 floor from
the alpha_s term alone is ~1.7 for the current target set.

## Current status

- Production N3LO perturbation engine (arXiv:2205.12608 exact constants).
  Validated: GR consistency relation `r = -8 n_T` to 0.01%; Starobinsky
  `n_s = 0.9751` at N=55, chi^2 = 1.83 with linear xi.
- Full background EOM integration with numerically stable Friedmann solver
  (q-formula) and multi-basin end-of-inflation detection.
- Exact `c_S^2`, `c_T^2` from WZW sector equations (arXiv:1707.08020).
- Joint PySR search with normalize-then-score, N-marginalization over [50, 110],
  fiducial-xi pass-1 seeding, and refinement stage. Search ranking is physically
  calibrated (chi^2 floor ~1.7; previous runs returned chi^2 ~ 495 due to
  inverted scoring).
- Relic GW spectrum covering PTA (~nHz), LISA (~mHz), and ground-based (~100 Hz)
  bands with per-detector detectability summary.
- CosmoRAG MCP + arXiv MCP available to the agent during search sessions.
- Local RAG (FAISS + BM25) over PDF/TeX corpus; requires separate indexing step.
- In progress: multi-field inflation, PBH-generating spectral features.

## Acknowledgements

Architecture and design borrowed from *DeepInflation*
(Peng, Yuan, Lai, Jiang, Ye, Zhang, Piao; arXiv:2601.14288, 2026).
N3LO master formulas vendored from the ancillary files of arXiv:2205.12608
(Pierre Auclair and Christophe Ringeval).
