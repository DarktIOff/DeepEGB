# DeepEGB

An AI agent for research and model discovery of inflation in Einstein–Gauss–Bonnet
gravity. Inspired by *DeepInflation* ([arXiv:2601.14288](https://arxiv.org/abs/2601.14288))
by Peng et al., extended to discover both the scalar potential `V(φ)` and the
Gauss–Bonnet coupling `ξ(φ)` jointly via symbolic regression, with a physics
kernel adapted to EGB slow-roll inflation.

DeepEGB is built to run primarily against **local LLMs** served by `llama.cpp`
(OpenAI-compatible `/v1` endpoint) on a Strix Halo box, with optional fallback
to API providers (Anthropic Claude, OpenAI, Z.AI GLM, …) for harder reasoning
steps.

## What it does

1. **Joint symbolic search** — uses `PySR` (genetic programming) to discover
   pairs `(V(φ), ξ(φ))` whose predictions for the inflationary observables
   `(n_s, r, …)` match a user-specified target (e.g. ACT DR6 + BICEP/Keck
   constraints, or a hypothetical relic-GW bump).
2. **Verification & visualization** — for any candidate, computes observables
   from the EGB slow-roll EOMs and produces a diagnostic plot
   (potential, GB coupling, slow-roll parameters, `n_s–r` overlay).
3. **Agent-driven exploration** — an `Agno` main agent orchestrates a
   symbolic-regression sub-agent and a set of analysis tools, taking
   natural-language requests and translating them into search jobs.
4. **(v2) RAG over EGB literature** — index your `~/University/PhD/PhD/papers`
   folder + a curated EGB inflation reading list to ground LLM answers.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     User (CLI / notebook / chat)                    │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Main Agent (Agno)         ── llm.py picks provider:                │
│   ─ orchestrates                local llama.cpp /v1   (default)     │
│   ─ tool calls:                 Anthropic / OpenAI / GLM (fallback) │
│      • search_egb_potentials                                        │
│      • analyze_egb_model                                            │
│      • plot_egb_model                                               │
│      • [v2] retrieve_literature                                     │
│                                                                     │
│  SR Sub-Agent              ── translates user goals → PySR config   │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  EGB Physics Kernel  (src/deepegb/physics/egb_slow_roll.py)         │
│   ─ slow-roll EOMs with Gauss–Bonnet coupling ξ(φ)                  │
│   ─ end-of-inflation, e-fold counting, n_s, r                       │
│   ─ Julia mirror (kernel.jl) for hot-loop PySR loss                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
cd ~/University/PhD/PhD/DeepEGB
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

Then point DeepEGB at it (`.env`):

```bash
DEEPEGB_LLM_BASE_URL=http://127.0.0.1:8080/v1
DEEPEGB_LLM_API_KEY=sk-no-key-needed
DEEPEGB_LLM_MODEL=local
# Optional fallbacks:
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
ZAI_API_KEY=...
```

## Quick start

```bash
# Pure SR run, no agent — fastest sanity check:
deepegb search --ns 0.974 --ns-sigma 0.003 \
               --r 0.0   --r-sigma  0.018 \
               --N 55 --niters 40

# Agent chat (uses local LLM by default):
deepegb chat

# Agent with Claude as fallback for hard reasoning:
deepegb chat --provider anthropic
```

See `notebooks/01_smoke_test.ipynb` for a non-agent walk-through.

## Status

- ✅ **Production-grade EGB perturbation kernel** (Python; Julia mirror stubbed)
  — modified tensor speed `c_T²(φ)`, full `P_T`, `P_S`, `n_s`, `n_T`, `r`,
  `α_s = dn_s/dlnk`. Formulas pinned to Hwang-Noh 2005, KLT 2014, YGS 2018.
  See [`docs/PHYSICS_PRODUCTION.md`](docs/PHYSICS_PRODUCTION.md).
- ✅ Validated against GR limits (Starobinsky, m²φ²) — consistency relation
  `r = -8 n_T` reproduced to 0.01% in pure GR; `c_T² = 1` exactly.
- ✅ PySR joint search for `(V, ξ)` with the production χ², including
  optional targets on `ln(10¹⁰ A_s)`, `α_s`, `n_T`, `c_T²`.
- ✅ `analyze` (single model) and `plot` (6-panel diagnostic) tools.
- ✅ Agno agent with main + SR sub-agent.
- ✅ CLI: `search`, `chat`, `analyze`, `plot`.
- ✅ llama.cpp / Anthropic / OpenAI / GLM provider switch.
- ✅ **Full background EOM integration** — `egb_background.py` solves
  Friedmann + Klein-Gordon with `solve_ivp`, no slow-roll truncation.
- ✅ **Full scalar sound speed** `c_S²(φ)` — Kawai–Soda formula, deviates
  from 1 with the GB coupling.
- ✅ **Mukhanov-Sasaki mode integration** — `egb_modes.py` integrates the
  canonical tensor and scalar modes from sub-horizon Bunch–Davies to end
  of inflation, returning `P_T(k)`, `P_S(k)` exactly.
- ✅ **Relic GW spectrum** — `relic_gw.py` computes `Ω_GW(f) h²` today
  with RD/MD transfer function and `g_*(T)` thresholds, mapping inflation
  `k` to today's frequency in Hz. CLI: `deepegb relic-gw …`
- 🚧 RAG over local PDFs (stubbed; design in `src/deepegb/rag/__init__.py`).
- 🚧 Multi-field, hybrid exit, PBH-generating bumps.

## Acknowledgements

Architecture and many design choices borrowed from *DeepInflation*
(Peng, Yuan, Lai, Jiang, Ye, Zhang, Piao — arXiv:2601.14288, 2026).
Their codebase: <https://github.com/pengzy-cosmo/DeepInflation>.
