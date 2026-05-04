# Development & extension guide

## Setup

```bash
cd ~/University/PhD/PhD/DeepEGB
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# One-time Julia toolchain bootstrap for PySR:
python -c "import pysr; pysr.install()"

# Optional: RAG extras
pip install -e ".[rag]"

cp .env.example .env   # edit with your endpoints / keys
```

## Running

```bash
# Quick offline kernel test (no PySR, no LLM):
python scripts/smoke_test.py

# Unit tests:
pytest -q tests/

# Pure SR run (no LLM):
deepegb search --ns 0.974 --r 0.0 --N 55 --niters 30

# Verify a single model:
deepegb analyze "(1-exp(-sqrt(2/3)*phi))**2" "0.05*exp(-0.4*phi)" --N 55

# Diagnostic plot:
deepegb plot "(1-exp(-sqrt(2/3)*phi))**2" "0.05*exp(-0.4*phi)" \
             --N 55 --out outputs/star_egb.png

# Agent chat (defaults to local llama.cpp at $DEEPEGB_LLM_BASE_URL):
deepegb chat
deepegb chat --provider anthropic  # use Claude
deepegb chat -m "Find me an EGB model with n_s=0.97 and r<0.01."
```

## Running the local LLM

DeepEGB uses an OpenAI-compatible `/v1` endpoint, which `llama.cpp` exposes
out of the box:

```bash
# Strix Halo (AMD) — build llama.cpp with -DGGML_HIPBLAS=1 -DAMDGPU_TARGETS=gfx1151
llama-server -m ~/models/qwen2.5-coder-32b-q4_k_m.gguf \
             -c 32768 -ngl 99 \
             --host 127.0.0.1 --port 8080
```

Then `DEEPEGB_LLM_BASE_URL=http://127.0.0.1:8080/v1` in `.env`.

Recommended local models for tool-calling reliability:
- `Qwen2.5-Coder-32B-Instruct` (best in this size class)
- `Llama-3.1-70B-Instruct` (if you can fit it)
- `DeepSeek-V3.x` distills
- `Qwen3-Next-80B` (newer, supports very long context)

## Extending

### Add an EGB observable
Edit `physics/egb_slow_roll.py`. Keep the `Observables` dataclass append-only
so nothing downstream breaks.

### Tighten the SR loss
Replace `chi2_for_expressions` in `search/pysr_search.py` with a Julia loss
kernel (see `physics/kernel.jl`). Pass the kernel to PySR via
`PySRRegressor(loss_function="...")`.

### Add a new tool to the agent
1. Add a Python function in `agent/tools.py` with rich type annotations and a
   docstring (Agno parses both).
2. Add it to `all_tools()` and (if it should be on the SR sub-agent rather
   than the main one) plumb it through `agent/runtime.py`.

### RAG over your papers
1. `pip install -e ".[rag]"`.
2. Drop your PDFs in `~/University/PhD/PhD/papers/`.
3. Implement `rag/index.py` (build FAISS + BM25 from chunks) and
   `rag/retrieve.py` (combine dense+sparse). Expose
   `retrieve_literature(query, k)` as a tool.
4. Add it to the main agent's tool list.

## Conventions

* Python ≥ 3.10. Type hints everywhere.
* `M_pl = 1` natural units throughout.
* Sympy strings use `phi` as the field name.
* All file outputs go to `outputs/` or `runs/`, both in `.gitignore`.
