# Running DeepEGB locally on your Mac (z.ai-driven)

This is a step-by-step recipe to get DeepEGB working on macOS using **GLM
models from Z.AI** as the LLM backend. The same flow works on Linux; on
Strix Halo you can later swap in a local llama.cpp endpoint without
changing any code.

## 1. Prerequisites

You need:

* **Python ≥ 3.10**. Check with `python3 --version`. If you're below 3.10,
  install via Homebrew: `brew install python@3.12`.
* **Julia** (for PySR's genetic-programming backend). PySR ships an
  installer; you don't need to install Julia separately.
* **Xcode CLI tools** (for building any wheels that need a C compiler):
  `xcode-select --install`.

You will need a Z.AI API key. Get one at <https://z.ai/manage-apikey/apikey-list>
and copy it; we'll paste it into a `.env` file in step 4.

## 2. Install DeepEGB

```bash
cd ~/University/PhD/PhD/DeepEGB
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -e ".[dev]"
```

Then install PySR's Julia toolchain (one-time, takes a few minutes):

```bash
python -c "import pysr; pysr.install()"
```

If `pysr.install()` fails on macOS, do this instead:

```bash
# Install Julia manually
brew install --cask julia
# Then re-try
python -c "import pysr; pysr.install()"
```

Sanity check (no LLM yet):

```bash
PYTHONPATH=src python3 scripts/smoke_test.py
PYTHONPATH=src python3 -m pytest tests/ -q
```

You should see `23 passed, 1 skipped`.

## 3. Decide which GLM model to use

Z.AI offers several models compatible with the OpenAI SDK schema. The
relevant ones for an agent that has to call tools are:

| Model           | Tool-calling | Speed | Cost           | Recommendation |
| --------------- | -----------: | ----: | -------------: | -------------- |
| `glm-4.6`       | very good    | fast  | low            | **default for DeepEGB**  |
| `glm-4.7`       | very good    | fast  | low–med        | newer; try if 4.6 misbehaves |
| `glm-5`         | excellent    | med   | med            | for harder reasoning |
| `glm-5.1`       | excellent    | med   | higher         | most capable; best for SR planning |
| `glm-4.5`       | good         | fast  | lowest         | only if budget-tight |
| `glm-5-turbo`   | good         | fastest | lowest       | for quick chats, not SR |

Start with `glm-4.6`. Bump up to `glm-5.1` for hard searches.

## 4. Configure environment

```bash
cp .env.example .env
nano .env   # or open in your editor of choice
```

Fill in:

```dotenv
DEEPEGB_PROVIDER=zai

ZAI_API_KEY=sk-zai-...your-key-here...
ZAI_BASE_URL=https://api.z.ai/api/paas/v4/
ZAI_MODEL=glm-4.6
```

Leave the local-LLM and Anthropic/OpenAI sections blank — you can fill them
in later.

## 5. Verify the connection

```bash
source .venv/bin/activate
PYTHONPATH=src python3 -c "
from deepegb.agent import get_model, resolve_provider
cfg = resolve_provider('zai')
print('Using:', cfg.name, cfg.model, '@', cfg.base_url)
model = get_model('zai')
print('Agno model object:', type(model).__name__)
"
```

You should see something like:

```
Using: zai glm-4.6 @ https://api.z.ai/api/paas/v4/
Agno model object: OpenAIChat
```

If you see an error about missing `ZAI_API_KEY`, your `.env` isn't being
loaded — make sure you're running from the project root.

## 6. Try it

### a) Pure SR run (no LLM, fastest)

```bash
deepegb search --ns 0.974 --r 0.0 --N 55 --niters 20 --populations 20
```

This runs PySR with the **slow-roll closed-form χ²** loss and prints the
top candidates. No GLM calls; takes 30 s – 2 min depending on your CPU.

### b) Single model verification

```bash
deepegb analyze "(1-exp(-sqrt(2/3)*phi))**2" "0" --N 55
deepegb plot    "(1-exp(-sqrt(2/3)*phi))**2" "0.05*exp(-0.4*phi)" \
                --N 55 --out outputs/star_egb.png
```

`plot` produces the 6-panel diagnostic with V, ξ, ε, c_T², ln P_S, and
the (n_s, r) plane.

### c) Relic GW spectrum

```bash
deepegb relic-gw "1e-10*(1-exp(-sqrt(2/3)*phi))**2" "0" \
                 --N 55 --decades 12 --T-reh 1e15 \
                 --out outputs/star_relic_gw.png
```

This runs the full background EOMs, integrates Mukhanov-Sasaki for ~30
modes, applies the transfer function, and produces the `Ω_GW(f) h²` plot
across PTA → LISA → DECIGO → ET/LIGO bands.

### d) Search targeting LISA-band relic GWs

```bash
deepegb search --ns 0.974 --r 0.05 \
               --gw-target 1e-3:1e-13:5e-14 \
               --N 55 --niters 8 --populations 12 \
               --maxsize 20
```

This tries to discover `(V, ξ)` pairs that simultaneously hit
`n_s = 0.974` and produce `Ω_GW h² ≈ 10⁻¹³` at 1 mHz (LISA band). Note
the **slow loss** (≈1 s/eval) — keep `--niters` and `--populations` small.

For "make this loud across LISA":

```bash
deepegb search --ns 0.965 --r 0.05 \
               --gw-band-min 1e-4:1e-1:1e-14 \
               --N 55 --niters 6 --populations 10
```

### e) Agent chat

```bash
deepegb chat
# or one-shot:
deepegb chat -m "Find me an EGB inflation model with n_s=0.974 and r<0.01."
```

The agent will (i) interpret your request, (ii) call the SR sub-agent which
calls `search_egb_potentials`, (iii) `analyze_egb_model_tool` the best
candidates, and (iv) optionally `plot_egb_model_tool`. All tool calls are
logged to stdout so you can see what the agent is doing.

If GLM hallucinates or returns malformed tool calls, switch to a more
capable model:

```bash
ZAI_MODEL=glm-5.1 deepegb chat
```

## 7. Where things live

```
~/University/PhD/PhD/DeepEGB/
├── outputs/             ← diagnostic plots, relic-GW spectra
├── runs/                ← PySR scratch + per-search results.json
├── src/deepegb/         ← code
└── docs/                ← physics + architecture + this file
```

The `outputs/` and `runs/` directories are git-ignored. Search results
JSON-dumps in `runs/results.json` after each `deepegb search`.

## 8. Common issues and fixes

* **`pysr.install()` hangs on Apple Silicon.** Install Julia via Homebrew
  (`brew install --cask julia`), then re-run.
* **`No module named deepegb`.** You forgot to `pip install -e ".[dev]"`,
  or you're not in the venv. `source .venv/bin/activate` and re-install.
* **Z.AI returns 401.** Check your `ZAI_API_KEY` and that you're using the
  exact key string (no surrounding whitespace).
* **Tool calls fail with "function not found".** Sometimes GLM emits
  malformed JSON. Bump to `glm-5.1` or fall back to Claude with
  `deepegb chat --provider anthropic` (set `ANTHROPIC_API_KEY` first).
* **`production_gw` loss is too slow.** Reduce `--niters` and
  `--populations`. Each `(V, ξ)` evaluation takes ≈1 s with one GW target,
  so a 10×10 search ≈ 100 s of pure loss + the genetic-search overhead.
* **Search produces only `xi = 0` candidates.** That means the GR
  (`xi = 0`) family already saturates your targets. Add a GW target to
  break the GR limit, or include `--target-cT2 1.05` to demand a non-trivial
  tensor sound speed.

## 9. Next moves

* When you eventually run on the Strix Halo box, switch
  `DEEPEGB_PROVIDER=local` and point `DEEPEGB_LLM_BASE_URL` at your
  llama.cpp `/v1` endpoint. Nothing else changes.
* For thesis-quality runs, use `--loss production_gw` with a band target
  matched to your detector of interest (LISA at 1e-4 to 1e-1 Hz; DECIGO
  at 0.01 to 10; PTAs at 1e-9 to 1e-7).
* If the GLM agent is the bottleneck (because tool-calling is flaky),
  use the **CLI directly** (`deepegb search …`) — it doesn't need any LLM
  and is the fastest way to actually find candidates. Then ask the agent
  to *interpret* the candidates rather than discover them.
