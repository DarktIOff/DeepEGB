# Architecture notes

## Modules

```
src/deepegb/
├── physics/        EGB slow-roll kernel (Python + Julia)
├── search/         PySR-based joint symbolic regression for V, ξ
├── analysis/       analyze_egb_model + plot_egb_model
├── agent/          Agno main agent + SR sub-agent + LLM provider abstraction
├── rag/            (v2) RAG over local EGB-inflation papers
└── cli.py          Click-based CLI: search / chat / analyze / plot
```

## Why this split

* `physics` has zero LLM and zero PySR dependencies — useful for unit testing
  and for invoking from notebooks. The Julia mirror exists to be plugged into
  PySR's `loss_function` for ~100× speedup, when wanted.
* `search` is the only place that imports PySR. It can be called directly
  (no agent) — that is what the CLI `search` subcommand does.
* `analysis` is shared between the CLI and the agent.
* `agent.tools` is the only file LLMs introspect; keep its function
  signatures simple and well-documented.
* `agent.llm.get_model` is the single decision point about *which* model
  backend to use. Add new providers there.

## Provider switch in one place

```python
from deepegb.agent import build_agent_team
agent = build_agent_team(provider="local")        # llama.cpp
agent = build_agent_team(provider="anthropic")    # Claude
agent = build_agent_team(provider="openai")       # GPT
agent = build_agent_team(provider="zai")          # GLM
```

All four routes go through Agno, so the rest of the code is provider-agnostic.

## Two-pass vs joint multi-output search

The MVP ships a **two-pass** strategy (search V with ξ=0, then search ξ
holding V fixed). This is robust against PySR's unstable behavior with
multi-output objectives on small problems and is much cheaper.

A genuine multi-output joint search is wired up under `mode="joint"` in
`SearchConfig` for future use, but is not the default. Worthwhile when
the V↔ξ coupling materially shapes the search landscape, which is more
common in higher-curvature regimes.

## Tool calling reliability with local LLMs

Smaller local models (≤ ~13B) often emit malformed JSON in tool-call
arguments. Mitigations already applied:

* `agent/runtime.py` instructions are explicit about argument names.
* `agent/llm.py` does not depend on structured-output features.
* `tools.py` returns plain JSON strings, not Pydantic objects.
* `configs/default.yaml.llm.tool_call_retries` is honored by Agno.

If you find a particular local model is not tool-calling well, fall back
to `--provider anthropic` for orchestration while keeping local models for
the RAG embedding side (when v2 lands).
