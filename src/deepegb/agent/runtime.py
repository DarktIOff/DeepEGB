"""Agno multi-agent setup.

Architecture mirrors DeepInflation, extended with two literature tools:

    ┌────────────────────────────────────────────────────────────┐
    │  Main Agent (orchestrator)                                 │
    │  tools: analyze_egb_model_tool, plot_egb_model_tool,       │
    │         relic_gw_spectrum_tool, retrieve_literature_tool,  │
    │         [arXiv MCP tools — search/fetch/read papers]       │
    │  delegates → SR Sub-Agent                                  │
    └────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌────────────────────────────────────────────────────────────┐
    │  SR Sub-Agent                                              │
    │  tools: search_egb_potentials                              │
    └────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import asyncio
from typing import Optional

from .llm import get_model
from .mcp_tools import build_arxiv_mcp_tools, has_arxiv_mcp_configured
from .tools import (
    analyze_egb_model_tool,
    plot_egb_model_tool,
    relic_gw_spectrum_tool,
    retrieve_literature_tool,
    search_egb_potentials,
)

try:
    from agno.agent import Agent
except ImportError:  # pragma: no cover
    Agent = None  # type: ignore


SR_SUBAGENT_INSTRUCTIONS = """\
You are the Symbolic Regression sub-agent for DeepEGB.

Your job: translate a user request into a symbolic regression search for
V(φ) and ξ(φ) in Einstein-Gauss-Bonnet inflation, run it via the
`search_egb_potentials` tool, and report the best candidates back.

Guidelines:
- Default to N = 55 e-folds before end of inflation.
- If only n_s is given, set target_r=0 and sigma_r≈0.05 (BICEP/Keck cap).
- Keep maxsize ≤ 25 to favor simple expressions.
- After the tool returns, summarize the top 1-3 (V, ξ) pairs in plain prose,
  noting their predicted (n_s, r) and any closed-form simplification.
- Mention χ² values so the main agent can compare runs.
- NEVER invent results; only report what the tool returned.
"""


MAIN_AGENT_INSTRUCTIONS = """\
You are the main agent of DeepEGB, an AI assistant for inflationary cosmology
research in Einstein-Gauss-Bonnet (EGB) gravity.

Available tools:
1. analyze_egb_model_tool(V_expr, xi_expr, N) — verify (n_s, r, n_T, r,
   c_T², ...) for a single model using the slow-roll closed-form kernel.
2. plot_egb_model_tool(V_expr, xi_expr, N, out_path) — 6-panel diagnostic.
3. relic_gw_spectrum_tool(V_expr, xi_expr, ...) — compute Ω_GW(f) h² across
   PTA / LISA / DECIGO / ET frequency bands using full Mukhanov-Sasaki.
4. retrieve_literature_tool(query, k) — search the LOCAL RAG index of
   PDFs/TeX/HTML the user has on disk. Use this BEFORE making claims about
   specific papers or models so you can cite real sources.
5. (optional, may be unavailable) arXiv MCP tools — for live search/fetch
   from arXiv when the local RAG doesn't have what you need.
6. (delegated) search_egb_potentials(target_ns, target_r, ...) — discover
   new (V, ξ) pairs via PySR symbolic regression.

Workflow guidance:
- For "find me a model" requests: delegate to the SR sub-agent. After the
  candidates come back, analyze + plot the top 1-2.
- For "tell me about X model" requests: FIRST call retrieve_literature_tool
  to ground in the user's local papers. If nothing relevant, fall back to
  the arXiv MCP if available, then to background knowledge clearly flagged
  as unsourced.
- For "what would the relic GW signal look like" requests: call
  relic_gw_spectrum_tool. Quote Ω_GW h² values at LISA (1 mHz), DECIGO
  (0.1 Hz), and PTA (1 nHz) bands.
- Always cite tool outputs (filenames + section titles for RAG hits;
  arXiv IDs for MCP hits). Never invent paper titles or numerical results.

Conventions:
- Reduced Planck mass M_pl = 1.
- Express V and ξ as Sympy strings in the variable `phi`.
- Flag |ε| > 0.05 as outside slow-roll.

Be concise and physical. Avoid bullet-point dumps unless the user asks.
"""


def build_agent_team(
    provider: Optional[str] = None,
    *,
    enable_arxiv_mcp: bool = True,
    enable_local_rag: bool = True,
) -> "Agent":
    """Build the orchestrator Agent + SR sub-agent.

    Parameters
    ----------
    provider          : LLM provider name (`local`, `anthropic`, `openai`,
                        `zai`). Defaults to env `DEEPEGB_PROVIDER`.
    enable_arxiv_mcp  : Whether to attempt connecting the arXiv MCP server.
                        Silently skipped if not configured / not installed.
    enable_local_rag  : Whether to expose the `retrieve_literature` tool
                        (requires a built RAG index — the tool reports
                        gracefully if there's none).
    """
    if Agent is None:
        raise RuntimeError("Agno is not installed. `pip install agno`.")

    model = get_model(provider)

    sr_agent = Agent(
        name="SR Sub-Agent",
        model=model,
        instructions=SR_SUBAGENT_INSTRUCTIONS,
        tools=[search_egb_potentials],
        markdown=True,
    )

    main_tools: list = [
        analyze_egb_model_tool,
        plot_egb_model_tool,
        relic_gw_spectrum_tool,
    ]
    if enable_local_rag:
        main_tools.append(retrieve_literature_tool)

    if enable_arxiv_mcp and has_arxiv_mcp_configured():
        # Connect to the arXiv MCP server. This must be done from within an
        # event loop; we run it synchronously here so the Agent constructor
        # sees the tools list. If MCP fails to start, fall through silently.
        try:
            mcp_tools = asyncio.get_event_loop().run_until_complete(
                build_arxiv_mcp_tools()
            )
            main_tools.append(mcp_tools)
            print("[DeepEGB] arXiv MCP connected.")
        except RuntimeError:
            # No running loop ⇒ create a fresh one
            try:
                mcp_tools = asyncio.run(build_arxiv_mcp_tools())
                main_tools.append(mcp_tools)
                print("[DeepEGB] arXiv MCP connected.")
            except Exception as exc:
                print(f"[DeepEGB] arXiv MCP not connected: {exc}")
        except Exception as exc:
            print(f"[DeepEGB] arXiv MCP not connected: {exc}")

    main_agent = Agent(
        name="DeepEGB Main Agent",
        model=model,
        instructions=MAIN_AGENT_INSTRUCTIONS,
        tools=main_tools,
        team=[sr_agent],
        markdown=True,
    )
    return main_agent


def run_chat(
    initial_message: Optional[str] = None,
    provider: Optional[str] = None,
    *,
    enable_arxiv_mcp: bool = True,
    enable_local_rag: bool = True,
) -> None:
    """Interactive REPL with the agent team. Quits on /exit or Ctrl-D."""
    agent = build_agent_team(
        provider=provider,
        enable_arxiv_mcp=enable_arxiv_mcp,
        enable_local_rag=enable_local_rag,
    )
    if initial_message:
        agent.print_response(initial_message, stream=True)

    print("\nDeepEGB ready — type your request (/exit to quit).\n")
    while True:
        try:
            msg = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        if msg in {"/exit", "/quit", ":q"}:
            break
        agent.print_response(msg, stream=True)
