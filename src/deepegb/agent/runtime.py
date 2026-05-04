"""Agno multi-agent setup.

Architecture mirrors DeepInflation:

    ┌────────────────────────────────┐
    │  Main Agent (orchestrator)     │
    │  tools: analyze, plot          │
    │  delegates → SR Sub-Agent      │
    └────────────────────────────────┘
                │
                ▼
    ┌────────────────────────────────┐
    │  SR Sub-Agent                  │
    │  tools: search_egb_potentials  │
    └────────────────────────────────┘
"""
from __future__ import annotations

from typing import Optional

from .llm import get_model
from .tools import (
    analyze_egb_model_tool,
    plot_egb_model_tool,
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

You have three tools:
1. analyze_egb_model_tool(V_expr, xi_expr, N) — verify (n_s, r) for a model.
2. plot_egb_model_tool(V_expr, xi_expr, N, out_path) — diagnostic plot.
3. (delegated) search_egb_potentials(target_ns, target_r, sigma_ns, sigma_r,
   N_pivot, niterations, populations, maxsize) — discover new V(φ), ξ(φ).

When the user asks for a model or wants to explore the EGB landscape:
- Delegate to the SR sub-agent with explicit (n_s, r) targets.
- After receiving candidates, analyze and plot the top 1-2.
- Summarize in physics terms: is the model plateau-like? hilltop?
  natural-inflation-style? And contextualise vs. ACT DR6 / Planck / BICEP/Keck.

When the user asks about a specific named model:
- Use your background knowledge to describe it; if the RAG tool is available,
  use it to ground statements.
- Then call analyze_egb_model_tool to verify the model's predictions.

Conventions:
- Reduced Planck mass M_pl = 1.
- Express V and ξ as Sympy strings in the variable `phi`.
- Be explicit about which slow-roll formulas are leading-order; flag any
  result with |ε|>0.05 as outside slow-roll.

Be concise and physical. Avoid bullet-point dumps unless the user asks.
"""


def build_agent_team(provider: Optional[str] = None) -> "Agent":
    """Build the orchestrator Agent + SR sub-agent."""
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

    main_agent = Agent(
        name="DeepEGB Main Agent",
        model=model,
        instructions=MAIN_AGENT_INSTRUCTIONS,
        tools=[analyze_egb_model_tool, plot_egb_model_tool],
        team=[sr_agent],
        markdown=True,
    )
    return main_agent


def run_chat(
    initial_message: Optional[str] = None,
    provider: Optional[str] = None,
) -> None:
    """Interactive REPL with the agent team. Quits on /exit or Ctrl-D."""
    agent = build_agent_team(provider=provider)
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
