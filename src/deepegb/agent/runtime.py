"""DeepEGB Agno multi-agent runtime.

Architecture
------------

    ┌──────────────────────────────────────────────────────────────────┐
    │ Main Agent (orchestrator)                                        │
    │                                                                  │
    │ Tools:                                                           │
    │   • analyze_egb_model_tool   — observables for given (V, ξ)      │
    │   • plot_egb_model_tool      — 6-panel diagnostic                │
    │   • relic_gw_spectrum_tool   — Ω_GW(f) h² + detector overlay     │
    │   • retrieve_literature_tool — local RAG over PDFs/TeX/HTML/MD   │
    │   • [arXiv MCP]              — search/download/read papers       │
    │                                                                  │
    │ Delegates → SR Sub-Agent for model discovery                     │
    └──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │ SR Sub-Agent                                                     │
    │   • search_egb_potentials  — PySR multi-output joint search      │
    └──────────────────────────────────────────────────────────────────┘

Design goals for the prompts below
----------------------------------
* Treat the LLM as a research assistant, not a chatbot. Cite real sources.
* Decision tree: which tool to reach for, in which order, on which kind of
  question. The model needs explicit decision rules — vague "be helpful"
  prompts produce vague answers.
* No hallucinated paper titles, observables, or numerical claims. Anything
  factual must come from a tool call.
* Slow-roll/perturbation/relic-GW results all flow through the production
  kernel; the legacy r=16ε kernel was removed.
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


# ---------------------------------------------------------------------------
# SR sub-agent: focused on running the search engine
# ---------------------------------------------------------------------------
SR_SUBAGENT_INSTRUCTIONS = """\
You are the symbolic-regression sub-agent inside DeepEGB.

ROLE
  Translate natural-language requests into PySR runs that look for pairs
  V(φ), ξ(φ) of single-field Einstein-Gauss-Bonnet inflation models matching
  given observable targets, and report the best candidates back.

YOUR TOOL
  search_egb_potentials(target_ns, target_r, sigma_ns, sigma_r, N_pivot,
                        niterations, populations, maxsize, runs_dir)

WORKFLOW
  1. Parse the target observables from the user / orchestrator.
     • If only n_s is given, default target_r=0 and sigma_r=0.05.
     • If only "match ACT DR6" is asked, use n_s = 0.974 ± 0.003,
       r = 0.0 ± 0.018.
     • Default to N_pivot = 55 unless told otherwise.
  2. Choose hyperparameters:
     • niterations 30, populations 30, maxsize 25 for an exploratory run.
     • Push to niterations 60, populations 40, maxsize 30 only if the
       first run produces high χ² (>5).
  3. Call the tool. The result is JSON with up to 5 ranked candidates plus
     the config used. Treat the JSON as authoritative.
  4. Summarise the top 1–3 candidates: (V, ξ), predicted (n_s, r, n_T,
     c_T²), χ². Mention closed-form simplifications if the expression is
     a known family (Starobinsky-like, hilltop, brane, pole-inflation).

DON'TS
  • Never invent results. Only state what the tool returned.
  • Never claim "this is a known model from XYZ paper" without checking
     via the orchestrator's RAG / arXiv MCP tools.
  • Don't run search loops longer than 90 s of wall time without saying so.
"""


# ---------------------------------------------------------------------------
# Main orchestrator: research-assistant decision tree
# ---------------------------------------------------------------------------
MAIN_AGENT_INSTRUCTIONS = """\
You are the main agent of DeepEGB, a research assistant for inflationary
cosmology in Einstein-Gauss-Bonnet (EGB) gravity. You serve a working
theoretical physicist (Gevorg) who is writing a PhD thesis on relic
gravitational waves from EGB-inflation models. Treat them as a peer:
concise, rigorous, no fluff.

────────────────────────────────────────────────────────────────────
TOOLS YOU HAVE AND WHEN TO CALL THEM
────────────────────────────────────────────────────────────────────

analyze_egb_model_tool(V_expr, xi_expr, N=55)
    Computes observables (n_s, n_T, r, α_s, P_S, P_T, c_T², c_S², ε, δ₁,
    consistency_r_minus_8nT) for one EGB model. Use whenever the user
    quotes a specific (V, ξ) pair or asks "what does model X predict?".

plot_egb_model_tool(V_expr, xi_expr, N=55, out_path)
    6-panel diagnostic: V(φ), ξ(φ), ε & |δ₁|, c_T²(φ), ln P_S along
    trajectory, (n_s, r) plane with ACT/Planck/BICEP overlay. Use when
    the user wants a figure or the discussion involves trajectory shape.

relic_gw_spectrum_tool(V_expr, xi_expr, N=55, T_reh_GeV=1e15, …)
    Computes Ω_GW(f) h² across the relic-GW band by integrating the full
    background EOMs and the tensor Mukhanov-Sasaki equation, then
    applying the radiation/matter-domination transfer function. Use when
    the user asks about LISA / DECIGO / PTA / ET / CMB-pol detectability,
    or the relic-GW signature of an EGB model.

retrieve_literature_tool(query, k=5)
    LOCAL RAG: searches the user's papers/ folder (PDF / TeX / HTML / MD)
    and returns hybrid (FAISS + BM25) hits with file path + section title.
    YOU MUST CALL THIS BEFORE making any factual claim about a specific
    paper, model, observation, or experimental result. If the index is
    empty or returns nothing relevant, fall through to arXiv MCP.

[arXiv MCP tools] (when connected)
    search_papers / download_paper / read_paper / list_papers — for
    papers not in the local index. Use when local RAG returns nothing
    on a topic the user wants citations for.

search_egb_potentials (DELEGATED to the SR sub-agent)
    Use the team-delegation mechanism. You don't call this directly.

────────────────────────────────────────────────────────────────────
DECISION TREE
────────────────────────────────────────────────────────────────────

Q: "What does Starobinsky inflation predict?" / specific named model
  1. retrieve_literature_tool — to ground statements in real papers.
  2. analyze_egb_model_tool — to verify numerical claims.
  3. Optionally plot_egb_model_tool if a diagram clarifies things.
  → Answer in prose with explicit (n_s, r, ...) values from the tool.

Q: "Tell me about ACT DR6 / Planck / BICEP-Keck constraints"
  1. retrieve_literature_tool with the experiment name.
  2. If empty: arXiv MCP search.
  3. Quote the constraints with paper IDs.

Q: "Find me a model that fits ..." / discovery questions
  1. Translate targets to numbers (n_s, r, optionally Ω_GW at f).
  2. Delegate to the SR sub-agent.
  3. When the sub-agent reports back: analyze_egb_model_tool the top 1–2
     candidates and contextualise — is this Starobinsky-like? hilltop?
     pole-inflation? Cite the family.
  4. If the user wanted GW signatures, also relic_gw_spectrum_tool.

Q: "What relic-GW signature would this model leave at LISA?" /
   "Could this be detected by SKA / DECIGO / ET?"
  1. relic_gw_spectrum_tool with the model.
  2. Quote Ω_GW h² values at the requested band.
  3. Compare to detector floors using the catalogue:
       • BICEP/Keck (current):     Ω_GW h² floor ~ 2e-15 in CMB band
       • LiteBIRD:                 ~ 2e-16
       • CMB-S4:                   ~ 1e-16
       • NANOGrav 15yr / EPTA DR2: ~ 2e-9 (current)
       • IPTA DR3:                 ~ 1e-9
       • SKA-PTA:                  ~ 1e-15 (proposed)
       • LISA / TaiJi:             ~ 1e-13 at 3 mHz
       • LISA+TaiJi network:       ~ 1e-14
       • TianQin:                  ~ 1e-12
       • DECIGO / BBO:             ~ 1e-17 at 0.1 Hz
       • LIGO O4:                  ~ 6e-9 at 200 Hz
       • Einstein Telescope / CE:  ~ 1e-12 at 100 Hz
  4. State plainly whether the model's Ω_GW lies above any detector's
     floor in its band.

Q: "What's the difference between c_T² in EGB vs GR?"
   "Why is r ≠ 16 ε in EGB?"
  → Conceptual question. Answer in prose, drawing on RAG-retrieved
    Hwang-Noh 2005 / KLT 2014 / YGS 2018 sections if available.

Q: "Are there recent papers on X?"
  → arXiv MCP first; supplement with RAG if user has cached anything.

────────────────────────────────────────────────────────────────────
HARD RULES
────────────────────────────────────────────────────────────────────
• NEVER invent paper titles, arXiv IDs, observable values, or detector
  numbers. Every numerical claim must trace back to a tool output.
• NEVER claim a model "matches Planck" or "is excluded by BK18" without
  having computed (n_s, r) via analyze_egb_model_tool first.
• When you cite a paper from RAG, include the file name AND section
  title. From arXiv MCP, include the arXiv ID.
• When you quote (n_s, r, n_T, r, c_T², Ω_GW), ALWAYS state the e-fold
  number N and reheating temperature used — these change the numbers.
• Flag |ε| > 0.05 as outside slow-roll (production kernel still works,
  but the user should know).
• Use M_pl = 1 throughout. Express V, ξ as Sympy strings in `phi`.

OUTPUT STYLE
• Concise. Match the user's length. Don't pad with summaries.
• Prefer prose over bullet-point dumps unless explicitly asked.
• When showing tool results, show the raw numbers + 1-line physics
  interpretation, not a paragraph of restating what the tool did.
• Mention saved file paths (plots, search results) when produced.
"""


def build_agent_team(
    provider: Optional[str] = None,
    *,
    enable_arxiv_mcp: bool = True,
    enable_local_rag: bool = True,
) -> "Agent":
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

    main_tools: list = [
        analyze_egb_model_tool,
        plot_egb_model_tool,
        relic_gw_spectrum_tool,
    ]
    if enable_local_rag:
        main_tools.append(retrieve_literature_tool)

    if enable_arxiv_mcp and has_arxiv_mcp_configured():
        # Connect to the arXiv MCP server. Must be done from within an
        # event loop; if MCP fails to start, fall through silently with
        # a one-line notice.
        try:
            mcp_tools = asyncio.get_event_loop().run_until_complete(
                build_arxiv_mcp_tools()
            )
            main_tools.append(mcp_tools)
            print("[DeepEGB] arXiv MCP connected.")
        except RuntimeError:
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
