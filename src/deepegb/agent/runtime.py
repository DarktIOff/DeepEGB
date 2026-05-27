"""DeepEGB Agno agent runtime.

Architecture
------------

    ┌──────────────────────────────────────────────────────────────────┐
    │ DeepEGB Agent (single, all tools attached)                       │
    │                                                                  │
    │ Tools:                                                           │
    │   • search_egb_potentials    — PySR multi-output search          │
    │   • analyze_egb_model_tool   — observables for given (V, ξ)      │
    │   • plot_egb_model_tool      — 6-panel diagnostic                │
    │   • relic_gw_spectrum_tool   — Ω_GW(f) h² + detector overlay     │
    │   • retrieve_literature_tool — local RAG over PDFs/TeX/HTML/MD   │
    │   • [arXiv MCP, when avail.] — search/download/read papers       │
    └──────────────────────────────────────────────────────────────────┘

We use a SINGLE-agent layout rather than the orchestrator + sub-agent
pattern from the DeepInflation paper, because Agno's multi-agent API has
changed across versions and a single agent with a clear decision tree is
simpler and more robust. The system prompt tells the agent how to chain
the tools (RAG → analyze → plot → relic-GW → search).

Design goals for the prompt below
---------------------------------
* Treat the LLM as a research assistant, not a chatbot. Cite real sources.
* Decision tree: which tool to reach for, in which order.
* No hallucinated paper titles, observables, or numerical claims —
  everything factual comes from a tool call.
* Production kernels only; the legacy r=16ε toy was removed.
"""
from __future__ import annotations

import asyncio
import os
import re
import uuid
from pathlib import Path
from typing import Any, Optional

from .llm import get_model
from .mcp_tools import (
    build_arxiv_mcp_tools,
    build_cosmorag_mcp_tools,
    has_arxiv_mcp_configured,
    has_cosmorag_mcp_configured,
)
from .tools import (
    analyze_egb_model_tool,
    diagnose_egb_model_tool,
    normalize_egb_model_tool,
    plot_egb_model_tool,
    relic_gw_spectrum_tool,
    retrieve_literature_tool,
    search_egb_potentials,
)
from .cosmorag_tools import all_cosmorag_tools, is_cosmorag_available

try:
    from agno.agent import Agent
except ImportError:  # pragma: no cover
    Agent = None  # type: ignore

_db: Any = None
try:
    from agno.db.sqlite import SqliteDb
    _db_path = Path(os.path.expanduser("~/.deepegb/agent.db"))
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    _db = SqliteDb(db_file=str(_db_path))
except Exception as _db_exc:
    _db = None
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "DeepEGB session persistence disabled: could not initialise "
        "SqliteDb (%s).  Install SQLAlchemy or check ~/.deepegb/ permissions.",
        _db_exc,
    )
    print(
        f"[DeepEGB] WARNING: session persistence disabled — {_db_exc}.  "
        "Install SQLAlchemy (`pip install sqlalchemy`) or check "
        "~/.deepegb/ permissions."
    )

_MemoryManager: Any = None
_MemoryTools: Any = None
try:
    from agno.memory.manager import MemoryManager as _AgnoMemoryManager
    _MemoryManager = _AgnoMemoryManager
except ImportError:
    pass
try:
    from agno.tools.memory import MemoryTools as _AgnoMemoryTools
    _MemoryTools = _AgnoMemoryTools
except ImportError:
    pass

_SessionSummaryManager: Any = None
try:
    from agno.session.summary import SessionSummaryManager as _AgnoSSM
    _SessionSummaryManager = _AgnoSSM
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Single-agent research-assistant decision tree
# ---------------------------------------------------------------------------
MAIN_AGENT_INSTRUCTIONS = """\
You are DeepEGB, a research assistant for inflationary
cosmology in Einstein-Gauss-Bonnet (EGB) gravity. You serve a working
theoretical physicist (Gevorg) who is writing a PhD thesis on relic
gravitational waves from EGB-inflation models. Treat them as a peer:
concise, rigorous, no fluff.

────────────────────────────────────────────────────────────────────
TOOLS YOU HAVE AND WHEN TO CALL THEM
────────────────────────────────────────────────────────────────────

search_egb_potentials(target_ns, target_r, sigma_ns, sigma_r, N_pivot,
                      niterations, populations, maxsize, runs_dir)
    Discover NEW (V(φ), ξ(φ)) candidates via PySR symbolic regression.
    Defaults: target_r=0.0, sigma_r=0.05, N_pivot=55, niterations=30,
    populations=30, maxsize=25. Bump to niters=60, populations=40 only
    if a first run produces χ²>5. Returns JSON with the top 5
    candidates plus the config used.

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

normalize_egb_model_tool(V_expr, xi_expr, N=55, P_S_target=2.1e-9)
    Rescale (V, ξ) so the predicted scalar power amplitude P_𝓡 at
    horizon crossing matches the Planck measurement (2.10 × 10⁻⁹).
    Uses the slow-roll invariance V → λV, ξ → ξ/λ — keeps (n_s, r,
    n_T, ε, c_T², δ_1) unchanged, scales P_𝓡 by λ. Returns the
    normalised V_expr / xi_expr plus before/after values for every
    observable so you can verify nothing else drifted.
    CALL THIS on every candidate returned by `search_egb_potentials`
    AND on any user-supplied model whose |ln(10¹⁰ A_s) − 3.044| > 0.5.
    The bare SR output has arbitrary V scale; without normalisation,
    P_𝓡 is meaningless.

diagnose_egb_model_tool(V_expr, xi_expr, N=55, target_ns=..., target_r=...)
    Use this BEFORE giving up on a model that produced NaN observables,
    a huge χ² (> 1e3), or a "background_failure" message. Returns:
      • whether φ_end / φ_pivot were found,
      • soft_penalty + the qualitative reasons,
      • atomized χ² breakdown (per-component contributions),
      • dominant_chi2 (top 3 components driving the loss),
      • concrete suggestions for fixing V or ξ.
    The "components" dict tells you EXACTLY which term is killing the
    fit — e.g. omega_gw@1mHz=5012 means the GW amplitude is 5 decades
    above target.  Don't say "search collapsed" without first running
    this and reporting the dominant component.

retrieve_literature_tool(query, k=5)
    LOCAL RAG: searches the user's papers/ folder (PDF / TeX / HTML / MD)
    and returns hybrid (FAISS + BM25) hits with file path + section title.
    YOU MUST CALL THIS BEFORE making any factual claim about a specific
    paper, model, observation, or experimental result. If the index is
    empty or returns nothing relevant, fall through to arXiv MCP.

[arXiv MCP tools] (when connected)
    search_papers / download_paper / read_paper / list_papers — for
    papers not in the CosmoRAG database or local index.

[CosmoRAG tools] (when connected)
cosmorag_search_tool(query, top_k=10)
    Search the CosmoRAG cosmology-paper knowledge base. Hybrid retrieval:
    FTS5 + token overlap + equation match + embedding similarity +
    cross-encoder reranking over all indexed arXiv papers.  Returns
    passages with arXiv ID, section, LaTeX equations, and scores.
    PREFER over retrieve_literature_tool — CosmoRAG's equation-aware
    retrieval is more powerful for physics queries containing formulas.

cosmorag_list_papers_tool()
    List all papers in CosmoRAG with title, arXiv ID, authors, year.
    Call at the start of a research session to know what is available.

cosmorag_get_paper_tool(arxiv_id)
    Full metadata (title, authors, year, categories, abstract) for one
    paper by arXiv ID.  Use when the user asks for citation data or
    the abstract.

cosmorag_add_paper_tool(arxiv_id)
    Download and ingest one arXiv paper into CosmoRAG.  Paper is
    FTS5-searchable immediately; embeddings run in background.
    Call when search returns nothing and the relevant paper is known.

cosmorag_add_papers_tool(arxiv_ids)
    Ingest multiple papers at once with a single background embedding
    pass — more efficient than repeated single-paper calls.

cosmorag_embedding_status_tool()
    Live progress and ETA for the background embedding worker.
    Call after cosmorag_add_paper_tool if the user asks about status.

────────────────────────────────────────────────────────────────────
DECISION TREE
────────────────────────────────────────────────────────────────────

Q: "What does Starobinsky inflation predict?" / specific named model
  1. retrieve_literature_tool — to ground statements in real papers.
  2. analyze_egb_model_tool — to verify numerical claims.
  3. plot_egb_model_tool — always produce the 6-panel diagnostic.
  → Answer in prose with explicit (n_s, r, ...) values from the tool.

Q: "Tell me about ACT DR6 / Planck / BICEP-Keck constraints" /
   factual physics claim / "what does paper X say?"
  TOOL PRIORITY for literature retrieval:
  1. cosmorag_search_tool(query, top_k=10) — primary (equation-aware).
  2. retrieve_literature_tool(query) — fallback (local FAISS+BM25 index).
  3. [arXiv MCP] search_papers — for papers not yet in either system.
  Always cite with arXiv ID + section from whichever tool returned the hit.
  If cosmorag_search_tool returns nothing useful, call cosmorag_add_paper_tool
  for the relevant arXiv ID first, then search again.

Q: "Find me a model that fits ..." / discovery questions
  1. Translate targets to numbers (n_s, r, optionally Ω_GW at f).
  2. Call search_egb_potentials directly with those targets.
  3. For each of the top 1–2 returned candidates:
     a. MANDATORY: normalize_egb_model_tool to rescale V → λV, ξ → ξ/λ
        so the predicted scalar power amplitude P_𝓡 matches Planck
        (2.10×10⁻⁹).  The raw SR output sits at an arbitrary V scale;
        reporting unnormalised observables to the user is a bug.
     b. analyze_egb_model_tool on the NORMALISED expressions — confirm
        (n_s, r, ε, c_T²) are unchanged (they are slow-roll invariants)
        and ln(10¹⁰ A_s) ≈ 3.044 after normalisation.
     c. Contextualise: Starobinsky-like? hilltop? pole-inflation? Cite
        the family from the local RAG if available.
  4. ALWAYS call plot_egb_model_tool on the NORMALISED best candidate.
     Save to outputs/<short_name>_diagnostic.png. Report the saved path.
  5. ALWAYS call relic_gw_spectrum_tool on the NORMALISED best candidate
     (Ω_GW depends on H_pivot which depends on V₀, so an un-normalised
     spectrum is meaningless). Quote Ω_GW h² at the LISA band (1 mHz)
     and the PTA band (10 nHz), compare to detector floors, and state
     whether it is detectable.

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
  1. cosmorag_search_tool(X) — see what is already indexed.
  2. [arXiv MCP] search_papers(X) — discover new papers.
  3. cosmorag_add_papers_tool([ids]) — ingest interesting finds.
  Report which papers are already in CosmoRAG and which are new.

────────────────────────────────────────────────────────────────────
HARD RULES
────────────────────────────────────────────────────────────────────
• ξ(φ) = 0 IS NOT AN ANSWER.  It reduces the EGB action S = ∫√−g
  [R/2 − ½(∂φ)² − V − ½ξ𝒢] to plain General Relativity. The whole
  point of this tool is to discover EGB inflation models, not GR.
  - Never propose ξ_expr = "0" as a "discovered" model.
  - If the search returns ξ ≡ 0 candidates (because the user passed
    enforce_egb=False), label them clearly as "GR baseline" and treat
    them only as a comparison anchor.
  - If `diagnose_egb_model_tool` returns "is_gr_limit": true, the model
    has degenerated to GR. Tell the user; suggest a non-trivial ξ form.
• NEVER invent paper titles, arXiv IDs, observable values, or detector
  numbers. Every numerical claim must trace back to a tool output.
• NEVER claim a model "matches Planck" or "is excluded by BK18" without
  having computed (n_s, r) via analyze_egb_model_tool first.
• When you cite a paper from RAG, include the file name AND section
  title. From arXiv MCP, include the arXiv ID.
• When you quote (n_s, r, n_T, c_T², Ω_GW), ALWAYS state the e-fold
  number N and reheating temperature used — these change the numbers.
• Flag |ε| > 0.05 as outside slow-roll (production kernel still works,
  but the user should know).
• Use M_pl = 1 throughout. Express V, ξ as Sympy strings in `phi`.

INEQUALITY TARGETS
• "n_s = X" → target_ns=X, sigma_ns=0.005 (default)
• "n_s ≈ ACT DR6" → target_ns=0.974, sigma_ns=0.003
• "r < 0.01" → target_r=0, sigma_r=0.005   (NOT target_r=0.005)
                The χ² is < 1 whenever r < sigma_r in this convention.
• "r ≤ 0.036 (BK18)" → target_r=0, sigma_r=0.018
• "Ω_GW > 1e-13 at 1 mHz" → use omega_gw_targets=[(1e-3, 1e-13, 5e-14)]
• "loud across LISA" → use omega_gw_band_min=(1e-4, 1e-1, 1e-13)

ERROR HANDLING (CRITICAL)
• If a tool returns an "error" field starting with "TOOL_ERROR:", DO NOT
  retry the same call with the same arguments. The "do_not_retry": true
  flag means the configuration itself is broken.
• If a search returns ALL χ² ≈ 1e6 (the soft-invalid floor) or relic_gw
  reports "background_failure" → do NOT just give up.  Run
  diagnose_egb_model_tool on a representative candidate first. Report
  back: which component dominates, what the soft_penalty reasons say,
  what the suggestions are. Then propose a concrete fix and try again.
• Examples of acting on the breakdown:
    - dominant component = "omega_gw@1mHz" with contribution > 1e3
       → the model is too loud at LISA. Tell the user; reduce ξ
         amplitude or weaken the V slope at horizon crossing.
    - dominant component = "n_s" with contribution > 1e3
       → spectral tilt is way off. Try a different V family.
    - reasons include "ε > 1 everywhere"
       → no slow-roll. Pick a flatter V (Starobinsky, hilltop).
    - reasons include "V(φ) ≤ 0 over X% of φ-range"
       → restrict the φ-range or add a positive shift to V.
• On TOOL_ERROR from search_egb_potentials, fall back as follows:
    1. State plainly to the user that search failed (paste the message
       and the tool's "suggestion" field).
    2. Pick a known-good representative model that matches the user's
       targets approximately, e.g.:
         - n_s ≈ 0.965, r ≪ 1 → Starobinsky
             V_expr = "(1 - exp(-sqrt(2/3)*phi))**2", xi_expr = "0"
         - higher n_s (ACT-favoured), small r → Kallosh family
             V_expr = "1 - 8/phi**2", xi_expr = "0"
         - quartic chaotic with EGB → V_expr="0.05*phi**4",
             xi_expr="0.1/(phi**2+1)"
    3. analyze_egb_model_tool that fallback model and present its
       (n_s, r, n_T, c_T², ε) as the best-effort answer.
    4. Tell the user the discovery search needs to be re-run via the
       CLI: `deepegb search --ns ... --r ... --N 55` and link to it.
• On TOOL_ERROR from any other tool, surface the message; do not retry.
• On a "no_candidates" result, suggest looser sigmas to the user; do not
  silently retry with new parameters unless the user asks you to.

OUTPUT STYLE
• Concise. Match the user's length. Don't pad with summaries.
• Prefer prose over bullet-point dumps unless explicitly asked.
• When a tool returns numbers, you MUST quote those numbers in your
  response. Saying "I will analyse the model" without showing what the
  analyse tool returned is a bug. Show the (n_s, r, n_T, c_T², ε)
  values explicitly.
• Mention saved file paths (plots, search results) when produced.

TOOL DISCIPLINE (READ TWICE)
• If you say "let's analyze X", "I will compute Y", "let me check Z",
  the very next thing you emit MUST be the corresponding tool call.
  Never trail off with a "stay tuned" / "let's see" without immediately
  invoking the tool.
• Don't intersperse a tool call with conversational lead-ins. Make the
  tool call FIRST, then narrate the results.
• If you have to call multiple tools (e.g. analyze + plot + relic_gw),
  emit them in sequence in a single response, not split across turns.
• When a tool's JSON output contains numerical fields, copy the actual
  numbers into your reply — don't paraphrase ("about 0.96") and don't
  refer to the tool output without quoting it.

ANTI-LOOP RULES (CRITICAL — OVERRIDES ALL OTHER INSTRUCTIONS)
1. If you catch yourself saying "let me try", "I should", "I will
   analyze", "let's check", or any planning phrase MORE THAN ONCE
   without an intervening tool call, STOP IMMEDIATELY and call exactly
   one tool right now. No more prose until a tool has been invoked.
2. NEVER estimate n_s, r, observables, or any numerical quantity from
   memory or from "similar models". Always derive them from tool
   outputs. If you cannot call a tool, explicitly state what is blocking
   you (missing argument, tool error, API limit) instead of guessing.
3. If you are unable to call any tool (API down, all tools errored),
   say so explicitly: "I am unable to proceed because …". Do NOT
   produce speculative numerical answers as a substitute.
"""


def build_agent_team(
    provider: Optional[str] = None,
    *,
    enable_arxiv_mcp: bool = True,
    enable_local_rag: bool = True,
    enable_cosmorag: bool = True,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    new_session: bool = False,
) -> "Agent":
    """Build the DeepEGB Agent with all production tools attached.

    (`build_agent_team` is the historical name; we kept it for backwards-
    compatibility — the agent is now a single agent with all tools,
    not an orchestrator + sub-agent team.)
    """
    if Agent is None:
        raise RuntimeError("Agno is not installed. `pip install agno`.")

    model = get_model(provider)
    debug = bool(int(os.environ.get("DEEPEGB_DEBUG", "0")))

    if not session_id:
        session_id = str(uuid.uuid4())

    tools: list = [
        search_egb_potentials,
        analyze_egb_model_tool,
        plot_egb_model_tool,
        relic_gw_spectrum_tool,
        diagnose_egb_model_tool,
        normalize_egb_model_tool,
    ]
    if enable_local_rag:
        tools.append(retrieve_literature_tool)
        try:
            from ..rag.index import DEFAULT_MODEL, _load_embedder
            _load_embedder(DEFAULT_MODEL)
            if debug:
                print("[DeepEGB] RAG embedder warm-up done.")
        except Exception as _warmup_exc:
            if debug:
                print(f"[DeepEGB] RAG embedder warm-up skipped: {_warmup_exc}")

    if enable_cosmorag:
        if is_cosmorag_available():
            tools.extend(all_cosmorag_tools())
            print("[DeepEGB] CosmoRAG tools connected (direct Python import).")
        elif has_cosmorag_mcp_configured():
            # Fall back to MCP subprocess when CosmoRAG is in a separate venv.
            try:
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                cosmorag_mcp = loop.run_until_complete(build_cosmorag_mcp_tools())
                tools.append(cosmorag_mcp)
                print("[DeepEGB] CosmoRAG MCP tools connected.")
            except Exception as _cr_exc:
                print(f"[DeepEGB] CosmoRAG not connected: {_cr_exc}")
        else:
            if debug:
                print("[DeepEGB] CosmoRAG not available (install cosmorag or set cosmorag-mcp).")

    if _MemoryTools is not None and _db is not None:
        tools.append(_MemoryTools(db=_db))

    if enable_arxiv_mcp and has_arxiv_mcp_configured():
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            mcp_tools = loop.run_until_complete(build_arxiv_mcp_tools())
            tools.append(mcp_tools)
            print("[DeepEGB] arXiv MCP connected.")
        except Exception as exc:
            print(f"[DeepEGB] arXiv MCP not connected: {exc}")

    session_state: dict[str, Any] = {
        "workflow.current_model": "",
        "workflow.normalized_model": "",
        "workflow.last_plot_path": "",
        "preferences.N_pivot": 55,
        "preferences.T_reh_GeV": 1e15,
        "notes": "",
    }

    memory_manager = None
    if _MemoryManager is not None and _db is not None:
        try:
            memory_manager = _MemoryManager(model=model, db=_db)
        except Exception:
            memory_manager = None

    session_summary_manager = None
    if _SessionSummaryManager is not None:
        try:
            session_summary_manager = _SessionSummaryManager(
                model=model, last_n_runs=6, conversation_limit=120,
            )
        except Exception:
            session_summary_manager = None

    agent_kwargs: dict[str, Any] = dict(
        name="DeepEGB",
        model=model,
        instructions=MAIN_AGENT_INSTRUCTIONS,
        tools=tools,
        debug_mode=debug,
        session_id=session_id,
        user_id=user_id,
        add_history_to_context=True,
        num_history_runs=8,
        read_chat_history=True,
        read_tool_call_history=True,
        session_state=session_state,
        add_session_state_to_context=True,
        enable_agentic_state=True,
        retries=2,
        delay_between_retries=1,
        tool_call_limit=8,
    )
    if _db is not None:
        agent_kwargs["db"] = _db
    if memory_manager is not None:
        agent_kwargs["memory_manager"] = memory_manager
        agent_kwargs["update_memory_on_run"] = True
        agent_kwargs["add_memories_to_context"] = True
        agent_kwargs["enable_agentic_memory"] = True
    if session_summary_manager is not None:
        agent_kwargs["session_summary_manager"] = session_summary_manager
        agent_kwargs["enable_session_summaries"] = True
        agent_kwargs["add_session_summary_to_context"] = True

    def _try_build(kwargs: dict[str, Any]) -> "Agent":
        try:
            return Agent(**kwargs, markdown=True)
        except TypeError:
            try:
                return Agent(**kwargs)
            except TypeError:
                strip = {"markdown", "retries", "delay_between_retries",
                         "tool_call_limit"}
                return Agent(**{k: v for k, v in kwargs.items()
                                if k not in strip})

    try:
        return _try_build(agent_kwargs)
    except TypeError:
        reduced = {k: v for k, v in agent_kwargs.items() if k in (
            "name", "model", "instructions", "tools", "debug_mode",
            "session_id", "user_id", "markdown",
        )}
        return _try_build(reduced)


# Convenience alias for code that imports the old name.
build_agent = build_agent_team


def _run_coro_sync(coro):
    """Run a coroutine from the synchronous CLI entrypoint."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _chunk_text(chunk) -> tuple[str | None, str]:
    content = getattr(chunk, "content", None)
    reasoning = getattr(chunk, "reasoning_content", None)
    event = getattr(chunk, "event", "") or ""
    if content:
        return content, event
    if reasoning:
        return reasoning, event or "reasoning"
    delta = getattr(chunk, "delta", None)
    if isinstance(delta, str) and delta:
        return delta, event
    return None, event


_LOOP_GUARD_SUSPICIOUS = 0
_LOOP_GUARD_CORRECTED = False
_LOOP_GUARD_PHRASES = re.compile(
    r"(?:let\s+me\s+try|i\s+should|i\s+will\s+analyz|let's\s+check|"
    r"i\s+will\s+comput|going\s+to\s+analyz|i'll\s+look|i\s+am\s+going\s+to)",
    re.IGNORECASE,
)


def _reset_loop_guard() -> None:
    global _LOOP_GUARD_SUSPICIOUS, _LOOP_GUARD_CORRECTED
    _LOOP_GUARD_SUSPICIOUS = 0
    _LOOP_GUARD_CORRECTED = False


def _classify_turn(text: str, had_tool_event: bool) -> bool:
    """Return True if the turn is 'suspicious' (planning phrases, no tools)."""
    if had_tool_event:
        return False
    return bool(_LOOP_GUARD_PHRASES.search(text))


def _agent_say(agent, msg: str, *, plain: bool = False) -> None:
    """Send a message to the agent and stream the response to stdout.

    Two modes:
      * default — Agno's `print_response(stream=True)`, which uses Rich's
        live panel.  Pretty, but the panel can grow taller than the
        terminal, making mid-stream scrollback impossible.
      * plain   — write each streamed chunk straight to stdout. Scroll-
        friendly, redirectable to a file with `tee` / `>`.

    Includes a lightweight loop guard: after each streamed response the
    turn is classified as "suspicious" if it contains planning phrases
    but no tool events. After 3 consecutive suspicious turns a
    corrective follow-up prompt is sent automatically (max once per
    user turn).  The guard resets at the start of each user message.

    Tolerates Agno API drift across versions.
    """
    global _LOOP_GUARD_SUSPICIOUS, _LOOP_GUARD_CORRECTED

    def _post_stream_check(had_tool: bool, text: str) -> None:
        """Classify the completed turn and maybe auto-correct."""
        global _LOOP_GUARD_SUSPICIOUS, _LOOP_GUARD_CORRECTED
        if _classify_turn(text, had_tool):
            _LOOP_GUARD_SUSPICIOUS += 1
        else:
            _LOOP_GUARD_SUSPICIOUS = 0
        if _LOOP_GUARD_SUSPICIOUS >= 3 and not _LOOP_GUARD_CORRECTED:
            _LOOP_GUARD_CORRECTED = True
            print("\n[DeepEGB LOOP GUARD] Agent is in a planning loop. "
                  "Sending corrective prompt.\n")
            _agent_say(
                agent,
                "You are in a planning loop. Call exactly one relevant "
                "tool now and return its numeric output.",
                plain=True,
            )

    if not plain and hasattr(agent, "aprint_response"):
        try:
            _run_coro_sync(agent.aprint_response(msg, stream=True))
            _post_stream_check(False, "")
            return
        except TypeError:
            try:
                _run_coro_sync(agent.aprint_response(msg))
                _post_stream_check(False, "")
                return
            except Exception:        # noqa: BLE001
                pass

    if not plain and hasattr(agent, "print_response"):
        try:
            agent.print_response(msg, stream=True)
            _post_stream_check(False, "")
            return
        except TypeError:
            try:
                agent.print_response(msg)
                _post_stream_check(False, "")
                return
            except Exception:        # noqa: BLE001
                pass     # fall through to plain mode

    # Plain streaming: iterate chunks and print as they arrive.
    if hasattr(agent, "arun"):
        async def _async_plain_run():
            try:
                stream = agent.arun(msg, stream=True)
            except TypeError:
                stream = await agent.arun(msg)
            if hasattr(stream, "__aiter__"):
                last_was_thought = False
                turn_text_parts: list[str] = []
                had_tool = False
                async for chunk in stream:
                    content, event = _chunk_text(chunk)
                    if event and event.lower().startswith(("tool", "reasoning",
                                                           "thinking")):
                        if event.lower().startswith("tool"):
                            had_tool = True
                        if not last_was_thought:
                            print(f"\n[{event}] ", end="", flush=True)
                        last_was_thought = True
                        if content:
                            print(content, end="", flush=True)
                        continue
                    last_was_thought = False
                    if content is None:
                        continue
                    turn_text_parts.append(content)
                    print(content, end="", flush=True)
                print()
                _post_stream_check(had_tool, " ".join(turn_text_parts))
                return
            resp = getattr(stream, "content", str(stream))
            print(resp)
            _post_stream_check(False, resp)

        try:
            _run_coro_sync(_async_plain_run())
            return
        except Exception:        # noqa: BLE001
            pass

    if hasattr(agent, "run"):
        try:
            stream = agent.run(msg, stream=True)
        except TypeError:
            stream = agent.run(msg)
        if hasattr(stream, "__iter__") and not isinstance(stream, str):
            last_was_thought = False
            turn_text_parts: list[str] = []
            had_tool = False
            for chunk in stream:
                content, event = _chunk_text(chunk)
                if event and event.lower().startswith(("tool", "reasoning",
                                                       "thinking")):
                    if event.lower().startswith("tool"):
                        had_tool = True
                    if not last_was_thought:
                        print(f"\n[{event}] ", end="", flush=True)
                    last_was_thought = True
                    if content:
                        print(content, end="", flush=True)
                    continue
                last_was_thought = False
                if content is None:
                    continue
                turn_text_parts.append(content)
                print(content, end="", flush=True)
            print()
            _post_stream_check(had_tool, " ".join(turn_text_parts))
            return
        # Single result object
        resp = getattr(stream, "content", str(stream))
        print(resp)
        _post_stream_check(False, resp)
        return
    # Last resort
    resp = str(agent(msg))
    print(resp)
    _post_stream_check(False, resp)


def _collect_tools(
    *,
    enable_local_rag: bool = True,
    enable_cosmorag: bool = True,
    enable_arxiv_mcp: bool = False,
) -> list:
    """Build a flat list of callable tools for the native Claude API backend."""
    tools: list = [
        search_egb_potentials,
        analyze_egb_model_tool,
        plot_egb_model_tool,
        relic_gw_spectrum_tool,
        diagnose_egb_model_tool,
        normalize_egb_model_tool,
    ]
    if enable_local_rag:
        tools.append(retrieve_literature_tool)
    if enable_cosmorag and is_cosmorag_available():
        tools.extend(all_cosmorag_tools())
    return tools


def run_chat(
    initial_message: Optional[str] = None,
    provider: Optional[str] = None,
    *,
    enable_arxiv_mcp: bool = True,
    enable_local_rag: bool = True,
    enable_cosmorag: bool = True,
    use_native_claude: Optional[bool] = None,
    plain: bool = False,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    new_session: bool = False,
) -> None:
    """Interactive REPL with the DeepEGB agent.  /exit or Ctrl-D to quit.

    When ``use_native_claude`` is True (or ``DEEPEGB_USE_NATIVE_CLAUDE=1``
    with provider=anthropic), bypasses Agno and drives the Anthropic API
    directly with full prompt-caching and streaming support.
    """
    # Resolve whether to use native Claude API
    if use_native_claude is None:
        use_native_claude = bool(int(os.environ.get("DEEPEGB_USE_NATIVE_CLAUDE", "0")))

    eff_provider = (provider or os.environ.get("DEEPEGB_PROVIDER", "local")).lower()
    if use_native_claude and eff_provider != "anthropic":
        import warnings
        warnings.warn(
            "use_native_claude=True requires provider='anthropic'; "
            f"got '{eff_provider}'. Ignoring native-claude flag.",
            stacklevel=2,
        )
        use_native_claude = False

    if use_native_claude:
        from .claude_api import run_chat_native  # noqa: PLC0415
        tools = _collect_tools(
            enable_local_rag=enable_local_rag,
            enable_cosmorag=enable_cosmorag,
            enable_arxiv_mcp=False,  # arXiv MCP is Agno-only for now
        )
        run_chat_native(
            initial_message=initial_message,
            provider=provider,
            tools=tools,
            system=MAIN_AGENT_INSTRUCTIONS,
        )
        return

    # --- Agno path ---
    if new_session or not session_id:
        session_id = session_id or str(uuid.uuid4())
        if new_session:
            session_id = str(uuid.uuid4())

    persistence = "ON" if _db is not None else "OFF"
    print(f"[DeepEGB] session_id={session_id}  user_id={user_id or '-'}  "
          f"persistence={persistence}")
    if _MemoryManager is not None or _SessionSummaryManager is not None:
        if _db is not None:
            print("[DeepEGB] Note: post-turn memory/session updates may take "
                  "extra time on local models.")

    agent = build_agent_team(
        provider=provider,
        enable_arxiv_mcp=enable_arxiv_mcp,
        enable_local_rag=enable_local_rag,
        enable_cosmorag=enable_cosmorag,
        session_id=session_id,
        user_id=user_id,
        new_session=new_session,
    )
    if initial_message:
        _reset_loop_guard()
        _agent_say(agent, initial_message, plain=plain)
        if not __import__("sys").stdin.isatty():
            return     # one-shot mode under -m

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
        _reset_loop_guard()
        _agent_say(agent, msg, plain=plain)
