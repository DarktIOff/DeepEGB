# Research patterns

A field-guide for using DeepEGB as a real research assistant rather than
a chatbot. Each pattern shows the canonical conversation, which tools the
agent should reach for, and what output to expect.

These patterns are baked into the main-agent system prompt
(`src/deepegb/agent/runtime.py`). If you're seeing the agent miss them,
either your local LLM has poor tool-calling reliability — try a bigger
GLM model — or the conversation has drifted; remind the agent of the
relevant pattern.

---

## Pattern 1 — "What does this specific model predict?"

**You ask:** *"What are the predictions of Starobinsky inflation at N=55,
and how does adding a dilatonic GB coupling ξ = α exp(−λφ) modify them?"*

**Agent should:**

1. `retrieve_literature_tool("Starobinsky inflation predictions n_s r")`
   to ground the GR baseline in your local copy of the literature.
2. `analyze_egb_model_tool("(1-exp(-sqrt(2/3)*phi))**2", "0", N=55)`
   for the GR baseline numbers.
3. `analyze_egb_model_tool("(1-exp(-sqrt(2/3)*phi))**2",
   "0.05*exp(-0.4*phi)", N=55)` for the EGB-modified numbers.
4. Compare in a sentence; quote `c_T²`, the consistency relation
   `r/(−8 n_T)`, and how each shifted.

**Look for in the answer:** explicit numerical (n_s, r, c_T², r/(−8 n_T))
for both GR and EGB cases, with a citation tag like `[starobinsky.md §
Observables]` from RAG.

---

## Pattern 2 — "Find me a model that fits ACT DR6 + a LISA-detectable signal"

**You ask:** *"Discover an EGB inflation model with n_s ≈ 0.974 (ACT DR6),
small r, and Ω_GW h² ≥ 10⁻¹³ at 1 mHz."*

**Agent should:**

1. Decide that the standard slow-roll loss isn't enough — Ω_GW target
   means `--loss production_gw`. Mentally translate to:
   - target_ns = 0.974, sigma_ns = 0.003
   - target_r = 0.0, sigma_r = 0.018
   - omega_gw_targets = [(1e-3, 1e-13, 5e-14)]
2. Delegate to the SR sub-agent. Sub-agent runs `search_egb_potentials`.
3. When the JSON comes back, pick the top 1–2 candidates with χ² < ~50.
4. `analyze_egb_model_tool` each finalist to confirm n_s, r are inside
   1σ of targets.
5. `relic_gw_spectrum_tool` each finalist to confirm the LISA prediction.
6. `plot_egb_model_tool` the winner.
7. Final answer cites: V, ξ, predicted (n_s, r, Ω_GW @ 1 mHz), how each
   compares to the relevant detector floor (LISA, TaiJi, BBO, ...), and
   any prior literature on similar models retrieved via RAG / arXiv MCP.

**Look for:** the agent NEVER quotes Ω_GW from inflation knowledge alone
— every Ω_GW number should be a `relic_gw_spectrum_tool` output.

---

## Pattern 3 — "Is this model already in the literature?"

**You ask:** *"My SR run produced V = exp(−0.42/φ). Has anyone published
this potential before?"*

**Agent should:**

1. `retrieve_literature_tool("V = exp(-1/phi) inverse exponential plateau
   inflation")`. If hits, cite them.
2. If no local hits, arXiv MCP: `search_papers` with the same query. Pull
   abstracts, scan for the form.
3. If still nothing, say so plainly; do **not** pretend to recognize a
   classic model. Mention the closest known family (e.g. "this is
   asymptotic to the brane / pole-inflation V ~ 1 − μ^k / φ^k family").

**Look for:** real arXiv IDs and file names. No invented "Smith et al.
2017" citations.

---

## Pattern 4 — "What relic-GW signal does this leave at every detector?"

**You ask:** *"Compute Ω_GW(f) h² for V = ½(0.05)φ⁴ × ξ = 0.1/(φ²+1) and
list which detectors could see it."*

**Agent should:**

1. `relic_gw_spectrum_tool` over a wide band (10 decades, T_reh = 1e15 GeV).
2. Read the resulting (f, Ω_GW h²) array. Compare to detector floors
   listed in the system prompt or via the catalogue:
   - PTA bands (NANOGrav 15yr, EPTA, IPTA, SKA)
   - Space (LISA, TaiJi, TianQin, LISA+TaiJi network, DECIGO, BBO)
   - Ground (LIGO O4, ET, CE)
   - CMB (BICEP/Keck, LiteBIRD, CMB-S4)
3. State, per detector: "above floor" / "below by X decades" /
   "out of band". Order by likelihood of detection.
4. Optionally `plot_egb_model_tool` for the diagnostic AND mention that
   `deepegb relic-gw` from the CLI gives the multi-detector overlay.

**Look for:** a per-detector summary, not just "yes LISA might see it".

---

## Pattern 5 — Calibrating the SR loss to a thesis target

**You ask:** *"I want to scan the EGB landscape for any model that
satisfies (i) Planck 2018 (n_s, r), (ii) ACT DR6 (n_s shift), and (iii)
gives Ω_GW h² > 10⁻¹⁴ across the LISA band."*

**Agent should:**

1. Realize this combines pointwise + band targets → compose:
   ```
   omega_gw_targets   = []         # nothing pointwise
   omega_gw_band_min  = (1e-4, 1e-1, 1e-14)
   target_ns          = 0.974, sigma_ns = 0.003
   target_r           = 0.0,   sigma_r = 0.018
   ```
2. Delegate to SR sub-agent with these settings AND
   `loss_kind = "production_gw"`. Warn the user this is ~1 s/eval and
   suggest niters=8, populations=12 first.
3. Iterate: tighten/loosen sigmas based on what the sub-agent reports.

---

## Pattern 6 — Sanity-checking the engine itself

**You ask:** *"Verify that the kernel reproduces Starobinsky's
GR predictions at N=55 to within 1%."*

**Agent should:**

1. Direct call to `analyze_egb_model_tool` with the Starobinsky V and ξ=0.
2. Compare result to the textbook closed-form: n_s = 1 − 2/N = 0.96364;
   r ≈ 12/N² = 0.00397; n_T ≈ −2ε; consistency r = −8 n_T.
3. State percent-deviation per quantity. If any deviates by > 1% the
   kernel may have a bug; flag it.

This pattern doubles as a smoke test you can run before trusting any
research-grade output.

---

## The GR-limit ban

**`ξ(φ) = 0` is not an EGB inflation model.** The action

$$
S = \int d^4x \sqrt{-g}\left[\tfrac{R}{2} - \tfrac{1}{2}(\partial\phi)^2
                              - V(\phi) - \tfrac{1}{2}\xi(\phi)\,\mathcal G\right]
$$

reduces to plain General Relativity in the limit $\xi \to 0$ — every EGB
signature ($c_T^2 \neq 1$, broken consistency relation, modified $r$,
relic-GW spectral features) vanishes with it. DeepEGB enforces this at
three layers:

1. **In the χ² loss.** `chi2_full_breakdown` adds an `egb_penalty`
   component $\sim 10^3 e^{-|\delta_1|/\tau}$ with $\tau =$ `egb_min_delta1`
   (default $10^{-4}$). Models with $|\delta_1(\phi_N)| < \tau$ pay a
   penalty that dominates any small-χ² fit.

2. **In the SR pipeline.** `run_joint_search` no longer includes `"0"` as
   a candidate ξ in its second pass when `enforce_egb=True` (the default).
   The user has to opt in via `--allow-gr` or `enforce_egb=False`.

3. **In the diagnose tool.** `diagnose_egb_model_tool` reports
   `"is_gr_limit": true` whenever $|\delta_1| < 10^{-8}$ at the pivot.
   The agent's prompt explicitly tells it never to present such models
   as "discovered EGB candidates".

To opt out — e.g. for a controlled comparison run that should include
the GR baseline as a reference point — pass:

```bash
deepegb search --ns 0.974 --r 0.0 --N 55 --allow-gr ...
```

or in the agent:

```
search_egb_potentials(target_ns=..., target_r=..., enforce_egb=False)
```

## Anti-patterns (call out and reject)

* **"I think the answer is..."** — production tools exist; if a question
  is answerable by a tool, the tool should be called. Don't accept an
  opinion-style answer for a numerical question.
* **"Smith & Jones (2024) showed..."** — every paper citation should
  carry a real arXiv ID or local-file pointer. If the agent makes up
  citations, your local LLM is hallucinating; switch to a bigger model.
* **"The relic GW signal would be detectable at LISA"** without a number
  — always demand Ω_GW h² @ 1 mHz with the detector floor next to it.
* **Mixing M_pl conventions** — DeepEGB uses M_pl = 1 throughout. If you
  cite a paper that uses M_pl explicitly, restate the formula in our
  convention before claiming numerical agreement.

---

## Practical knobs while you work

In any chat session you can constrain the agent's tool palette:

```bash
deepegb chat                    # all tools
deepegb chat --no-arxiv         # local only (offline)
deepegb chat --no-rag           # skip the local index
deepegb chat --no-arxiv --no-rag   # tools-only mode, no literature grounding
```

Forcing a higher-capability model when your local GLM-4.6 hallucinates:

```bash
ZAI_MODEL=glm-5.1 deepegb chat
deepegb chat --provider anthropic     # Claude as a stronger orchestrator
```
