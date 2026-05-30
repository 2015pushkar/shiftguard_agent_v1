# ShiftGuard — Architecture Plan

## Context

ShiftGuard is a fully-local "action agent" that audits hourly timecards before
payroll. It must (a) retrieve payroll policy via RAG, (b) run deterministic
Python tools for all math/actions, and (c) autonomously decide — via a ReAct
loop with **no hard-coded triggers** — when to retrieve, when to act, and how to
synthesize an answer.

This plan is optimized for the **four grading criteria** (architecture choice,
agent autonomy, prompt robustness, engineering quality), **not** feature count.
Target hardware is an **AMD Ryzen AI 7 350 (8-core Zen 5) APU, 24GB RAM,
Windows, CPU-only** — the Radeon 860M iGPU and XDNA NPU are unusable by Ollama
today (see *Hardware reality check*), so all inference runs on the CPU. This
constraint drives every performance trade-off below. The demo is an accepted
**detailed log file**; a frontend is out of scope.

---

## Hardware reality check (CPU-only — verified vs. assumed)

**Honest caveat:** these figures were not measured on the target box. Model
*sizes* and Ollama *behaviors* are verified from docs; *speed* numbers are
calculated estimates from chip specs — confirm with `ollama run <model>
--verbose` once installed.

- **Machine:** Ryzen AI 7 350 (8c/16t Zen 5), 24GB RAM, LPDDR5x-8000 / DDR5-5600
  dual-channel (~90–128 GB/s bandwidth), Windows.
- **Inference path = CPU only, and that's the only working path:** ROCm doesn't
  support the Radeon 860M (gfx1152) on Windows; Ollama's *experimental* Vulkan
  backend currently fails to detect the 860M (open issue #14562); the XDNA NPU
  can't be used by Ollama at all (AMD's separate Lemonade/GAIA stack can, but
  it's immature and has no tool/ReAct tooling). This isn't a failure to
  optimize — there is no working accelerator path for this iGPU/NPU today.
- **Memory fit — non-issue (fits, ~10GB headroom):** Ollama 7B Q4 process
  ~6–7GB (4.7GB weights + ~1GB KV @ 8k ctx + overhead) · `nomic-embed-text` via
  Ollama ~0.3GB · Qdrant embedded <0.1GB · Windows baseline ~5–6GB →
  **peak ≈ 12–14GB / 24GB.** Both models load in the one Ollama runtime, and
  embedding only runs hard at *index-build* time, not in the loop.
- **Speed — MEASURED on this box:** 7B Q4 = **~10 tok/s** generation (two smoke
  tests: ~9.75 and a re-run at **10.41 tok/s** eval rate / 17.03 tok/s prompt /
  ~9.4s one-time load — confirms the 8–12 estimate). With Ollama
  prefix-caching across loop steps, **~15–30s/step → ~2–4 min for the
  5-step headline query**, and **~20–40 min for a full eval run**. 3B is ~2–3×
  faster (chosen for dev iteration).
- **Practical risks:** pin `num_ctx` (§1); one-time online setup (pre-pull models
  for offline grading); a 28W APU will thermally throttle over long eval runs
  (batch the evals, stay on AC).

---

## 1. Local LLM model choice

**Runner: Ollama.** Named in the brief, single-binary local install, and — as of
early 2026 — has mature **native tool-calling** *and* **JSON-schema-constrained
structured outputs** (the `format` field). Rejected: vLLM (needs GPU), raw
llama.cpp (lower-level, more glue), LM Studio (GUI-first).

**Model (default): `qwen2.5:7b-instruct` (Q4_K_M, ~4.7GB).** Best-in-class
tool-calling reliability in the small-model class and strong instruction
following — the two things that matter most here.

**Why reliability over speed:** per-call reliability *compounds* across a ReAct
loop (~95%/call → ~66% over 8 steps). Since the demo is a log file, not a
latency-bound UI, correctness wins. CPU generation is slow (~tens of seconds per
step), so the loop is designed for a **short step horizon** (§3).

**Two-model posture (chosen):** default `qwen2.5:7b-instruct` for the recorded
demo and the final eval run (reliability is graded), and `qwen2.5:3b-instruct`
for fast day-to-day iteration. Both are real Ollama tags; the model is a single
config value (`OLLAMA_MODEL`) so it swaps with no code change — confirm the exact
tag with `ollama list` before relying on it. (`qwen3:4b` is a viable alternative
but verify the tag and run it in **non-thinking mode**, since our loop already
supplies an explicit `thought` field and extended thinking only bloats context
and CPU time.)

**Why not 14B, despite 24GB fitting it:** on a CPU the binding constraint is
memory *bandwidth*, not RAM *capacity*. A 14B would roughly halve tokens/sec and
make the 5–6 step loop impractical. 7B is the reliability/speed sweet spot here.

**Context window must be pinned (correctness, not just perf):** the loop runs
with an explicit `num_ctx` (~8192). Ollama defaults to 2048 and **silently
truncates oldest tokens on overflow**, which can corrupt a multi-step loop. Set
it via `OLLAMA_CONTEXT_LENGTH` / `options.num_ctx` and budget the prompt to stay
under it.

---

## 2. Vector store + embedding + chunking

**Vector store: Qdrant in embedded/local mode** via `qdrant-client`
(`QdrantClient(path="./data/qdrant")`). **No Docker required** — pushback on
over-scope; a path-backed embedded collection is zero-ops and still "Qdrant" as
the brief suggests. Docker stays an optional note in the README.

**Embedding model: `nomic-embed-text` (768-dim) via Ollama.** Switched from the
original fastembed/`bge-small` plan after checking the box: Python is **3.14**
(no `onnxruntime`/`fastembed` wheels yet) and `nomic-embed-text` is already
pulled. Embedding through Ollama keeps the whole stack on **one runner**, drops
the `onnxruntime` dependency, and removes the wheel-availability risk. We call
Ollama's embeddings endpoint, upsert vectors into Qdrant, and query by embedding
the user's text the same way. (`bge-small`/fastembed stays a viable swap if a
supported Python/onnxruntime combo is preferred later.)

**Chunking: structure-aware, one chunk per policy rule/subsection.** Policy docs
are authored as Markdown with headings; we split on headings so each chunk is a
*self-contained rule* (e.g., "Overtime threshold"), capped ~512 tokens with a
sentence-boundary fallback for long sections, no blind fixed-size splitting.
Each chunk carries metadata (`doc`, `section`) for **citations** in answers.
This directly targets the "logical data chunking" criterion and keeps retrieved
context coherent for a small model.

---

## 3. ReAct loop design + autonomous routing

**Core idea: ReAct with JSON-structured actions.** Each step, the LLM returns a
schema-constrained object — either an action or a final answer:

```json
{"thought": "short reasoning", "action": {"tool": "search_policy", "args": {"query": "overtime threshold"}}}
{"thought": "I have everything I need", "final_answer": "..."}
```

This keeps the CoT/ReAct structure the brief recommends **and** the robustness of
schema-validated JSON (vs. brittle free-text "Thought:/Action:" parsing on a
small CPU model).

**Routing with no hard-coded triggers:** RAG is exposed as *just another tool*
(`search_policy`). The model chooses among all tools — including retrieval —
based purely on the tool descriptions in the system prompt. There is **no**
`if "overtime" in query` logic anywhere. The same loop + prompt handles every
query category; only the query changes. This is the heart of the "agent
autonomy" criterion.

**Reliability contingency:** the custom `thought`+`action` JSON is primary
(transparent CoT, schema-guaranteed). If the 7B's tool *selection* proves shaky
on the long chain, switch to Ollama's **native tool-calling API** (`tools=[...]`)
— Qwen2.5 was fine-tuned on that exact format and may select tools more
reliably; the cost is losing the explicit `thought` field. This is a swap
isolated to `llm.py`, not a redesign.

**Loop (max ~6 steps):**
1. Build messages: system prompt (role, rules, tool catalog, output contract,
   few-shot) + conversation.
2. Call LLM with structured-output schema → parse to a Pydantic `AgentStep`.
3. `final_answer` → validate and return.
4. `action` → validate args against the tool's schema → execute deterministic
   Python tool → capture a compact `Observation` (or a structured error).
5. Append thought/action/observation; loop.

**Guardrails:** max-step budget, repeated-action detector (breaks loops),
and tool errors fed back as observations so the model can **recover** (the
"handle a failed tool call" criterion).

**Context management (brief hint):** inject only top-k=3 retrieved chunks,
truncated; keep tool outputs as compact JSON; trim the scratchpad if it grows.
Run with the pinned `num_ctx` (~8192) from §1 and budget the prompt (system +
tool catalog + few-shot + chunks + scratchpad) to stay well under it, since
overflow silently drops tokens.

---

## 4. Minimal deterministic tool set (5)

All math and all side effects live here — the LLM never computes.

| Tool | Role | Notes |
|------|------|-------|
| `get_timecards(employee?, range?)` | Load timecard records from local JSON | Data access; returns structured error for unknown employee |
| `search_policy(query)` | **RAG retrieval** over policy KB | Returns top-k chunks + citations; routing is the model's choice |
| `compute_hours(punches)` | Regular/OT hours, missed clock-out & rounding flags | **All punch math** |
| `estimate_payroll_impact(hours, rate, policy_params)` | $ impact of an issue | **All money math** |
| `create_review_ticket(...)` | The external **action** | Appends to `data/tickets.jsonl` |

Single-responsibility tools are easier for a small model to call correctly and
better engineering than one mega-tool. **Pushback on over-scope:** no separate
tool per policy nuance (breaks/holiday/double-time), no Jira/Slack/email
integration (ticket = local JSONL), no database (sample data = JSON files).

---

## 5. Prompt-robustness measures

- **System prompt:** explicit role; hard rules (never compute — use tools for
  all math; answer only from retrieved policy or tool output; cite policy
  section; refuse out-of-scope politely); the JSON output contract; 1–2 few-shot
  ReAct steps.
- **JSON enforcement:** Ollama structured outputs (JSON schema in `format`) +
  Pydantic validation of every step. **Constrain `action.args` to an object** in
  the schema — a smoke test showed a loose schema lets the model emit a bare
  string (`"args": "overtime threshold"`) instead of `{"query": "..."}`.
- **Retry on bad output:** on invalid JSON or failed arg-schema validation,
  re-prompt with the validation error appended (bounded, ~2 retries).
- **Failed tool calls:** tool exceptions are caught and returned as a structured
  error observation; the agent can retry or escalate instead of crashing.
- **Anti-hallucination:** "if policy isn't found in retrieval, say so — never
  invent policy numbers"; cite chunk metadata; never fabricate timecard data.
- **Out-of-scope refusal:** prompt instructs a polite `final_answer` decline for
  non-payroll/timecard queries; proven in evals (§6).
- **Loop safety:** max steps + repeated-action detection.

---

## 6. Evals — proving autonomous routing

A small harness (`evals/run_evals.py`, mirrored in `tests/test_routing.py`) runs
the **full agent** over labeled scenarios and asserts on the **tool trajectory**
(which tools fired) and answer properties — not exact wording. Categories:

- **RAG-only:** "What's our overtime threshold?" → `search_policy` fires; compute
  tools do **not**.
- **Tool-only:** "How many hours did Maria work Tuesday?" → `get_timecards` +
  `compute_hours`.
- **Multi-step RAG+tool (headline demo):** "Audit Maria's week for overtime risk
  and open a ticket if needed." → `get_timecards` → `search_policy` →
  `compute_hours` → `estimate_payroll_impact` → `create_review_ticket`.
- **Out-of-scope:** "What's the weather?" → refusal, **zero** tools.
- **Failure recovery:** unknown employee → tool error → graceful explanation, no
  crash.

**Known risk (the 5-step chain):** the headline trajectory is 5 tool calls,
right at the edge of a 7B's multi-step coherence. Mitigations: a few-shot that
demonstrates this exact trajectory, compact observations, retries, and the
native-tool-calling contingency (§3). If it still won't hold reliably,
`create_review_ticket` is the step to make optional — a 4-step audit
(`get_timecards → search_policy → compute_hours → final answer with
recommendation`) still demonstrates RAG + tool-use + multi-step. The eval
measures this directly, so the decision is data-driven, not guessed.

**Why this proves autonomy:** the loop and prompt are identical across
categories; only the query differs, yet the tool trajectory changes correctly —
demonstrating routing without hard-coded triggers. Deterministic tool math is
unit-tested separately (exact assertions). The headline scenario's full
thought/action/observation trace **is** the "detailed log file" deliverable.

---

## 7. File / folder layout

```
shiftguard_agent_v1/
├── README.md                 # stack justification (deliverable)
├── PLAN.md                   # this document
├── pyproject.toml            # deps + console entrypoint
├── .env.example              # OLLAMA_MODEL, OLLAMA_CONTEXT_LENGTH (8192), paths, top_k, max_steps
├── data/
│   ├── policies/             # synthetic payroll policy markdown (RAG corpus)
│   │   ├── overtime.md
│   │   ├── timekeeping.md
│   │   └── rounding.md
│   └── timecards.json        # sample timecard data
├── src/shiftguard/
│   ├── config.py             # pydantic-settings
│   ├── llm.py                # Ollama wrapper: structured call + retry
│   ├── rag/
│   │   ├── chunking.py       # heading-aware chunker
│   │   ├── index.py          # build/load Qdrant + Ollama embeddings
│   │   └── retriever.py      # search_policy backend (top-k + citations)
│   ├── tools/
│   │   ├── registry.py       # name -> (schema, fn)
│   │   ├── timecards.py
│   │   ├── compute.py
│   │   └── tickets.py
│   ├── agent/
│   │   ├── react.py          # the ReAct loop
│   │   ├── prompts.py        # system prompt + few-shot
│   │   └── schemas.py        # AgentStep, Action, ToolResult
│   ├── logging_setup.py      # structured trace -> console + file
│   └── cli.py                # `shiftguard "query"`
├── evals/
│   ├── scenarios.jsonl
│   ├── run_evals.py          # runs agent, writes report + per-scenario logs
│   └── report/               # generated logs (demo deliverable)
└── tests/
    ├── test_tools.py         # exact-math unit tests
    ├── test_chunking.py
    └── test_routing.py       # integration: routing per category
```

**Dependencies (deliberately thin):** `ollama`, `qdrant-client` (no `fastembed`
extra — we embed via Ollama), `pydantic`, `pydantic-settings`, on a **Python
3.13** venv (3.14 lacks wheels for some deps). **No agent framework.** Pushback: LangChain /
LlamaIndex / CrewAI would hide the autonomy logic the brief is grading, add
context bloat (bad for a small CPU model), and obscure "engineering quality." A
hand-rolled ~150-line ReAct loop is more transparent and demonstrates real
understanding. (LangChain noted in README as the considered-and-rejected
alternative.)

---

## Over-scope items explicitly cut

- Frontend / web UI (log-file demo accepted)
- Docker requirement (Qdrant embedded mode)
- Real ticketing/notification integrations (local JSONL)
- Database (JSON sample data)
- Agent framework (hand-rolled loop)
- Multi-agent orchestration / async streaming
- Per-policy-nuance tool sprawl (5 tools total)

---

## Verification (after implementation)

1. **Setup:** create a **Python 3.13** venv (`py -3.13 -m venv .venv`);
   `pip install -e .`; ensure `qwen2.5:7b-instruct` + `nomic-embed-text` are
   pulled (both already present) plus `qwen2.5:3b-instruct` for dev; then
   `python -m shiftguard.rag.index` to build the Qdrant collection. **Pre-pull
   all models if the grader runs offline.**
2. **Sanity-check speed first:** `ollama run qwen2.5:7b-instruct --verbose "hi"`
   to read *real* tokens/sec on this box and confirm the estimates above before
   committing the 7B as the demo default.
3. **Unit tests:** `pytest tests/test_tools.py tests/test_chunking.py` — exact
   math and chunk-boundary assertions (no LLM).
4. **Routing/integration:** `pytest tests/test_routing.py` — asserts the tool
   trajectory per query category (proves autonomous routing + out-of-scope
   refusal + failure recovery).
5. **End-to-end demo:** `shiftguard "Audit Maria's week for overtime risk and
   open a ticket if needed"` → inspect `evals/report/` for the full
   thought/action/observation trace and the appended `data/tickets.jsonl`.
6. **Eval report:** `python evals/run_evals.py` → per-scenario logs + a
   pass/fail summary across all five categories.

---

## Build order

1. Scaffolding: `pyproject.toml`, `config.py`, `logging_setup.py`, sample
   `data/` (policies + timecards).
2. RAG: `chunking.py` → `index.py` → `retriever.py` (+ chunking unit test).
3. Tools: `compute.py`, `timecards.py`, `tickets.py`, `registry.py` (+ exact
   math unit tests).
4. Agent: `schemas.py` → `prompts.py` → `llm.py` → `react.py` → `cli.py`.
5. Evals: `scenarios.jsonl` → `run_evals.py` → `test_routing.py`.
6. `README.md` stack justification + update `CLAUDE.md` run commands.
