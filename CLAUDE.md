# ShiftGuard

## What this is
ShiftGuard is a fully-local AI "action agent" being built. It audits hourly
employee timecards before payroll: retrieves payroll policy via RAG, runs deterministic 
Python tools to detect issues (missed clock-outs, overtime risk, rounding), estimates payroll impact, and
creates manager review tickets. Everything runs locally — no cloud APIs.

## Hard constraints (the assessment is graded on these)
- Fully local: LLM (Ollama), vector store (Qdrant), and orchestration all run
  on this machine. No external LLM/API calls.
- The LLM only plans, routes, and explains. ALL math/calculations are done by
  deterministic Python tools, never by the model.
- Agentic reasoning: a ReAct loop decides when to retrieve (RAG) vs. when to
  call a tool. No hard-coded trigger sequences.
- Graded on: architecture choice, agent autonomy, prompt robustness, clean
  modular code with real error handling.

## Architecture (LOCKED — full design + rationale in PLAN.md)
- **Runner:** Ollama. **Default model `qwen2.5:7b-instruct`** (Q4, demo/eval —
  reliability is graded); `qwen2.5:3b-instruct` for dev iteration. Swappable via
  `OLLAMA_MODEL`, no code change. **Pin `num_ctx`≈8192** (Ollama defaults to 2048
  and silently truncates → corrupts the loop).
- **Performance:** CPU-only is the only working path (iGPU/NPU unusable — see
  PLAN.md; do NOT try to "fix" them). Measured ~9 tok/s on the 7B, so keep
  per-step output compact to hold the 5–6 step loop in the ~2–4 min range.
- **RAG:** Qdrant **embedded** (`QdrantClient(path=...)`, no Docker) +
  `nomic-embed-text` embeddings via Ollama. Heading-aware chunking, one chunk per
  policy rule, with `doc`/`section` metadata for citations.
- **Agent:** hand-rolled ReAct loop (no LangChain/framework). Each step the LLM
  returns a schema-constrained JSON `AgentStep` — either
  `{thought, action:{tool, args}}` or `{thought, final_answer}`. **RAG is just
  another tool (`search_policy`)** — routing is the model's choice, zero
  hard-coded triggers. Guardrails: max-step budget, repeated-action detector,
  tool errors fed back as observations for recovery; ~2 retries on bad JSON.
- **5 deterministic tools** (all math/side-effects live here, never the model):
  `get_timecards`, `search_policy`, `compute_hours`,
  `estimate_payroll_impact`, `create_review_ticket` (appends `data/tickets.jsonl`).
- **Layout / build order:** see PLAN.md §7 and §Build order. Code under
  `src/shiftguard/` (`config`, `llm`, `rag/`, `tools/`, `agent/`, `cli`),
  data under `data/`, evals under `evals/`, tests under `tests/`.

## Status
**Implemented and verified.** All 6 build-order steps are complete. 28 unit tests
pass; the eval harness passes **5/5** (RAG-only, tool-only, multi-step headline,
out-of-scope, failure-recovery) via both `evals/run_evals.py` and the
`tests/test_routing.py` pytest mirror. Stack: `ollama`, `qdrant-client`,
`pydantic`, `pydantic-settings` (+ `pytest` dev) on a **Python 3.13** venv (3.14
lacks some wheels). See README.md for stack justification; PLAN.md for full design.

## Run commands
```
py -3.13 -m venv .venv && .venv\Scripts\activate      # Python 3.13, not 3.14
pip install -e ".[dev]"
ollama pull qwen2.5:7b-instruct && ollama pull nomic-embed-text
python -m shiftguard.rag.index                        # build the Qdrant policy index
pytest tests/test_tools.py tests/test_chunking.py     # fast unit tests (no LLM)
shiftguard "Audit Maria's week for overtime risk and open a ticket if needed."
python evals/run_evals.py                             # full eval -> evals/report/summary.md
pytest tests/test_routing.py                          # integration mirror (slow, live LLM)
```

## How I want you to work
- Don't assume — if a requirement is ambiguous, ask or surface the options.
- Prefer the simplest thing that satisfies the requirement. No speculative
  abstractions.
- End every task with a short plain-language summary: what changed, what's
  next, any risks. Under 6 lines.
