# ShiftGuard

ShiftGuard is a **fully-local AI action agent** that audits hourly employee timecards before payroll. Given a plain-English request, it looks up the relevant payroll policy via RAG, runs deterministic Python tools to detect issues (overtime, rounding, missed clock-outs), estimates the dollar impact, and opens a manager review ticket. It decides on its own when to look something up, when to calculate, and when to act.

No cloud APIs: the LLM, the vector store, and the orchestration all run on one machine. The LLM only **plans, routes, and explains**. Every calculation is done by tested Python, never by the model.

---

## What it demonstrates

- **Autonomous routing, no hard-coded triggers.** RAG is just another tool (`search_policy`); the model picks among all five tools from their descriptions alone. Same loop and prompt for every query, and only the tool path changes.
- **Tools own all math and side-effects.** Hours, rounding, overtime tiers, dollar amounts, and ticket creation live in deterministic, unit-tested functions.
- **Schema-constrained ReAct.** Each turn the model emits one JSON object: a `thought` plus either an `action` or a `final_answer`, validated with Pydantic and bounded by retries.
- **Real error handling.** A failed or mis-called tool returns a structured error that is fed back as an observation, so the agent recovers instead of crashing.

---

## High-level architecture

A request enters a loop. Each turn the model thinks, then either calls **one** tool or returns the final answer. The tool's result comes back as an observation and the loop repeats until the model is done.

```mermaid
flowchart TD
    U(["Plain-English request"]) --> A["ReAct loop · agent/react.py<br/>think, then act<br/><i>the LLM only plans and routes</i>"]
    A -->|"emits JSON: action"| R{{"picks ONE of 5 tools<br/>by its description"}}
    A -->|"emits JSON: final_answer"| OUT(["Answer + citation"])

    R --> T1["1 · search_policy (RAG)"]
    R --> T2["2 · get_timecards"]
    R --> T3["3 · compute_hours"]
    R --> T4["4 · estimate_payroll_impact"]
    R --> T5["5 · create_review_ticket"]

    T1 & T2 & T3 & T4 & T5 -->|observation fed back| A

    T1 -. reads .-> QD[("embedded Qdrant<br/>+ nomic-embed-text")]
    T2 -. reads .-> JS[/"timecards.json (INPUT)"/]
    T5 -. writes .-> TK[/"tickets.jsonl (OUTPUT, review queue)"/]

    classDef tool fill:#eef6ff,stroke:#5b8def;
    classDef store fill:#f4f4f4,stroke:#999,stroke-dasharray:3 3;
    classDef io fill:#fff7e6,stroke:#d98f00;
    classDef answer fill:#e6f4ea,stroke:#2f9e44,stroke-width:2px;
    class T1,T2,T3,T4,T5 tool;
    class QD store;
    class JS,TK io;
    class OUT answer;
```

Everything runs locally under **Ollama** (LLM + embeddings) with **embedded Qdrant**. Tickets append to `data/tickets.jsonl`, the local manager-review queue (no external notifications, by design).

**Each run, before you see an answer:** every model turn is schema-constrained and Pydantic-validated (bad JSON is retried); the loop caps steps and blocks repeated actions; tool errors come back as observations rather than crashes. The full trace is written to `logs/*.log`. The eval harness (below) is a separate offline test, not the live path.

---

## Two questions, two routes (the agent decides each one)

Both questions hit the **same loop, same prompt, same five tools**. Nothing about the path is scripted; the model reads the tool descriptions and chooses. The route differs only because the task does.

### Route A: a policy question (document lookup only)

> *"What is our overtime threshold?"*

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant A as ReAct agent
    participant P as search_policy (RAG)

    U->>A: What is our overtime threshold?
    Note over A: thought - policy question, just look it up
    A->>P: search_policy(overtime threshold)
    P-->>A: top 3 policy chunks
    Note over A: thought - I have the rule
    A-->>U: 40 h per workweek (cites Overtime Threshold)
```

**Answer:** Our overtime threshold is 40 hours per workweek (Monday 12:00 AM through Sunday 11:59 PM). Cited from *Overtime Policy > Overtime Threshold*.

**Path:** `search_policy -> final answer` &nbsp;(1 of 5 tools; no timecards, no math, no ticket)

### Route B: a full audit (document lookup *and* tool execution)

> *"Audit Maria's week for overtime risk and open a manager review ticket if needed."*
> This is the case that needs both a policy lookup and tool execution.

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant A as ReAct agent
    participant TC as get_timecards
    participant CH as compute_hours
    participant P as search_policy (RAG)
    participant EP as estimate_payroll_impact
    participant TK as create_review_ticket

    U->>A: Audit Maria's week, open a ticket if needed
    A->>TC: get_timecards(Maria)
    TC-->>A: raw punches, 5 shifts
    A->>CH: compute_hours(shifts)
    CH-->>A: 40.25 h total, 0.25 h overtime + flags
    A->>P: search_policy(overtime auth, missed clock-out)
    P-->>A: Overtime Authorization, Missed Clock-Out
    A->>EP: estimate_payroll_impact(rate, hours)
    EP-->>A: 2.81 USD overtime premium
    A->>TK: create_review_ticket(...)
    TK-->>A: TKT-29531b26 created
    A-->>U: Ticket TKT-29531b26 opened, premium 2.81 USD
```

**Answer:** A manager review ticket (TKT-29531b26) was opened for overtime worked without authorization and a missed clock-out punch. Overtime premium: $2.81.

**Path:** `get_timecards -> compute_hours -> search_policy -> estimate_payroll_impact -> create_review_ticket -> final answer` &nbsp;(5 of 5 tools; RAG lookup *and* tool execution)

Note the discipline in Route B: every number ($2.81, 40.25 h) comes from a Python tool, never the model; every citation is one `search_policy` actually returned; and the ticket is reported as created only because the observation confirmed it.

---

## Stack and why each piece was chosen

| Choice | What | Why this one |
|---|---|---|
| **Runner** | Ollama | One local binary serves both the LLM and the embeddings, and constrains output to a JSON schema natively, which the ReAct loop relies on. |
| **LLM** | `qwen2.5:7b-instruct` (dev: `:3b`) | Strong instruction-following and native function-calling at the 7B size (tracked on the Berkeley Function-Calling Leaderboard). In a multi-step loop one bad step derails the rest, so per-step reliability matters more than tokens/sec. Swappable via `OLLAMA_MODEL`. |
| **Why not 14B** | n/a | CPU token generation re-reads every weight per token, so it is bound by memory bandwidth: a 14B moves ~2x the bytes and runs at ~half the speed. Too slow for a 5-6 step loop. |
| **Vector store** | Qdrant **embedded** | The same Qdrant engine as the server, run in-process from a file path: real ANN search, zero ops, no Docker. |
| **Embeddings** | `nomic-embed-text` via Ollama | Runs on the same Ollama runner (no extra ML dependencies). It is task-prefixed: queries are embedded as `search_query:` and policy chunks as `search_document:`, which is how it was trained and helps asymmetric RAG (short question vs long rule). |
| **Chunking** | One chunk per policy rule | Split on headings so each vector is a complete, citable rule carrying `doc`/`section` metadata. Fixed-size cuts would split a rule across chunks and break citations. |
| **Agent** | Hand-rolled ReAct | ~150 lines of explicit loop. The autonomy logic is what is being evaluated, so it stays visible rather than buried in a framework that also bloats context for a small model. |
| **Context window** | Pinned `num_ctx=8192` | Ollama defaults to 2048 and, on overflow, silently drops the oldest tokens (the system prompt and early observations) with no error, quietly corrupting the loop. 8192 holds the whole trajectory. |

---

## Prompt robustness and error handling

Each of these targets a failure actually observed while running the agent:

- **Structured output, validation, retry.** Ollama's `format` constrains output to the step schema; Pydantic enforces "exactly one of action / final_answer"; invalid JSON is re-prompted with the error.
- **Strict tool args.** A mis-named argument surfaces as a recoverable error instead of being silently dropped (which had once produced a plausible-but-wrong $0 estimate).
- **Anti-hallucination.** The model must `search_policy` before citing a rule, and may cite only the exact strings the tool returned.
- **Action honesty.** It claims success only when the observation confirms it (e.g. a ticket marked `created`); otherwise it retries or reports the failure.
- **Loop safety.** Max-step budget, a repeated-action detector, and tool exceptions caught and returned as structured errors.

---

## Run it

Needs **Python 3.13** and a running **Ollama**. No Docker, no cloud account.

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate                  # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"

ollama pull qwen2.5:7b-instruct         # pulled once, cached, works offline after
ollama pull nomic-embed-text

python -m shiftguard.rag.index          # build the embedded-Qdrant policy index

shiftguard "Audit Maria's week for overtime risk and open a ticket if needed."
shiftguard "What is our overtime threshold?"
```

Tests and evals: `pytest tests/test_tools.py tests/test_chunking.py` (fast, no LLM), `python evals/run_evals.py` (full agent, 5/5 passing), `pytest tests/test_routing.py` (live-LLM mirror; skips if Ollama is down). Config is via `.env` (see `.env.example`); every setting has a safe default.

---

## Project layout

```
src/shiftguard/
├── cli.py               # shiftguard "<query>"
├── config.py            # settings (env/.env, safe defaults)
├── logging_setup.py     # full trace -> console + run file
├── rag/                 # chunking.py, index.py, retriever.py   (RAG / search_policy)
├── tools/               # timecards.py, compute.py, tickets.py, registry.py
└── agent/               # schemas.py, prompts.py, llm.py, react.py   (the ReAct loop)
data/   policies/*.md (RAG corpus), timecards.json, tickets.jsonl (generated)
evals/  scenarios.jsonl, run_evals.py, report/
tests/  test_tools.py, test_chunking.py, test_routing.py
```

---

## Considered and rejected

- **LangChain / LlamaIndex / CrewAI:** would hide the autonomy logic and bloat context for a small CPU model.
- **Docker Qdrant:** embedded mode is real Qdrant with zero ops.
- **14B model:** CPU is bandwidth-bound; too slow for the loop.

## Known limitations

- **Speed:** CPU-only is about 9 tok/s on the 7B, so roughly 2 to 5 minutes for the multi-step audit (the only working inference path on the target hardware).
- **Small-model variance:** multi-step coherence on a 7B is at the edge. Structured output, strict arg schemas, retries, and the repeated-action and max-step guards are the mitigations, and the eval measures the trajectory directly.
</content>
