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
    U(["Plain-English request"]) --> A["ReAct loop · agent/react.py<br/>think then act<br/><i>the LLM only plans and routes</i>"]
    A -->|"emits JSON: action"| R{{"picks ONE of 5 tools<br/>by its description"}}
    A -->|"emits JSON: final_answer"| OUT(["Answer + citations"])

    R --> T1["search_policy<br/>(RAG)"]
    R --> T2["get_timecards"]
    R --> T3["compute_hours"]
    R --> T4["estimate_payroll_impact"]
    R --> T5["create_review_ticket"]

    T1 & T2 & T3 & T4 & T5 -->|observation fed back| A

    T1 -.-> QD[("embedded Qdrant<br/>+ nomic-embed-text")]
    T2 -.-> JS[("timecards.json")]
    T5 -.-> TK[("tickets.jsonl")]

    classDef tool fill:#eef6ff,stroke:#5b8def;
    classDef store fill:#f4f4f4,stroke:#999,stroke-dasharray:3 3;
    class T1,T2,T3,T4,T5 tool;
    class QD,JS,TK store;
```

Everything runs locally under **Ollama** (the LLM and the embeddings) with **embedded Qdrant** as the vector store. Raised tickets are appended to `data/tickets.jsonl`, which acts as the local manager-review queue; there are no external notifications, by design, to keep the system fully local.

**On each run, before any answer reaches you:** every model turn is generated against a JSON schema, then validated with Pydantic (bad JSON is re-prompted, with bounded retries); the loop enforces a max-step budget and a repeated-action detector; tool exceptions are caught and returned as observations. The full thought, action, and observation trace is written to a `logs/*.log` file, so every audit is reviewable afterward. The eval harness (below) is a separate offline test, not part of the live request path.

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
| **Runner** | Ollama | One-binary local install; mature native structured outputs and tool-calling. |
| **LLM** | `qwen2.5:7b-instruct` (dev: `:3b`) | Best small-model instruction-following and tool-calling. Reliability compounds across a multi-step loop, so correctness beats raw speed. Swappable via `OLLAMA_MODEL` with no code change. |
| **Why not 14B** | n/a | On CPU the limit is memory bandwidth, not capacity; a 14B roughly halves tok/s and makes the loop impractical. |
| **Vector store** | Qdrant **embedded** | Real Qdrant, zero ops, no Docker. |
| **Embeddings** | `nomic-embed-text` via Ollama | Whole stack on one runner (no extra wheels); uses nomic's asymmetric query/document prefixes for better retrieval. |
| **Chunking** | Heading-aware, one chunk per rule | Each chunk is a self-contained rule with `doc`/`section` metadata for citations, never blind fixed-size cuts. |
| **Agent** | Hand-rolled ReAct + structured JSON | A ~150-line loop is the autonomy logic, in the open. A framework would hide it and bloat context for a small CPU model. |
| **Context window** | Pinned `num_ctx=8192` | Ollama defaults to 2048 and silently truncates on overflow, corrupting a multi-step loop. |

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
