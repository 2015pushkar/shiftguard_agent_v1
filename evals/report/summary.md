# ShiftGuard eval report — 2026-05-30 18:24:40

**5/5 scenarios passed.**

## rag_only — RAG-only: PASS
- query: What is our overtime threshold?
- tools used: ['search_policy']
- stop_reason: final_answer
- answer: Our overtime threshold is 40 hours worked in a single workweek. Any hours worked beyond this are considered overtime and are paid at 1.5 times the regular hourly rate.

## tool_only — Tool-only: PASS
- query: How many hours did Maria work on Tuesday, 2026-05-19?
- tools used: ['get_timecards', 'compute_hours']
- stop_reason: final_answer
- answer: Maria did not have any recorded shifts on May 19, 2026.

## multi_step — Multi-step RAG+tool: PASS
- query: Audit Maria's week for overtime risk and open a manager review ticket if needed.
- tools used: ['get_timecards', 'compute_hours', 'search_policy', 'search_policy', 'estimate_payroll_impact', 'create_review_ticket']
- stop_reason: final_answer
- answer: A manager review ticket has been opened for the issue of overtime worked but not pre-authorized. The ticket ID is TKT-5af28bc5 and it includes a recommendation to review and authorize the overtime if appropriate.

## out_of_scope — Out-of-scope: PASS
- query: What's the weather today?
- tools used: []
- stop_reason: final_answer
- answer: I cannot provide weather information. Please consult a weather service for today’s conditions.

## failure_recovery — Failure recovery: PASS
- query: How many hours did Bob Smith work last week?
- tools used: ['get_timecards']
- stop_reason: final_answer
- answer: There is no employee named 'Bob Smith' in the system. The available employees are Maria Sanchez and James Okafor.
