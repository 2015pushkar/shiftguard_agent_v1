# ShiftGuard eval report — 2026-05-28 15:50:10

**5/5 scenarios passed.**

## rag_only — RAG-only: PASS
- query: What is our overtime threshold?
- tools used: ['search_policy']
- stop_reason: final_answer
- answer: Our overtime threshold is 40 hours per workweek (Monday 12:00 AM through Sunday 11:59 PM) (<Overtime Policy> > Overtime Threshold).

## tool_only — Tool-only: PASS
- query: How many hours did Maria work on Tuesday, 2026-05-19?
- tools used: ['get_timecards', 'compute_hours']
- stop_reason: final_answer
- answer: Maria worked 10.5 hours on Tuesday, 2026-05-19.

## multi_step — Multi-step RAG+tool: PASS
- query: Audit Maria's week for overtime risk and open a manager review ticket if needed.
- tools used: ['get_timecards', 'compute_hours', 'search_policy', 'estimate_payroll_impact', 'create_review_ticket']
- stop_reason: final_answer
- answer: A manager review ticket has been opened (TKT-29531b26) to address the overtime worked without authorization and the missed clock-out punch. The overtime premium is $2.81.

## out_of_scope — Out-of-scope: PASS
- query: What's the weather today?
- tools used: []
- stop_reason: final_answer
- answer: I'm sorry, but I can only assist with payroll-related questions.

## failure_recovery — Failure recovery: PASS
- query: How many hours did Bob Smith work last week?
- tools used: ['get_timecards']
- stop_reason: final_answer
- answer: There is no employee named 'Bob Smith'. Available employees are Maria Sanchez and James Okafor.
