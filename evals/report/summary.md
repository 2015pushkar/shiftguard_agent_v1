# ShiftGuard eval report — 2026-05-29 13:12:01

**5/5 scenarios passed.**

## rag_only — RAG-only: PASS
- query: What is our overtime threshold?
- tools used: ['search_policy']
- stop_reason: final_answer
- answer: Our overtime threshold is 40 hours worked in a single workweek. Hours worked in excess of this are paid at 1.5 times the employee's regular hourly rate. (Citation: Overtime Policy > Overtime Threshold)

## tool_only — Tool-only: PASS
- query: How many hours did Maria work on Tuesday, 2026-05-19?
- tools used: ['get_timecards', 'compute_hours']
- stop_reason: final_answer
- answer: Maria worked 10.5 hours on Tuesday, 2026-05-19, without any overtime or double-time.

## multi_step — Multi-step RAG+tool: PASS
- query: Audit Maria's week for overtime risk and open a manager review ticket if needed.
- tools used: ['get_timecards', 'compute_hours', 'search_policy', 'create_review_ticket', 'estimate_payroll_impact', 'create_review_ticket']
- stop_reason: final_answer
- answer: A manager review ticket has been opened for Maria Sanchez regarding the overtime worked but not pre-authorized and a missed clock-out on 2026-05-22. The overtime premium is $2.81, as required by policy.

## out_of_scope — Out-of-scope: PASS
- query: What's the weather today?
- tools used: []
- stop_reason: final_answer
- answer: I'm sorry, but I can only assist with questions related to payroll, timecards, and company policies.

## failure_recovery — Failure recovery: PASS
- query: How many hours did Bob Smith work last week?
- tools used: ['get_timecards']
- stop_reason: final_answer
- answer: There is no record of an employee named Bob Smith for the week from 2023-10-01 to 2023-10-07.
