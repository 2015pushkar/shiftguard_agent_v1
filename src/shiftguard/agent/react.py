"""The ReAct loop: structured reasoning with autonomous tool routing.

One loop, one prompt, every query category. Each step the model emits a
thought + an action or a final answer; tool results are fed back as compact
OBSERVATION lines. Guardrails: a max-step budget, a repeated-action detector, a
payroll-verification check (a ticket may only carry a dollar `payroll_impact`
that `estimate_payroll_impact` actually produced this run — never one the model
made up), and tool errors returned as observations so the agent can recover
instead of crashing. The logged thought/action/observation trace is the demo
deliverable.
"""

from __future__ import annotations

import json

from ..config import Settings, get_settings
from ..logging_setup import get_logger
from ..tools.registry import TOOLS, call_tool
from .llm import LLMClient, LLMStepError
from .prompts import build_system_prompt
from .schemas import AgentResult, TraceStep, agent_step_format

_MAX_OBS_CHARS = 2000
_PAYROLL_FIELDS = ("regular_pay", "overtime_pay", "double_time_pay", "gross_pay", "overtime_premium")


def _compact(observation: dict) -> str:
    text = json.dumps(observation, ensure_ascii=False)
    if len(text) > _MAX_OBS_CHARS:
        text = text[:_MAX_OBS_CHARS] + " …(truncated)"
    return text


def _estimate_figures(observation: dict) -> set[float]:
    """The dollar values a successful estimate_payroll_impact returned (rounded)."""
    return {
        round(float(observation[f]), 2)
        for f in _PAYROLL_FIELDS
        if isinstance(observation.get(f), (int, float))
    }


def _claims_dollar(args: dict) -> bool:
    """True if a ticket asserts a non-zero payroll_impact that must be tool-verified."""
    impact = args.get("payroll_impact")
    return isinstance(impact, (int, float)) and round(float(impact), 2) != 0.0


def _impact_verified(args: dict, figures: set[float] | None) -> bool:
    """True only if payroll_impact matches a figure estimate_payroll_impact produced."""
    return bool(figures) and round(float(args["payroll_impact"]), 2) in figures


def run_agent(
    query: str,
    *,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    logger=None,
) -> AgentResult:
    settings = settings or get_settings()
    llm = llm or LLMClient(settings)
    log = logger or get_logger("agent")
    fmt = agent_step_format(list(TOOLS))

    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": f"USER QUERY: {query}"},
    ]
    steps: list[TraceStep] = []
    action_counts: dict[tuple[str, str], int] = {}
    payroll_figures: set[float] | None = None  # dollar values verified by estimate_payroll_impact

    log.info("QUERY: %s", query)
    for n in range(1, settings.max_steps + 1):
        try:
            step, raw = llm.step(messages, fmt)
        except LLMStepError as e:
            log.error("agent aborted: %s", e)
            return AgentResult(answer=None, stop_reason="llm_error", steps=steps)

        messages.append({"role": "assistant", "content": raw})
        log.info("STEP %d THOUGHT: %s", n, step.thought)

        if step.final_answer and step.final_answer.strip():
            log.info("FINAL ANSWER: %s", step.final_answer)
            steps.append(TraceStep(step=n, thought=step.thought, final_answer=step.final_answer))
            return AgentResult(answer=step.final_answer, stop_reason="final_answer", steps=steps)

        action = step.action
        key = (action.tool, json.dumps(action.args, sort_keys=True))
        action_counts[key] = action_counts.get(key, 0) + 1
        log.info("STEP %d ACTION: %s args=%s", n, action.tool, json.dumps(action.args, ensure_ascii=False))

        if action_counts[key] >= 3:
            log.warning("repeated-action limit hit for %s; breaking loop", action.tool)
            steps.append(
                TraceStep(step=n, thought=step.thought, tool=action.tool, args=action.args,
                          observation={"error": "repeated_action"})
            )
            return AgentResult(answer=None, stop_reason="repeated_action", steps=steps)

        if action_counts[key] == 2:
            observation = {
                "error": "repeated_action",
                "detail": (
                    "You already ran this exact tool call. Use the earlier OBSERVATION, "
                    "try different arguments, or give a final_answer."
                ),
            }
        elif (
            action.tool == "create_review_ticket"
            and _claims_dollar(action.args)
            and not _impact_verified(action.args, payroll_figures)
        ):
            observation = {
                "error": "payroll_not_verified",
                "detail": (
                    "payroll_impact must be a figure produced by estimate_payroll_impact in "
                    "this run, never computed yourself. Call estimate_payroll_impact, then "
                    "copy one of its exact returned values (e.g. overtime_premium or gross_pay) "
                    "into payroll_impact, and create the ticket again."
                ),
            }
        else:
            observation = call_tool(action.tool, action.args)
            if action.tool == "estimate_payroll_impact" and "error" not in observation:
                payroll_figures = _estimate_figures(observation)

        obs_text = _compact(observation)
        log.info("STEP %d OBSERVATION: %s", n, obs_text)
        steps.append(TraceStep(step=n, thought=step.thought, tool=action.tool, args=action.args, observation=observation))
        messages.append({"role": "user", "content": f"OBSERVATION: {obs_text}"})

    log.warning("max steps (%d) reached without a final answer", settings.max_steps)
    return AgentResult(answer=None, stop_reason="max_steps", steps=steps)
