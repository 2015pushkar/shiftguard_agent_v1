"""The ReAct loop: structured reasoning with autonomous tool routing.

One loop, one prompt, every query category. Each step the model emits a
thought + an action or a final answer; tool results are fed back as compact
OBSERVATION lines. Guardrails: a max-step budget, a repeated-action detector,
and tool errors returned as observations so the agent can recover instead of
crashing. The logged thought/action/observation trace is the demo deliverable.
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


def _compact(observation: dict) -> str:
    text = json.dumps(observation, ensure_ascii=False)
    if len(text) > _MAX_OBS_CHARS:
        text = text[:_MAX_OBS_CHARS] + " …(truncated)"
    return text


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
        else:
            observation = call_tool(action.tool, action.args)

        obs_text = _compact(observation)
        log.info("STEP %d OBSERVATION: %s", n, obs_text)
        steps.append(TraceStep(step=n, thought=step.thought, tool=action.tool, args=action.args, observation=observation))
        messages.append({"role": "user", "content": f"OBSERVATION: {obs_text}"})

    log.warning("max steps (%d) reached without a final answer", settings.max_steps)
    return AgentResult(answer=None, stop_reason="max_steps", steps=steps)
