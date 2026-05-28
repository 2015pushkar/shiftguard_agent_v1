"""Schemas for the ReAct agent's structured step output and run trace.

Each LLM turn must produce an `AgentStep`: a `thought` plus EITHER an `action`
(call a tool) OR a `final_answer`. The JSON schema sent to Ollama's `format`
field is intentionally hand-built and ref-free (`agent_step_format`) and pins
`action.args` to an object — a loose schema lets a small model emit a bare
string for args.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: str
    args: dict = Field(default_factory=dict)


class AgentStep(BaseModel):
    """One ReAct step: a thought plus exactly one of `action` or `final_answer`."""

    model_config = ConfigDict(extra="ignore")
    thought: str = ""
    action: Action | None = None
    final_answer: str | None = None

    @model_validator(mode="after")
    def _exactly_one_outcome(self) -> "AgentStep":
        has_action = self.action is not None
        has_final = bool(self.final_answer and self.final_answer.strip())
        if has_action == has_final:
            raise ValueError(
                "each step needs exactly one of: an 'action', or a non-empty 'final_answer'"
            )
        return self


def agent_step_format(tool_names: list[str]) -> dict:
    """JSON schema for Ollama's structured-output `format` field."""
    return {
        "type": "object",
        "properties": {
            "thought": {"type": "string"},
            "action": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "enum": tool_names},
                    "args": {"type": "object"},
                },
                "required": ["tool", "args"],
            },
            "final_answer": {"type": "string"},
        },
        "required": ["thought"],
    }


@dataclass
class TraceStep:
    step: int
    thought: str
    tool: str | None = None
    args: dict | None = None
    observation: dict | None = None
    final_answer: str | None = None


@dataclass
class AgentResult:
    answer: str | None
    stop_reason: str  # final_answer | max_steps | repeated_action | llm_error
    steps: list[TraceStep] = field(default_factory=list)

    @property
    def tool_trajectory(self) -> list[str]:
        return [s.tool for s in self.steps if s.tool]
