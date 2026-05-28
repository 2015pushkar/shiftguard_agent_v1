"""Tool catalog: maps a tool name to its description, argument schema, and the
deterministic function behind it.

The agent picks tools purely from these descriptions (no hard-coded routing).
`call_tool` validates the LLM-supplied args against the tool's Pydantic schema,
then executes it, returning either the tool's result dict or a structured error
dict so the agent loop can retry or recover instead of crashing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, ValidationError

from ..rag.retriever import search_policy
from .compute import compute_hours, estimate_payroll_impact
from .tickets import create_review_ticket
from .timecards import get_timecards


class _Args(BaseModel):
    # forbid extras so a mis-named argument surfaces as a recoverable
    # invalid_arguments error instead of being silently dropped (which produced
    # a plausible-but-wrong $0 payroll estimate).
    model_config = ConfigDict(extra="forbid")


class GetTimecardsArgs(_Args):
    employee: str | None = None
    start: str | None = None
    end: str | None = None


class SearchPolicyArgs(_Args):
    query: str
    top_k: int | None = None


class ComputeHoursArgs(_Args):
    shifts: list[dict]
    overtime_threshold: float = 40.0
    double_time_threshold: float = 60.0


class EstimatePayrollImpactArgs(_Args):
    hourly_rate: float
    regular_hours: float = 0.0
    overtime_hours: float = 0.0
    double_time_hours: float = 0.0
    overtime_multiplier: float = 1.5
    double_time_multiplier: float = 2.0


class CreateReviewTicketArgs(_Args):
    employee: str
    issue: str
    recommended_action: str | None = None
    payroll_impact: float | None = None
    citations: list[str] | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[..., dict]

    def run(self, args: dict) -> dict:
        validated = self.args_model(**(args or {}))
        return self.fn(**validated.model_dump())


TOOLS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in [
        ToolSpec(
            name="get_timecards",
            description=(
                "Load timecard records (shifts with clock-in/out punches) for employees "
                "from the local data file. Use this to fetch an employee's actual punches "
                "before any hours calculation. Filter by `employee` (name or id). The "
                "`start`/`end` date bounds (YYYY-MM-DD) are OPTIONAL: OMIT them to get the "
                "employee's full current pay period, and set them ONLY when the user names "
                "specific dates. Never guess or invent a date range."
            ),
            args_model=GetTimecardsArgs,
            fn=get_timecards,
        ),
        ToolSpec(
            name="search_policy",
            description=(
                "Retrieve relevant payroll policy passages from the company policy knowledge "
                "base via semantic search. Use whenever you need a policy rule, threshold, or "
                "definition (e.g. overtime threshold, time rounding, missed-punch handling, "
                "overtime authorization). Returns the top matching policy chunks with citations."
            ),
            args_model=SearchPolicyArgs,
            fn=search_policy,
        ),
        ToolSpec(
            name="compute_hours",
            description=(
                "Compute worked hours from timecard `shifts`: applies quarter-hour rounding, "
                "deducts unpaid breaks, splits totals into regular/overtime/double-time tiers, "
                "and flags missed clock-outs and rounding. Use this for ALL hours math — never "
                "calculate hours yourself."
            ),
            args_model=ComputeHoursArgs,
            fn=compute_hours,
        ),
        ToolSpec(
            name="estimate_payroll_impact",
            description=(
                "Compute payroll dollar amounts. Required args: `hourly_rate`, plus the hour "
                "tiers `regular_hours`, `overtime_hours`, and `double_time_hours` taken directly "
                "from the `compute_hours` result — do NOT pass a single combined total. Returns "
                "regular/overtime/double-time pay, gross pay, and the overtime premium (extra cost "
                "above straight time). Use this for ALL money math — never calculate dollars yourself."
            ),
            args_model=EstimatePayrollImpactArgs,
            fn=estimate_payroll_impact,
        ),
        ToolSpec(
            name="create_review_ticket",
            description=(
                "Open a manager review ticket for a flagged payroll issue — the only external "
                "action. Use after you have identified and quantified an issue needing manager "
                "attention. Records the `employee`, `issue`, optional `recommended_action`, "
                "`payroll_impact` (a single dollar amount as a NUMBER, e.g. the overtime premium "
                "2.81 — not an object), and `citations` (a list of citation strings from search_policy)."
            ),
            args_model=CreateReviewTicketArgs,
            fn=create_review_ticket,
        ),
    ]
}


def tool_catalog() -> str:
    """Render the tool list as a prompt-ready catalog (name: description)."""
    return "\n".join(f"- {s.name}: {s.description}" for s in TOOLS.values())


def call_tool(name: str, args: dict | None = None) -> dict:
    """Validate args and execute a tool, returning its result or a structured error."""
    spec = TOOLS.get(name)
    if spec is None:
        return {"error": "unknown_tool", "tool": name, "available_tools": list(TOOLS)}
    try:
        return spec.run(args or {})
    except ValidationError as e:
        detail = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
        return {"error": "invalid_arguments", "tool": name, "detail": detail}
    except Exception as e:  # tool blew up unexpectedly — surface, don't crash the loop
        return {"error": "tool_error", "tool": name, "detail": str(e)}
