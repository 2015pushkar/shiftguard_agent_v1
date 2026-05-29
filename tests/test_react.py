"""Unit tests for the ReAct loop's payroll-verification guardrail.

Fully offline: a scripted fake LLM drives the loop and `call_tool` is stubbed,
so no Ollama call and no ticket file is written. The guardrail must block a
ticket whose dollar `payroll_impact` was not produced by `estimate_payroll_impact`
this run, then let the corrected ticket through (the recover-via-observation path).
"""

from __future__ import annotations

from shiftguard.agent import react
from shiftguard.agent.schemas import Action, AgentStep

_ESTIMATE_RESULT = {
    "regular_pay": 900.0,
    "overtime_pay": 8.44,
    "double_time_pay": 0.0,
    "gross_pay": 908.44,
    "overtime_premium": 2.81,
}


class _FakeLLM:
    """Returns pre-scripted steps; ignores the prompt entirely."""

    def __init__(self, steps: list[AgentStep]):
        self._steps = list(steps)

    def step(self, messages, fmt):
        s = self._steps.pop(0)
        return s, s.model_dump_json()


def _stub_call_tool(calls):
    def _call(name, args):
        calls.append((name, dict(args)))
        if name == "estimate_payroll_impact":
            return dict(_ESTIMATE_RESULT)
        if name == "create_review_ticket":
            return {"status": "created", "ticket_id": "TKT-test"}
        return {}

    return _call


# --- pure helpers -----------------------------------------------------------

def test_estimate_figures_collects_dollar_values():
    assert react._estimate_figures(_ESTIMATE_RESULT) == {900.0, 8.44, 0.0, 908.44, 2.81}


def test_claims_dollar_only_for_nonzero_numbers():
    assert react._claims_dollar({"payroll_impact": 2.81}) is True
    assert react._claims_dollar({"payroll_impact": 0.0}) is False
    assert react._claims_dollar({"payroll_impact": None}) is False
    assert react._claims_dollar({}) is False


def test_impact_verified_requires_a_match():
    figs = {900.0, 2.81}
    assert react._impact_verified({"payroll_impact": 2.81}, figs) is True
    assert react._impact_verified({"payroll_impact": 14.79375}, figs) is False
    assert react._impact_verified({"payroll_impact": 2.81}, None) is False  # nothing computed yet


# --- loop behaviour ---------------------------------------------------------

def test_unverified_dollar_blocked_then_recovers(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(react, "call_tool", _stub_call_tool(calls))

    bad_args = {"employee": "Maria", "issue": "OT", "payroll_impact": 14.79375}
    good_args = {"employee": "Maria", "issue": "OT", "payroll_impact": 2.81}
    scripted = [
        AgentStep(thought="ticket it", action=Action(tool="create_review_ticket", args=bad_args)),
        AgentStep(
            thought="compute first",
            action=Action(
                tool="estimate_payroll_impact",
                args={"hourly_rate": 22.5, "regular_hours": 40.0, "overtime_hours": 0.25},
            ),
        ),
        AgentStep(thought="ticket with verified figure", action=Action(tool="create_review_ticket", args=good_args)),
        AgentStep(thought="done", final_answer="Ticket TKT-test opened."),
    ]

    result = react.run_agent("audit", llm=_FakeLLM(scripted))

    # The fabricated-dollar ticket never reached the tool.
    assert ("create_review_ticket", bad_args) not in calls
    # It was blocked with the recoverable error observation, exactly once.
    blocks = [s for s in result.steps if s.observation and s.observation.get("error") == "payroll_not_verified"]
    assert len(blocks) == 1
    # After estimate_payroll_impact ran, the corrected ticket executed.
    assert ("estimate_payroll_impact", {"hourly_rate": 22.5, "regular_hours": 40.0, "overtime_hours": 0.25}) in calls
    assert ("create_review_ticket", good_args) in calls
    assert result.stop_reason == "final_answer"


def test_zero_impact_ticket_not_blocked(monkeypatch):
    """A genuine $0 / no-impact ticket needs no prior estimate call."""
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(react, "call_tool", _stub_call_tool(calls))

    args = {"employee": "Sam", "issue": "missed clock-out only", "payroll_impact": 0.0}
    scripted = [
        AgentStep(thought="no money impact", action=Action(tool="create_review_ticket", args=args)),
        AgentStep(thought="done", final_answer="Ticket opened."),
    ]

    result = react.run_agent("audit", llm=_FakeLLM(scripted))

    assert ("create_review_ticket", args) in calls
    assert result.stop_reason == "final_answer"
