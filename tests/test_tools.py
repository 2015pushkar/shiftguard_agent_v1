"""Exact-math + behavior tests for the deterministic tools (no LLM, no Qdrant)."""

import json

import pytest

from shiftguard import config
from shiftguard.tools import tickets
from shiftguard.tools.compute import _round_punch, compute_hours, estimate_payroll_impact
from shiftguard.tools.registry import call_tool
from shiftguard.tools.timecards import get_timecards

# Maria's real week (also in data/timecards.json) — the headline audit case.
MARIA_SHIFTS = [
    {"date": "2026-05-18", "clock_in": "08:00", "clock_out": "18:30", "unpaid_break_min": 30},
    {"date": "2026-05-19", "clock_in": "07:30", "clock_out": "18:30", "unpaid_break_min": 30},
    {"date": "2026-05-20", "clock_in": "08:00", "clock_out": "18:30", "unpaid_break_min": 30},
    {"date": "2026-05-21", "clock_in": "08:06", "clock_out": "18:21", "unpaid_break_min": 30},
    {"date": "2026-05-22", "clock_in": "08:00", "clock_out": None, "unpaid_break_min": 30},
]


# --- 7-minute rounding rule ---

@pytest.mark.parametrize(
    "clock,expected_min,expected_changed",
    [
        ("08:00", 480, False),  # on a quarter mark
        ("08:15", 495, False),  # on a quarter mark
        ("08:06", 480, True),   # rem 6 -> down
        ("08:21", 495, True),   # rem 6 -> down
        ("08:07", 480, True),   # rem 7 -> still down
        ("08:08", 495, True),   # rem 8 -> up
        ("08:53", 540, True),   # rem 8 -> up, rolls to 09:00
    ],
)
def test_round_punch(clock, expected_min, expected_changed):
    assert _round_punch(clock) == (expected_min, expected_changed)


# --- compute_hours ---

def test_compute_hours_headline_case():
    r = compute_hours(MARIA_SHIFTS)
    assert r["total_hours"] == 40.25
    assert r["regular_hours"] == 40.0
    assert r["overtime_hours"] == 0.25
    assert r["double_time_hours"] == 0.0
    assert r["has_overtime"] is True
    assert r["all_shifts_complete"] is False


def test_compute_hours_per_shift_flags():
    r = compute_hours(MARIA_SHIFTS)
    by_date = {s["date"]: s for s in r["per_shift"]}
    assert by_date["2026-05-21"]["worked_hours"] == 9.75
    assert by_date["2026-05-21"]["rounding_applied"] is True
    assert by_date["2026-05-22"]["complete"] is False
    assert by_date["2026-05-22"]["worked_hours"] is None
    assert "missed_clock_out on 2026-05-22" in r["flags"]
    assert "rounding_applied on 2026-05-21" in r["flags"]


def test_compute_hours_double_time_tier():
    # Five 13h shifts (no break) = 65h -> 40 regular, 20 overtime, 5 double-time.
    shifts = [
        {"date": f"2026-06-0{i}", "clock_in": "06:00", "clock_out": "19:00", "unpaid_break_min": 0}
        for i in range(1, 6)
    ]
    r = compute_hours(shifts)
    assert r["total_hours"] == 65.0
    assert r["regular_hours"] == 40.0
    assert r["overtime_hours"] == 20.0
    assert r["double_time_hours"] == 5.0


def test_compute_hours_empty_returns_error():
    assert compute_hours([]) == {"error": "no shifts provided"}


# --- estimate_payroll_impact ---

def test_estimate_payroll_impact_headline_case():
    r = estimate_payroll_impact(22.50, regular_hours=40.0, overtime_hours=0.25)
    assert r["regular_pay"] == pytest.approx(900.00, abs=0.005)
    assert r["overtime_pay"] == pytest.approx(8.44, abs=0.005)
    assert r["double_time_pay"] == pytest.approx(0.0, abs=0.005)
    assert r["gross_pay"] == pytest.approx(908.44, abs=0.005)
    assert r["overtime_premium"] == pytest.approx(2.81, abs=0.005)


def test_estimate_payroll_impact_rejects_negative():
    assert "error" in estimate_payroll_impact(-1.0)
    assert "error" in estimate_payroll_impact(22.5, overtime_hours=-3.0)


# --- get_timecards (reads the committed sample data) ---

def test_get_timecards_matches_by_name():
    r = get_timecards("Maria")
    assert "error" not in r
    assert len(r["employees"]) == 1
    emp = r["employees"][0]
    assert emp["name"] == "Maria Sanchez"
    assert emp["overtime_authorization"] == "unknown"
    assert len(emp["shifts"]) == 5


def test_get_timecards_date_filter():
    r = get_timecards("Maria", start="2026-05-19", end="2026-05-19")
    assert [s["date"] for s in r["employees"][0]["shifts"]] == ["2026-05-19"]


def test_get_timecards_unknown_employee_is_structured_error():
    r = get_timecards("Bob")
    assert r["error"].startswith("no employee matching")
    assert "Maria Sanchez" in r["available_employees"]


# --- create_review_ticket (isolated to a temp file) ---

def test_create_review_ticket_appends(tmp_path, monkeypatch):
    settings = config.Settings(data_dir=tmp_path)
    monkeypatch.setattr(tickets, "get_settings", lambda: settings)

    res = tickets.create_review_ticket(
        "Maria Sanchez",
        "Unauthorized overtime",
        recommended_action="Manager review",
        payroll_impact=2.81,
        citations=["Overtime Policy > Overtime Authorization"],
    )
    assert res["status"] == "created"
    assert res["ticket_id"].startswith("TKT-")

    lines = (tmp_path / "tickets.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["employee"] == "Maria Sanchez"
    assert record["status"] == "open"
    assert record["payroll_impact"] == 2.81


def test_create_review_ticket_requires_fields(tmp_path, monkeypatch):
    settings = config.Settings(data_dir=tmp_path)
    monkeypatch.setattr(tickets, "get_settings", lambda: settings)
    assert "error" in tickets.create_review_ticket("", "issue")
    assert not (tmp_path / "tickets.jsonl").exists()


# --- registry routing + validation ---

def test_call_tool_routes_and_computes():
    r = call_tool("compute_hours", {"shifts": MARIA_SHIFTS})
    assert r["total_hours"] == 40.25


def test_call_tool_unknown_tool():
    r = call_tool("nonexistent", {})
    assert r["error"] == "unknown_tool"
    assert "compute_hours" in r["available_tools"]


def test_call_tool_invalid_arguments():
    # estimate_payroll_impact requires hourly_rate.
    r = call_tool("estimate_payroll_impact", {})
    assert r["error"] == "invalid_arguments"
    assert "hourly_rate" in r["detail"]
