"""Backend for the `get_timecards` tool: load the timecard file and return
employees (optionally filtered by employee identity and shift date range).
"""

from __future__ import annotations

import json

from ..config import get_settings


def get_timecards(
    employee: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """Return employees and their shifts, filtered by `employee` and date range.

    `employee` matches a case-insensitive `id` or a case-insensitive substring
    of `name`. `start`/`end` are inclusive ISO date bounds on each shift.
    """
    data = json.loads(get_settings().timecards_path.read_text(encoding="utf-8"))
    all_employees: list[dict] = data["employees"]

    if employee is not None:
        needle = employee.lower()
        matched = [
            e
            for e in all_employees
            if needle == e["id"].lower() or needle in e["name"].lower()
        ]
        if not matched:
            return {
                "error": f"no employee matching '{employee}'",
                "available_employees": [e["name"] for e in all_employees],
            }
    else:
        matched = all_employees

    return {
        "pay_period": data["pay_period"],
        "employees": [
            {
                "id": e["id"],
                "name": e["name"],
                "hourly_rate": e["hourly_rate"],
                "overtime_authorization": e["overtime_authorization"],
                "shifts": [
                    s
                    for s in e["shifts"]
                    if (start is None or s["date"] >= start)
                    and (end is None or s["date"] <= end)
                ],
            }
            for e in matched
        ],
    }
