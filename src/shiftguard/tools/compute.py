"""Deterministic punch/money math for ShiftGuard: quarter-hour ("7-minute")
rounding, worked-hours aggregation with overtime tiers, and payroll-impact
estimation. Pure functions — no config, files, or network.
"""

from __future__ import annotations


def _round_punch(clock: str) -> tuple[int, bool]:
    """Apply the 7-minute quarter-hour rule to a "HH:MM" time.

    Returns (rounded total minutes, whether rounding changed the value).
    """
    hh, mm = clock.split(":")
    t = int(hh) * 60 + int(mm)
    rem = t % 15
    if rem == 0:
        return t, False
    if rem <= 7:
        return t - rem, True
    return t + (15 - rem), True


def compute_hours(
    shifts: list[dict],
    overtime_threshold: float = 40.0,
    double_time_threshold: float = 60.0,
) -> dict:
    """Compute per-shift and aggregate worked hours with overtime tiers."""
    if not shifts:
        return {"error": "no shifts provided"}

    per_shift: list[dict] = []
    flags: list[str] = []
    total_hours = 0.0
    all_complete = True

    for shift in shifts:
        date = shift.get("date")
        clock_in = shift.get("clock_in")
        clock_out = shift.get("clock_out")

        if clock_out is None:
            all_complete = False
            per_shift.append(
                {
                    "date": date,
                    "worked_hours": None,
                    "complete": False,
                    "rounding_applied": False,
                }
            )
            flags.append(f"missed_clock_out on {date}")
            continue

        rounded_in, in_rounded = _round_punch(clock_in)
        rounded_out, out_rounded = _round_punch(clock_out)
        rounding_applied = in_rounded or out_rounded
        break_min = shift.get("unpaid_break_min", 0) or 0
        worked_minutes = rounded_out - rounded_in - break_min
        worked_hours = round(worked_minutes / 60, 2)
        total_hours += worked_hours

        per_shift.append(
            {
                "date": date,
                "worked_hours": worked_hours,
                "complete": True,
                "rounding_applied": rounding_applied,
            }
        )
        if rounding_applied:
            flags.append(f"rounding_applied on {date}")

    total_hours = round(total_hours, 2)
    regular_hours = round(min(total_hours, overtime_threshold), 2)
    overtime_hours = round(
        min(
            max(total_hours - overtime_threshold, 0.0),
            double_time_threshold - overtime_threshold,
        ),
        2,
    )
    double_time_hours = round(max(total_hours - double_time_threshold, 0.0), 2)

    return {
        "per_shift": per_shift,
        "total_hours": total_hours,
        "regular_hours": regular_hours,
        "overtime_hours": overtime_hours,
        "double_time_hours": double_time_hours,
        "has_overtime": (overtime_hours + double_time_hours) > 0,
        "all_shifts_complete": all_complete,
        "flags": flags,
    }


def estimate_payroll_impact(
    hourly_rate: float,
    regular_hours: float = 0.0,
    overtime_hours: float = 0.0,
    double_time_hours: float = 0.0,
    overtime_multiplier: float = 1.5,
    double_time_multiplier: float = 2.0,
) -> dict:
    """Estimate pay and the overtime premium (extra cost above straight time)."""
    if hourly_rate < 0 or regular_hours < 0 or overtime_hours < 0 or double_time_hours < 0:
        return {"error": "hourly_rate and hours must be non-negative"}

    regular_pay = regular_hours * hourly_rate
    overtime_pay = overtime_hours * hourly_rate * overtime_multiplier
    double_time_pay = double_time_hours * hourly_rate * double_time_multiplier
    gross_pay = regular_pay + overtime_pay + double_time_pay
    overtime_premium = overtime_hours * hourly_rate * (overtime_multiplier - 1) + (
        double_time_hours * hourly_rate * (double_time_multiplier - 1)
    )

    return {
        "regular_pay": round(regular_pay, 2),
        "overtime_pay": round(overtime_pay, 2),
        "double_time_pay": round(double_time_pay, 2),
        "gross_pay": round(gross_pay, 2),
        "overtime_premium": round(overtime_premium, 2),
    }
