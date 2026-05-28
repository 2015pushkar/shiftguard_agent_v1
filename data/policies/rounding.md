# Time Rounding Policy

## Quarter-Hour Rounding (7-Minute Rule)
Punch times are rounded to the nearest quarter-hour (:00, :15, :30, :45). A
punch 1 to 7 minutes past a quarter-hour mark rounds down to that mark; a punch
8 to 14 minutes past a quarter-hour mark rounds up to the next mark. This is the
standard "7-minute rule."

## Rounding Neutrality
Rounding must be neutral over time and may not systematically favor the
employer. Any shift whose raw punches do not fall exactly on quarter-hour marks
is flagged so the rounding adjustment is visible and auditable.

## Application Order
Rounding is applied to the raw clock-in and clock-out punches first. Worked
hours, the unpaid meal-break deduction, and overtime are then calculated from
the rounded punch times.
