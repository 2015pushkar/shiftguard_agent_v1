"""Backend for the `create_review_ticket` tool: open a manager review ticket
by appending one JSON record per line to the local tickets file.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from ..config import get_settings


def create_review_ticket(
    employee: str,
    issue: str,
    recommended_action: str | None = None,
    payroll_impact: float | None = None,
    citations: list[str] | None = None,
) -> dict:
    """Append a manager review ticket to the tickets file and return its record.

    Returns an `{"error": ...}` dict (writing nothing) when `employee` or
    `issue` is blank.
    """
    if not employee or not employee.strip() or not issue or not issue.strip():
        return {"error": "employee and issue are required"}

    ticket = {
        "ticket_id": "TKT-" + uuid.uuid4().hex[:8],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "employee": employee,
        "issue": issue,
        "recommended_action": recommended_action,
        "payroll_impact": payroll_impact,
        "citations": citations or [],
        "status": "open",
    }

    tickets_path = get_settings().tickets_path
    tickets_path.parent.mkdir(parents=True, exist_ok=True)
    with tickets_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ticket) + "\n")

    return {
        "status": "created",
        "ticket_id": ticket["ticket_id"],
        "path": str(tickets_path),
        "ticket": ticket,
    }
