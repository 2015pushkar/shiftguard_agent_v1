"""Eval harness: run the full agent over labeled scenarios and assert on the
TOOL TRAJECTORY (which tools fired) and answer properties — not exact wording.

This proves autonomous routing: the loop and prompt are identical across
categories; only the query changes, yet the trajectory changes correctly. Each
scenario's thought/action/observation trace is written to `evals/report/` (the
demo deliverable), alongside a pass/fail `summary.md`.

Run from the repo root:  python evals/run_evals.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from shiftguard.agent.react import run_agent
from shiftguard.logging_setup import setup_logging
from shiftguard.rag.retriever import get_retriever

EVALS_DIR = Path(__file__).resolve().parent
SCENARIOS_PATH = EVALS_DIR / "scenarios.jsonl"
REPORT_DIR = EVALS_DIR / "report"


def load_scenarios(path: Path = SCENARIOS_PATH) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def evaluate(result, scenario: dict) -> list[str]:
    """Return a list of failure messages; empty means the scenario passed."""
    failures: list[str] = []
    trajectory = result.tool_trajectory
    used = set(trajectory)

    for tool in scenario.get("required_tools", []):
        if tool not in used:
            failures.append(f"missing required tool '{tool}' (trajectory={trajectory})")
    for tool in scenario.get("forbidden_tools", []):
        if tool in used:
            failures.append(f"used forbidden tool '{tool}' (trajectory={trajectory})")
    if scenario.get("expect_zero_tools") and trajectory:
        failures.append(f"expected zero tools, got {trajectory}")
    if scenario.get("expect_answer", True) and not result.answer:
        failures.append(f"no final answer (stop_reason={result.stop_reason})")
    return failures


def run_scenario(scenario: dict) -> dict:
    setup_logging(run_name=scenario["id"], log_dir=REPORT_DIR)
    try:
        result = run_agent(scenario["query"])
        failures = evaluate(result, scenario)
    except Exception as e:  # a scenario must never abort the whole run
        return {"scenario": scenario, "result": None, "failures": [f"crashed: {e!r}"]}
    return {"scenario": scenario, "result": result, "failures": failures}


def _write_report(outcomes: list[dict]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    passed = sum(1 for o in outcomes if not o["failures"])
    lines = [
        f"# ShiftGuard eval report — {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        f"**{passed}/{len(outcomes)} scenarios passed.**",
        "",
    ]
    for o in outcomes:
        sc, res, fails = o["scenario"], o["result"], o["failures"]
        status = "PASS" if not fails else "FAIL"
        lines += [
            f"## {sc['id']} — {sc['category']}: {status}",
            f"- query: {sc['query']}",
            f"- tools used: {res.tool_trajectory if res else '(crashed)'}",
            f"- stop_reason: {res.stop_reason if res else '(crashed)'}",
            f"- answer: {res.answer if res else '(crashed)'}",
        ]
        if fails:
            lines.append(f"- failures: {fails}")
        lines.append("")
    report_path = REPORT_DIR / "summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> int:
    outcomes = [run_scenario(sc) for sc in load_scenarios()]
    report_path = _write_report(outcomes)
    get_retriever().close()

    passed = sum(1 for o in outcomes if not o["failures"])
    print(f"\n==== EVAL SUMMARY: {passed}/{len(outcomes)} passed ====")
    for o in outcomes:
        status = "PASS" if not o["failures"] else "FAIL"
        print(f"  [{status}] {o['scenario']['id']:<16} tools={o['result'].tool_trajectory if o['result'] else '(crashed)'}")
        for f in o["failures"]:
            print(f"          - {f}")
    print(f"report: {report_path}")
    return 0 if passed == len(outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
