"""Command-line entry point: `shiftguard "<query>"`.

Sets up console+file trace logging, runs the agent on the query, and prints the
final answer plus the tool trajectory. The log file is the demo deliverable.
"""

from __future__ import annotations

import argparse

from .agent.react import run_agent
from .logging_setup import setup_logging
from .rag.retriever import get_retriever


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="shiftguard", description="Audit hourly timecards before payroll.")
    parser.add_argument("query", help="natural-language audit request")
    args = parser.parse_args(argv)

    setup_logging("agent")
    try:
        result = run_agent(args.query)
    finally:
        # Close the embedded-Qdrant connection while imports still work, so its
        # client isn't finalized mid-interpreter-shutdown (noisy __del__ error).
        get_retriever().close()

    print()
    print(result.answer if result.answer else f"[no final answer — stopped: {result.stop_reason}]")
    print(f"\n(tools used: {', '.join(result.tool_trajectory) or 'none'})")
    return 0 if result.answer else 1


if __name__ == "__main__":
    raise SystemExit(main())
