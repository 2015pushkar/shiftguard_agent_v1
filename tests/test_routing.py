"""Integration tests: run the full agent per scenario and assert on the tool
trajectory + answer properties (mirrors evals/run_evals.py).

Slow — each test runs the live LLM. Marked `integration` and skipped if Ollama
is unreachable. Tickets are redirected to a temp file so runs don't pollute
data/tickets.jsonl.
"""

import urllib.request

import pytest

from run_evals import evaluate, load_scenarios

from shiftguard.agent.react import run_agent
from shiftguard.config import Settings, get_settings
from shiftguard.tools import tickets


def _ollama_up() -> bool:
    try:
        urllib.request.urlopen(f"{get_settings().ollama_host}/api/version", timeout=2)
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _ollama_up(), reason="Ollama not reachable"),
]


@pytest.fixture(autouse=True)
def _isolate_tickets(tmp_path, monkeypatch):
    monkeypatch.setattr(tickets, "get_settings", lambda: Settings(data_dir=tmp_path))


@pytest.mark.parametrize("scenario", load_scenarios(), ids=lambda s: s["id"])
def test_routing_trajectory(scenario):
    result = run_agent(scenario["query"])
    failures = evaluate(result, scenario)
    assert not failures, f"[{scenario['id']}] " + "; ".join(failures)
