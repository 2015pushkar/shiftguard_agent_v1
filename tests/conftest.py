"""Shared test setup: make the eval harness importable and close the embedded
Qdrant connection cleanly at the end of the session (avoids the noisy
__del__-at-shutdown traceback)."""

import sys
from pathlib import Path

import pytest

# Let `tests/test_routing.py` reuse the scenario list + checker from evals/.
_EVALS_DIR = Path(__file__).resolve().parents[1] / "evals"
if str(_EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(_EVALS_DIR))


@pytest.fixture(scope="session", autouse=True)
def _close_retriever():
    yield
    from shiftguard.rag.retriever import get_retriever

    get_retriever().close()
