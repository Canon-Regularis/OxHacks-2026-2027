"""Settings shared by every backend test."""

import pytest


@pytest.fixture(autouse=True)
def _serial_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fit in the test process: pytest-xdist already uses every CPU, so a process pool per
    algorithm run would only oversubscribe them. test_parallel.py covers the pool."""
    monkeypatch.setenv("GRIDPULSE_WORKERS", "1")
