"""Settings shared by every backend test, and optional sharding so CI can spread the suite
over several runners.

Set ``PYTEST_SHARD=<index>/<count>`` (for example ``1/3``) to run every ``count``-th test,
starting from ``index``. Dealing tests out in turn mixes the slow Orchid files into every shard.
Every test lands in exactly one shard, and the choice does not depend on the process, so
pytest-xdist workers still agree on which tests exist.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def _serial_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fit in the test process: pytest-xdist already uses every CPU, so a process pool per
    algorithm run would only oversubscribe them. test_parallel.py covers the pool."""
    monkeypatch.setenv("GRIDPULSE_WORKERS", "1")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    spec = os.environ.get("PYTEST_SHARD", "").strip()
    if not spec:
        return
    index, _, count = spec.partition("/")
    shard, shards = int(index), int(count)
    if not 1 <= shard <= shards:
        raise pytest.UsageError(
            f"PYTEST_SHARD={spec!r} must be <index>/<count> with 1 <= index <= count"
        )
    kept, dropped = [], []
    for position, item in enumerate(items):
        (kept if position % shards == shard - 1 else dropped).append(item)
    config.hook.pytest_deselected(items=dropped)
    items[:] = kept
