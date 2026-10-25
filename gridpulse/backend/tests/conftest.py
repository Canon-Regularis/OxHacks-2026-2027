"""Optional sharding, so CI can spread the suite over several runners.

Set ``PYTEST_SHARD=<index>/<count>`` (for example ``1/3``) to run every ``count``-th test,
starting from ``index``. Dealing tests out in turn mixes the slow Orchid files into every shard.
Every test lands in exactly one shard, and the choice does not depend on the process, so
pytest-xdist workers still agree on which tests exist.
"""

import os

import pytest


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
