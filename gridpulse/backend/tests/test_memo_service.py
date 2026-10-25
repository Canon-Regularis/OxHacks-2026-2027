"""run_algorithm hands memo algorithms their stored entries and saves the new ones."""

from contextlib import nullcontext
from datetime import UTC, datetime

import pandas as pd
import pytest

from gridpulse.analysis import service
from gridpulse.analysis.algorithms import ALGORITHMS
from gridpulse.analysis.contract import Algorithm

START = datetime(2026, 10, 1, tzinfo=UTC)


class FakeConn:
    def transaction(self):
        return nullcontext()


def frame(start, end, cutoff=None):
    index = pd.date_range(start, end, freq="30min", tz="UTC", inclusive="left")
    return pd.DataFrame({"price_gbp_mwh": 50.0}, index=index)


@pytest.fixture
def db(monkeypatch):
    """Stand-in queries; returns a record of the memo loads and saves."""
    calls = {"loads": [], "saves": [], "finished": []}
    q = service.queries
    monkeypatch.setattr(q, "dataset_loads", lambda conn, names: {})
    monkeypatch.setattr(q, "create_run", lambda *args: 7)
    monkeypatch.setattr(q, "insert_results", lambda conn, rows: len(list(rows)))
    monkeypatch.setattr(q, "insert_table", lambda *args: None)
    monkeypatch.setattr(q, "finish_run", lambda conn, run_id, status, **kw: calls["finished"].append(status))
    monkeypatch.setattr(service, "build_input_frame", lambda conn, start, end, cutoff: frame(start, end))

    def load(conn, namespace):
        calls["loads"].append(namespace)
        return {"old": "1", "stale": "2"}

    monkeypatch.setattr(q, "load_memo", load)
    monkeypatch.setattr(q, "save_memo", lambda conn, *args: calls["saves"].append(args))
    return calls


def memo_algorithm(seen):
    def run(frame, *, memo):
        seen.append(dict(memo))
        assert memo["old"] == "1"
        memo["new"] = "3"
        return pd.DataFrame({"x": 1.0}, index=frame.index)

    return Algorithm(id="memo_test", version="1", description="", run=run, inputs=("price_gbp_mwh",),
                     outputs={"x": ""}, memo=True)


def test_memo_algorithms_get_stored_entries_and_save_new_ones(db):
    seen = []
    algorithm = memo_algorithm(seen)
    service.run_algorithm(FakeConn(), algorithm, START, START.replace(day=2))
    namespace = service.memo_namespace(algorithm)
    assert db["loads"] == [namespace]
    assert seen == [{"old": "1", "stale": "2"}]
    ((algorithm_id, saved_namespace, new, read),) = db["saves"]
    assert (algorithm_id, saved_namespace, new, set(read)) == ("memo_test", namespace, {"new": "3"}, {"old"})
    assert db["finished"] == ["succeeded"]


def test_reuse_false_starts_empty_but_still_saves(db):
    def run(frame, *, memo):
        assert dict(memo) == {}
        memo["new"] = "3"
        return pd.DataFrame({"x": 1.0}, index=frame.index)

    algorithm = Algorithm(id="memo_test", version="1", description="", run=run, inputs=("price_gbp_mwh",),
                          outputs={"x": ""}, memo=True)
    service.run_algorithm(FakeConn(), algorithm, START, START.replace(day=2), reuse=False)
    assert db["loads"] == []
    assert db["saves"][0][2] == {"new": "3"}


def test_other_algorithms_never_see_a_memo(db):
    def run(frame):  # no memo argument
        return pd.DataFrame({"x": 1.0}, index=frame.index)

    algorithm = Algorithm(id="plain", version="1", description="", run=run, inputs=("price_gbp_mwh",),
                          outputs={"x": ""})
    service.run_algorithm(FakeConn(), algorithm, START, START.replace(day=2))
    assert db["loads"] == db["saves"] == []
    assert db["finished"] == ["succeeded"]


def test_namespace_depends_on_id_version_and_code():
    a = memo_algorithm([])
    b = Algorithm(**{**a.__dict__, "version": "2"})
    c = Algorithm(**{**a.__dict__, "id": "other"})
    assert service.memo_namespace(a) == service.memo_namespace(memo_algorithm([]))
    assert len({service.memo_namespace(x) for x in (a, b, c)}) == 3
    assert len(service._algorithms_code()) == 64


def test_only_the_backtests_use_memo():
    assert {a.id for a in ALGORITHMS.values() if a.memo} == {
        "price_forecast_backtest", "price_baseline_backtest", "price_regression_backtest",
    }
