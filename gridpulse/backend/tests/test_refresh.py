"""gridpulse refresh: which algorithms are stale, the windows they run over, and failures."""

from datetime import UTC, date, datetime

import pytest

from gridpulse.analysis import refresh as r
from gridpulse.analysis.algorithms import ALGORITHMS

FORECAST = ALGORITHMS["price_forecast_backtest"]
DAY_AHEAD = ALGORITHMS["price_day_ahead_forecast"]


def test_every_algorithm_has_a_window():
    assert set(r.WINDOWS) == set(ALGORITHMS)


def test_backtest_windows_fit_max_folds():
    for algorithm in ALGORITHMS.values():
        if "max_folds" in algorithm.params:
            assert r.WINDOWS[algorithm.id][0] <= algorithm.params["max_folds"], algorithm.id


@pytest.mark.parametrize(
    ("latest", "expected"),
    [
        (datetime(2026, 10, 23, 22, 30, tzinfo=UTC), date(2026, 10, 23)),  # last period of 23 Oct (BST)
        (datetime(2026, 10, 23, 22, 0, tzinfo=UTC), date(2026, 10, 22)),  # 23 Oct still incomplete
        (datetime(2026, 10, 25, 23, 30, tzinfo=UTC), date(2026, 10, 25)),  # period 50 of 25 Oct
    ],
)
def test_latest_complete_day(monkeypatch, latest, expected):
    monkeypatch.setattr(r.queries, "latest_price_bucket", lambda conn: latest)
    assert r.latest_complete_day(None) == expected


def test_windows_end_at_the_latest_day():
    assert r.window(FORECAST, date(2026, 10, 23)) == (date(2026, 8, 29), date(2026, 10, 23))
    assert r.window(DAY_AHEAD, date(2026, 10, 23)) == (date(2026, 10, 23), date(2026, 10, 23))
    fourier = ALGORITHMS["calendar_fourier_terms"]
    assert r.window(fourier, date(2026, 10, 23))[1] == date(2026, 10, 30)


def stored_run(start, end, datasets=None):
    return {"run_id": 9, "range_start_utc": start, "range_end_utc": end, "datasets": datasets}


@pytest.fixture
def window_utc():
    return r.range_utc(date(2026, 10, 1), date(2026, 10, 23))


def test_stale_reasons(monkeypatch, window_utc):
    start, end = window_utc
    found = {}
    monkeypatch.setattr(r.queries, "find_run", lambda *args: found.get("run"))
    monkeypatch.setattr(r.service, "reloaded_warnings", lambda conn, run: found.get("reloaded", []))
    days = (date(2026, 10, 1), date(2026, 10, 23))

    assert r.stale_reason(None, FORECAST, *days) == "no run"
    found["run"] = stored_run(start, end.replace(day=20))
    assert "covers" in r.stale_reason(None, FORECAST, *days)
    found["run"] = stored_run(start, end)
    assert "predates" in r.stale_reason(None, FORECAST, *days)
    found["run"] = stored_run(start, end, {"elexon_system_prices": "2026-10-24T14:00:00+00:00"})
    assert r.stale_reason(None, FORECAST, *days) is None
    found["reloaded"] = ["elexon_system_prices was loaded again"]
    assert "loaded again" in r.stale_reason(None, FORECAST, *days)


def test_day_ahead_must_cover_its_horizon(monkeypatch):
    start, end = r.range_utc(date(2026, 10, 23), date(2026, 10, 23))
    run = stored_run(start, end, {})  # covers the input day but not the forecast day after it
    monkeypatch.setattr(r.queries, "find_run", lambda *args: run)
    assert "covers" in r.stale_reason(None, DAY_AHEAD, date(2026, 10, 23), date(2026, 10, 23))


def test_refresh_reruns_only_stale_algorithms_and_survives_failures(monkeypatch):
    stale = {"price_forecast_backtest": "no run", "price_day_ahead_forecast": "no run"}
    monkeypatch.setattr(r, "stale_reason", lambda conn, algorithm, *days: stale.get(algorithm.id))

    def run_algorithm(conn, algorithm, start, end, reuse):
        assert reuse  # stored backtest folds are reused unless forced
        if algorithm.id == "price_forecast_backtest":
            raise ValueError("boom")
        return 41

    monkeypatch.setattr(r.service, "run_algorithm", run_algorithm)
    chosen = [FORECAST, DAY_AHEAD, ALGORITHMS["price_rolling_signals"]]
    outcomes = {o.algorithm_id: o for o in r.refresh(None, chosen, through=date(2026, 10, 23))}
    assert outcomes["price_forecast_backtest"].status == "failed"
    assert "ValueError: boom" in outcomes["price_forecast_backtest"].detail
    assert (outcomes["price_day_ahead_forecast"].status, outcomes["price_day_ahead_forecast"].run_id) == ("succeeded", 41)
    assert outcomes["price_rolling_signals"].status == "current"


def test_forced_refresh_refits_from_scratch(monkeypatch):
    reused = []
    monkeypatch.setattr(r.service, "run_algorithm", lambda conn, a, start, end, reuse: reused.append(reuse) or 41)
    r.refresh(None, [FORECAST], through=date(2026, 10, 23), force=True)
    assert reused == [False]


def test_refresh_needs_prices(monkeypatch):
    monkeypatch.setattr(r.queries, "latest_price_bucket", lambda conn: None)
    with pytest.raises(RuntimeError, match="No settlement prices"):
        r.refresh(None, [FORECAST])
