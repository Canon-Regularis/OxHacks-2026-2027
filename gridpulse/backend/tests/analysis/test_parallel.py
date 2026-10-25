"""The process pool gives the same results as fitting in this process."""

import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from gridpulse.analysis.algorithms import _parallel, price_baseline_backtest
from gridpulse.analysis.algorithms._backtest import WINDOW
from gridpulse.analysis.algorithms.errors import InsufficientDataError
from gridpulse.analysis.algorithms.forecasting import Naive, SeasonalNaive
from gridpulse.analysis.algorithms.series import HalfHourlySeries
from gridpulse.analysis.contract import AlgorithmResult

START = datetime(2026, 9, 1, tzinfo=UTC)


def prices(n: int) -> list[float]:
    rng = np.random.default_rng(1)
    t = np.arange(n)
    return [float(v) for v in 80 + 30 * np.sin(2 * np.pi * t / 48) + rng.normal(0, 8, n)]


def test_workers_follows_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_parallel.WORKERS_ENV, "3")
    assert _parallel.workers() == 3
    monkeypatch.delenv(_parallel.WORKERS_ENV)
    assert 1 <= _parallel.workers() <= _parallel.DEFAULT_MAX_WORKERS
    for bad in ("0", "two"):
        monkeypatch.setenv(_parallel.WORKERS_ENV, bad)
        with pytest.raises(ValueError, match=_parallel.WORKERS_ENV):
            _parallel.workers()


def test_run_all_keeps_task_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_parallel.WORKERS_ENV, "2")
    tasks = [float(n) for n in range(20)]
    assert _parallel.run_all(math.sqrt, tasks) == [math.sqrt(n) for n in tasks]


def test_forecast_all_returns_errors_from_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_parallel.WORKERS_ENV, "2")
    history = HalfHourlySeries(START, prices(20))
    naive, seasonal = _parallel.forecast_all([Naive(), SeasonalNaive(48)], history, 4)
    assert not isinstance(naive, Exception)
    assert naive.mean == Naive().forecast(history, 4).mean
    assert isinstance(seasonal, InsufficientDataError)
    assert (seasonal.required, seasonal.available) == (49, 20)


def test_backtest_is_identical_in_a_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    values = prices(WINDOW + 3 * 48)
    values[WINDOW + 60] = math.nan  # the second fold of every model is missing_data
    index = pd.date_range(START, periods=len(values), freq="30min")
    frame = pd.DataFrame({"price_gbp_mwh": values}, index=index)
    params = {**price_baseline_backtest.ALGORITHM.params, "window": 672, "max_folds": 3}

    def run(workers: str) -> AlgorithmResult:
        monkeypatch.setenv(_parallel.WORKERS_ENV, workers)
        return price_baseline_backtest.run(frame.copy(), **params)

    serial, pooled = run("1"), run("3")
    assert serial.series is not None
    assert pooled.series is not None
    assert serial.series.equals(pooled.series)
    assert list(serial.tables) == list(pooled.tables)
    for name, table in serial.tables.items():
        assert table.equals(pooled.tables[name]), name
    assert set(serial.tables["folds"]["status"]) == {"ok", "missing_data"}
