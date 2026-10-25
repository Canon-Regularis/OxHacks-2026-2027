"""Backtest folds kept in memo: a run with stored folds returns exactly what a cold run does."""

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import pytest

from gridpulse.analysis.algorithms import (
    _backtest,
    _parallel,
    price_baseline_backtest,
    price_forecast_backtest,
    price_regression_backtest,
)
from gridpulse.analysis.algorithms._backtest import WINDOW
from gridpulse.analysis.algorithms._frames import half_hourly
from gridpulse.analysis.contract import AlgorithmResult

FAST: dict[Any, dict[str, Any]] = {
    price_baseline_backtest: {"window": 672},
    price_regression_backtest: {"window": 336},
    price_forecast_backtest: {"window": 336},
}


def frame_of(values: Sequence[float]) -> pd.DataFrame:
    index = pd.date_range("2026-09-01", periods=len(values), freq="30min", tz="UTC")
    return pd.DataFrame({"price_gbp_mwh": values}, index=index)


def prices(days: int) -> list[float]:
    rng = np.random.default_rng(3)
    t = np.arange(WINDOW + days * 48)
    return [float(v) for v in 80 + 30 * np.sin(2 * np.pi * t / 48) + rng.normal(0, 8, len(t))]


def run(module: Any, values: Sequence[float], memo: dict[str, str] | None) -> AlgorithmResult:
    params = {**module.ALGORITHM.params, **FAST[module], "max_folds": 60}
    return module.run(frame_of(values), **params, memo=memo)  # type: ignore[no-any-return]


def assert_identical(a: AlgorithmResult, b: AlgorithmResult) -> None:
    assert a.series is not None
    assert b.series is not None
    assert a.series.equals(b.series)
    assert list(a.tables) == list(b.tables)
    for name, table in a.tables.items():
        assert table.equals(b.tables[name]), name


@pytest.fixture
def fits(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """How many folds each run_all call fits."""
    counts: list[int] = []
    real = _parallel.run_all

    def counting(function: Callable[[Any], Any], tasks: Sequence[Any]) -> list[Any]:
        counts.append(len(tasks))
        return real(function, tasks)

    monkeypatch.setattr(_parallel, "run_all", counting)
    return counts


@pytest.mark.parametrize("module", list(FAST), ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_stored_folds_give_the_cold_result(module: Any, fits: list[int]) -> None:
    values = prices(2)
    models = len(module.MODELS)
    cold = run(module, values, None)
    memo: dict[str, str] = {}
    first = run(module, values, memo)
    again = run(module, values, memo)
    assert fits == [2 * models, 2 * models, 0]
    assert len(memo) == 2 * models
    assert_identical(cold, first)
    assert_identical(cold, again)


def test_a_new_day_fits_only_the_new_folds(fits: list[int]) -> None:
    module = price_forecast_backtest
    values = prices(3)
    memo: dict[str, str] = {}
    run(module, values[:-48], memo)  # the run before the third day settled
    later = run(module, values, memo)
    assert fits[1] == len(module.MODELS)
    assert_identical(later, run(module, values, None))


def test_a_revised_price_refits_the_folds_that_contain_it(fits: list[int]) -> None:
    module = price_baseline_backtest
    values = prices(2)
    memo: dict[str, str] = {}
    run(module, values, memo)
    values[WINDOW + 50] += 100.0  # a target of fold 2, so only fold 2's span changes
    revised = run(module, values, memo)
    assert fits[1] == len(module.MODELS)
    assert_identical(revised, run(module, values, None))


def test_unreadable_entries_are_refitted(fits: list[int]) -> None:
    module = price_regression_backtest
    values = prices(1)
    memo: dict[str, str] = {}
    cold = run(module, values, memo)
    for key in list(memo)[:2]:
        memo[key] = "not json"
    memo[list(memo)[2]] = '{"forecast": null}'
    again = run(module, values, memo)
    assert fits[1] == 3
    assert_identical(cold, again)
    assert all(text.startswith("{") and '"origin"' in text for text in memo.values())


def test_missing_data_folds_are_stored_too(fits: list[int]) -> None:
    module = price_baseline_backtest
    values = prices(2)
    values[WINDOW + 60] = math.nan
    memo: dict[str, str] = {}
    cold = run(module, values, memo)
    again = run(module, values, memo)
    assert fits[1] == 0
    assert_identical(cold, again)
    assert set(again.tables["folds"]["status"]) == {"ok", "missing_data"}


def test_folds_round_trip_exactly() -> None:
    module = price_forecast_backtest
    values = prices(1)
    span = half_hourly(frame_of(values)["price_gbp_mwh"]).window(WINDOW - 336, WINDOW + 48)
    forecasters = _backtest.build(module.MODELS, module.ALGORITHM.params["models"])
    for forecaster in forecasters.values():
        fold = _backtest._fold((forecaster, span), window=336, horizon=48, season_length=48)
        assert fold.forecast is not None
        assert _backtest._decoded(_backtest._encoded(fold)) == fold
    arima = _backtest._fold((forecasters["arima"], span), window=336, horizon=48, season_length=48)
    assert arima.forecast is not None
    assert arima.forecast.params["order"] == (2, 0, 1)  # tuples come back as tuples


def test_fold_keys_cover_model_settings_and_prices() -> None:
    span = half_hourly(frame_of(prices(1))["price_gbp_mwh"]).window(0, 400)
    forecasters = _backtest.build(
        price_baseline_backtest.MODELS, price_baseline_backtest.ALGORITHM.params["models"]
    )
    day, week = forecasters["seasonal_naive_day"], forecasters["seasonal_naive_week"]
    key = _backtest._fold_key(day, span, 352, 48, 48)
    assert key == _backtest._fold_key(day, span, 352, 48, 48)
    assert key != _backtest._fold_key(week, span, 352, 48, 48)
    assert key != _backtest._fold_key(day, span, 352, 48, 24)
    assert key != _backtest._fold_key(day, span.window(1, 400), 351, 48, 48)
