"""price_forecast_backtest: fold geometry, availability, gaps and the stored tables."""

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from gridpulse.analysis.algorithms import price_forecast_backtest as pfb
from gridpulse.analysis.algorithms.errors import InvalidInputError
from gridpulse.analysis.algorithms.forecasting import ETS
from gridpulse.analysis.algorithms.series import HalfHourlySeries
from gridpulse.analysis.contract import AlgorithmResult
from gridpulse.analysis.service import check_output, check_tables

FOLDS = 2
WINDOW = 336  # a 7-day fit keeps the tests fast; the range still starts after pfb.WINDOW


def frame_of(values: list[float]) -> pd.DataFrame:
    index = pd.date_range("2026-09-01", periods=len(values), freq="30min", tz="UTC")
    return pd.DataFrame({"price_gbp_mwh": values}, index=index)


def params(**overrides: Any) -> dict[str, Any]:
    return {**pfb.ALGORITHM.params, "window": WINDOW, "max_folds": FOLDS, **overrides}


@pytest.fixture(scope="module")
def prices() -> list[float]:
    rng = np.random.default_rng(1)
    t = np.arange(pfb.WINDOW + FOLDS * 48)
    return list(80 + 30 * np.sin(2 * np.pi * t / 48) + rng.normal(0, 8, len(t)))


@pytest.fixture(scope="module")
def result(prices: list[float]) -> AlgorithmResult:
    return pfb.run(frame_of(prices), **params())


def table(result: AlgorithmResult, name: str) -> pd.DataFrame:
    return result.tables[name]


def forecasts(result: AlgorithmResult) -> pd.DataFrame:
    assert result.series is not None
    return result.series


def test_every_range_period_gets_one_forecast_per_model(
    prices: list[float], result: AlgorithmResult
) -> None:
    frame = frame_of(prices)
    series = check_output(pfb.ALGORITHM, result.series, frame.index[pfb.WINDOW], frame.index[-1])
    assert len(series) == FOLDS * 48 - 1  # check_output's end is exclusive
    assert series.notna().all().all()
    assert list(series.columns) == list(pfb.OUTPUTS)
    assert (series["ets_lower_95"] <= series["ets_lower_80"]).all()
    assert (series["ets_lower_80"] <= series["ets_forecast"]).all()
    assert (series["ets_forecast"] <= series["ets_upper_80"]).all()
    check_tables(pfb.ALGORITHM, result.tables)


def test_forecast_is_the_model_fitted_on_the_window_before_the_origin(
    prices: list[float], result: AlgorithmResult
) -> None:
    frame = frame_of(prices)
    origin = pfb.WINDOW + 48  # the second fold
    history = HalfHourlySeries(
        frame.index[origin - WINDOW].to_pydatetime(), prices[origin - WINDOW : origin]
    )
    expected = ETS(**pfb.ALGORITHM.params["models"]["ets"]).forecast(history, 48).mean
    got = forecasts(result)["ets_forecast"].iloc[origin - pfb.WINDOW : origin - pfb.WINDOW + 48]
    assert list(got) == pytest.approx(expected)


def test_forecasts_ignore_prices_after_their_origin(
    prices: list[float], result: AlgorithmResult
) -> None:
    changed = list(prices)
    changed[pfb.WINDOW + 10] += 500.0  # inside the first fold's targets
    again = pfb.run(frame_of(changed), **params())
    first, second = slice(0, 48), slice(48, 96)
    assert (
        forecasts(result).iloc[first].equals(forecasts(again).iloc[first])
    )  # made before the spike
    assert not forecasts(result).iloc[second].equals(forecasts(again).iloc[second])  # trained on it


def test_a_gap_excludes_only_the_folds_that_contain_it(prices: list[float]) -> None:
    gappy = list(prices)
    gappy[pfb.WINDOW + 60] = math.nan  # a target of fold 2 only
    out = pfb.run(frame_of(gappy), **params())
    folds = table(out, "folds")
    assert list(folds.groupby("status").size().items()) == [("missing_data", 3), ("ok", 3)]
    assert folds.loc[folds["status"] == "missing_data", "message"].str.startswith("1 of its").all()
    assert forecasts(out).iloc[:48].notna().all().all()
    assert forecasts(out).iloc[48:].isna().all().all()
    metrics = table(out, "metrics").set_index("model")
    assert (metrics["n_ok_folds"] == 1).all()
    assert (metrics["n_excluded_folds"] == 1).all()


def test_metrics_tables(result: AlgorithmResult) -> None:
    metrics = table(result, "metrics").set_index("model")
    assert list(metrics.index) == list(pfb.MODELS)
    assert metrics[["n_ok_folds", "n_excluded_folds"]].to_numpy().tolist() == [[FOLDS, 0]] * 3
    assert metrics.notna().all().all()
    assert ((metrics["coverage_80"] >= 0) & (metrics["coverage_95"] <= 1)).all()
    assert metrics.at["seasonal_naive", "interval_method"] == "seasonal-random-walk"

    by_horizon = table(result, "by_horizon")
    assert len(by_horizon) == 3 * 48
    assert list(by_horizon["horizon_step"][:2]) == [1, 2]
    assert by_horizon["lead_hours"].iloc[47] == 24.0

    folds = table(result, "folds")
    assert len(folds) == 3 * FOLDS
    assert (folds["train_size"] == WINDOW).all()
    origins = folds["origin_utc"].drop_duplicates()
    assert list(origins.diff().dropna()) == [pd.Timedelta(hours=24)] * (FOLDS - 1)

    fitted = table(result, "fold_params")
    by_model = fitted.groupby("model")["parameter"].unique()
    assert {"alpha", "gamma"} <= set(by_model["ets"])
    assert {"sigma2", "aicc"} <= set(by_model["arima"])
    assert "seasonal_naive" not in by_model


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"max_folds": FOLDS - 1}, "exceed max_folds"),
        ({"window": pfb.WINDOW + 1}, "window must be"),
        ({"models": {"ets": {}}}, "models must configure"),
        ({"horizon": 0}, "horizon must be"),
    ],
)
def test_settings_are_checked(prices: list[float], overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(InvalidInputError, match=match):
        pfb.run(frame_of(prices), **params(**overrides))


def test_a_frame_shorter_than_the_window_gives_no_folds() -> None:
    out = pfb.run(frame_of([80.0] * (pfb.WINDOW + 10)), **params())
    assert forecasts(out).isna().all().all()
    assert table(out, "folds").empty
    assert (table(out, "metrics")["n_ok_folds"] == 0).all()


def test_tables_have_the_declared_columns(result: AlgorithmResult) -> None:
    assert {name: list(t.columns) for name, t in result.tables.items()} == pfb.TABLE_COLUMNS
