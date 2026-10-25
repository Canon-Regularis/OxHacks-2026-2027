"""price_p1_backtest: the P1 models through the shared folds, the ensemble and dm_tests."""

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from gridpulse.analysis.algorithms import price_p1_backtest as p1
from gridpulse.analysis.algorithms.errors import InvalidInputError
from gridpulse.analysis.contract import AlgorithmResult
from gridpulse.analysis.service import check_output, check_tables

FOLDS = 2
WINDOW = 336  # a 7-day fit keeps the tests fast; the range still starts after p1.WINDOW
MEMBERS = p1.ALGORITHM.params["models"]["ensemble"]["members"]


def frame_of(values: list[float]) -> pd.DataFrame:
    index = pd.date_range("2026-09-01", periods=len(values), freq="30min", tz="UTC")
    return pd.DataFrame({"price_gbp_mwh": values}, index=index)


def params(**overrides: Any) -> dict[str, Any]:
    return {**p1.ALGORITHM.params, "window": WINDOW, "max_folds": FOLDS, **overrides}


def models(**changes: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {**p1.ALGORITHM.params["models"], **changes}


@pytest.fixture(scope="module")
def prices() -> list[float]:
    rng = np.random.default_rng(1)
    t = np.arange(p1.WINDOW + FOLDS * 48)
    return list(80 + 30 * np.sin(2 * np.pi * t / 48) + rng.normal(0, 8, len(t)))


@pytest.fixture(scope="module")
def result(prices: list[float]) -> AlgorithmResult:
    return p1.run(frame_of(prices), **params())


def forecasts(result: AlgorithmResult) -> pd.DataFrame:
    assert result.series is not None
    return result.series


def test_every_range_period_gets_one_forecast_per_model(
    prices: list[float], result: AlgorithmResult
) -> None:
    frame = frame_of(prices)
    series = check_output(p1.ALGORITHM, result.series, frame.index[p1.WINDOW], frame.index[-1])
    assert len(series) == FOLDS * 48 - 1  # check_output's end is exclusive
    assert series.notna().all().all()
    assert list(series.columns) == list(p1.OUTPUTS)
    assert "seasonal_naive_lower_80" not in series
    for name in p1.BANDED:
        assert (series[f"{name}_lower_95"] <= series[f"{name}_lower_80"]).all(), name
        assert (series[f"{name}_lower_80"] <= series[f"{name}_forecast"]).all(), name
        assert (series[f"{name}_forecast"] <= series[f"{name}_upper_80"]).all(), name
    stored = check_tables(p1.ALGORITHM, result.tables)
    assert {name: table["columns"] for name, table in stored.items()} == p1.TABLE_COLUMNS


def test_the_ensemble_is_the_mean_of_its_members(result: AlgorithmResult) -> None:
    out = forecasts(result)
    members = out[[f"{name}_forecast" for name in MEMBERS]].mean(axis=1)
    assert list(out["ensemble_forecast"]) == pytest.approx(list(members))


def test_forecasts_ignore_prices_after_their_origin(
    prices: list[float], result: AlgorithmResult
) -> None:
    changed = list(prices)
    changed[p1.WINDOW + 10] += 500.0  # inside the first fold's targets
    again = forecasts(p1.run(frame_of(changed), **params()))
    assert forecasts(result).iloc[:48].equals(again.iloc[:48])  # made before the spike
    assert not forecasts(result).iloc[48:].equals(again.iloc[48:])  # trained on it


def test_a_gap_excludes_only_the_folds_that_contain_it(prices: list[float]) -> None:
    gappy = list(prices)
    gappy[p1.WINDOW + 60] = math.nan  # a target of fold 2 only
    out = p1.run(frame_of(gappy), **params())
    folds = out.tables["folds"]
    assert dict(folds.groupby("status").size()) == {"missing_data": 6, "ok": 6}
    assert forecasts(out).iloc[:48].notna().all().all()
    assert forecasts(out).iloc[48:].isna().all().all()


def test_metrics_and_fitted_parameters(result: AlgorithmResult) -> None:
    metrics = result.tables["metrics"].set_index("model")
    assert list(metrics.index) == list(p1.MODELS)
    assert (metrics["n_ok_folds"] == FOLDS).all()
    assert metrics.at["discount_dlm", "interval_method"] == "student-t"
    assert metrics.at["ensemble", "interval_method"] == "quantile-averaging"

    fitted = result.tables["fold_params"].groupby("model")["parameter"].unique()
    assert {"alpha", "drift"} <= set(fitted["theta"])
    assert {"ar.L1", "ar.L2", "sigma2"} <= set(fitted["ar"])
    assert "observation_variance" in set(fitted["discount_dlm"])
    assert {"variance.level", "variance.irregular", "log_likelihood"} <= set(fitted["structural"])
    assert "seasonal_naive" not in fitted


def test_dm_tests_with_too_few_folds_explain_why(result: AlgorithmResult) -> None:
    dm = result.tables["dm_tests"]
    assert list(dm.columns) == p1.DM_COLUMNS
    assert len(dm) == 2 * (len(p1.MODELS) - 1)  # two leads, every model but the benchmark
    assert set(dm["benchmark"]) == {"seasonal_naive"}
    assert list(dm["horizon_step"].unique()) == [1, 48]
    assert list(dm["lead_hours"].unique()) == [0.5, 24.0]
    assert (dm["n"] == FOLDS).all()
    assert (dm["status"] == "insufficient_data").all()
    assert dm["mean_loss_model"].notna().all()


def dm_case(folds: int, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    """Forecasts in fold order: the benchmark is off by 10 everywhere and the theta by 1; the
    other models equal the benchmark, with noise so that their differential is not constant."""
    rng = np.random.default_rng(0)
    n = folds * horizon
    index = pd.date_range("2026-09-29", periods=n, freq="30min", tz="UTC")
    actual = pd.Series(rng.normal(80, 20, n), index=index)
    sign = rng.choice([-1.0, 1.0], n)
    out = pd.DataFrame(index=index)
    for name in p1.MODELS:
        out[f"{name}_forecast"] = actual + 10 * sign + rng.normal(0, 1, n)
    out["theta_forecast"] = actual + sign
    return out, actual


def test_dm_tests_align_errors_by_lead_and_sign_them_for_the_model() -> None:
    out, actual = dm_case(folds=40, horizon=4)
    out.iloc[3::4, out.columns.get_loc("theta_forecast")] = np.nan  # an excluded lead-4 target
    dm = p1.dm_tests(
        out, actual, horizon=4, benchmark="seasonal_naive", leads=[1, 4], loss="absolute"
    ).set_index(["model", "horizon_step"])
    theta = dm.loc[("theta", 1)]
    assert theta["status"] == "ok"
    assert theta["n"] == 40
    assert theta["statistic"] > 0  # the model was the more accurate
    assert theta["p_value"] < 0.001
    assert theta["mean_loss_model"] == pytest.approx(1.0)
    assert dm.loc[("theta", 4), "n"] == 0
    assert dm.loc[("theta", 4), "status"] == "insufficient_data"
    assert dm.loc[("ar", 1), "status"] == "ok"
    assert dm.loc[("ar", 1), "p_value"] > 0.01


def test_dm_tests_use_the_squared_loss_when_asked() -> None:
    out, actual = dm_case(folds=40, horizon=2)
    dm = p1.dm_tests(
        out, actual, horizon=2, benchmark="seasonal_naive", leads=[1], loss="squared"
    ).set_index("model")
    assert dm.at["theta", "mean_loss_model"] == pytest.approx(1.0)
    assert dm.at["theta", "mean_loss_benchmark"] == pytest.approx(101.0, rel=0.05)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"models": {"theta": {}}}, "models must configure"),
        ({"models": models(ensemble={"members": ["theta", "ensemble"]})}, "distinct models"),
        ({"models": models(ensemble={"members": ["theta", "theta"]})}, "distinct models"),
        ({"models": models(ensemble={"members": "theta"})}, "list of model names"),
        ({"models": models(ensemble={"members": ["theta"]})}, "two or more members"),
        ({"dm_benchmark": "naive"}, "dm_benchmark must be"),
        ({"dm_leads": [49]}, "dm_leads must be"),
        ({"dm_loss": "pinball"}, "dm_loss must be"),
        ({"max_folds": FOLDS - 1}, "exceed max_folds"),
    ],
)
def test_settings_are_checked(prices: list[float], overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(InvalidInputError, match=match):
        p1.run(frame_of(prices), **params(**overrides))
