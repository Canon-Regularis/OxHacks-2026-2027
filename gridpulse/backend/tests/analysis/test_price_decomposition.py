"""price_decomposition: components, seasonal indices, strengths, smoothers and the sample."""

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from gridpulse.analysis.algorithms import price_decomposition as pd_
from gridpulse.analysis.algorithms.errors import InvalidInputError
from gridpulse.analysis.contract import AlgorithmResult
from gridpulse.analysis.service import check_output, check_tables

DAYS = 14


def frame_of(values: list[float]) -> pd.DataFrame:
    index = pd.date_range("2026-09-01", periods=len(values), freq="30min", tz="UTC")
    return pd.DataFrame({"price_gbp_mwh": values}, index=index)


def params(**overrides: Any) -> dict[str, Any]:
    return {**pd_.ALGORITHM.params, **overrides}


@pytest.fixture(scope="module")
def prices() -> list[float]:
    rng = np.random.default_rng(3)
    t = np.arange(48 * DAYS)
    return list(80 + 30 * np.sin(2 * np.pi * t / 48) + rng.normal(0, 5, len(t)))


@pytest.fixture(scope="module")
def result(prices: list[float]) -> AlgorithmResult:
    return pd_.run(frame_of(prices), **params())


def series(result: AlgorithmResult) -> pd.DataFrame:
    assert result.series is not None
    return result.series


def test_honours_the_contract(prices: list[float], result: AlgorithmResult) -> None:
    frame = frame_of(prices)
    out = check_output(pd_.ALGORITHM, result.series, frame.index[0], frame.index[-1])
    assert list(out.columns) == list(pd_.OUTPUTS)
    stored = check_tables(pd_.ALGORITHM, result.tables)
    assert {name: table["columns"] for name, table in stored.items()} == pd_.TABLE_COLUMNS
    assert list(result.tables["methods"]["status"]) == ["ok", "ok"]


@pytest.mark.parametrize("method", pd_.METHODS)
def test_components_add_up_to_the_price(
    prices: list[float], result: AlgorithmResult, method: str
) -> None:
    out = series(result)
    total = out[f"{method}_trend"] + out[f"{method}_seasonal"] + out[f"{method}_remainder"]
    defined = total.notna()
    assert defined.sum() >= len(prices) - 48  # classical loses half a period at each end
    assert list(total[defined]) == pytest.approx(list(np.array(prices)[defined.to_numpy()]))


def test_seasonal_indices_recover_the_daily_shape(result: AlgorithmResult) -> None:
    indices = result.tables["seasonal_indices"]
    assert list(indices["phase"]) == list(range(48))
    assert indices["first_utc"].iloc[1] - indices["first_utc"].iloc[0] == pd.Timedelta("30min")
    expected = 30 * np.sin(2 * np.pi * np.arange(48) / 48)
    for method in pd_.METHODS:
        assert list(indices[f"{method}_gbp_mwh"]) == pytest.approx(list(expected), abs=3.0)


def test_strengths_show_a_strong_season_and_no_trend(result: AlgorithmResult) -> None:
    row = result.tables["strength"].iloc[0]
    assert row["status"] == "ok"
    assert row["seasonal_strength"] > 0.9
    assert row["trend_strength"] < 0.1


def test_the_median_removes_a_spike_and_smoothers_are_centred(prices: list[float]) -> None:
    spiky = list(prices)
    spiky[200] += 1000.0
    out = series(pd_.run(frame_of(spiky), **params()))
    assert abs(out[f"price_median_{pd_.MEDIAN}"].iloc[200] - prices[200]) < 30
    for name, window in (("median", pd_.MEDIAN), ("savgol", pd_.SAVGOL), ("lowess", pd_.LOWESS)):
        column = out[f"price_{name}_{window}"]
        assert column.iloc[: window // 2].isna().all(), name
        assert column.iloc[window // 2 : -(window // 2)].notna().all(), name


def test_a_gap_limits_the_decomposition_to_the_longest_stretch(prices: list[float]) -> None:
    gappy = list(prices)
    gappy[100] = math.nan
    out = pd_.run(frame_of(gappy), **params())
    sample = out.tables["sample"].iloc[0]
    assert sample["periods"] == len(prices) - 101
    assert sample["excluded_periods"] == 101
    assert series(out)["stl_trend"].iloc[:101].isna().all()
    assert (
        series(out)[f"price_median_{pd_.MEDIAN}"].iloc[:96].notna().any()
    )  # smoothers run per run


def test_max_periods_keeps_the_latest_periods(prices: list[float]) -> None:
    out = pd_.run(frame_of(prices), **params(max_periods=48 * 8))
    assert out.tables["sample"].iloc[0]["periods"] == 48 * 8
    assert series(out)["stl_seasonal"].iloc[: 48 * (DAYS - 8)].isna().all()


def test_too_little_history_reports_each_method(prices: list[float]) -> None:
    out = pd_.run(frame_of(prices[: 48 * 3]), **params())  # STL needs 7 cycles; classical 2
    methods = out.tables["methods"].set_index("method")
    assert methods.at["stl", "status"] == "insufficient_data"
    assert methods.at["classical", "status"] == "ok"
    assert series(out)["stl_trend"].isna().all()
    assert out.tables["seasonal_indices"]["stl_gbp_mwh"].isna().all()
    check_tables(pd_.ALGORITHM, out.tables)


def test_no_prices_gives_empty_components() -> None:
    out = pd_.run(frame_of([math.nan] * 96), **params())
    assert series(out).isna().all().all()
    assert (out.tables["methods"]["status"] == "insufficient_data").all()
    assert out.tables["strength"].iloc[0]["status"] == "insufficient_data"
    check_tables(pd_.ALGORITHM, out.tables)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [({"period": 1}, "period must be"), ({"max_periods": 0}, "max_periods must be")],
)
def test_settings_are_checked(prices: list[float], overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(InvalidInputError, match=match):
        pd_.run(frame_of(prices), **params(**overrides))
