"""price_stationarity: unit-root rows, differencing orders, the spectrum and the sample."""

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from gridpulse.analysis.algorithms import price_stationarity as ps
from gridpulse.analysis.algorithms.errors import InvalidInputError
from gridpulse.analysis.service import check_tables

DAYS = 14


def frame_of(values: Any) -> pd.DataFrame:
    index = pd.date_range("2026-09-01", periods=len(values), freq="30min", tz="UTC")
    return pd.DataFrame({"price_gbp_mwh": list(values)}, index=index)


def params(**overrides: Any) -> dict[str, Any]:
    return {**ps.ALGORITHM.params, **overrides}


def rows(table: pd.DataFrame, series: str, test: str, regression: str = "c") -> pd.Series:
    match = table[
        (table.series == series) & (table.test == test) & (table.regression == regression)
    ]
    assert len(match) == 1
    return match.iloc[0]


@pytest.fixture(scope="module")
def walk() -> np.ndarray:
    """56 days, the refresh window: over two weeks 48 lags leave both tests little power."""
    return 80 + np.cumsum(np.random.default_rng(5).normal(0, 5, 48 * 56))


@pytest.fixture(scope="module")
def cycle() -> np.ndarray:
    t = np.arange(48 * DAYS)
    return 80 + 30 * np.sin(2 * np.pi * t / 48) + np.random.default_rng(5).normal(0, 5, len(t))


def test_honours_the_contract(cycle: np.ndarray) -> None:
    out = ps.run(frame_of(cycle), **params())
    stored = check_tables(ps.ALGORITHM, out.tables)
    assert {name: table["columns"] for name, table in stored.items()} == ps.TABLE_COLUMNS
    unit_root = out.tables["unit_root"]
    assert len(unit_root) == 3 * 2 * 2  # series x regressions x tests
    assert (unit_root["status"] == "ok").all()
    assert set(unit_root["series"]) == {"price", "price_change_1", "price_change_48"}


def test_a_random_walk_has_a_unit_root_and_its_change_does_not(walk: np.ndarray) -> None:
    table = ps.run(frame_of(walk), **params()).tables["unit_root"]
    assert rows(table, "price", "adf")["rejected_at"] == "none"
    assert rows(table, "price", "kpss")["rejected_at"] == "1%, 2.5%, 5%, 10%"
    assert rows(table, "price_change_1", "adf")["rejected_at"] == "1%, 5%, 10%"
    assert rows(table, "price_change_1", "kpss")["rejected_at"] == "none"


def test_critical_values_and_observations(cycle: np.ndarray) -> None:
    table = ps.run(frame_of(cycle), **params()).tables["unit_root"]
    adf, kpss = rows(table, "price", "adf"), rows(table, "price", "kpss")
    assert math.isnan(adf["critical_2_5"])
    assert not math.isnan(kpss["critical_2_5"])
    assert adf["critical_1"] < adf["critical_5"] < adf["critical_10"] < 0
    assert kpss["critical_1"] > kpss["critical_5"] > kpss["critical_10"] > 0
    assert kpss["n"] == 48 * DAYS
    assert rows(table, "price_change_48", "kpss")["n"] == 48 * (DAYS - 1)
    assert adf["null_hypothesis"] == "unit root"
    assert rows(table, "price", "kpss", "ct")["null_hypothesis"] == "trend stationary"


def test_difference_orders_tested_and_fixed(cycle: np.ndarray) -> None:
    orders = (
        ps.run(frame_of(cycle), **params()).tables["difference_orders"].set_index("seasonal_choice")
    )
    assert orders.at["tested", "seasonal"] == 1  # a strong daily cycle
    assert orders.at["fixed", "seasonal"] == 1
    assert (orders["status"] == "ok").all()


def test_the_spectrum_peaks_at_the_daily_cycle(cycle: np.ndarray) -> None:
    out = ps.run(frame_of(cycle), **params())
    spectrum = out.tables["spectrum"]
    assert len(spectrum) == ps.ALGORITHM.params["nfft"] // 2 + 1
    assert spectrum["frequency"].iloc[-1] == pytest.approx(0.5)
    assert math.isnan(spectrum["period_hours"].iloc[0])
    assert spectrum.loc[spectrum["density"].idxmax(), "period_hours"] == pytest.approx(24.0)
    fit = out.tables["spectrum_fit"].iloc[0]
    assert fit["status"] == "ok"
    assert fit["order"] == 48
    assert fit["innovation_variance"] > 0


def test_short_or_constant_samples_are_reported_not_raised() -> None:
    short = ps.run(frame_of([80.0 + i % 3 for i in range(20)]), **params())
    assert "needs at least" in short.tables["spectrum_fit"].iloc[0]["message"]
    assert (short.tables["unit_root"]["status"] == "insufficient_data").all()
    assert (short.tables["difference_orders"]["status"] == "insufficient_data").all()
    assert short.tables["spectrum"].empty
    assert short.tables["spectrum_fit"].iloc[0]["status"] == "insufficient_data"
    check_tables(ps.ALGORITHM, short.tables)

    medium = ps.run(frame_of([80.0 + (i * 7) % 5 for i in range(120)]), **params())
    medium_rows = medium.tables["unit_root"].set_index(["series", "test", "regression"])
    assert medium_rows.loc[("price", "kpss", "c"), "status"] == "insufficient_data"
    assert medium_rows.loc[("price", "adf", "c"), "status"] == "ok"  # 2 * 48 + 5 <= 120

    flat = ps.run(frame_of([80.0] * 480), **params())
    assert flat.tables["spectrum_fit"].iloc[0]["status"] == "degenerate_data"


def test_no_prices_gives_empty_tables() -> None:
    out = ps.run(frame_of([math.nan] * 96), **params())
    assert out.tables["sample"].iloc[0]["periods"] == 0
    assert out.tables["unit_root"].empty
    check_tables(ps.ALGORITHM, out.tables)


def test_a_gap_limits_the_tests_to_the_longest_stretch(cycle: np.ndarray) -> None:
    gappy = cycle.copy()
    gappy[100] = math.nan
    out = ps.run(frame_of(gappy), **params())
    assert out.tables["sample"].iloc[0]["periods"] == len(cycle) - 101
    assert rows(out.tables["unit_root"], "price", "kpss")["n"] == len(cycle) - 101


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"regressions": ["n"]}, "regressions must be"),
        ({"regressions": []}, "regressions must be"),
        ({"season_length": 2}, "season_length must be"),
    ],
)
def test_settings_are_checked(cycle: np.ndarray, overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(InvalidInputError, match=match):
        ps.run(frame_of(cycle), **params(**overrides))
