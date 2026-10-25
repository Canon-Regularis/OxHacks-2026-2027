"""Inputs shared by the analysis tests, and the RUNSLOW gate on slow upstream tests."""

import math
import os
import random
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from gridpulse.analysis.algorithms.series import HALF_HOUR, LONDON, HalfHourlySeries
from gridpulse.analysis.features import input_frame_from_rows

START = datetime(2026, 9, 1, tzinfo=UTC)

# Upstream state_space tests that take several seconds each, one over ten; they run with
# RUNSLOW=1, as the ported smoothing suite's slow tests do.
_SLOW_STATE_SPACE_TESTS = (
    "test_state_space.py::TestSmoother::test_smoothed_cov_nonnegative_multistate",
    "test_state_space.py::TestComponents::test_seasonal_recovers_pattern",
    "test_state_space.py::TestComponents::test_cycle_recovers_frequency",
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if os.environ.get("RUNSLOW", "").strip().lower() not in {"", "0", "false", "no", "off"}:
        return
    skip = pytest.mark.skip(reason="slow test -- set RUNSLOW=1 to run it")
    for item in items:
        if item.nodeid.endswith(_SLOW_STATE_SPACE_TESTS):
            item.add_marker(skip)


def _settlement_day_starts(day: date) -> list[datetime]:
    first = datetime(day.year, day.month, day.day, tzinfo=LONDON).astimezone(UTC)
    following = day + timedelta(days=1)
    stop = datetime(following.year, following.month, following.day, tzinfo=LONDON)
    return [first + i * HALF_HOUR for i in range((stop.astimezone(UTC) - first) // HALF_HOUR)]


def _synthetic_values(
    n: int,
    *,
    level: float = 50.0,
    amplitude: float = 10.0,
    phi: float = 0.5,
    noise: float = 2.0,
    seed: int = 0,
) -> list[float]:
    rng = random.Random(seed)
    values, shock = [], 0.0
    for t in range(n):
        shock = phi * shock + rng.gauss(0.0, noise)
        values.append(level + amplitude * math.sin(2 * math.pi * t / 48) + shock)
    return values


@pytest.fixture
def settlement_day_starts() -> Callable[[date], list[datetime]]:
    """UTC start of every settlement period of a GB settlement day (46, 48 or 50)."""
    return _settlement_day_starts


@pytest.fixture
def synthetic_values() -> Callable[..., list[float]]:
    """Daily-seasonal AR(1) values; ``level=0`` gives frequent negative prices."""
    return _synthetic_values


@pytest.fixture
def make_series() -> Callable[..., HalfHourlySeries]:
    """A complete synthetic series starting at ``START`` unless ``start`` is given."""

    def build(
        n: int,
        *,
        start: datetime = START,
        level: float = 50.0,
        amplitude: float = 10.0,
        phi: float = 0.5,
        noise: float = 2.0,
        seed: int = 0,
    ) -> HalfHourlySeries:
        values = _synthetic_values(
            n, level=level, amplitude=amplitude, phi=phi, noise=noise, seed=seed
        )
        return HalfHourlySeries(start, values)

    return build


@pytest.fixture
def input_frame() -> pd.DataFrame:
    """Five synthetic weeks shaped exactly like build_input_frame output, with some gaps.

    Long enough for the 28-day training window of price_forecast_backtest plus a week."""
    index = pd.date_range("2026-09-19", periods=48 * 35, freq="30min", tz="UTC")
    rng = np.random.default_rng(0)
    n = len(index)
    price = rng.normal(80, 40, n)
    price[::50] = np.nan  # missing periods must stay missing
    rows = zip(
        index.to_pydatetime(),
        price,
        rng.normal(0, 300, n),  # niv_mwh
        rng.uniform(1_000, 15_000, n),  # wind_mw
        rng.uniform(1_000, 15_000, n),  # wind_outturn_mw
        rng.uniform(20_000, 40_000, n),  # demand_mw
        rng.uniform(20_000, 40_000, n),  # demand_outturn_mw
        strict=True,
    )
    return input_frame_from_rows(list(rows))
