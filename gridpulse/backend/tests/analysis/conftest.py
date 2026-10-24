"""Series builders shared by the adapter tests."""

import math
import random
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

import pytest

from gridpulse.analysis.algorithms.series import HALF_HOUR, LONDON, HalfHourlySeries

START = datetime(2026, 9, 1, tzinfo=UTC)


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
