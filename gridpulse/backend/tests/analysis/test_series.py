import math
from collections.abc import Callable
from datetime import UTC, date, datetime

import pytest

from gridpulse.analysis.algorithms.errors import InvalidInputError, MissingDataError
from gridpulse.analysis.algorithms.series import (
    HALF_HOUR,
    LONDON,
    HalfHourlySeries,
    crosses_clock_change,
)

START = datetime(2026, 10, 1, tzinfo=UTC)
DayStarts = Callable[[date], list[datetime]]


def test_start_is_normalised_to_utc() -> None:
    series = HalfHourlySeries(datetime(2026, 7, 1, 1, 30, tzinfo=LONDON), [1.0])
    assert series.start == datetime(2026, 7, 1, 0, 30, tzinfo=UTC)
    assert series.start.utcoffset() == UTC.utcoffset(None)


@pytest.mark.parametrize(
    "start",
    [
        datetime(2026, 10, 1),  # naive
        datetime(2026, 10, 1, 0, 15, tzinfo=UTC),
        datetime(2026, 10, 1, 0, 0, 1, tzinfo=UTC),
    ],
)
def test_rejects_naive_or_misaligned_start(start: datetime) -> None:
    with pytest.raises(InvalidInputError):
        HalfHourlySeries(start, [1.0])


def test_nan_is_missing_and_numbers_become_floats() -> None:
    values = HalfHourlySeries(START, [1, math.nan, None, -2.5]).values
    assert values == (1.0, None, None, -2.5)
    assert [type(value) for value in values] == [float, type(None), type(None), float]


@pytest.mark.parametrize("bad", [True, "1.0", math.inf, -math.inf, 10**400])
def test_rejects_non_numeric_and_infinite_values(bad: object) -> None:
    with pytest.raises(InvalidInputError):
        HalfHourlySeries(START, [1.0, bad])  # type: ignore[list-item]


@pytest.mark.parametrize(("day", "periods"), [(date(2026, 3, 29), 46), (date(2026, 10, 25), 50)])
def test_grid_spans_clock_change_days(
    settlement_day_starts: DayStarts, day: date, periods: int
) -> None:
    starts = settlement_day_starts(day)
    series = HalfHourlySeries(starts[0], [0.0] * len(starts))
    assert len(series) == periods
    assert series.timestamps() == tuple(starts)
    assert series.end.astimezone(LONDON).hour == 0


def test_window_keeps_timestamps() -> None:
    series = HalfHourlySeries(START, [0.0, 1.0, 2.0, 3.0])
    window = series.window(1, 3)
    assert (window.start, window.values) == (START + HALF_HOUR, (1.0, 2.0))
    with pytest.raises(InvalidInputError):
        series.window(2, 5)


def test_complete_values_reports_missing_indices() -> None:
    with pytest.raises(MissingDataError) as excinfo:
        HalfHourlySeries(START, [1.0, None, 3.0, math.nan]).complete_values()
    assert excinfo.value.indices == (1, 3)


@pytest.mark.parametrize(
    ("start", "stop", "expected"),
    [
        (datetime(2026, 10, 24, 22, tzinfo=UTC), datetime(2026, 10, 25, 6, tzinfo=UTC), True),
        (datetime(2026, 10, 25, 1, tzinfo=UTC), datetime(2026, 11, 20, tzinfo=UTC), False),
        # The last period before the change starts at 00:30 UTC.
        (datetime(2026, 10, 24, tzinfo=UTC), datetime(2026, 10, 25, 1, tzinfo=UTC), False),
        (datetime(2026, 3, 28, tzinfo=UTC), datetime(2026, 4, 10, tzinfo=UTC), True),
        (datetime(2026, 6, 1, tzinfo=UTC), datetime(2026, 8, 31, tzinfo=UTC), False),
        # London-local instants around the repeated hour give the same answers as UTC.
        (
            datetime(2026, 10, 25, 1, tzinfo=UTC).astimezone(LONDON),
            datetime(2026, 10, 25, 2, tzinfo=UTC).astimezone(LONDON),
            False,
        ),
        (
            datetime(2026, 10, 25, tzinfo=UTC).astimezone(LONDON),
            datetime(2026, 10, 25, 1, 30, tzinfo=UTC).astimezone(LONDON),
            True,
        ),
    ],
)
def test_crosses_clock_change(start: datetime, stop: datetime, expected: bool) -> None:
    assert crosses_clock_change(start, stop) is expected


def test_crosses_clock_change_rejects_naive_datetimes() -> None:
    with pytest.raises(InvalidInputError):
        crosses_clock_change(datetime(2026, 10, 25), datetime(2026, 10, 26, tzinfo=UTC))
