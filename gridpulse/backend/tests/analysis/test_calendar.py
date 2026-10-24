from collections.abc import Callable
from datetime import UTC, date, datetime

import pytest

from gridpulse.analysis.algorithms.calendar import fourier_terms
from gridpulse.analysis.algorithms.errors import InvalidInputError
from gridpulse.analysis.algorithms.orchid import forecasting_baseline as fb
from gridpulse.analysis.algorithms.series import LONDON

DayStarts = Callable[[date], list[datetime]]


def _daily_row(slot: int, harmonics: int) -> list[float]:
    row: list[float] = fb.fourier_features(1, 48, harmonics, start=slot)[0]
    return row


def test_rows_are_orchid_features_at_the_local_clock_slot(
    settlement_day_starts: DayStarts,
) -> None:
    starts = settlement_day_starts(date(2026, 7, 15))  # BST: local time is UTC + 1
    terms = fourier_terms(starts, daily_harmonics=3, weekly_harmonics=2)
    week_start = starts[0].astimezone(LONDON).weekday() * 48
    for slot, row in enumerate(terms.rows):
        weekly: list[float] = fb.fourier_features(1, 336, 2, start=week_start + slot)[0]
        assert row == pytest.approx(_daily_row(slot, 3) + weekly, abs=1e-12)
    assert terms.columns[:2] == ("daily_sin_1", "daily_cos_1")
    assert terms.columns[-2:] == ("weekly_sin_2", "weekly_cos_2")


def test_46_period_day_skips_the_missing_local_hour(settlement_day_starts: DayStarts) -> None:
    terms = fourier_terms(settlement_day_starts(date(2026, 3, 29)), daily_harmonics=2)
    # Settlement period 3 starts at 02:00 BST: slot 4, not slot 2.
    assert terms.rows[2] == pytest.approx(_daily_row(4, 2), abs=1e-12)
    assert len(terms.rows) == 46


def test_50_period_day_repeats_the_local_hour(settlement_day_starts: DayStarts) -> None:
    rows = fourier_terms(settlement_day_starts(date(2026, 10, 25)), daily_harmonics=2).rows
    assert rows[2:4] == rows[4:6]  # 01:00 and 01:30 occur in BST, then again in GMT
    assert rows[6] == pytest.approx(_daily_row(4, 2), abs=1e-12)


def test_nyquist_sine_and_duplicated_weekly_harmonics_are_omitted() -> None:
    timestamps = [datetime(2026, 10, 5, tzinfo=UTC)]
    assert len(fourier_terms(timestamps, daily_harmonics=24).columns) == 47
    columns = fourier_terms(timestamps, daily_harmonics=1, weekly_harmonics=14).columns
    assert "weekly_cos_7" not in columns
    assert "weekly_cos_14" in columns


def test_weekly_harmonic_seven_equals_daily_harmonic_one(
    settlement_day_starts: DayStarts,
) -> None:
    starts = settlement_day_starts(date(2026, 10, 7))
    weekly = fourier_terms(starts, weekly_harmonics=7)
    daily = fourier_terms(starts, daily_harmonics=1)
    position = weekly.columns.index("weekly_cos_7")
    assert [row[position] for row in weekly.rows] == pytest.approx(
        [row[1] for row in daily.rows], abs=1e-9
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"daily_harmonics": 25},
        {"weekly_harmonics": -1},
        {"daily_harmonics": True},
    ],
)
def test_rejects_invalid_harmonics(kwargs: dict[str, object]) -> None:
    with pytest.raises(InvalidInputError):
        fourier_terms([datetime(2026, 10, 5, tzinfo=UTC)], **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "timestamp", [datetime(2026, 10, 5), datetime(2026, 10, 5, 0, 10, tzinfo=UTC)]
)
def test_rejects_naive_or_misaligned_timestamps(timestamp: datetime) -> None:
    with pytest.raises(InvalidInputError):
        fourier_terms([timestamp], daily_harmonics=1)
