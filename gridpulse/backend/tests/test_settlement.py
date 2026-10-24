from datetime import UTC, date, datetime

import pytest

from gridpulse.settlement import (
    bucket_starts,
    period_start_utc,
    periods_in_day,
    range_utc,
    settlement_period_of,
)

CLOCKS_BACK = date(2026, 10, 25)  # 50 periods
CLOCKS_FORWARD = date(2026, 3, 29)  # 46 periods


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


@pytest.mark.parametrize("day, expected", [(date(2026, 10, 24), 48), (CLOCKS_BACK, 50), (CLOCKS_FORWARD, 46)])
def test_periods_in_day(day, expected):
    assert periods_in_day(day) == expected


def test_clocks_back_day_starts_in_bst_and_ends_in_gmt():
    assert period_start_utc(CLOCKS_BACK, 1) == utc(2026, 10, 24, 23, 0)
    assert period_start_utc(CLOCKS_BACK, 50) == utc(2026, 10, 25, 23, 30)


def test_repeated_local_hour_maps_to_distinct_periods():
    # 01:00 local happens twice on 25 Oct: once in BST (00:00 UTC), once in GMT (01:00 UTC).
    assert settlement_period_of(utc(2026, 10, 25, 0, 0)) == (CLOCKS_BACK, 3)
    assert settlement_period_of(utc(2026, 10, 25, 1, 0)) == (CLOCKS_BACK, 5)


@pytest.mark.parametrize("day", [date(2026, 10, 24), CLOCKS_BACK, CLOCKS_FORWARD])
def test_round_trip_every_period(day):
    for period in range(1, periods_in_day(day) + 1):
        assert settlement_period_of(period_start_utc(day, period)) == (day, period)


def test_invalid_period_rejected():
    with pytest.raises(ValueError):
        period_start_utc(CLOCKS_FORWARD, 47)


def test_daily_buckets_follow_london_days():
    start, end = range_utc(date(2026, 10, 24), date(2026, 10, 26))
    assert bucket_starts(start, end, "1d") == [
        utc(2026, 10, 23, 23, 0),
        utc(2026, 10, 24, 23, 0),
        utc(2026, 10, 26, 0, 0),
    ]


def test_half_hour_buckets_match_period_count():
    start, end = range_utc(CLOCKS_BACK, CLOCKS_BACK)
    assert len(bucket_starts(start, end, "30m")) == 50
    assert len(bucket_starts(start, end, "1h")) == 25
