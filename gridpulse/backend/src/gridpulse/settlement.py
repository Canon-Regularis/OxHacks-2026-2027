"""GB settlement-period and time-bucket arithmetic.

A settlement day is a Europe/London calendar day split into 30-minute periods numbered from 1.
Clock changes give 46 (March) or 50 (October) periods, so never assume 48. All instants are UTC.
"""

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

LONDON = ZoneInfo("Europe/London")
PERIOD = timedelta(minutes=30)

# Bucket widths the API serves. Must match the {family}_{resolution} relations in the database.
RESOLUTIONS = {
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "1d": timedelta(days=1),
}


def day_start_utc(settlement_date: date) -> datetime:
    """UTC instant of the local midnight that starts a settlement day."""
    return datetime.combine(settlement_date, time(0), tzinfo=LONDON).astimezone(UTC)


def range_utc(from_date: date, to_date: date) -> tuple[datetime, datetime]:
    """Half-open UTC interval covering settlement days from_date..to_date inclusive."""
    return day_start_utc(from_date), day_start_utc(to_date + timedelta(days=1))


def periods_in_day(settlement_date: date) -> int:
    start, end = range_utc(settlement_date, settlement_date)
    return (end - start) // PERIOD


def period_start_utc(settlement_date: date, settlement_period: int) -> datetime:
    if not 1 <= settlement_period <= periods_in_day(settlement_date):
        raise ValueError(f"{settlement_date} has no settlement period {settlement_period}")
    return day_start_utc(settlement_date) + (settlement_period - 1) * PERIOD


def settlement_period_of(instant: datetime) -> tuple[date, int]:
    """(settlement_date, settlement_period) containing a UTC instant."""
    instant = instant.astimezone(UTC)
    settlement_date = instant.astimezone(LONDON).date()
    return settlement_date, (instant - day_start_utc(settlement_date)) // PERIOD + 1


def bucket_starts(start: datetime, end: datetime, resolution: str) -> list[datetime]:
    """Bucket start instants in [start, end), matching the database's time_bucket output.

    Daily buckets are London settlement days (23, 24 or 25 hours long); the others are uniform.
    """
    start, end = start.astimezone(UTC), end.astimezone(UTC)
    if resolution == "1d":
        starts = []
        day = start.astimezone(LONDON).date()
        while (day_start := day_start_utc(day)) < end:
            if day_start >= start:
                starts.append(day_start)
            day += timedelta(days=1)
        return starts

    step = RESOLUTIONS[resolution]
    return [start + i * step for i in range((end - start) // step)]
