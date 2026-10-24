"""Shared request handling: database connections, date ranges, resolution choice and errors."""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime

from fastapi import HTTPException, Query, Request
from psycopg import Connection

from gridpulse.config import get_settings
from gridpulse.settlement import RESOLUTIONS, range_utc


def api_error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def get_conn(request: Request) -> Iterator[Connection]:
    pool = request.app.state.pool
    if pool is None:
        raise api_error(503, "database_unavailable", "No database is configured (set DATABASE_URL).")
    with pool.connection(timeout=5) as conn:  # PoolTimeout becomes a 503 in main.py
        yield conn


@dataclass(frozen=True)
class DateRange:
    from_date: date
    to_date: date
    start_utc: datetime
    end_utc: datetime


def date_range(
    from_date: date = Query(alias="from", description="First settlement date, inclusive"),
    to_date: date = Query(alias="to", description="Last settlement date, inclusive"),
) -> DateRange:
    if to_date < from_date:
        raise api_error(400, "invalid_range", "`to` must be on or after `from`.")
    max_days = get_settings().max_range_days
    if (to_date - from_date).days + 1 > max_days:
        raise api_error(400, "range_too_large", f"Ranges are limited to {max_days} days.")
    start, end = range_utc(from_date, to_date)
    return DateRange(from_date, to_date, start, end)


def choose_resolution(dates: DateRange, requested: str) -> str:
    """Finest resolution that fits within max_points, or the requested one if it fits."""
    max_points = get_settings().max_points
    span = dates.end_utc - dates.start_utc
    if requested != "auto":
        if span / RESOLUTIONS[requested] > max_points:
            raise api_error(400, "too_many_points", f"Over {max_points} points; use a coarser resolution or auto.")
        return requested
    return next((r for r in ("30m", "1h") if span / RESOLUTIONS[r] <= max_points), "1d")


def as_utc(instant: datetime) -> datetime:
    if instant.tzinfo is None:
        raise api_error(400, "naive_datetime", "Datetimes need an explicit offset, e.g. 2026-10-24T12:00Z.")
    return instant.astimezone(UTC)
