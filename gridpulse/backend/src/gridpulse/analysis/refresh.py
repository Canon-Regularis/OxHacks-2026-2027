"""Keeps every algorithm's served run current: `gridpulse refresh`, run on a schedule.

Each algorithm has a standard window ending at the latest complete settlement day of prices. An
algorithm is rerun when its served run does not cover that window, was made before the data it
read was loaded again, or predates load-time recording. Otherwise it is left alone, so a
refresh after no new data costs a few queries.
"""

import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta

from psycopg import Connection

from gridpulse.analysis import service
from gridpulse.analysis.contract import Algorithm
from gridpulse.db import queries
from gridpulse.settlement import PERIOD, day_start_utc, range_utc, settlement_period_of

# Algorithm id -> (settlement days up to and including the latest complete day, days after it).
# The backtests stay within max_folds (60); the day-ahead forecast needs only the last day.
WINDOWS: dict[str, tuple[int, int]] = {
    "wind_price_correlation": (120, 0),
    "price_naive_backtest": (120, 0),
    "price_forecast_backtest": (56, 0),
    "price_baseline_backtest": (56, 0),
    "price_regression_backtest": (56, 0),
    "price_p1_backtest": (56, 0),
    "price_day_ahead_forecast": (1, 0),
    "price_rolling_signals": (120, 0),
    "price_driver_correlograms": (56, 0),
    "price_residual_diagnostics": (56, 0),
    "price_decomposition": (56, 0),
    "price_stationarity": (56, 0),
    "calendar_fourier_terms": (56, 7),  # from the clock, so it can run ahead of the data
}


@dataclass(frozen=True)
class Outcome:
    algorithm_id: str
    from_date: date
    to_date: date
    status: str  # "current", "succeeded" or "failed"
    run_id: int | None
    detail: str
    seconds: float


def latest_complete_day(conn: Connection) -> date | None:
    """The last settlement day whose final period has a settled price."""
    latest = queries.latest_price_bucket(conn)
    if latest is None:
        return None
    day, _ = settlement_period_of(latest)
    complete = latest + PERIOD == day_start_utc(day + timedelta(days=1))
    return day if complete else day - timedelta(days=1)


def window(algorithm: Algorithm, through: date) -> tuple[date, date]:
    back, ahead = WINDOWS[algorithm.id]
    return through - timedelta(days=back - 1), through + timedelta(days=ahead)


def stale_reason(conn: Connection, algorithm: Algorithm, from_date: date, to_date: date) -> str | None:
    """Why the served run does not cover this window with current data, or None if it does."""
    start, end = range_utc(from_date, to_date)
    run = queries.find_run(conn, algorithm.id, algorithm.version, service.with_params(algorithm), start, end)
    if run is None:
        return "no run"
    if run["range_start_utc"] > start or run["range_end_utc"] < end + algorithm.horizon:
        return f"run {run['run_id']} covers {run['range_start_utc']:%Y-%m-%d} to {run['range_end_utc']:%Y-%m-%d} only"
    if run["datasets"] is None:
        return f"run {run['run_id']} predates recording data load times"
    if service.reloaded_warnings(conn, run):
        return f"data loaded again since run {run['run_id']}"
    return None


def refresh(
    conn: Connection,
    algorithms: Iterable[Algorithm],
    through: date | None = None,
    force: bool = False,
) -> list[Outcome]:
    """Bring each algorithm up to date, in order; a failure is recorded and the rest continue.
    `force` reruns every algorithm from scratch, ignoring stored memo entries."""
    through = through or latest_complete_day(conn)
    if through is None:
        raise RuntimeError("No settlement prices are loaded, so there is nothing to refresh.")
    outcomes = []
    for algorithm in algorithms:
        from_date, to_date = window(algorithm, through)
        began = time.monotonic()
        reason = "forced" if force else stale_reason(conn, algorithm, from_date, to_date)
        if reason is None:
            outcome = Outcome(algorithm.id, from_date, to_date, "current", None, "up to date", 0.0)
        else:
            start, end = range_utc(from_date, to_date)
            try:
                run_id = service.run_algorithm(conn, algorithm, start, end, reuse=not force)
                outcome = Outcome(algorithm.id, from_date, to_date, "succeeded", run_id, reason, 0.0)
            except Exception as exc:  # run_algorithm has stored the failure on its run
                detail = f"{reason}; {type(exc).__name__}: {exc}"
                outcome = Outcome(algorithm.id, from_date, to_date, "failed", None, detail, 0.0)
        outcomes.append(
            Outcome(**{**outcome.__dict__, "seconds": round(time.monotonic() - began, 1)})
        )
    return outcomes
