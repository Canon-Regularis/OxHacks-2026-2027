"""Source-data endpoints: ticker snapshot, chart series, top/bottom periods and period detail."""

from datetime import date, datetime
from typing import Literal

import pandas as pd
from fastapi import APIRouter, Depends, Query
from psycopg import Connection

from gridpulse.analysis import service
from gridpulse.analysis.algorithms import ALGORITHMS
from gridpulse.analysis.features import build_input_frame
from gridpulse.api.deps import DateRange, api_error, as_utc, choose_resolution, date_range, get_conn
from gridpulse.api.schemas import (
    Column,
    Extreme,
    ExtremesResponse,
    PeriodDetail,
    PeriodIndicator,
    SeriesInfo,
    SeriesResponse,
    Snapshot,
    TickerItem,
    Vintage,
)
from gridpulse.catalog import FORECAST_VINTAGES, SERIES, SNAPSHOT_SERIES, series_info
from gridpulse.db import queries
from gridpulse.settlement import PERIOD, bucket_starts, period_start_utc, settlement_period_of

router = APIRouter(tags=["market"])


def _series_ids(ids: str) -> list[str]:
    series_ids = [s.strip() for s in ids.split(",") if s.strip()]
    unknown = [s for s in series_ids if s not in SERIES]
    if unknown or not series_ids:
        raise api_error(400, "unknown_series", f"Unknown series {unknown}; choose from {list(SERIES)}.")
    return series_ids


def _float(value) -> float | None:
    return None if pd.isna(value) else float(value)


def _floats(values: pd.Series) -> list[float | None]:
    return [_float(v) for v in values]


@router.get("/snapshot", response_model=Snapshot)
def snapshot(
    at: datetime | None = Query(None, description="Replay the ticker as of this instant (default: latest data)"),
    conn: Connection = Depends(get_conn),
) -> Snapshot:
    """Latest settled period and its change from the previous period, for the ticker strip."""
    as_of = queries.latest_price_bucket(conn, as_utc(at) if at else None)
    if as_of is None:
        raise api_error(404, "no_data", "No settlement prices are loaded (gridpulse.price_30m is empty).")
    previous = as_of - PERIOD
    frame = build_input_frame(conn, previous, as_of + PERIOD)  # both periods, every series, one query

    items = []
    for sid in SNAPSHOT_SERIES:
        series = SERIES[sid]
        value, prev = _float(frame.at[as_of, series.column]), _float(frame.at[previous, series.column])
        change = None if value is None or prev is None else value - prev
        items.append(TickerItem(id=sid, label=series.label, unit=series.unit, value=value, previous=prev, change=change))

    settlement_date, settlement_period = settlement_period_of(as_of)
    return Snapshot(
        as_of_utc=as_of,
        settlement_date=settlement_date,
        settlement_period=settlement_period,
        items=items,
        data_updated_at_utc=queries.data_updated_at(conn),
    )


@router.get("/series", response_model=SeriesResponse)
def series(
    dates: DateRange = Depends(date_range),
    ids: str = Query("price,wind,demand", description="Comma-separated series ids from /api/catalog"),
    resolution: Literal["auto", "30m", "1h", "1d"] = "auto",
    cutoff: datetime | None = Query(None, description="Only forecasts published by this instant (30m only)"),
    conn: Connection = Depends(get_conn),
) -> SeriesResponse:
    """Several series on one shared time axis, downsampled in the database to fit max_points."""
    series_ids = _series_ids(ids)
    res = choose_resolution(dates, resolution)

    if cutoff is None:
        t = bucket_starts(dates.start_utc, dates.end_utc, res)
        by_series = queries.series_values(
            conn, {sid: SERIES[sid] for sid in series_ids}, res, dates.start_utc, dates.end_utc
        )
        values = {sid: [by_bucket.get(bucket) for bucket in t] for sid, by_bucket in by_series.items()}
    else:
        # What was known at the cutoff, built the same way as algorithm inputs.
        if res != "30m":
            raise api_error(400, "cutoff_needs_30m", "cutoff works at 30m resolution; narrow the date range.")
        frame = build_input_frame(conn, dates.start_utc, dates.end_utc, as_utc(cutoff))
        unsupported = [sid for sid in series_ids if SERIES[sid].column not in frame.columns]
        if unsupported:
            raise api_error(400, "cutoff_unsupported", f"Series {unsupported} are not available with cutoff.")
        t = list(frame.index.to_pydatetime())
        values = {sid: _floats(frame[SERIES[sid].column]) for sid in series_ids}

    periods = [settlement_period_of(bucket) for bucket in t] if res == "30m" else None
    return SeriesResponse(
        from_=dates.from_date,
        to=dates.to_date,
        resolution=res,
        cutoff_utc=as_utc(cutoff) if cutoff else None,
        t=t,
        settlement_date=[d for d, _ in periods] if periods else None,
        settlement_period=[p for _, p in periods] if periods else None,
        series={
            sid: Column(label=SERIES[sid].label, unit=SERIES[sid].unit, values=values[sid]) for sid in series_ids
        },
    )


@router.get("/extremes", response_model=ExtremesResponse)
def extremes(
    dates: DateRange = Depends(date_range),
    series: str = Query("price", description="Series id from /api/catalog"),
    order: Literal["desc", "asc"] = "desc",
    limit: int = Query(10, ge=1, le=100),
    conn: Connection = Depends(get_conn),
) -> ExtremesResponse:
    """Highest or lowest settlement periods in the range: a screener for finding events."""
    if series not in SERIES:
        raise api_error(400, "unknown_series", f"Unknown series {series!r}; choose from {list(SERIES)}.")
    info = SERIES[series]
    rows = []
    for bucket, value in queries.extremes(conn, info, dates.start_utc, dates.end_utc, limit, order == "desc"):
        settlement_date, settlement_period = settlement_period_of(bucket)
        rows.append(Extreme(delivery_start_utc=bucket, settlement_date=settlement_date,
                            settlement_period=settlement_period, value=value))
    return ExtremesResponse(
        series=SeriesInfo(**series_info(series)),
        from_=dates.from_date,
        to=dates.to_date,
        order=order,
        rows=rows,
    )


@router.get("/periods/{settlement_date}/{settlement_period}", response_model=PeriodDetail)
def period(settlement_date: date, settlement_period: int, conn: Connection = Depends(get_conn)) -> PeriodDetail:
    """Everything about one settlement period: values, forecast history and analysis outputs."""
    try:
        start = period_start_utc(settlement_date, settlement_period)
    except ValueError as exc:
        raise api_error(404, "no_such_period", str(exc)) from exc

    row = build_input_frame(conn, start, start + PERIOD).iloc[0]
    values = {sid: _float(row[s.column]) for sid, s in SERIES.items()}
    vintages = {
        name: [Vintage(published_at_utc=p, value=v) for p, v in queries.forecast_vintages(conn, table, column, start)]
        for name, (table, column) in FORECAST_VINTAGES.items()
    }
    analysis = []
    for row in service.period_results(conn, list(ALGORITHMS.values()), start):
        algorithm = ALGORITHMS.get(row["algorithm_id"])
        unit = algorithm.outputs.get(row["indicator"]) if algorithm else None
        analysis.append(PeriodIndicator(**row, unit=unit))

    return PeriodDetail(
        settlement_date=settlement_date,
        settlement_period=settlement_period,
        delivery_start_utc=start,
        delivery_end_utc=start + PERIOD,
        values=values,
        vintages=vintages,
        analysis=analysis,
    )
