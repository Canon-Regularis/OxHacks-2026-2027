"""All application SQL. Every read is bounded by a time range or a LIMIT.

The API reads the `gridpulse.*` views (see db/migrations/002_series_views.sql), never `raw` tables
directly, except for the loader's raw.dataset_catalog. Bucketing, latest-forecast selection and
joins happen in Tiger Data; Python receives at most a few thousand rows per call.
"""

from collections.abc import Iterable
from datetime import datetime

from psycopg import Connection, sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from gridpulse.catalog import Series
from gridpulse.settlement import RESOLUTIONS


def _dicts(conn: Connection, query, params=None) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(query, params).fetchall()


def _view(name: str) -> sql.Identifier:
    return sql.Identifier("gridpulse", name)


def _bucket(resolution: str, column: str) -> sql.Composed:
    """time_bucket expression taking %(width)s. Daily buckets follow London settlement days;
    finer buckets stay in UTC so the repeated local hour on clock-change days is not merged."""
    if resolution == "1d":
        return sql.SQL("time_bucket(%(width)s, {}, 'Europe/London')").format(sql.Identifier(column))
    return sql.SQL("time_bucket(%(width)s, {})").format(sql.Identifier(column))


# --- Source series ----------------------------------------------------------------------------


def series_values(
    conn: Connection, series: Series, resolution: str, start: datetime, end: datetime
) -> dict[datetime, float | None]:
    query = sql.SQL(
        "SELECT {bucket}, {agg}({column}) FROM {view}"
        " WHERE bucket >= %(start)s AND bucket < %(end)s GROUP BY 1"
    ).format(
        bucket=_bucket(resolution, "bucket"),
        agg=sql.SQL(series.agg),
        column=sql.Identifier(series.column),
        view=_view(series.relation),
    )
    params = {"width": RESOLUTIONS[resolution], "start": start, "end": end}
    return dict(conn.execute(query, params).fetchall())


def extremes(
    conn: Connection, series: Series, start: datetime, end: datetime, limit: int, descending: bool
) -> list[tuple[datetime, float]]:
    """Settlement periods with the highest (or lowest) values of a series."""
    query = sql.SQL(
        "SELECT bucket, {column} FROM {view}"
        " WHERE bucket >= %s AND bucket < %s AND {column} IS NOT NULL"
        " ORDER BY {column} {direction} LIMIT %s"
    ).format(
        column=sql.Identifier(series.column),
        view=_view(series.relation),
        direction=sql.SQL("DESC" if descending else "ASC"),
    )
    return conn.execute(query, (start, end, limit)).fetchall()


def latest_price_bucket(conn: Connection, at: datetime | None = None) -> datetime | None:
    """Most recent settlement period with a price, optionally no later than `at`."""
    row = conn.execute(
        "SELECT max(bucket) FROM gridpulse.price_30m"
        " WHERE price_gbp_mwh IS NOT NULL AND bucket <= coalesce(%s::timestamptz, 'infinity')",
        (at,),
    ).fetchone()
    return row[0]


def forecast_vintages(
    conn: Connection, view: str, column: str, delivery_start: datetime, limit: int = 500
) -> list[tuple[datetime, float | None]]:
    """Every stored published version of a forecast for one period, oldest first."""
    query = sql.SQL(
        "SELECT published_at_utc, {column} FROM {view}"
        " WHERE delivery_start_utc = %s ORDER BY published_at_utc DESC LIMIT %s"
    ).format(column=sql.Identifier(column), view=_view(view))
    return conn.execute(query, (delivery_start, limit)).fetchall()[::-1]


# --- Algorithm inputs -------------------------------------------------------------------------

INPUT_ROW_COLUMNS = (
    "delivery_start_utc",
    "price_gbp_mwh",
    "niv_mwh",
    "wind_mw",
    "wind_outturn_mw",
    "demand_mw",
    "demand_outturn_mw",
)

_GRID = (
    "generate_series(%(start)s::timestamptz, %(end)s::timestamptz - INTERVAL '30 minutes',"
    " INTERVAL '30 minutes') AS grid(t)"
)

# Latest pre-delivery forecasts, and every price and outturn, for each period in [start, end).
_LATEST_INPUTS = f"""
SELECT grid.t, p.price_gbp_mwh, p.niv_mwh, w.wind_mw, wo.wind_outturn_mw, d.demand_mw, dout.demand_outturn_mw
FROM {_GRID}
LEFT JOIN gridpulse.price_30m          p    ON p.bucket = grid.t    AND p.bucket >= %(start)s    AND p.bucket < %(end)s
LEFT JOIN gridpulse.wind_30m           w    ON w.bucket = grid.t    AND w.bucket >= %(start)s    AND w.bucket < %(end)s
LEFT JOIN gridpulse.wind_outturn_30m   wo   ON wo.bucket = grid.t   AND wo.bucket >= %(start)s   AND wo.bucket < %(end)s
LEFT JOIN gridpulse.demand_30m         d    ON d.bucket = grid.t    AND d.bucket >= %(start)s    AND d.bucket < %(end)s
LEFT JOIN gridpulse.demand_outturn_30m dout ON dout.bucket = grid.t AND dout.bucket >= %(start)s AND dout.bucket < %(end)s
ORDER BY grid.t
"""

# What was known at `cutoff`: forecasts published by then (and, as in the *_30m views, before the
# period started), and prices/outturns for periods that had ended by then. Publication delays of
# actuals (e.g. NESO demand, ~21 days) are not modelled.
_AS_OF_INPUTS = f"""
WITH wind AS (
    SELECT delivery_start_utc AS t, last(wind_mw, published_at_utc) AS wind_mw
    FROM gridpulse.wind_forecast_vintages
    WHERE delivery_start_utc >= %(start)s AND delivery_start_utc < %(end)s
      AND published_at_utc <= %(cutoff)s AND published_at_utc <= delivery_start_utc
    GROUP BY 1
),
demand AS (
    SELECT delivery_start_utc AS t, last(demand_mw, published_at_utc) AS demand_mw
    FROM gridpulse.demand_forecast_vintages
    WHERE delivery_start_utc >= %(start)s AND delivery_start_utc < %(end)s
      AND published_at_utc <= %(cutoff)s AND published_at_utc <= delivery_start_utc
    GROUP BY 1
),
ended AS (
    SELECT t FROM {_GRID} WHERE t + INTERVAL '30 minutes' <= %(cutoff)s
)
SELECT grid.t, p.price_gbp_mwh, p.niv_mwh, wind.wind_mw, wo.wind_outturn_mw, demand.demand_mw, dout.demand_outturn_mw
FROM {_GRID}
LEFT JOIN wind   ON wind.t = grid.t
LEFT JOIN demand ON demand.t = grid.t
LEFT JOIN ended  ON ended.t = grid.t
LEFT JOIN gridpulse.price_30m          p    ON p.bucket = ended.t    AND p.bucket >= %(start)s    AND p.bucket < %(end)s
LEFT JOIN gridpulse.wind_outturn_30m   wo   ON wo.bucket = ended.t   AND wo.bucket >= %(start)s   AND wo.bucket < %(end)s
LEFT JOIN gridpulse.demand_outturn_30m dout ON dout.bucket = ended.t AND dout.bucket >= %(start)s AND dout.bucket < %(end)s
ORDER BY grid.t
"""


def input_rows(conn: Connection, start: datetime, end: datetime, cutoff: datetime | None = None) -> list[tuple]:
    """One row per settlement period in [start, end), columns as INPUT_ROW_COLUMNS."""
    if cutoff is None:
        return conn.execute(_LATEST_INPUTS, {"start": start, "end": end}).fetchall()
    return conn.execute(_AS_OF_INPUTS, {"start": start, "end": end, "cutoff": cutoff}).fetchall()


# --- Analysis runs and results ----------------------------------------------------------------

_RUN_COLUMNS = (
    "run_id, algorithm_id, algorithm_version, params,"
    " range_start_utc, range_end_utc, cutoff_utc, finished_at_utc"
)


def create_run(
    conn: Connection,
    algorithm_id: str,
    version: str,
    params: dict,
    start: datetime,
    end: datetime,
    cutoff: datetime | None,
) -> int:
    row = conn.execute(
        "INSERT INTO gridpulse.analysis_runs"
        " (algorithm_id, algorithm_version, params, range_start_utc, range_end_utc, cutoff_utc, status)"
        " VALUES (%s, %s, %s, %s, %s, %s, 'running') RETURNING run_id",
        (algorithm_id, version, Jsonb(params), start, end, cutoff),
    ).fetchone()
    return row[0]


def finish_run(
    conn: Connection, run_id: int, status: str, result_rows: int | None = None, error: str | None = None
) -> None:
    conn.execute(
        "UPDATE gridpulse.analysis_runs SET status = %s, result_rows = %s, error = %s, finished_at_utc = now()"
        " WHERE run_id = %s",
        (status, result_rows, error, run_id),
    )


def insert_results(conn: Connection, rows: Iterable[tuple[int, datetime, str, float | None]]) -> int:
    """Bulk-load (run_id, delivery_start_utc, indicator, value) rows with COPY."""
    count = 0
    with conn.cursor() as cur:
        with cur.copy(
            "COPY gridpulse.analysis_results (run_id, delivery_start_utc, indicator, value) FROM STDIN"
        ) as copy:
            for row in rows:
                copy.write_row(row)
                count += 1
    return count


def insert_table(conn: Connection, run_id: int, name: str, data: dict) -> None:
    """Store one result table, {"columns": [...], "rows": [[...], ...]}."""
    conn.execute(
        "INSERT INTO gridpulse.analysis_tables (run_id, name, row_count, data) VALUES (%s, %s, %s, %s)",
        (run_id, name, len(data["rows"]), Jsonb(data)),
    )


def table_names(conn: Connection, run_id: int) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM gridpulse.analysis_tables WHERE run_id = %s ORDER BY name", (run_id,)
    ).fetchall()
    return [name for (name,) in rows]


def get_tables(conn: Connection, run_id: int, name: str | None = None) -> dict[str, dict]:
    """Stored tables of one run (or just `name`), keyed by table name."""
    rows = conn.execute(
        "SELECT name, data FROM gridpulse.analysis_tables"
        " WHERE run_id = %s AND (%s::text IS NULL OR name = %s) ORDER BY name",
        (run_id, name, name),
    ).fetchall()
    return dict(rows)


def find_run(conn: Connection, algorithm_id: str, version: str, start: datetime, end: datetime) -> dict | None:
    """Latest successful retrospective run of this algorithm version overlapping [start, end).

    Runs with full coverage come first. Cutoff runs are only served when asked for by run id.
    """
    rows = _dicts(
        conn,
        f"SELECT {_RUN_COLUMNS} FROM gridpulse.analysis_runs"
        " WHERE algorithm_id = %(id)s AND algorithm_version = %(version)s AND status = 'succeeded'"
        "   AND cutoff_utc IS NULL AND range_start_utc < %(end)s AND range_end_utc > %(start)s"
        " ORDER BY (range_start_utc <= %(start)s AND range_end_utc >= %(end)s) DESC, finished_at_utc DESC"
        " LIMIT 1",
        {"id": algorithm_id, "version": version, "start": start, "end": end},
    )
    return rows[0] if rows else None


def get_run(conn: Connection, run_id: int) -> dict | None:
    rows = _dicts(
        conn,
        f"SELECT {_RUN_COLUMNS} FROM gridpulse.analysis_runs WHERE run_id = %s AND status = 'succeeded'",
        (run_id,),
    )
    return rows[0] if rows else None


def latest_runs(conn: Connection) -> list[dict]:
    """Most recent successful retrospective run of each algorithm (what /api/analysis serves)."""
    return _dicts(
        conn,
        f"SELECT DISTINCT ON (algorithm_id) {_RUN_COLUMNS} FROM gridpulse.analysis_runs"
        " WHERE status = 'succeeded' AND cutoff_utc IS NULL ORDER BY algorithm_id, finished_at_utc DESC",
    )


def result_buckets(
    conn: Connection, run_id: int, start: datetime, end: datetime, resolution: str
) -> list[tuple[datetime, str, float | None]]:
    """(bucket, indicator, mean value) for one run, bucketed like the source series."""
    query = sql.SQL(
        "SELECT {bucket} AS bucket, indicator, avg(value) FROM gridpulse.analysis_results"
        " WHERE run_id = %(run_id)s AND delivery_start_utc >= %(start)s AND delivery_start_utc < %(end)s"
        " GROUP BY 1, 2 ORDER BY 1"
    ).format(bucket=_bucket(resolution, "delivery_start_utc"))
    params = {"width": RESOLUTIONS[resolution], "run_id": run_id, "start": start, "end": end}
    return conn.execute(query, params).fetchall()


def period_results(conn: Connection, delivery_start: datetime) -> list[dict]:
    """Latest successful value of every (algorithm, indicator) for one settlement period."""
    return _dicts(
        conn,
        "SELECT DISTINCT ON (r.algorithm_id, res.indicator)"
        "       r.algorithm_id, r.algorithm_version, r.run_id, res.indicator, res.value"
        " FROM gridpulse.analysis_results res JOIN gridpulse.analysis_runs r ON r.run_id = res.run_id"
        " WHERE res.delivery_start_utc = %s AND r.status = 'succeeded'"
        " ORDER BY r.algorithm_id, res.indicator, r.finished_at_utc DESC",
        (delivery_start,),
    )


# --- Loaded datasets --------------------------------------------------------------------------


def coverage(conn: Connection) -> list[dict]:
    """Datasets in `raw`, as recorded by the data loader in raw.dataset_catalog."""
    return _dicts(
        conn,
        "SELECT table_name AS dataset, source, notes, first_time AS first_utc, last_time AS last_utc,"
        "       loaded_at AS loaded_at_utc,"
        "       approximate_row_count(to_regclass('raw.' || quote_ident(table_name))) AS approx_rows"
        " FROM raw.dataset_catalog ORDER BY table_name",
    )


def data_updated_at(conn: Connection) -> datetime | None:
    return conn.execute("SELECT max(loaded_at) FROM raw.dataset_catalog").fetchone()[0]
