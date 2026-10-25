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
    conn: Connection, series: dict[str, Series], resolution: str, start: datetime, end: datetime
) -> dict[str, dict[datetime, float | None]]:
    """Several series in one query: series id -> {bucket: value}.

    Each view is joined once onto the period grid, however many series read it. Every join repeats
    the range because Postgres does not carry a range filter across a join.
    """
    relations = list(dict.fromkeys(s.relation for s in series.values()))
    alias = {relation: sql.Identifier(f"r{i}") for i, relation in enumerate(relations)}
    columns = [
        sql.SQL("{agg}({alias}.{column})").format(
            agg=sql.SQL(s.agg), alias=alias[s.relation], column=sql.Identifier(s.column)
        )
        for s in series.values()
    ]
    joins = [
        sql.SQL(
            "LEFT JOIN {view} {alias} ON {alias}.bucket = grid.t"
            " AND {alias}.bucket >= %(start)s AND {alias}.bucket < %(end)s"
        ).format(view=_view(relation), alias=alias[relation])
        for relation in relations
    ]
    query = sql.SQL("SELECT {bucket}, {columns} FROM {grid} {joins} GROUP BY 1").format(
        bucket=_bucket(resolution, "t"),
        columns=sql.SQL(", ").join(columns),
        grid=sql.SQL(_GRID),
        joins=sql.SQL(" ").join(joins),
    )
    params = {"width": RESOLUTIONS[resolution], "start": start, "end": end}
    rows = conn.execute(query, params).fetchall()
    return {sid: {row[0]: row[i] for row in rows} for i, sid in enumerate(series, start=1)}


def extremes(
    conn: Connection, series: Series, start: datetime, end: datetime, limit: int, descending: bool
) -> list[tuple[datetime, float]]:
    """Settlement periods with the highest (or lowest) values of a series."""
    query = sql.SQL(
        "SELECT bucket, {column} FROM {view}"
        " WHERE bucket >= %s AND bucket < %s AND {column} IS NOT NULL"
        " ORDER BY {column} {direction}, bucket LIMIT %s"  # both halves of a WINDFOR hour tie
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
        " WHERE delivery_start_utc = %(t)s"
        "   AND source_start_utc > %(t)s - INTERVAL '1 hour' AND source_start_utc <= %(t)s"
        " ORDER BY published_at_utc DESC LIMIT %(limit)s"
    ).format(column=sql.Identifier(column), view=_view(view))
    return conn.execute(query, {"t": delivery_start, "limit": limit}).fetchall()[::-1]


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
# actuals (e.g. NESO demand, ~21 days) are not modelled. The source_start_utc bounds let the
# vintage views skip chunks (a source row starts at most an hour before its delivery period).
_AS_OF_INPUTS = f"""
WITH wind AS (
    SELECT delivery_start_utc AS t, last(wind_mw, published_at_utc) AS wind_mw
    FROM gridpulse.wind_forecast_vintages
    WHERE delivery_start_utc >= %(start)s AND delivery_start_utc < %(end)s
      AND source_start_utc > %(start)s::timestamptz - INTERVAL '1 hour' AND source_start_utc < %(end)s
      AND published_at_utc <= %(cutoff)s AND published_at_utc <= delivery_start_utc
    GROUP BY 1
),
demand AS (
    SELECT delivery_start_utc AS t, last(demand_mw, published_at_utc) AS demand_mw
    FROM gridpulse.demand_forecast_vintages
    WHERE delivery_start_utc >= %(start)s AND delivery_start_utc < %(end)s
      AND source_start_utc > %(start)s::timestamptz - INTERVAL '1 hour' AND source_start_utc < %(end)s
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
    "run_id, algorithm_id, algorithm_version, params, datasets,"
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
    datasets: dict[str, datetime | None],
) -> int:
    loads = {name: loaded.isoformat() if loaded else None for name, loaded in datasets.items()}
    row = conn.execute(
        "INSERT INTO gridpulse.analysis_runs"
        " (algorithm_id, algorithm_version, params, datasets, range_start_utc, range_end_utc, cutoff_utc, status)"
        " VALUES (%s, %s, %s, %s, %s, %s, %s, 'running') RETURNING run_id",
        (algorithm_id, version, Jsonb(params), Jsonb(loads), start, end, cutoff),
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


MEMO_KEEP_DAYS = 7


def load_memo(conn: Connection, namespace: str) -> dict[str, str]:
    """Every stored memo entry of one namespace: key -> value."""
    rows = conn.execute("SELECT key, value FROM gridpulse.analysis_memo WHERE namespace = %s", (namespace,))
    return dict(rows.fetchall())


def save_memo(
    conn: Connection, algorithm_id: str, namespace: str, new: dict[str, str], reused: Iterable[str]
) -> None:
    """Store new entries, mark reused ones as used now, and drop entries unused for a week."""
    conn.execute(
        "UPDATE gridpulse.analysis_memo SET used_at = now() WHERE namespace = %s AND key = ANY(%s)",
        (namespace, list(reused)),
    )
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO gridpulse.analysis_memo (namespace, key, algorithm_id, value) VALUES (%s, %s, %s, %s)"
            " ON CONFLICT (namespace, key) DO UPDATE SET value = EXCLUDED.value, used_at = now()",
            [(namespace, key, algorithm_id, value) for key, value in new.items()],
        )
    conn.execute(
        "DELETE FROM gridpulse.analysis_memo WHERE used_at < now() - make_interval(days => %s)",
        (MEMO_KEEP_DAYS,),
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


def find_run(
    conn: Connection,
    algorithm_id: str,
    version: str,
    params: dict,
    start: datetime | None = None,
    end: datetime | None = None,
) -> dict | None:
    """Latest successful retrospective run of this algorithm version with exactly these params.

    Params compare as JSON values, so a run made with overrides never stands in for the defaults
    (or another variant). With a range, only runs overlapping [start, end) count and runs covering
    all of it come first. Cutoff runs are only served when asked for by run id.
    """
    rows = _dicts(
        conn,
        f"SELECT {_RUN_COLUMNS} FROM gridpulse.analysis_runs"
        " WHERE algorithm_id = %(id)s AND algorithm_version = %(version)s AND params = %(params)s"
        "   AND status = 'succeeded' AND cutoff_utc IS NULL"
        "   AND (%(start)s::timestamptz IS NULL OR (range_start_utc < %(end)s AND range_end_utc > %(start)s))"
        " ORDER BY (%(start)s::timestamptz IS NULL OR (range_start_utc <= %(start)s AND range_end_utc >= %(end)s))"
        "   DESC, finished_at_utc DESC, run_id DESC"
        " LIMIT 1",
        {"id": algorithm_id, "version": version, "params": Jsonb(params), "start": start, "end": end},
    )
    return rows[0] if rows else None


def served_runs(conn: Connection, algorithms: list[tuple[str, str, dict]]) -> dict[str, dict]:
    """find_run without a range for many algorithms in one query: algorithm id -> its latest
    successful retrospective run with exactly these params. `algorithms` is (id, version, params)."""
    if not algorithms:
        return {}
    ids, versions, params = zip(*algorithms, strict=True)
    columns = ", ".join(f"r.{column.strip()}" for column in _RUN_COLUMNS.split(","))
    rows = _dicts(
        conn,
        f"SELECT DISTINCT ON (r.algorithm_id) {columns} FROM gridpulse.analysis_runs r"
        " JOIN unnest(%(ids)s::text[], %(versions)s::text[], %(params)s::jsonb[]) AS a(id, version, params)"
        "   ON r.algorithm_id = a.id AND r.algorithm_version = a.version AND r.params = a.params"
        " WHERE r.status = 'succeeded' AND r.cutoff_utc IS NULL"
        " ORDER BY r.algorithm_id, r.finished_at_utc DESC, r.run_id DESC",
        {"ids": list(ids), "versions": list(versions), "params": [Jsonb(p) for p in params]},
    )
    return {row["algorithm_id"]: row for row in rows}


def get_run(conn: Connection, run_id: int) -> dict | None:
    rows = _dicts(
        conn,
        f"SELECT {_RUN_COLUMNS} FROM gridpulse.analysis_runs WHERE run_id = %s AND status = 'succeeded'",
        (run_id,),
    )
    return rows[0] if rows else None


def list_runs(conn: Connection, algorithm_id: str, limit: int) -> list[dict]:
    """Successful runs of one algorithm, newest first: every version, params and cutoff."""
    return _dicts(
        conn,
        f"SELECT {_RUN_COLUMNS} FROM gridpulse.analysis_runs"
        " WHERE algorithm_id = %s AND status = 'succeeded' ORDER BY finished_at_utc DESC, run_id DESC LIMIT %s",
        (algorithm_id, limit),
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


def period_results(
    conn: Connection, delivery_start: datetime, algorithms: list[tuple[str, str, dict]]
) -> list[dict]:
    """Every indicator of one settlement period from each algorithm's served run.

    `algorithms` is (id, version, params) per algorithm. For each, the run is the one find_run
    would serve for this period, so every indicator of an algorithm comes from the same run.
    """
    if not algorithms:
        return []
    ids, versions, params = zip(*algorithms, strict=True)
    return _dicts(
        conn,
        "WITH served AS ("
        "  SELECT DISTINCT ON (r.algorithm_id) r.run_id, r.algorithm_id, r.algorithm_version"
        "  FROM gridpulse.analysis_runs r"
        "  JOIN unnest(%(ids)s::text[], %(versions)s::text[], %(params)s::jsonb[]) AS a(id, version, params)"
        "    ON r.algorithm_id = a.id AND r.algorithm_version = a.version AND r.params = a.params"
        "  WHERE r.status = 'succeeded' AND r.cutoff_utc IS NULL"
        "    AND r.range_start_utc <= %(t)s AND r.range_end_utc > %(t)s"
        "  ORDER BY r.algorithm_id, r.finished_at_utc DESC, r.run_id DESC"
        ")"
        " SELECT s.algorithm_id, s.algorithm_version, s.run_id, res.indicator, res.value"
        " FROM served s JOIN gridpulse.analysis_results res ON res.run_id = s.run_id"
        " WHERE res.delivery_start_utc = %(t)s"
        " ORDER BY s.algorithm_id, res.indicator",
        {"ids": list(ids), "versions": list(versions), "params": [Jsonb(p) for p in params], "t": delivery_start},
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


def dataset_loads(conn: Connection, names: list[str]) -> dict[str, datetime | None]:
    """When each named raw dataset was last loaded; None if it is not in raw.dataset_catalog."""
    rows = conn.execute(
        "SELECT table_name, loaded_at FROM raw.dataset_catalog WHERE table_name = ANY(%s)", (names,)
    ).fetchall()
    found = dict(rows)
    return {name: found.get(name) for name in names}


def data_updated_at(conn: Connection) -> datetime | None:
    return conn.execute("SELECT max(loaded_at) FROM raw.dataset_catalog").fetchone()[0]


# --- Weather context (retrospective archived forecasts, not proven as-of inputs) -----------------------------

def weather_locations(conn: Connection) -> list[dict]:
    return _dicts(conn, """
        SELECT location AS id, initcap(replace(location, '_', ' ')) AS label,
               min(latitude) AS latitude, min(longitude) AS longitude
        FROM gridpulse.weather_hourly GROUP BY location ORDER BY location
    """)


def weather_values(conn: Connection, location: str, start: datetime, end: datetime) -> list[dict]:
    return _dicts(conn, """
        SELECT time_bucket('1 hour', bucket) AS bucket,
               avg(temperature_c) AS temperature, avg(pressure_hpa) AS pressure
        FROM gridpulse.weather_hourly
        WHERE location = %(location)s AND bucket >= %(start)s AND bucket < %(end)s
        GROUP BY 1 ORDER BY 1
    """, {"location": location, "start": start, "end": end})


def hourly_weather_prices(conn: Connection, start: datetime, end: datetime) -> list[dict]:
    # A partial hour must not be treated as a complete hourly price observation.
    return _dicts(conn, """
        SELECT time_bucket('1 hour', bucket) AS bucket,
               CASE WHEN count(price_gbp_mwh) = 2 AND count(DISTINCT bucket) = 2
                    THEN avg(price_gbp_mwh) END AS price
        FROM gridpulse.price_30m
        WHERE bucket >= %(start)s AND bucket < %(end)s
        GROUP BY 1 ORDER BY 1
    """, {"start": start, "end": end})
