"""Runs algorithms on stored data and serves their stored results.

Batch (CLI `gridpulse analyse`): build input frame -> algorithm.run -> check output -> store.
The API only reads stored results, so rendering a chart never triggers a recomputation.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from psycopg import Connection

from gridpulse.analysis.contract import MAX_TABLE_ROWS, Algorithm, AlgorithmResult
from gridpulse.analysis.features import build_input_frame
from gridpulse.catalog import datasets_for
from gridpulse.db import queries
from gridpulse.settlement import bucket_starts


class AlgorithmError(Exception):
    """An algorithm returned something that breaks the contract in analysis/contract.py."""


class UnknownParameterError(ValueError):
    """An override names a parameter the algorithm does not declare."""


def with_params(algorithm: Algorithm, overrides: dict | None = None) -> dict:
    """The algorithm's defaults with `overrides` applied: what a run is given and stored with."""
    unknown = sorted(set(overrides or {}) - set(algorithm.params))
    if unknown:
        raise UnknownParameterError(f"{algorithm.id} has no parameters {unknown}; it takes {sorted(algorithm.params)}")
    return {**algorithm.params, **(overrides or {})}


def overrides_of(algorithm: Algorithm | None, params: dict) -> dict:
    """The stored params of a run that differ from the algorithm's current defaults."""
    defaults = json.loads(json.dumps(with_params(algorithm))) if algorithm else {}  # as stored: tuples -> lists
    return {name: value for name, value in params.items() if name not in defaults or defaults[name] != value}


def run_algorithm(
    conn: Connection,
    algorithm: Algorithm,
    start: datetime,
    end: datetime,
    params: dict | None = None,
    cutoff: datetime | None = None,
) -> int:
    """Run one algorithm over [start, end) and store its results. Returns the run id."""
    params = with_params(algorithm, params)
    range_end = end + algorithm.horizon
    loads = queries.dataset_loads(conn, datasets_for(algorithm.inputs))
    run_id = queries.create_run(conn, algorithm.id, algorithm.version, params, start, range_end, cutoff, loads)
    try:
        frame = build_input_frame(conn, start - algorithm.warmup, end, cutoff)
        series, tables = split_result(algorithm, algorithm.run(frame[list(algorithm.inputs)].copy(), **params))
        series = check_output(algorithm, series, start, range_end)
        tables = check_tables(algorithm, tables)
        with conn.transaction():  # results become visible together with the 'succeeded' status
            count = queries.insert_results(conn, result_rows(run_id, series))
            for name, data in tables.items():
                queries.insert_table(conn, run_id, name, data)
            queries.finish_run(conn, run_id, "succeeded", result_rows=count)
    except Exception as exc:
        queries.finish_run(conn, run_id, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    return run_id


def split_result(algorithm: Algorithm, result: object) -> tuple[object, dict]:
    """(series, tables) from whatever run() returned."""
    if isinstance(result, AlgorithmResult):
        return result.series, result.tables
    if isinstance(result, pd.DataFrame):
        return result, {}
    raise AlgorithmError(f"{algorithm.id} returned {type(result).__name__}, expected a DataFrame or AlgorithmResult")


def check_output(algorithm: Algorithm, output: object, start: datetime, end: datetime) -> pd.DataFrame:
    """Enforce the series contract and trim to [start, end)."""
    if not algorithm.outputs:
        if output is not None and not (isinstance(output, pd.DataFrame) and output.empty):
            raise AlgorithmError(f"{algorithm.id} returned series but declares no outputs")
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    if not isinstance(output, pd.DataFrame):
        raise AlgorithmError(f"{algorithm.id} returned {type(output).__name__}, expected a DataFrame")
    if not isinstance(output.index, pd.DatetimeIndex) or output.index.tz is None:
        raise AlgorithmError(f"{algorithm.id} output must be indexed by tz-aware delivery_start_utc")
    if set(output.columns) != set(algorithm.outputs):
        raise AlgorithmError(
            f"{algorithm.id} returned columns {sorted(output.columns)}, declared {sorted(algorithm.outputs)}"
        )
    output = output.astype("float64").replace([np.inf, -np.inf], np.nan)
    return output[(output.index >= start) & (output.index < end)]


def check_tables(algorithm: Algorithm, tables: dict) -> dict[str, dict]:
    """Enforce the table contract and convert each table to {"columns": [...], "rows": [[...]]}."""
    if set(tables) != set(algorithm.tables):
        raise AlgorithmError(f"{algorithm.id} returned tables {sorted(tables)}, declared {sorted(algorithm.tables)}")
    return {name: table_data(algorithm, name, table) for name, table in tables.items()}


def table_data(algorithm: Algorithm, name: str, table: object) -> dict:
    if not isinstance(table, pd.DataFrame):
        raise AlgorithmError(f"{algorithm.id} table {name!r} is {type(table).__name__}, expected a DataFrame")
    if len(table) > MAX_TABLE_ROWS:
        raise AlgorithmError(f"{algorithm.id} table {name!r} has {len(table)} rows; the limit is {MAX_TABLE_ROWS}")
    if not isinstance(table.index, pd.RangeIndex):
        table = table.reset_index()  # keep a meaningful index, e.g. lag or origin time, as a column
    naive = [str(c) for c in table.columns if pd.api.types.is_datetime64_dtype(table[c].dtype)]
    if naive:
        raise AlgorithmError(f"{algorithm.id} table {name!r} has timezone-naive datetime columns {naive}; use UTC")
    # pandas writes tz-aware datetimes as ISO strings ending in Z, and NaN/inf as null.
    split = json.loads(table.to_json(orient="split", index=False, date_format="iso", date_unit="s"))
    return {"columns": [str(c) for c in table.columns], "rows": split["data"]}


def result_rows(run_id: int, output: pd.DataFrame) -> Iterator[tuple[int, datetime, str, float | None]]:
    for indicator in output.columns:
        for timestamp, value in output[indicator].items():
            yield run_id, timestamp.to_pydatetime(), indicator, None if np.isnan(value) else float(value)


@dataclass
class StoredResults:
    run: dict
    t: list[datetime]
    indicators: dict[str, list[float | None]]
    tables: list[str]  # names; fetch contents with get_tables
    warnings: list[str]


def get_results(
    conn: Connection,
    algorithm: Algorithm,
    start: datetime,
    end: datetime,
    resolution: str,
    run_id: int | None = None,
    overrides: dict | None = None,
) -> StoredResults | None:
    """Stored results for [start, end) from a given run, or the latest run of the current version
    made with the default params plus `overrides` (UnknownParameterError for an undeclared name)."""
    if run_id is not None:
        run = queries.get_run(conn, run_id)
    else:
        run = queries.find_run(conn, algorithm.id, algorithm.version, with_params(algorithm, overrides), start, end)
    if run is None or run["algorithm_id"] != algorithm.id:
        return None

    by_indicator: dict[str, dict[datetime, float | None]] = {name: {} for name in algorithm.outputs}
    for bucket, indicator, value in queries.result_buckets(conn, run["run_id"], start, end, resolution):
        by_indicator.setdefault(indicator, {})[bucket] = value
    t = bucket_starts(start, end, resolution)

    warnings = []
    if run["range_start_utc"] > start or run["range_end_utc"] < end:
        warnings.append(
            f"Run {run['run_id']} covers {run['range_start_utc']:%Y-%m-%d %H:%M} to "
            f"{run['range_end_utc']:%Y-%m-%d %H:%M} UTC only; other periods are empty."
        )
    if run["cutoff_utc"] is not None:
        warnings.append(f"Inputs limited to data available at {run['cutoff_utc']:%Y-%m-%d %H:%M} UTC.")
    if run["algorithm_version"] != algorithm.version:
        warnings.append(f"Results are from version {run['algorithm_version']}; current is {algorithm.version}.")
    warnings += reloaded_warnings(conn, run)
    changed = overrides_of(algorithm, run["params"])
    if changed:
        settings = ", ".join(f"{name}={json.dumps(value)}" for name, value in changed.items())
        warnings.append(f"Run {run['run_id']} uses non-default parameters: {settings}.")

    indicators = {name: [values.get(bucket) for bucket in t] for name, values in by_indicator.items()}
    return StoredResults(run, t, indicators, queries.table_names(conn, run["run_id"]), warnings)


def run_datasets(algorithm: Algorithm | None, run: dict) -> dict[str, str | None]:
    """Dataset -> load time (ISO string) that a run used. Runs made before loads were recorded
    list the algorithm's current datasets with unknown (None) load times."""
    if run["datasets"] is not None:
        return dict(sorted(run["datasets"].items()))
    return {name: None for name in datasets_for(algorithm.inputs)} if algorithm else {}


def reloaded_warnings(conn: Connection, run: dict) -> list[str]:
    """One warning per dataset loaded again after the run read it."""
    used = {name: loaded for name, loaded in (run["datasets"] or {}).items() if loaded is not None}
    if not used:
        return []
    current = queries.dataset_loads(conn, list(used))
    return [
        f"{name} was loaded again at {current[name]:%Y-%m-%d %H:%M} UTC, after this run; "
        "rerun the algorithm to include any new or revised data."
        for name, loaded in used.items()
        if current[name] is not None and current[name] > datetime.fromisoformat(loaded)
    ]


def served_run(conn: Connection, algorithm: Algorithm) -> dict | None:
    """The run /api/analysis serves by default: latest retrospective run with the default params."""
    return queries.find_run(conn, algorithm.id, algorithm.version, with_params(algorithm))


def period_results(conn: Connection, algorithms: list[Algorithm], delivery_start: datetime) -> list[dict]:
    """Every algorithm's indicators for one period, from the run that would be served by default."""
    served = [(a.id, a.version, with_params(a)) for a in algorithms]
    return queries.period_results(conn, delivery_start, served)


def list_runs(conn: Connection, algorithm: Algorithm, limit: int) -> list[dict]:
    return queries.list_runs(conn, algorithm.id, limit)


def get_tables(conn: Connection, run_id: int, name: str | None = None) -> tuple[dict, dict[str, dict]] | None:
    """(run, {name: table}) for a successful run, or None if there is no such run."""
    run = queries.get_run(conn, run_id)
    if run is None:
        return None
    return run, queries.get_tables(conn, run_id, name)
