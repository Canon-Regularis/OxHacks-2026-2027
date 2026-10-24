"""Stored algorithm results. Runs are produced by `gridpulse analyse`, never by a page load."""

import json
from dataclasses import dataclass
from typing import Literal

from fastapi import APIRouter, Depends, Query
from psycopg import Connection

from gridpulse.analysis import service
from gridpulse.analysis.algorithms import ALGORITHMS
from gridpulse.analysis.contract import Algorithm
from gridpulse.api.deps import DateRange, api_error, choose_resolution, date_range, get_conn
from gridpulse.api.schemas import AlgorithmInfo, AnalysisResponse, Column, RunInfo, RunList, RunTables, Table

router = APIRouter(tags=["analysis"])


def run_info(run: dict) -> RunInfo:
    overrides = service.overrides_of(ALGORITHMS.get(run["algorithm_id"]), run["params"])
    return RunInfo(**run, overrides=overrides)


def algorithm_info(algorithm: Algorithm, latest_run: dict | None = None) -> AlgorithmInfo:
    return AlgorithmInfo(
        id=algorithm.id,
        version=algorithm.version,
        description=algorithm.description,
        inputs=list(algorithm.inputs),
        outputs=algorithm.outputs,
        tables=algorithm.tables,
        params=json.loads(json.dumps(algorithm.params)),
        param_descriptions=algorithm.param_descriptions,
        limitations=algorithm.limitations,
        latest_run=run_info(latest_run) if latest_run else None,
    )


def registered_algorithm(algorithm: str = Query(description="Algorithm id from /api/catalog")) -> Algorithm:
    found = ALGORITHMS.get(algorithm)
    if found is None:
        raise api_error(404, "unknown_algorithm", f"Unknown algorithm {algorithm!r}; choose from {list(ALGORITHMS)}.")
    return found


@dataclass(frozen=True)
class Selection:
    algorithm: Algorithm
    run_id: int | None
    overrides: dict


def selection(
    algo: Algorithm = Depends(registered_algorithm),
    run_id: int | None = Query(None, description="A specific run; default is the latest matching run"),
    params: str | None = Query(
        None,
        description='JSON object of parameter overrides, e.g. {"horizon_periods": 24}. '
        "Serves the latest run made with the defaults plus these; omit for the default run",
    ),
) -> Selection:
    """Which run to serve, validated before a database connection is taken."""
    if params is None:
        return Selection(algo, run_id, {})
    if run_id is not None:
        raise api_error(400, "invalid_params", "Give run_id or params, not both.")
    try:
        overrides = json.loads(params)
    except json.JSONDecodeError:
        overrides = None
    if not isinstance(overrides, dict):
        raise api_error(400, "invalid_params", 'params must be a JSON object, e.g. {"horizon_periods": 24}.')
    try:
        service.with_params(algo, overrides)
    except service.UnknownParameterError as exc:
        raise api_error(400, "unknown_param", f"{exc}.") from exc
    return Selection(algo, None, overrides)


@router.get("/analysis", response_model=AnalysisResponse)
def analysis(
    selected: Selection = Depends(selection),
    dates: DateRange = Depends(date_range),
    resolution: Literal["auto", "30m", "1h", "1d"] = "auto",
    conn: Connection = Depends(get_conn),
) -> AnalysisResponse:
    algo, run_id, overrides = selected.algorithm, selected.run_id, selected.overrides
    res = choose_resolution(dates, resolution)

    stored = service.get_results(conn, algo, dates.start_utc, dates.end_utc, res, run_id, overrides)
    if stored is None:
        if run_id is not None:
            raise api_error(404, "no_run", f"No successful {algo.id} run {run_id}.")
        flags = "".join(f" --param {name}={json.dumps(value)}" for name, value in overrides.items())
        raise api_error(
            404,
            "no_run",
            f"No completed {algo.id} v{algo.version} run with these parameters covers this range. "
            f"Run: gridpulse analyse {algo.id} --from {dates.from_date} --to {dates.to_date}{flags}",
        )

    return AnalysisResponse(
        algorithm=algorithm_info(algo),
        run=run_info(stored.run),
        from_=dates.from_date,
        to=dates.to_date,
        resolution=res,
        t=stored.t,
        indicators={
            name: Column(label=name, unit=algo.outputs.get(name, ""), values=values)
            for name, values in stored.indicators.items()
        },
        tables=stored.tables,
        warnings=stored.warnings,
    )


@router.get("/analysis/runs", response_model=RunList)
def runs(
    algo: Algorithm = Depends(registered_algorithm),
    limit: int = Query(20, ge=1, le=100),
    conn: Connection = Depends(get_conn),
) -> RunList:
    """Successful runs of one algorithm, newest first, so the UI can offer parameter variants."""
    return RunList(
        algorithm=algorithm_info(algo, service.served_run(conn, algo)),
        runs=[run_info(run) for run in service.list_runs(conn, algo, limit)],
    )


@router.get("/analysis/runs/{run_id}/tables", response_model=RunTables)
def run_tables(
    run_id: int,
    name: str | None = Query(None, description="Return only this table"),
    conn: Connection = Depends(get_conn),
) -> RunTables:
    """Non-time-series results of one run: scorecards, ACF by lag, fitted parameters, diagnostics."""
    found = service.get_tables(conn, run_id, name)
    if found is None:
        raise api_error(404, "no_run", f"No successful run {run_id}.")
    run, tables = found
    if name is not None and name not in tables:
        raise api_error(404, "no_table", f"Run {run_id} has no table {name!r}.")

    algorithm = ALGORITHMS.get(run["algorithm_id"])
    descriptions = algorithm.tables if algorithm else {}
    return RunTables(
        run=run_info(run),
        tables={n: Table(description=descriptions.get(n, ""), **data) for n, data in tables.items()},
    )
