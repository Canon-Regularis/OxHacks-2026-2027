"""Stored algorithm results. Runs are produced by `gridpulse analyse`, never by a page load."""

from typing import Literal

from fastapi import APIRouter, Depends, Query
from psycopg import Connection

from gridpulse.analysis import service
from gridpulse.analysis.algorithms import ALGORITHMS
from gridpulse.analysis.contract import Algorithm
from gridpulse.api.deps import DateRange, api_error, choose_resolution, date_range, get_conn
from gridpulse.api.schemas import AlgorithmInfo, AnalysisResponse, Column, RunInfo, RunTables, Table

router = APIRouter(tags=["analysis"])


def algorithm_info(algorithm: Algorithm, latest_run: dict | None = None) -> AlgorithmInfo:
    return AlgorithmInfo(
        id=algorithm.id,
        version=algorithm.version,
        description=algorithm.description,
        inputs=list(algorithm.inputs),
        outputs=algorithm.outputs,
        tables=algorithm.tables,
        limitations=algorithm.limitations,
        latest_run=RunInfo(**latest_run) if latest_run else None,
    )


@router.get("/analysis", response_model=AnalysisResponse)
def analysis(
    algorithm: str = Query(description="Algorithm id from /api/catalog"),
    dates: DateRange = Depends(date_range),
    resolution: Literal["auto", "30m", "1h", "1d"] = "auto",
    run_id: int | None = Query(None, description="A specific run; default is the latest matching run"),
    conn: Connection = Depends(get_conn),
) -> AnalysisResponse:
    algo = ALGORITHMS.get(algorithm)
    if algo is None:
        raise api_error(404, "unknown_algorithm", f"Unknown algorithm {algorithm!r}; choose from {list(ALGORITHMS)}.")
    res = choose_resolution(dates, resolution)

    stored = service.get_results(conn, algo, dates.start_utc, dates.end_utc, res, run_id)
    if stored is None:
        raise api_error(
            404,
            "no_run",
            f"No completed {algo.id} v{algo.version} run covers this range. "
            f"Run: gridpulse analyse {algo.id} --from {dates.from_date} --to {dates.to_date}",
        )

    return AnalysisResponse(
        algorithm=algorithm_info(algo),
        run=RunInfo(**stored.run),
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
        run=RunInfo(**run),
        tables={n: Table(description=descriptions.get(n, ""), **data) for n, data in tables.items()},
    )
