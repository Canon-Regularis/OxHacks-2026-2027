"""Health and catalog: what the frontend loads first to build its panels and status bar."""

from fastapi import APIRouter, Depends, Request
from psycopg import Connection

from gridpulse import __version__
from gridpulse.analysis.algorithms import ALGORITHMS
from gridpulse.api.analysis import algorithm_info
from gridpulse.api.deps import get_conn
from gridpulse.api.schemas import Catalog, DatasetCoverage, Health, Limits, SeriesInfo
from gridpulse.catalog import SERIES
from gridpulse.config import get_settings
from gridpulse.db import queries

router = APIRouter(tags=["system"])


@router.get("/health", response_model=Health)
def health(request: Request) -> Health:
    """Always answers, even with no database, so the UI can show what is wrong."""
    settings = get_settings()
    pool = request.app.state.pool
    if settings.gridpulse_mode == "replay":
        database = "not_used"
    elif pool is None:
        database = "not_configured"
    else:
        try:
            with pool.connection(timeout=2) as conn:
                conn.execute("SELECT 1")
            database = "ok"
        except Exception:
            database = "unavailable"
    status = "ok" if database in ("ok", "not_used") else "degraded"
    return Health(status=status, mode=settings.gridpulse_mode, database=database, version=__version__)


@router.get("/catalog", response_model=Catalog)
def catalog(conn: Connection = Depends(get_conn)) -> Catalog:
    """Series, data coverage and algorithms: enough to build command autocomplete and panels."""
    settings = get_settings()
    latest = {run["algorithm_id"]: run for run in queries.latest_runs(conn)}
    return Catalog(
        series=[SeriesInfo(id=sid, label=s.label, unit=s.unit) for sid, s in SERIES.items()],
        datasets=[DatasetCoverage(**row) for row in queries.coverage(conn)],
        algorithms=[algorithm_info(a, latest.get(a.id)) for a in ALGORITHMS.values()],
        limits=Limits(max_points=settings.max_points, max_range_days=settings.max_range_days),
        data_updated_at_utc=queries.data_updated_at(conn),
    )
