"""FastAPI application. Run from gridpulse/backend: `uvicorn gridpulse.main:app --reload`.

Endpoints are plain `def` functions: FastAPI runs them in a thread pool, which suits the
synchronous psycopg pool and keeps the code free of async/await.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from psycopg import OperationalError
from psycopg.errors import QueryCanceled
from psycopg_pool import PoolTimeout

from gridpulse import __version__
from gridpulse.api import analysis, market, system
from gridpulse.api.demo_cache import DemoCacheMiddleware
from gridpulse.config import get_settings
from gridpulse.db.connection import open_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    use_database = settings.database_url and settings.gridpulse_mode != "replay"
    app.state.pool = open_pool(settings.database_url, settings.query_timeout_seconds) if use_database else None
    yield
    if app.state.pool is not None:
        app.state.pool.close()


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"detail": {"code": code, "message": message}}, status_code=status)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="GridPulse API", version=__version__, lifespan=lifespan)

    # Middleware added last runs first: gzip -> CORS -> demo cache -> routes.
    if settings.gridpulse_mode in ("record", "replay"):
        app.add_middleware(DemoCacheMiddleware, mode=settings.gridpulse_mode, directory=settings.demo_cache_dir)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_origin],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["X-GridPulse-Mode"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    for module in (system, market, analysis):
        app.include_router(module.router, prefix="/api")

    @app.exception_handler(PoolTimeout)
    @app.exception_handler(OperationalError)
    def database_unavailable(request: Request, exc: Exception) -> JSONResponse:
        return _error(503, "database_unavailable", "The database is unreachable; check /api/health.")

    @app.exception_handler(QueryCanceled)
    def query_timeout(request: Request, exc: Exception) -> JSONResponse:
        return _error(504, "query_timeout", "The query took too long; narrow the date range.")

    return app


app = create_app()
