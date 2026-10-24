"""Offline demo fallback: record API responses to disk, then replay them with no database.

GRIDPULSE_MODE=record  serve normally and save every successful GET /api/* response.
GRIDPULSE_MODE=replay  serve only saved responses; unrecorded requests get a 404.
Click through the demo path once in record mode, then replay works without network access.
"""

import hashlib
from pathlib import Path
from urllib.parse import urlencode

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware


def cache_key(request: Request) -> str:
    query = urlencode(sorted(request.query_params.multi_items()))
    return hashlib.sha256(f"{request.url.path}?{query}".encode()).hexdigest()[:24]


class DemoCacheMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, mode: str, directory: Path):
        super().__init__(app)
        self.mode = mode
        self.directory = directory

    async def dispatch(self, request: Request, call_next):
        cacheable = request.method == "GET" and request.url.path.startswith("/api/")
        if not cacheable or request.url.path == "/api/health":
            return await call_next(request)

        path = self.directory / f"{cache_key(request)}.json"
        if self.mode == "replay":
            if not path.exists():
                detail = {"code": "not_recorded", "message": "This request was not recorded for the offline demo."}
                return JSONResponse({"detail": detail}, status_code=404)
            return Response(path.read_bytes(), media_type="application/json", headers={"X-GridPulse-Mode": "replay"})

        response = await call_next(request)
        if response.status_code != 200:
            return response
        body = b"".join([chunk async for chunk in response.body_iterator])
        self.directory.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return Response(body, status_code=200, headers=dict(response.headers), media_type=response.media_type)
