# syntax=docker/dockerfile:1
# GridPulse API. Build from gridpulse/:  docker build -f backend/Dockerfile -t gridpulse-backend .
# Settings come from the environment (see gridpulse/.env.example); none are baked in.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN useradd --system --no-create-home gridpulse

WORKDIR /app/gridpulse/backend

# Dependencies first, so a code change does not reinstall pandas and numpy.
COPY backend/pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# config.py finds the migrations and demo cache relative to its own path, so keep the
# repository layout under /app and install in editable mode, as in local development.
COPY db/migrations /app/gridpulse/db/migrations
COPY --chown=gridpulse data/demo /app/gridpulse/data/demo
COPY backend/src ./src
RUN pip install --no-deps -e .

USER gridpulse
EXPOSE 8000

# Agent admission, deduplication and short-lived approvals are process-local.
# Keep one async API worker until those records move to a shared coordinator.
ENV WEB_CONCURRENCY=1
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --start-interval=2s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"]

# Caddy terminates HTTPS in front of this container and forwards client details.
CMD ["uvicorn", "gridpulse.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
