#!/usr/bin/env bash
# Runs on the VPS from the directory holding docker-compose.yml and .env (/opt/gridpulse).
# The GitHub workflow logs in to ghcr.io and calls it with IMAGE_TAG=sha-<commit>.
set -euo pipefail
cd "$(dirname "$0")"

: "${IMAGE_TAG:?set IMAGE_TAG to sha-<commit> or latest}"
export IMAGE_TAG

if [[ ! -f .env ]]; then
    echo "$(pwd)/.env is missing: copy gridpulse/.env.example there and fill it in." >&2
    exit 1
fi

docker compose pull --quiet
# Migrations are re-runnable, so applying them on every deploy is safe.
docker compose run --rm --no-TTY backend gridpulse migrate
docker compose up --detach --remove-orphans --wait --wait-timeout 120
docker image prune --force
docker compose ps
