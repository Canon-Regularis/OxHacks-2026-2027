#!/usr/bin/env bash
# Runs on the VPS from one environment's directory (/opt/gridpulse or /opt/gridpulse-dev), which
# holds docker-compose.yml and .env. The GitHub workflow logs in to ghcr.io and calls it with
# IMAGE_TAG=sha-<commit>. RUN_MIGRATIONS=false skips `gridpulse migrate`.
set -euo pipefail
cd "$(dirname "$0")"

: "${IMAGE_TAG:?set IMAGE_TAG to sha-<commit>, main or production}"
export IMAGE_TAG
PROXY_DIR=${PROXY_DIR:-/opt/gridpulse-proxy}

for env_file in .env "$PROXY_DIR/.env"; do
    if [[ ! -f $env_file ]]; then
        echo "$env_file is missing; see gridpulse/deploy/README.md." >&2
        exit 1
    fi
done

# The proxy owns ports 80/443 and the network both environments join. Reloading picks up a new
# Caddyfile without dropping connections.
(cd "$PROXY_DIR" && docker compose up --detach --wait && docker compose exec -T caddy caddy reload --config /etc/caddy/Caddyfile)

docker compose pull --quiet
if [[ ${RUN_MIGRATIONS:-true} == true ]]; then
    # Migrations are re-runnable, so applying them on every deploy is safe.
    docker compose run --rm --no-TTY backend gridpulse migrate
fi
docker compose up --detach --remove-orphans --wait --wait-timeout 120
docker image prune --force
docker compose ps
