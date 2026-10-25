# Vultr deployment

One Vultr VPS runs two environments with Docker Compose. Tiger Cloud stays the database.

| Branch | Environment | URL (current VPS) | Directory on the VPS |
|---|---|---|---|
| `production` | production | <https://78-141-242-126.sslip.io> | `/opt/gridpulse` |
| `main` | development | <https://dev.78-141-242-126.sslip.io> | `/opt/gridpulse-dev` |

Work merges into `main` and appears on development. Promote to production with a pull request from `main` into `production`.

```text
push to main or production touching gridpulse/{backend,frontend,db,data/demo,deploy}
  → GitHub Actions (.github/workflows/deploy.yml)
      in parallel: backend tests (three shards, each pytest -n auto), frontend checks (once a package.json exists),
        and the gridpulse-backend and gridpulse-frontend image builds → ghcr.io (private)
      → once all of those pass, SSH to the VPS: start/reload the proxy, pull images,
        `gridpulse migrate`, restart that environment, wait for the health check
```

Images build alongside the tests, so the `main` and `production` tags can move before the tests finish; deploys use the `sha-<commit>` tag and wait for the tests. Pull requests run the tests and build only the images their diff can affect (both, if it touches the workflow or `deploy/`), without pushing or deploying. A manual run (Actions → CI/CD → Run workflow) deploys only when started from `main` or `production`.

## What runs on the VPS

```text
                     ┌ proxy (/opt/gridpulse-proxy) ┐
internet ─ 80/443 ─► │ Caddy: HTTPS, routes by host │
                     └──────────────┬───────────────┘
               gridpulse-edge network│
          ┌──────────────────────────┴───────────────────────┐
  gridpulse-production                              gridpulse-development
  web (SvelteKit Node) ─► backend (uvicorn)       web ─► backend
```

| Container | Defined in | Role |
|---|---|---|
| proxy `caddy` | [`proxy/docker-compose.yml`](proxy/docker-compose.yml), [`proxy/Caddyfile`](proxy/Caddyfile) | Shared by both environments. Gets the HTTPS certificates and routes each host name to that environment's `web`. |
| `web` | [`frontend/Dockerfile`](../frontend/Dockerfile) | Runs the SvelteKit Node server on port 80 and proxies `/api/*`, `/docs` and `/openapi.json` to its backend. Same origin, so no CORS. |
| `backend` | [`backend/Dockerfile`](../backend/Dockerfile) | `uvicorn gridpulse.main:app` with two workers, configured by the environment's `.env`. Not public: only its `web` and a localhost port on the VPS reach it. |

The frontend image builds with adapter-node and runs `node build`. `API_PROXY_TARGET` defaults to `http://backend:8000`; the shared edge Caddy still terminates HTTPS. `deploy/Caddyfile` is the earlier static-server reference and is not baked into the Node image.

**Separate databases.** Each environment has its own Tiger Cloud service, set by `DATABASE_URL` in its `.env`: production uses `prod-koxford-terminator`, development the original `koxford-terminator`. Each deploy runs `gridpulse migrate` against its own environment's database, so migrations reach production only when the code does. Production's copy of the `raw` data does not follow loads into development's service; load new data into both. To stop an environment from migrating, set its GitHub environment variable `RUN_MIGRATIONS=false`.

## One-time setup

The current VPS (`78.141.242.126`) is already set up. These steps are for rebuilding it.

**1. Create the VPS.** Vultr Cloud Compute, London, Debian or Ubuntu, 2 GB RAM or more. Add your own SSH key when creating it.

**2. Prepare it.** As root:

```bash
curl -fsSL https://get.docker.com | sh
useradd --create-home --shell /bin/bash --groups docker deploy
install -d -o deploy -g deploy /opt/gridpulse /opt/gridpulse-dev /opt/gridpulse-proxy /opt/gridpulse-proxy/conf
ufw allow 80,443/tcp && ufw allow 443/udp
```

**3. Give GitHub a deploy key.** On your laptop:

```bash
ssh-keygen -t ed25519 -N '' -C gridpulse-deploy -f gridpulse_deploy
```

Then on the VPS, as root, authorise the public half (`gridpulse_deploy.pub`) for `deploy`:

```bash
install -d -m 700 -o deploy -g deploy /home/deploy/.ssh
echo 'PASTE gridpulse_deploy.pub HERE' >> /home/deploy/.ssh/authorized_keys
chown deploy:deploy /home/deploy/.ssh/authorized_keys && chmod 600 /home/deploy/.ssh/authorized_keys
```

**4. Write the server settings**, each file owned by `deploy` with mode `600`. Docker Compose expands `$` in these files, so wrap any value containing `$` in single quotes.

`/opt/gridpulse-proxy/.env`: the public host names. Use a domain with a DNS A record pointing at the VPS, or the VPS IP with dashes under `sslip.io` for HTTPS without owning a domain.

```bash
PRODUCTION_ADDRESS=78-141-242-126.sslip.io
DEVELOPMENT_ADDRESS=dev.78-141-242-126.sslip.io
```

`/opt/gridpulse/.env` and `/opt/gridpulse-dev/.env`: application settings from [`gridpulse/.env.example`](../.env.example), at least `DATABASE_URL`, plus:

| Setting | Production | Development |
|---|---|---|
| `DEPLOY_ENV` | `production` | `development` |
| `IMAGE_TAG` | `production` | `main` |
| `BACKEND_PORT` | `8000` | `8001` |

**5. Configure the repository.** Settings → Secrets and variables → Actions, or the commands below. Repository-level:

| Name | Kind | Value |
|---|---|---|
| `VPS_HOST` | variable | VPS IP address. Deploys are skipped while it is unset. |
| `VPS_USER` | variable | `deploy` (the default) |
| `VPS_SSH_KEY` | secret | The private key file `gridpulse_deploy` |
| `VPS_SSH_KNOWN_HOSTS` | variable, optional | Output of `ssh-keyscan <ip>`, to pin the host key. Without it the workflow trusts the key it sees. |

Per GitHub environment (`production`, `development`; Settings → Environments):

| Name | Value |
|---|---|
| `PUBLIC_URL` | Optional. That environment's URL; after deploying, the workflow fetches `/api/health` there and warns if it is not `ok`. |
| `RUN_MIGRATIONS` | Optional. `false` skips `gridpulse migrate` for that environment. |

```bash
gh variable set VPS_HOST --body 203.0.113.10
gh variable set VPS_USER --body deploy
gh secret set VPS_SSH_KEY < gridpulse_deploy
gh variable set VPS_SSH_KNOWN_HOSTS --body "$(ssh-keyscan 203.0.113.10 2>/dev/null)"
gh variable set PUBLIC_URL --env production --body https://203-0-113-10.sslip.io
gh variable set PUBLIC_URL --env development --body https://dev.203-0-113-10.sslip.io
```

Then delete the local private key, or keep it somewhere safe. The images are private: each deploy logs the VPS in to ghcr.io with the workflow's own short-lived token and logs it out afterwards, so the VPS stores no registry credentials.

**6. Deploy.** Push to `main` and `production`, or run the workflow manually from each. Check both URLs, from another device too, then reboot the VPS once (`sudo reboot`) and check again: Docker starts on boot and every container restarts itself.

## Operating it

Run these on the VPS as `deploy`, from the environment's directory (`/opt/gridpulse` or `/opt/gridpulse-dev`).

| Task | How |
|---|---|
| Deploy | Push to the branch, or run the workflow manually from it |
| Roll back | Revert and push. Alternatively, open an earlier successful run in Actions and re-run its *Deploy* job, which redeploys that commit's images. |
| Logs | `docker compose logs -f backend` (or `web`); proxy: `cd /opt/gridpulse-proxy && docker compose logs -f` |
| Status | `docker compose ps`; `curl -s localhost:8000/api/health` (development: `8001`) |
| Change settings | Edit `.env`, then `docker compose up -d` |
| Run a CLI job | `docker compose run --rm backend gridpulse analyse wind_price_correlation --from 2026-07-01 --to 2026-10-23` |
| Refresh analysis results | Automatic every three hours (*Refresh analysis* workflow). By hand: run that workflow from Actions (optionally *force*), or `docker compose exec backend gridpulse refresh` |

**Analysis results** are computed per environment: each database only has the runs made against it. The *Refresh analysis* workflow ([`refresh-analysis.yml`](../../.github/workflows/refresh-analysis.yml)) runs `gridpulse refresh` inside each environment's backend container every three hours and on demand. It reruns only what is stale, so an environment shows results within three hours of a deploy or data load. It uses the same SSH settings as the deploy job. If the production environment requires approval, each scheduled run waits for it.

**Offline fallback.** Recorded responses committed under `gridpulse/data/demo/responses/` are built into the backend image. [`scripts/record_demo.py`](../scripts/record_demo.py) records them without clicking through a UI. To serve them, set `GRIDPULSE_MODE=replay` in that environment's `.env` and run `docker compose up -d`. Record locally and commit the files; responses recorded inside the container are lost on the next deploy.

## Running one environment locally

From `gridpulse/`:

```bash
docker build -f backend/Dockerfile -t ghcr.io/canon-regularis/gridpulse-backend:local .
docker build -f frontend/Dockerfile -t ghcr.io/canon-regularis/gridpulse-frontend:local .
docker network create gridpulse-edge
cp .env deploy/.env
printf 'DEPLOY_ENV=local\nIMAGE_TAG=local\n' >> deploy/.env
printf 'services:\n  web:\n    ports: ["8080:80"]\n' > deploy/docker-compose.override.yml
docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.override.yml up
```

Then open <http://localhost:8080>. This skips the HTTPS proxy. `deploy/.env` is gitignored like `gridpulse/.env`.

### Terminal AI

Set server-only `OPENROUTER_API_KEY`, `AGENT_PROVIDER=openrouter`, `AGENT_BASE_URL=https://openrouter.ai/api/v1`, `AGENT_MODEL=openai/gpt-5.4-mini`, `AGENT_REASONING_EFFORT=low`, and a random `AGENT_ACCESS_TOKEN` in the deployment `.env`. Compose forwards the proxy token to the SvelteKit server; it does not forward the inference key. Keep `WEB_CONCURRENCY=1`: current rate limits, deduplication and approvals are process-local. See [agent setup and limits](../../docs/gridpulse/AI_AGENT.md). Do not use the legacy static Caddy frontend configuration for the Node adapter; the dedicated same-origin POST proxy must run.
