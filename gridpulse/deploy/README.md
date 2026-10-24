# Vultr deployment

One Vultr VPS runs two containers with Docker Compose. Tiger Cloud stays the database.

```text
push to main touching gridpulse/{backend,frontend,db,data/demo,deploy}
  → GitHub Actions (.github/workflows/deploy.yml)
      backend tests (pytest) + frontend checks (once a package.json exists)
      → build gridpulse-backend and gridpulse-frontend images → ghcr.io (private)
      → SSH to the VPS: pull images, `gridpulse migrate`, restart, wait for the health check
```

| Container | Image | Role |
|---|---|---|
| `web` | [`frontend/Dockerfile`](../frontend/Dockerfile) | Caddy: serves the frontend's static build, gets HTTPS certificates, proxies `/api/*`, `/docs` and `/openapi.json` to the backend. Same origin, so no CORS. |
| `backend` | [`backend/Dockerfile`](../backend/Dockerfile) | `uvicorn gridpulse.main:app` with two workers, configured by `/opt/gridpulse/.env`. Not public: only Caddy and `127.0.0.1:8000` on the VPS reach it. |

Until the frontend framework is set up (no `frontend/package.json`), `web` serves a placeholder page, so the server, HTTPS and API can be deployed and checked now.

Files here: [`docker-compose.yml`](docker-compose.yml) and [`deploy.sh`](deploy.sh) are copied to `/opt/gridpulse` on every deploy; [`Caddyfile`](Caddyfile) is baked into the `web` image.

Pull requests run the tests and build both images without pushing or deploying.

## One-time setup

**1. Create the VPS.** Vultr Cloud Compute, London, Ubuntu 24.04, 2 GB RAM. Add your own SSH key when creating it.

**2. Prepare it.** As root on the VPS:

```bash
curl -fsSL https://get.docker.com | sh
useradd --create-home --shell /bin/bash --groups docker deploy
install -d -o deploy -g deploy /opt/gridpulse
ufw allow 80,443/tcp && ufw allow 443/udp
```

**3. Give GitHub a deploy key.** On your laptop:

```bash
ssh-keygen -t ed25519 -N '' -C gridpulse-deploy -f gridpulse_deploy
```

Then on the VPS, as root, authorise the public half (`gridpulse_deploy.pub`) for the `deploy` user:

```bash
install -d -m 700 -o deploy -g deploy /home/deploy/.ssh
echo 'PASTE gridpulse_deploy.pub HERE' >> /home/deploy/.ssh/authorized_keys
chown deploy:deploy /home/deploy/.ssh/authorized_keys && chmod 600 /home/deploy/.ssh/authorized_keys
```

**4. Write the server settings.** Create `/opt/gridpulse/.env` from [`gridpulse/.env.example`](../.env.example), owned by `deploy` with mode `600`. Set at least `DATABASE_URL` and `SITE_ADDRESS`:

- `SITE_ADDRESS=gridpulse.example.com` with a DNS A record pointing at the VPS, or `203-0-113-10.sslip.io` (the VPS IP with dashes) for HTTPS without owning a domain. Caddy fetches the certificate on first start. Leave it unset for plain HTTP on port 80.
- Docker Compose expands `$` in this file. Wrap any value containing `$` in single quotes.

**5. Configure the repository.** In GitHub, go to Settings → Secrets and variables → Actions, or use the commands below:

| Name | Kind | Value |
|---|---|---|
| `VPS_HOST` | variable | VPS IP address. Deploys are skipped while it is unset. |
| `VPS_USER` | variable | `deploy` (the default) |
| `VPS_SSH_KEY` | secret | The private key file `gridpulse_deploy` |
| `VPS_SSH_KNOWN_HOSTS` | variable, optional | Output of `ssh-keyscan <ip>`, to pin the host key. Without it the workflow trusts the key it sees. |
| `PUBLIC_URL` | variable, optional | e.g. `https://gridpulse.example.com`. After deploying, the workflow fetches `/api/health` there and warns if it is not `ok`. |

```bash
gh variable set VPS_HOST --body 203.0.113.10
gh variable set VPS_USER --body deploy
gh secret set VPS_SSH_KEY < gridpulse_deploy
gh variable set VPS_SSH_KNOWN_HOSTS --body "$(ssh-keyscan 203.0.113.10 2>/dev/null)"
gh variable set PUBLIC_URL --body https://203-0-113-10.sslip.io
```

Then delete the local private key, or keep it somewhere safe. The images are private: each deploy logs the VPS in to ghcr.io with the workflow's own short-lived token and logs it out afterwards, so the VPS stores no registry credentials.

**6. Deploy.** Actions → CI/CD → Run workflow, or push to `main`. Check `https://<SITE_ADDRESS>/api/health`, from another device too, then reboot the VPS once (`sudo reboot`) and check again: Docker starts on boot and both containers restart themselves.

## Operating it

Run these on the VPS as `deploy`, from `/opt/gridpulse`.

| Task | How |
|---|---|
| Deploy | Push to `main`, or run the workflow manually |
| Roll back | Revert and push. Alternatively, open an earlier successful run in Actions and re-run its *Deploy to Vultr* job, which redeploys that commit's images. |
| Logs | `docker compose logs -f backend` (or `web`) |
| Status | `docker compose ps`; `curl -s localhost:8000/api/health` |
| Change settings | Edit `.env`, then `docker compose up -d` |
| Run a CLI job | `docker compose run --rm backend gridpulse analyse wind_price_correlation --from 2026-07-01 --to 2026-10-23` |

**Offline fallback.** Recorded responses committed under `gridpulse/data/demo/responses/` are built into the backend image. To serve them, set `GRIDPULSE_MODE=replay` in `.env` and run `docker compose up -d`. Record locally and commit the files; responses recorded inside the container are lost on the next deploy.

## Running the production stack locally

From `gridpulse/`, with ports 80 and 443 free (otherwise change the `web` ports in a `deploy/docker-compose.override.yml`):

```bash
docker build -f backend/Dockerfile -t ghcr.io/canon-regularis/gridpulse-backend:local .
docker build -f frontend/Dockerfile -t ghcr.io/canon-regularis/gridpulse-frontend:local .
cp .env deploy/.env
IMAGE_TAG=local docker compose -f deploy/docker-compose.yml up
```

Then open <http://localhost>. `deploy/.env` is gitignored like `gridpulse/.env`.
