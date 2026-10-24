# syntax=docker/dockerfile:1
# GridPulse web server: the frontend's static build behind Caddy, which also proxies the API
# to the backend container (see deploy/Caddyfile).
# Build from gridpulse/:  docker build -f frontend/Dockerfile -t gridpulse-frontend .

FROM node:22-alpine AS build
WORKDIR /app

COPY frontend/package*.json ./
RUN if [ -f package.json ]; then npm ci; fi
COPY frontend/ ./

# Until the frontend framework is set up there is no package.json. Serve this page instead, so
# the server, HTTPS and API proxy can be deployed and checked in the meantime.
COPY <<'EOF' /placeholder/index.html
<!doctype html>
<meta charset="utf-8">
<title>GridPulse</title>
<p>GridPulse: the frontend has not been built yet. API: <a href="/api/health">/api/health</a>, <a href="/docs">/docs</a>.</p>
EOF

# Vite writes dist/ and SvelteKit's adapter-static writes build/; set BUILD_DIR if yours differs.
ARG BUILD_DIR=""
RUN <<'EOF'
set -eu
if [ ! -f package.json ]; then
    cp -r /placeholder /site
    exit 0
fi
npm run build
if [ -z "$BUILD_DIR" ]; then
    if [ -d build ]; then BUILD_DIR=build; else BUILD_DIR=dist; fi
fi
cp -r "$BUILD_DIR" /site
EOF

FROM caddy:2.10-alpine
COPY deploy/Caddyfile /etc/caddy/Caddyfile
COPY --from=build /site /srv
