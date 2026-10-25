# Repository structure and ownership

This is the agreed layout. The FastAPI backend, weather integration, P0 algorithm layer, Vultr deployment and SvelteKit Overview are implemented. Explore, Methods and agent features remain to be implemented. Avoid introducing parallel locations for the same responsibility.

## Layout

Repository guidance, documentation and licence stay at the root. All application code, data, contracts, configuration and deployment files live under `gridpulse/`. Paths below are relative to the repository root.

```text
AGENTS.md                         # Repository-wide guidance and reuse sources
README.md                         # Project overview and setup status
LICENSE
.gitignore
.github/workflows/deploy.yml      # CI/CD: tests, images, Vultr deploy
docs/                             # Team documentation
gridpulse/                        # All application code and operational files
  .env.example                      # Server-side configuration placeholders
  frontend/                         # Frontend/product workstream
    src/
      pages/                        # Dashboard screens and request state
      components/
        ui/                         # Shared UI primitives; Components reference
        charts/                     # Chart presentation
      features/                     # Feature-specific frontend code
      lib/api.ts                    # Shared HTTP request boundary
      styles/                       # Styles/design tokens; WWW reference
    public/                         # Static assets
  backend/                          # Design: docs/gridpulse/BACKEND_DESIGN.md
    pyproject.toml                  # FastAPI, psycopg 3, pandas; `gridpulse` CLI entry point
    src/gridpulse/
      main.py                       # FastAPI app wiring, middleware, DB error handlers
      config.py                     # Server configuration (env / gridpulse/.env)
      settlement.py                 # Settlement period ↔ UTC, bucket grids (46/48/50)
      catalog.py                    # Series ids → database relations, labels, units
      cli.py                        # migrate | analyse | algorithms
      api/
        system.py                   # /health, /catalog
        market.py                   # /snapshot, /series, /extremes, /periods
        analysis.py                 # /analysis (stored results only)
        deps.py                     # Connection, date range, resolution, errors
        schemas.py                  # Runtime response validation
        demo_cache.py               # Record/replay offline demo fallback
      db/
        connection.py               # Tiger Data connections
        queries.py                  # All SQL reads/writes
        migrate.py                  # Applies gridpulse/db/migrations
      ingestion/csv_import.py       # Placeholder: the team's loader writes Tiger's raw schema
      analysis/
        contract.py                 # Algorithm interface and input columns
        features.py                 # ASOF/latest input frame built in SQL
        service.py                  # Prepare → calculate → check → store/retrieve
        algorithms/                 # Registered algorithms (see __init__.py) and the Orchid P0 adapters
          orchid/                   # Orchid modules vendored verbatim
      agent/
        service.py                  # Optional Gemini orchestration
        tools.py                    # Existing-service adapters
    tests/                          # pytest; no database needed
      ingestion/
      analysis/                     # Contract tests for every registered algorithm; adapter tests
        orchid/                     # Orchid's upstream test suites, ported
      api/
  db/migrations/                    # Re-runnable gridpulse schema: views over raw, analysis tables
  contracts/
    README.md                       # Shared API agreement
    examples/*.json                 # Labelled synthetic response per endpoint
  data/
    raw/                            # Ignored local CSVs
    demo/                           # Permitted real-data demo cache
  scripts/                          # Thin operational commands
  deploy/                           # Vultr: per-environment Compose stack, shared HTTPS proxy, deploy script
  backend/Dockerfile, frontend/Dockerfile  # Images; build context is gridpulse/
```

Python package directories also contain `__init__.py`. Framework-specific entry points, manifests and dependencies should be added after checking WWW/Components compatibility; do not infer a framework from the folder names. Root README and workstream READMEs must gain exact startup commands when these exist.

## Ownership

| Workstream | Primary files |
|---|---|
| Data/database | `gridpulse/backend/src/gridpulse/ingestion/`, `gridpulse/backend/src/gridpulse/db/`, `gridpulse/db/migrations/`, source-data documentation |
| Algorithms/backend | `gridpulse/backend/src/gridpulse/analysis/`, `gridpulse/backend/src/gridpulse/api/`, application wiring; later `agent/` |
| Frontend/product | `gridpulse/frontend/`, visual design, dashboard interactions and demo presentation |
| Shared agreements | `gridpulse/contracts/`, `gridpulse/.env.example`, root README and deployment coordination |

Agree edits to shared contracts with the other affected owner. No developer names are assigned here until the team confirms them.

## Request flow

Example target flow for a historical analysis request (not implemented yet):

```text
gridpulse/frontend/src/pages/ (dashboard screen)
  → gridpulse/frontend/src/lib/api.ts
  → GET /api/analysis?from=2026-10-24&to=2026-10-24
  → gridpulse/backend/src/gridpulse/api/analysis.py
  → gridpulse/backend/src/gridpulse/analysis/service.py
      → db/queries.py → Tiger Data
      → analysis/features.py
      → analysis/algorithms/wind_analysis.py
  → api/schemas.py response serialization
  → gridpulse/frontend/src/lib/api.ts
  → dashboard state
  → gridpulse/frontend/src/components/charts/
```

For expensive calculations, run analysis after an import and return stored completed results from the API. The frontend should not trigger a full recomputation each time a chart renders. The same analysis service is the integration point for batch commands and eventual Gemini tools.

## Boundaries to preserve

- Routes handle HTTP validation/status and call services; SQL belongs in `db/`.
- Analysis services coordinate inputs, preparation, algorithm calls and result provenance.
- Algorithms accept prepared data and parameters, returning documented results; keep HTTP and database access out of them.
- Ingestion validates actual CSV schemas and handles idempotent imports; do not duplicate the existing exporter.
- Frontend pages/features call the shared API client. UI/chart components receive data rather than implementing backend calculations.
- Keep schema SQL in `gridpulse/db/migrations/`; Python database access stays under the backend package.
- Shared contract examples describe API shapes; backend schemas enforce them at runtime once implemented.
- Gemini calls existing services through bounded tools. It is optional and must not gate dashboard startup.

## Reference code and fixtures

Inspect Orchid, WWW and Components before reuse. Keep reference checkouts outside the application tree; bring selected files into their responsibility folders with licence/attribution, and record upstream revisions in [AGENTS.md](../../AGENTS.md). Orchid's P0 modules are vendored in `analysis/algorithms/orchid/`. No WWW or Components code is included yet.

Synthetic API examples belong in `gridpulse/contracts/examples/` and must be visibly labelled during UI development. Real cached demo data belongs in `gridpulse/data/demo/` with provenance. Raw downloads stay in ignored `gridpulse/data/raw/`.

## Implementation status

The backend infrastructure runs end to end against Tiger Data. That covers the API, the `gridpulse` views over the loader's `raw` schema, the algorithm runner, result storage and the demo cache, with example algorithms. GitHub Actions tests each change and deploys `main` to development and `production` to production on Vultr ([deployment](../../gridpulse/deploy/README.md)). The Orchid P0 methods (issue #6) are implemented and tested as pure adapters in `analysis/algorithms/`, but no registered algorithm uses them yet. Data modelling and Gemini are not implemented; the local SvelteKit Overview is implemented; see [BACKEND_DESIGN.md](BACKEND_DESIGN.md#open-items). Do not present stubs or fixtures as implemented functionality. Add meaningful tests as each real implementation lands.

## Current frontend entry points

SvelteKit owns routing in `gridpulse/frontend/src/routes/`; the Overview route delegates to `src/features/overview/Overview.svelte`. Root layout context owns shared view preferences; browser storage persists the chosen date/location. Source charts are under `src/components/charts/`, and selected KCPC components are recorded in `src/lib/vendor/kcpc/README.md`. The same-origin GET API proxy is `src/routes/api/[...path]/+server.ts`.

Weather requests follow that proxy to backend `api/weather.py`, which reads the `gridpulse.weather_hourly` view through `db/queries.py` and calls `analysis/weather.py`. The entire day's weather statistics are returned at once; hover selects them locally. See the frontend README for exact run/check commands.
