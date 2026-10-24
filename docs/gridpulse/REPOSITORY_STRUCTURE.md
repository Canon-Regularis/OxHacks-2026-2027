# Repository structure and ownership

This is the agreed layout. The folders and base files are scaffolded; HTTP handlers, database integration, algorithms and frontend framework setup remain to be implemented. Avoid introducing parallel locations for the same responsibility.

## Layout

```text
AGENTS.md                         # Repository-wide guidance and reuse sources
README.md                         # Project overview and setup status
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
backend/
  pyproject.toml                  # Python package; runtime dependencies pending
  src/gridpulse/
    main.py                       # Application/framework wiring
    config.py                     # Server configuration
    api/
      analysis.py                 # HTTP analysis endpoint
      schemas.py                  # Runtime request/response validation
    db/
      connection.py               # Tiger Data connections
      queries.py                  # SQL reads/writes
    ingestion/csv_import.py        # Existing CSV exporter handoff
    analysis/
      features.py                 # ASOF joins and window calculations
      service.py                  # Query → prepare → calculate → store/retrieve
      algorithms/wind_analysis.py # Pure algorithm/Orchid adapter placeholder
    agent/
      service.py                  # Optional Gemini orchestration
      tools.py                    # Existing-service adapters
  tests/
    ingestion/
    analysis/
    api/
db/migrations/                    # Versioned database schema changes
contracts/
  README.md                       # Shared API agreement
  examples/analysis.json           # Labelled synthetic response fixture
data/
  raw/                            # Ignored local CSVs
  demo/                           # Permitted real-data demo cache
scripts/                          # Thin operational commands
deploy/                           # Vultr deployment configuration
docs/gridpulse/                   # Project documentation
```

Python package directories also contain `__init__.py`. Framework-specific entry points, manifests and dependencies should be added after checking WWW/Components compatibility; do not infer a framework from the folder names. Root README and workstream READMEs must gain exact startup commands when these exist.

## Ownership

| Workstream | Primary files |
|---|---|
| Data/database | `backend/src/gridpulse/ingestion/`, `backend/src/gridpulse/db/`, `db/migrations/`, source-data documentation |
| Algorithms/backend | `backend/src/gridpulse/analysis/`, `backend/src/gridpulse/api/`, application wiring; later `agent/` |
| Frontend/product | `frontend/`, visual design, dashboard interactions and demo presentation |
| Shared agreements | `contracts/`, `.env.example`, root README and deployment coordination |

Agree edits to shared contracts with the other affected owner. No developer names are assigned here until the team confirms them.

## Request flow

Example target flow for a historical analysis request (not implemented yet):

```text
frontend/src/pages/ (dashboard screen)
  → frontend/src/lib/api.ts
  → GET /api/analysis?from=2026-10-24&to=2026-10-24
  → backend/src/gridpulse/api/analysis.py
  → backend/src/gridpulse/analysis/service.py
      → db/queries.py → Tiger Data
      → analysis/features.py
      → analysis/algorithms/wind_analysis.py
  → api/schemas.py response serialization
  → frontend/src/lib/api.ts
  → dashboard state
  → frontend/src/components/charts/
```

For expensive calculations, run analysis after an import and return stored completed results from the API. The frontend should not trigger a full recomputation each time a chart renders. The same analysis service is the integration point for batch commands and eventual Gemini tools.

## Boundaries to preserve

- Routes handle HTTP validation/status and call services; SQL belongs in `db/`.
- Analysis services coordinate inputs, preparation, algorithm calls and result provenance.
- Algorithms accept prepared data and parameters, returning documented results; keep HTTP and database access out of them.
- Ingestion validates actual CSV schemas and handles idempotent imports; do not duplicate the existing exporter.
- Frontend pages/features call the shared API client. UI/chart components receive data rather than implementing backend calculations.
- Keep schema SQL in root `db/migrations/`; Python database access stays under the backend package.
- Shared contract examples describe API shapes; backend schemas enforce them at runtime once implemented.
- Gemini calls existing services through bounded tools. It is optional and must not gate dashboard startup.

## Reference code and fixtures

Inspect Orchid, WWW and Components before reuse. Keep reference checkouts outside the application tree; bring selected files into their responsibility folders with licence/attribution, and record upstream revisions in [AGENTS.md](../../AGENTS.md). No reference code is included in this scaffold.

Synthetic API examples belong in `contracts/examples/` and must be visibly labelled during UI development. Real cached demo data belongs in `data/demo/` with provenance. Raw downloads stay in ignored `data/raw/`.

## Implementation status

The scaffold establishes locations and package metadata only. It does not expose a working API, connect to Tiger Data, calculate indicators, run Gemini or deploy to Vultr. Do not present stubs or fixtures as implemented functionality. Add meaningful tests as each real implementation lands.
