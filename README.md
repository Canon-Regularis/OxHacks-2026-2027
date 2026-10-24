# GridPulse — OxHack 2026

An electricity-market analysis dashboard combining GB wind forecasts, demand forecasts and imbalance settlement prices with the team's existing analysis algorithms.

**Team:** Matthew, Mark, David. **Build window:** 25 hours. **Status:** repository skeleton created; application services and features are not yet implemented.

## MVP

```text
Elexon API → existing CSV exporter → backend ingestion → Tiger Data
                                                          ↓
                                                backend algorithms
                                                          ↓
                                                     backend API
                                                          ↓
                                                 frontend dashboard
```

- **Tiger Data:** central PostgreSQL/time-series store for imported source records and analysis results.
- **Backend:** imports CSVs, queries stored data, runs existing algorithms and serves chart-ready results.
- **Frontend:** polished overview, time-series charts, algorithm indicators and historical exploration.
- **Vultr:** target hosting platform for the frontend/backend; retain a local cached demo fallback.
- **Extension:** Gemini API agent using backend tools to analyse data and answer abstract questions.
- **Optional track exploration:** Snowflake API, only after identifying a useful capability and completing the core demo.

No model training is planned. Algorithm formulas, score semantics and supported outputs must be supplied by their owner before integration.

## Project docs

Start with the [project overview](docs/gridpulse/README.md), then the [data pipeline](docs/gridpulse/DATA_PIPELINE.md), [algorithms and API contracts](docs/gridpulse/ALGORITHMS_AND_ANALYSIS.md), [25-hour build and demo plan](docs/gridpulse/BUILD_AND_DEMO.md), and [extensions](docs/gridpulse/RESEARCH_AND_EXTENSIONS.md). The [sponsor reference](docs/sponsors.md) contains background on available tracks.

## Repository layout

- `frontend/`: dashboard, UI components, charts and API client.
- `backend/`: Python package for routes, ingestion, database access, algorithms and optional agent.
- `db/migrations/`: schema changes; `contracts/`: shared API examples.
- `data/`: ignored raw CSVs and a future cached demo; `scripts/`: operational helpers.
- `deploy/`: Vultr configuration; `docs/`: team documentation.

Follow the [full layout, ownership boundaries and request flow](docs/gridpulse/REPOSITORY_STRUCTURE.md). Workstream instructions are in [frontend/README.md](frontend/README.md) and [backend/README.md](backend/README.md).

## Setup and demo

Base modules and a Python package manifest are present. They contain no working server, database integration or algorithms, and the frontend framework is not bootstrapped. See the backend README for optional package installation. Copy `.env.example` to a local `.env` when configuring the implementation; automatic environment loading is not yet implemented. Keep database credentials and service keys server-side.

As implementations land, add exact database migration, CSV import, backend/frontend startup and Vultr deployment commands here. No runnable demo command exists yet.

The demo must show which data is cached, its source and freshness, algorithm version and limitations. This is analysis of electricity prices and forecasts; indicators are not claims of grid failure or guaranteed future outcomes.
