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
- **Vultr:** target hosting platform for the gridpulse/frontend/backend; retain a local cached demo fallback.
- **Extension:** Gemini API agent using backend tools to analyse data and answer abstract questions.
- **Optional track exploration:** Snowflake API, only after identifying a useful capability and completing the core demo.

No model training is planned. Algorithm formulas, score semantics and supported outputs must be supplied by their owner before integration.

## Project docs

Start with the [project overview](docs/gridpulse/README.md), then the [data pipeline](docs/gridpulse/DATA_PIPELINE.md), [algorithms and API contracts](docs/gridpulse/ALGORITHMS_AND_ANALYSIS.md), [25-hour build and demo plan](docs/gridpulse/BUILD_AND_DEMO.md), and [extensions](docs/gridpulse/RESEARCH_AND_EXTENSIONS.md). The [sponsor reference](docs/sponsors.md) contains background on available tracks.

The [frontend design brief](docs/gridpulse/FRONTEND_DESIGN.md) records confirmed KCPC reuse requirements and the synoptic/electricity-map and animated Gemini investigation extensions. Weather providers and exact agent UI tools remain undecided.

## Repository layout

**Confirmed frontend scope:** Overview, Explore and Methods are MVP pages. The immediate implementation focus is Overview only. Investigate/Gemini and the synoptic weather map remain extensions.


- `gridpulse/frontend/`: dashboard, UI components, charts and API client.
- `gridpulse/backend/`: Python package for routes, ingestion, database access, algorithms and optional agent.
- `gridpulse/db/migrations/`: schema changes; `gridpulse/contracts/`: shared API examples.
- `gridpulse/data/`: ignored raw CSVs and a future cached demo; `gridpulse/scripts/`: operational helpers.
- `gridpulse/deploy/`: Vultr configuration; `docs/`: team documentation.

Follow the [full layout, ownership boundaries and request flow](docs/gridpulse/REPOSITORY_STRUCTURE.md). Workstream instructions are in [gridpulse/frontend/README.md](gridpulse/frontend/README.md) and [gridpulse/backend/README.md](gridpulse/backend/README.md).

## Setup and demo

The backend (FastAPI over Tiger Data) runs end to end against the team's Tiger Cloud instance, reading the loader's `raw` schema through `gridpulse.*` views; data modelling is still to come. Its design is in [BACKEND_DESIGN.md](docs/gridpulse/BACKEND_DESIGN.md), and the migrate, analyse, test and serve commands are in the [backend README](gridpulse/backend/README.md). Put the Tiger Cloud credentials file in the repository root, or a `DATABASE_URL` in `gridpulse/.env`; both are loaded automatically and gitignored. Keep database credentials and service keys server-side. The frontend framework is not yet bootstrapped.

Add frontend startup and Vultr deployment commands here as they land.

The demo must show which data is cached, its source and freshness, algorithm version and limitations. This is analysis of electricity prices and forecasts; indicators are not claims of grid failure or guaranteed future outcomes.
