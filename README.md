# GridPulse — OxHack 2026

An electricity-market analysis dashboard combining GB wind forecasts, demand forecasts and imbalance settlement prices with the team's existing analysis algorithms.

**Team:** Matthew, Mark, David. **Build window:** 25 hours. **Status:** implementation plan; the documented services and features are not yet implemented in this repository.

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

## Setup and demo

Application code, dependency manifests and launch commands are not yet present. Implementation must add exact local setup, database migration, CSV import, backend/frontend startup and Vultr deployment instructions here. Document required configuration with placeholders in `.env.example`; keep database credentials and service keys server-side.

The demo must show which data is cached, its source and freshness, algorithm version and limitations. This is analysis of electricity prices and forecasts; indicators are not claims of grid failure or guaranteed future outcomes.
