# GridPulse — electricity-market analysis dashboard

**Status:** revised implementation plan, not a delivered system. **Team:** 3 developers. **Build window:** 25 hours.

## Project overview

GridPulse brings together GB electricity wind forecasts, demand forecasts and imbalance settlement prices. The backend applies the team's existing algorithms, and a polished frontend helps users explore the resulting indicators, changing conditions and historical events.

**Pitch:** “GridPulse turns electricity-market data into an interactive view of changing wind, demand and price conditions, with advanced analysis that users can inspect.”

## Architecture and technology roles

```text
Elexon API → CSV exporter → backend importer → Tiger Data
                                                ↓
                                     backend analysis algorithms
                                                ↓
                                           backend API
                                                ↓
                                        frontend dashboard

Extension: user question → Gemini agent → backend data/analysis tools → answer
Hosting: Vultr for the application; Tiger Data remains the database service.
Optional: Snowflake API integration after a useful role is established.
```

CSV is the current upstream handoff; reuse the exporter already being built. Tiger Data is the central application store. Keep database access and algorithms in the backend; the frontend consumes its API. Choose frontend/backend frameworks based on team familiarity in the first hour.

## MVP acceptance criteria

**Confirmed frontend scope:** Overview, Explore and Methods are MVP pages. The immediate implementation focus is Overview only. Investigate/Gemini and the synoptic weather map remain extensions.


1. Real exported CSV data can be imported repeatedly into Tiger Data without duplicate logical records.
2. The backend runs the existing algorithms on stored data and returns documented outputs.
3. The dashboard shows a selected period's forecasts, prices and available algorithm indicators with clear units.
4. Users can explore a historical event and inspect its underlying data and analysis.
5. Freshness, missing data, algorithm identity and limitations are visible.
6. A Vultr-hosted application is reachable, with a local cached demo available if hosting or connectivity fails.

Visual polish is a core goal. Prioritise a coherent overview and useful interactions before adding more screens.

## Analysis scope

No ML model training is part of this build: the classical forecasters (issue #6) re-estimate their statistical parameters (up to 64 regression coefficients) on every call, and no fitted model is persisted or reused. Do not promise classifier probabilities, learned thresholds, calibration or SHAP explanations. The algorithm owner defines supported calculations, required history, parameters, score meanings and runtime constraints. Suggested dashboard signals are candidates until that contract is confirmed.

Keep retrospective observations distinguishable from forecasts. If presenting an alert as having been available historically, use only inputs published by its declared analysis cutoff. Otherwise label the view as retrospective analysis.

## Extensions and hosting

- **Gemini API:** backend agent calling existing data and analysis tools to answer higher-level questions, with supporting periods and results.
- **Vultr:** deploy the application early enough to catch configuration and network issues; avoid creating a separate infrastructure project.
- **Snowflake API:** optional sponsor-track exploration. Decide on a useful role, such as comparing a hosted model's analysis of the same evidence, before implementation. No duplicate data platform is required for the MVP.

## Document map

- [FRONTEND_DESIGN.md](FRONTEND_DESIGN.md): confirmed reuse patterns, agent-driven view changes and synoptic-map requirements.
- [FRONTEND_API.md](FRONTEND_API.md): every endpoint, field, parameter and error the frontend uses.
- [ALGORITHM_DISPLAY.md](ALGORITHM_DISPLAY.md): how to present each algorithm's outputs and tables.

- [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md): agreed folders, ownership and frontend-to-backend request flow.

- [DATA_PIPELINE.md](DATA_PIPELINE.md): CSV import, database tables and data quality.
- [BACKEND_DESIGN.md](BACKEND_DESIGN.md): backend architecture, Tiger Data schema, API, algorithm contract and deployment.
- [ALGORITHMS_AND_ANALYSIS.md](ALGORITHMS_AND_ANALYSIS.md): integration contracts, backend API and validation.
- [BUILD_AND_DEMO.md](BUILD_AND_DEMO.md): ownership, 25-hour checkpoints, dashboard and hosting.
- [RESEARCH_AND_EXTENSIONS.md](RESEARCH_AND_EXTENSIONS.md): Gemini agent, optional Snowflake role and limitations.
