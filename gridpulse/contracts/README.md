# Shared API contracts

Frontend and backend owners agree changes here before integration. This directory is a shared specification, not another runtime service. Backend runtime validation lives in `gridpulse/backend/src/gridpulse/api/schemas.py`, and `tests/api/test_contracts.py` checks that every example here still parses with those schemas.

| Method/path | Purpose | Example |
|---|---|---|
| `GET /api/health` | Mode (`live`/`record`/`replay`) and database status | `health.json` |
| `GET /api/catalog` | Series, dataset coverage/freshness, algorithms and latest runs, limits | `catalog.json` |
| `GET /api/snapshot?at=` | Ticker values for the latest (or a historical) settled period | `snapshot.json` |
| `GET /api/series?ids=&from=&to=&resolution=auto&cutoff=` | Columnar series on one shared time axis | `series.json` |
| `GET /api/analysis?algorithm=&from=&to=&resolution=auto&run_id=` | Stored per-period algorithm results with run provenance and table names | `analysis.json` |
| `GET /api/analysis/runs/{run_id}/tables?name=` | A run's tables (scorecards, ACF by lag, diagnostics) as `{description, columns, rows}` | `analysis_tables.json` |
| `GET /api/extremes?series=&from=&to=&order=desc&limit=10` | Highest/lowest settlement periods | `extremes.json` |
| `GET /api/periods/{date}/{period}` | One period: values, forecast vintages and analysis outputs | `period.json` |
| any error | `{"detail": {"code", "message"}}` | `error.json` |

Details, including resolution rules, cutoff semantics and error codes, are in the [backend design](../../docs/gridpulse/BACKEND_DESIGN.md#api).

Conventions: `from`/`to` are inclusive settlement dates (`YYYY-MM-DD`). Instants are UTC with a `Z` suffix. Each series is a `values` array aligned with the shared `t` array, with `null` for missing data, which is never zero. At 30m resolution the response includes `settlement_date`/`settlement_period` arrays. Never assume 48 periods per day.

Every example carries a `_fixture` label and contains synthetic values only. Real responses do not have that key. Update examples and `schemas.py` together.


## Overview weather context

- `GET /api/weather/locations` → `{locations: [{id,label,latitude,longitude}], warnings}`.
- `GET /api/weather?date=YYYY-MM-DD&location=<id>` → `{location, points, source, warnings, window_hours:168}`.
- Each point: `t` (completed UTC hour end), `temperature` (°C), `pressure` (MSL hPa), `temperature_change`, `pressure_change`, `price_change` (£/MWh), `temperature_correlation`, `pressure_correlation`, `temperature_samples`, `pressure_samples`.

All numeric values except counts may be null. Correlations compare hourly changes over seven days, require 24 valid pairs and never include future hours. Source is stored Open-Meteo archived forecasts, not verified as-of observations. Unknown locations return `unknown_weather_location` (404); missing migration returns `weather_not_configured` (503). See runtime schemas and weather-analysis tests for the executable contract.
