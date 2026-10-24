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

## Planned algorithm adapters

`examples/algorithms/` holds a `/api/analysis` response (`<id>.analysis.json`) and, where the algorithm declares tables, a `/api/analysis/runs/{run_id}/tables` response (`<id>.tables.json`) for each adapter planned around the P0 Orchid modules. They are proposals to build the frontend against; the adapters are not implemented yet, and ids, outputs and table columns may change when they are.

| Algorithm id | Page | Series outputs | Tables |
|---|---|---|---|
| `price_forecast_backtest` | Overview, Explore | Day-ahead forecasts per model, ETS 80%/95% bands | `metrics`, `by_horizon`, `folds` |
| `price_volatility_signals` | Overview | EWMA, rolling std, 7-day p10/p90, period change | none |
| `price_driver_correlograms` | Explore | none (`t` and `indicators` empty) | `acf_price`, `pacf_price`, `ccf_residual_demand`, `ccf_wind` |
| `price_residual_diagnostics` | Methods | none | `ljung_box`, `tests` |

Every example carries a `_fixture` label and contains synthetic values only. Real responses do not have that key. Update examples and `schemas.py` together.
