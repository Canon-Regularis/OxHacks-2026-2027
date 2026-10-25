# Shared API contracts

Frontend and backend owners agree changes here before integration. This directory is a shared specification, not another runtime service. Backend runtime validation lives in `gridpulse/backend/src/gridpulse/api/schemas.py`, and `tests/api/test_contracts.py` checks that every example here still parses with those schemas.

| Method/path | Purpose | Example |
|---|---|---|
| `GET /api/health` | Mode (`live`/`record`/`replay`) and database status | `health.json` |
| `GET /api/catalog` | Series, dataset coverage/freshness, algorithms and latest runs, limits | `catalog.json` |
| `GET /api/snapshot?at=` | Ticker values for the latest (or a historical) settled period | `snapshot.json` |
| `GET /api/series?ids=&from=&to=&resolution=auto&cutoff=` | Columnar series on one shared time axis | `series.json` |
| `GET /api/analysis?algorithm=&from=&to=&resolution=auto&params=&run_id=` | Stored per-period algorithm results with run provenance and table names; default params unless `params` (JSON overrides) or `run_id` is given | `analysis.json` |
| `GET /api/analysis/runs?algorithm=&limit=` | Successful runs of one algorithm with their params and overrides, newest first | `analysis_runs.json` |
| `GET /api/analysis/runs/{run_id}/tables?name=` | A run's tables (scorecards, ACF by lag, diagnostics) as `{description, columns, rows}` | `analysis_tables.json` |
| `GET /api/extremes?series=&from=&to=&order=desc&limit=10` | Highest/lowest settlement periods | `extremes.json` |
| `GET /api/periods/{date}/{period}` | One period: values, forecast vintages and analysis outputs | `period.json` |
| any error | `{"detail": {"code", "message"}}` | `error.json` |

The frontend reference, covering every field, parameter and error code, is [FRONTEND_API.md](../../docs/gridpulse/FRONTEND_API.md); how to present algorithm results is [ALGORITHM_DISPLAY.md](../../docs/gridpulse/ALGORITHM_DISPLAY.md). TypeScript types are in [`frontend/src/lib/types.ts`](../frontend/src/lib/types.ts).

Conventions: `from`/`to` are inclusive settlement dates (`YYYY-MM-DD`). Instants are UTC with a `Z` suffix. Each series is a `values` array aligned with the shared `t` array, with `null` for missing data, which is never zero. At 30m resolution the response includes `settlement_date`/`settlement_period` arrays. Never assume 48 periods per day.

## Planned algorithm adapters

`examples/algorithms/` holds a `/api/analysis` response (`<id>.analysis.json`) and, where the algorithm declares tables, a `/api/analysis/runs/{run_id}/tables` response (`<id>.tables.json`) for each adapter planned around the P0 Orchid modules. Together they cover every public method in the 14 P0 families (see [the P0 table](../../docs/gridpulse/ALGORITHMS_AND_ANALYSIS.md#p0-algorithms-issue-6)). They are proposals to build the frontend against; the adapters are not implemented yet, and ids, outputs and table columns may change when they are.

| Algorithm id | Page | Series outputs | Tables | P0 methods |
|---|---|---|---|---|
| `price_baseline_backtest` | Explore | Day-ahead forecast per baseline | `metrics`, `by_horizon`, `folds` | `Naive`, `SeasonalNaive` (48, 336), `Naive2`, `Drift`, `Mean`, `MovingAverage`, `WeightedMovingAverage` |
| `price_forecast_backtest` | Overview, Explore | Seasonal naive, ETS and ARIMA forecasts with 80%/95% bands | `metrics`, `by_horizon`, `folds`, `fold_params` | `ETS`, `ARIMA` |
| `price_regression_backtest` | Explore | Day-ahead forecast per regression | `metrics`, `by_horizon`, `folds` | `TrendRegression`, `SeasonalDummyRegression`, `FourierRegression`, `CalendarRegression` |
| `price_day_ahead_forecast` | Overview | Next-day AutoETS and AutoARIMA forecasts with 80%/95% bands | `selected_models`, `fit_warnings` | `AutoETS`, `AutoARIMA` |
| `price_rolling_signals` | Overview | Trailing mean, weighted mean, EWMA, variance, std, median, RMS, p10/p90, changes | none | every `signals` function |
| `calendar_fourier_terms` | Methods | Daily and weekly sine/cosine terms | none | `calendar.fourier_terms` |
| `price_driver_correlograms` | Explore | none (`indicators` empty) | `acf_price`, `pacf_price`, `ccf_residual_demand`, `ccf_wind` | `autocorrelation`, `partial_autocorrelation`, `cross_correlation` |
| `price_residual_diagnostics` | Methods | none | `ljung_box`, `tests` | `residual_diagnostics` |

Every backtest uses `evaluation.backtest`, and its `metrics` table carries every accuracy and interval metric: MAE, RMSE, WMAPE, sMAPE, MASE, RMSSE, coverage, interval score, MSIS and pinball loss. The backtests share column names so the frontend can render them with one component. Run `AutoETS`, `ARIMA`, `AutoARIMA` and every backtest of a fitted model in batch jobs only.

Every example carries a `_fixture` label and contains synthetic values only. Real responses do not have that key. Update examples, `schemas.py` and `frontend/src/lib/types.ts` together.

## Overview weather context

- `GET /api/weather/locations` → `{locations: [{id,label,latitude,longitude}], warnings}`.
- `GET /api/weather?date=YYYY-MM-DD&location=<id>` → `{location, points, source, warnings, window_hours:168}`.
- Each point: `t` (completed UTC hour end), `temperature` (°C), `pressure` (MSL hPa), `temperature_change`, `pressure_change`, `price_change` (£/MWh), `temperature_correlation`, `pressure_correlation`, `temperature_samples`, `pressure_samples`.

All numeric values except counts may be null. Correlations compare hourly changes over seven days, require 24 valid pairs and never include future hours. Source is stored Open-Meteo archived forecasts, not verified as-of observations. Unknown locations return `unknown_weather_location` (404); missing migration returns `weather_not_configured` (503). See runtime schemas and weather-analysis tests for the executable contract.
