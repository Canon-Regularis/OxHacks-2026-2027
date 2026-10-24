# Existing algorithms and backend integration

## Scope

Integrate the advanced algorithms already developed by a team member. This project does not require training models. The P0 statistical methods are implemented and specified [below](#p0-algorithms-issue-6); other algorithms' formulas and capabilities are not yet specified in this repository, so do not replace them with invented rules or advertise unsupported outputs.

The backend owns CSV ingestion, database queries, algorithm execution and API responses. The frontend owns presentation and interaction.

## Agree the algorithm contract first

For each algorithm, its owner must provide:

- Name/version, entry point, dependencies and a representative input/output example.
- Required columns, units, time ordering, minimum history and forecast-version requirements.
- Parameters, defaults and whether it supports a historical cutoff or retrospective mode only.
- Output names, types, units, score ranges and what each result means.
- Missing-data behaviour, failure conditions, expected runtime and known limitations.

Wrap the implementation with a small adapter so database and UI code do not depend on its internal structures. Reuse that adapter for API requests and cached demo generation. Keep algorithm version, parameters and input provenance alongside results.

Candidate supporting calculations include residual demand, wind forecast revisions and period-to-period ramps. Implement only those required by the agreed algorithms or useful charts. A risk or anomaly score is an indicator unless its interpretation as a probability is supported.

## P0 algorithms (issue #6)

Pure adapters in `gridpulse.analysis.algorithms` wrap Orchid modules that are vendored verbatim at commit `ca8fc92`. Each method has one Orchid implementation; see [AGENTS.md](../../AGENTS.md) for provenance. The adapters perform no I/O. Algorithms registered with the runner (`analysis/contract.py`) are expected to call them; none does yet.

### Conventions

- **Input:** a `HalfHourlySeries(start, values)` holds one value per 30-minute UTC period, starting at an aware `start` on :00 or :30. `None` or NaN marks a missing period. Windows, lags and season lengths count elapsed half-hours. Only `fourier_terms` and `CalendarRegression` follow the Europe/London clock, so they are exact on 46- and 50-period days. A seasonal forecast whose history or horizon spans a clock change carries a warning.
- **Missing data:** features return a series aligned to their input, with `None` wherever the trailing window is incomplete or touches a gap; the EWMA restarts after a gap. Forecasters, metrics, diagnostics and backtests raise `MissingDataError`. Nothing is imputed.
- **Errors:** invalid values raise an `AlgorithmError`, including numbers of the wrong kind (a float or bool where an integer is expected, a string where a number is expected). Passing the wrong kind of object where a series or sequence is expected may raise `TypeError` or `AttributeError`.
  - `InvalidInputError` (also a `ValueError`) reports bad arguments. Its subclasses are:
    - `MissingDataError`
    - `InsufficientDataError` (`.required`, `.available`)
    - `DegenerateDataError`, raised when the values a statistic is computed from (the series, its differences or the squared residuals) have no variation
  - `CalculationError` means a calculation on validated input failed: Orchid raised an error or returned non-finite output, an intermediate value overflowed a float, or (in `backtest`) a forecaster returned a malformed forecast.
- **Availability:** features use trailing windows only. `backtest` fits each fold on data before its origin. `Forecast.residuals` and the diagnostics use parameters estimated on the whole history, so they are retrospective and must not be used as features.
- **Determinism:** every call fits a fresh Orchid object, with a fixed seed wherever Orchid takes one, so identical inputs give identical results.
- **Score meanings:** WMAPE and sMAPE are fractions, not percentages. P-values below about 1e-16 are reported as 0. Intervals are model-based; "sqrt-horizon" intervals are a heuristic. No output is a probability of a price event.

| # | Family | GridPulse entry points | Orchid source |
|---|---|---|---|
| 1 | Rolling-origin backtesting | `evaluation.backtest` | AR `RollingOriginSplitter`, used for fold geometry only |
| 2 | Naive baselines | `forecasting.Naive`, `SeasonalNaive`, `Naive2`, `Drift`, `Mean` | FB `Naive`, `SeasonalNaive`, `Naive2`, `Drift`, `MeanForecast` |
| 3 | Trailing MA and EWMA | `signals.trailing_mean`, `trailing_weighted_mean`, `exponential_moving_average`; `forecasting.MovingAverage`, `WeightedMovingAverage` | SM `simple_moving_average`, `weighted_moving_average`, `exponential_moving_average`; FB `MovingAverageForecaster`, `WeightedMovingAverage` |
| 4 | Rolling dispersion and quantiles | `signals.rolling_variance`, `rolling_std`, `rolling_median`, `rolling_quantile`, `rolling_rms` | FB `variance`, `std`, `median`, `quantile`; SP `rms` |
| 5 | Regular and seasonal differences | `signals.difference(lag=, order=)` | SM `difference` |
| 6 | Seasonal basis functions | `calendar.fourier_terms`; `forecasting.FourierRegression`, `SeasonalDummyRegression`, `CalendarRegression` | FB `fourier_features`, `FourierForecaster`, `SeasonalDummyForecaster`, `LinearRegressionForecaster` |
| 7 | ACF / PACF | `diagnostics.autocorrelation`, `partial_autocorrelation` | AR `acf`, `pacf` (Levinson–Durbin) |
| 8 | Cross-correlation | `diagnostics.cross_correlation` | AR `ccf` |
| 9 | Exponential smoothing | `forecasting.ETS`, `AutoETS` | FB `ETSForecaster`, `auto_ets` |
| 10 | ARIMA | `forecasting.ARIMA`, `AutoARIMA` | AR `ARIMA` (conditional least squares), `nsdiffs`, `ndiffs`, `auto_arima` |
| 11 | Linear and seasonal regression | `forecasting.TrendRegression`, `SeasonalDummyRegression`, `CalendarRegression` | FB `TrendForecaster`, `SeasonalDummyForecaster`, `LinearRegressionForecaster` |
| 12 | Forecast accuracy | `evaluation.mae`, `rmse`, `mase`, `rmsse`, `wmape`, `smape`, `pinball_loss` | AR functions of the same names |
| 13 | Prediction intervals | `Forecast.intervals`; `evaluation.empirical_coverage`, `interval_score`, `msis` | Model `forecast(level=…)`; AR functions of the same names |
| 14 | Residual diagnostics | `diagnostics.residual_diagnostics` | AR `ljung_box`, `durbin_watson`, `jarque_bera`, `heteroskedasticity_test` (ARCH-LM) |

AR, FB, SM and SP are Orchid's `autoregressive`, `forecasting_baseline`, `smoothing` and `spectral` modules.

### Forecasters

Every forecaster is a frozen dataclass that validates its parameters on construction and exposes `min_history`. Calling `forecast(history, horizon, *, levels=(0.8, 0.95))` returns a `Forecast` with these fields:

| Field | Contents |
|---|---|
| `source` | the Orchid callable used |
| `params` | configuration and data-driven choices, plus fitted smoothing parameters (`ETS`, `AutoETS`) and fitted coefficients and σ² (`ARIMA`, `AutoARIMA`) |
| `origin` | the first forecast period |
| `mean` | point forecasts |
| `intervals` | one per level, in ascending order |
| `interval_method` | how the intervals were produced |
| `residuals` | in-sample residuals, after the model's warm-up |
| `warnings` | conditions flagged during the fit |

| Forecaster | Minimum history | Intervals | Runtime, 4 weeks / 92 days |
|---|---|---|---|
| `Naive()` | 2 | `random-walk` | < 0.1 s / < 0.1 s |
| `SeasonalNaive(m)` | m + 1 | `seasonal-random-walk` | < 0.1 s / < 0.1 s |
| `Drift()` | 3 | `sqrt-horizon` | < 0.1 s / < 0.1 s |
| `Mean()` | 2 | `analytic` | < 0.1 s / < 0.1 s |
| `Naive2(m)` | 2m + 1 | `sqrt-horizon` | < 0.1 s / < 0.1 s |
| `MovingAverage(window)` | window + 1 | `sqrt-horizon` | < 0.1 s / < 0.1 s |
| `WeightedMovingAverage(weights)` | k + 1 for k weights | `sqrt-horizon` | < 0.1 s / < 0.1 s |
| `ETS(trend=, damped=, seasonal=, season_length=)` | 2m with a season; otherwise 2, plus 1 for a trend and 1 for damping | `analytic`; `sqrt-horizon` for a multiplicative season | 0.5–0.7 s / 2 s |
| `AutoETS(season_length=)` | 2m with a season; otherwise 4 | `analytic` | 6–25 s / 30–60 s |
| `ARIMA(p, d, q, seasonal_differencing=, season_length=)` | d + D·m + max(100, 10(p + q + 1)) | `analytic` | 3–7 s / 13–31 s |
| `AutoARIMA(season_length=, seasonal_differencing=, max_p=3, max_q=2)` | 1 + max(100, 10(max_p + max_q + 1)); with a season, plus m, and at least 6m when D is tested | `analytic` | 2 min / 6–7.5 min |
| `TrendRegression(degree=1)` | degree + 2 | `sqrt-horizon` | < 0.1 s / < 0.1 s |
| `SeasonalDummyRegression(m, degree=1)`, m ≤ 48 | m + degree + 1 | `sqrt-horizon` | 0.5–0.8 s / 2–4 s |
| `FourierRegression(m, harmonics, degree=1)` | max(m, 2 × harmonics + degree + 2) | `sqrt-horizon` | < 0.1 s / 0.1–0.8 s |
| `CalendarRegression(daily_harmonics, weekly_harmonics=0, degree=1)` | max(48, or 336 with weekly terms; columns + 1) | `sqrt-horizon` | < 0.1 s / 0.2–0.6 s |

Runtimes are wall-clock times for one fit and a 48-period forecast on synthetic half-hourly data with a daily cycle. They were measured single-threaded with Python 3.14 on a battery-powered laptop (Intel Core 7 150U). Repeated runs varied by up to 6×, so treat them as orders of magnitude.

The configurations measured were:

- `ETS`, `AutoETS` and `SeasonalDummyRegression` with a daily season (m = 48);
- `ARIMA(2, 0, 1)`, with and without a seasonal difference;
- `AutoARIMA` searching without a season;
- `FourierRegression` with 4 harmonics;
- `CalendarRegression` with 4 daily and 2 weekly harmonics.

Minimums are stricter than Orchid's where Orchid would accept a fit it cannot use:

- **Warm-up residuals:** Orchid's naive, drift and moving-average forecasters compute σ from warm-up placeholder residuals. The adapters drop those residuals, rescale the intervals, and require history for at least one informative residual.
- **Statistical tests:** `Naive2` needs more than two seasons for its seasonality test. `AutoARIMA` needs six seasons to test for a seasonal difference: with fewer, Orchid's test picks one even for white noise.
- **ARIMA:** needs at least max(100, 10(p + q + 1)) observations after differencing, and rejects a constant differenced series.
- **Clock slots:** with 23 or more daily harmonics and no weekly terms, `CalendarRegression` raises `InsufficientDataError` for a history whose clock change leaves too few distinct half-hour slots to identify the terms.

ARIMA offers no seasonal AR or MA terms, because Orchid's are infeasible at m = 48. It also offers no Box–Cox transform, which negative prices rule out. Fits run on data scaled to the differenced history's range. A fit whose residual standard deviation is below 1e-6 of that range reaches Orchid's σ² floor and raises `CalculationError`. Orchid's seasonal-difference test also fires on random-walk-like series without a seasonal cycle, so give `AutoARIMA` an explicit `seasonal_differencing` when it is known; for example, wind has no daily cycle to difference.

Run `AutoETS`, `ARIMA`, `AutoARIMA` and backtests of fitted models in batch jobs, never on the request path. For example, a 14-fold, day-ahead backtest on four weeks of data took about 5 s for seasonal `ETS` and about 30 s for `ARIMA(2, 0, 1)`.

### Evaluation and diagnostics

- **`backtest(forecaster, series, *, initial, horizon, step=None, window=None, max_folds=50, season_length=48, levels=(0.8, 0.95))`**
  - Uses an expanding window by default, or a sliding one when `window` is given; `step` defaults to `horizon`.
  - Reports every fold with a status of `ok`, `insufficient_data`, `degenerate_data` or `failed`, and counts them as `n_ok` and `n_excluded`. Requesting more than `max_folds` folds is an error, not a truncation.
  - `metrics` pools MAE, RMSE, coverage and interval score over the `ok` folds. MASE, RMSSE and MSIS are averaged over folds, each fold scaled by its own training window. `by_horizon` scores each lead step.
- **Metric wrappers** require equal-length, non-empty, complete inputs. The scaled metrics also need:
  - `len(training) > season_length`, otherwise `InsufficientDataError`;
  - a seasonal-naive scale above 1e-12 × max(1, mean |training|), otherwise `DegenerateDataError`. The scale is the mean absolute seasonal difference, or the root mean square for RMSSE, which must also exceed 1e-6.

  Otherwise Orchid would silently return 0.0, divide by its 1e-12 clamp, or return a value dominated by rounding error.
- **`autocorrelation` / `partial_autocorrelation(series, *, max_lag, alpha=0.05)`** return lags 1..K with a zero-centred band: Bartlett for the ACF, ±z/√n for the PACF.
- **`cross_correlation(response, driver, *, max_lag, alpha=0.05)`** covers lags −K..K, and the value at k is corr(response[t+k], driver[t]). A peak at k > 0 means the driver leads by k half-hours. The band assumes white noise, so it overstates significance between autocorrelated series. This is association, not causation.
- **`residual_diagnostics(residuals, *, ljung_box_lags, model_df=0, arch_lags=12)`** returns:
  - Ljung–Box, Jarque–Bera and ARCH-LM statistics with p-values (the Ljung–Box p is `None` where it cannot be tested);
  - the Durbin–Watson statistic.

  It requires at least 30 residuals and does not apply pass/fail thresholds.

### Registered algorithms

The P0 methods reach the API through eight algorithms in `analysis/algorithms/`. Runtimes are wall-clock times of `gridpulse analyse` from a laptop against the Tiger dev database, including loading inputs and storing results (about 15 s of each).

| Algorithm | Methods | Range to run | Runtime |
|---|---|---|---|
| `price_forecast_backtest` | `ETS`, `ARIMA`, `SeasonalNaive` via `backtest` | at most about 60 days (`max_folds`) | 110 s for 53 days |
| `price_baseline_backtest` | `Naive`, `SeasonalNaive`, `Naive2`, `Drift`, `Mean`, `MovingAverage`, `WeightedMovingAverage` | at most about 60 days | 22 s for 53 days |
| `price_regression_backtest` | `TrendRegression`, `SeasonalDummyRegression`, `FourierRegression`, `CalendarRegression` | at most about 60 days | 46 s for 53 days |
| `price_day_ahead_forecast` | `AutoETS`, `AutoARIMA`, `SeasonalNaive` | `--from` and `--to` the last settled day; forecasts the next | 20 s |
| `price_rolling_signals` | every `signals` function | any | 19 s for 115 days |
| `price_driver_correlograms` | `autocorrelation`, `partial_autocorrelation`, `cross_correlation` | any; uses its longest gap-free stretch | 14 s for 53 days |
| `price_residual_diagnostics` | `residual_diagnostics` on `ETS` and `ARIMA` residuals | any; fits the latest 56 gap-free days | 18 s for 53 days |
| `calendar_fourier_terms` | `fourier_terms` | any | 17 s for 61 days |

The three backtests share `_backtest.py`, which backtests one origin at a time so that a missing price excludes only the folds containing it (status `missing_data`). The correlograms and diagnostics use the longest gap-free stretch of the range and report it in a `sample` table. Run them with `gridpulse analyse <id> --from … --to …`; see the frontend's [display guide](ALGORITHM_DISPLAY.md) for how each is shown.

## Execution and result format

Start with explicit batch processing after an import. Store completed runs and serve their results quickly; avoid rerunning expensive algorithms whenever a chart renders. For interactive requests, bound date ranges and use caching or return a running status if needed. Never expose partial results as a completed run.

A period-level result should identify:

```text
run_id, algorithm_id, algorithm_version,
settlement_date, settlement_period, delivery_start_utc,
analysis_mode, optional_cutoff_utc,
indicator_name, value, unit, status
```

Statuses must distinguish a valid zero, missing inputs and a failed calculation. Forecast and actual values must be separately named.

## Proposed backend API

The implemented design, including `/api/catalog`, `/api/snapshot` and `/api/extremes`, is in [BACKEND_DESIGN.md](BACKEND_DESIGN.md#api), and the algorithm interface is in [its contract section](BACKEND_DESIGN.md#algorithm-contract-matthew). The table below is the original proposal.

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Backend readiness and database/cache mode |
| `GET /api/coverage` | Available dates, datasets, freshness and missingness |
| `GET /api/series?from=...&to=...` | Wind/demand forecasts and prices for a bounded range |
| `GET /api/analysis?from=...&to=...` | Completed algorithm results and run metadata |
| `GET /api/periods/{date}/{period}` | Source values and available analysis for a drill-down |

Return UTC timestamps, explicit units and source/algorithm metadata. Specify which forecast version is shown. If historical cutoff selection is supported, use the same explicit cutoff across charts and analysis. Use consistent error responses for invalid ranges, missing coverage and unavailable processing. Keep importing and recomputing behind internal commands or protected routes.

## Validation

Validate integration and analysis correctness rather than ML performance:

- Match the owner's known examples and expected outputs.
- Check ordering, units, joins, duplicates, missing inputs and negative prices.
- Exercise 46/48/50-period days and repeated local clock times.
- Repeat a run with the same inputs, parameters and version; explain any nondeterminism.
- Verify historical availability whenever results are presented as earlier-known alerts.
- Check frontend charts match backend values and stored run metadata.

Historical examples can compare indicators with observed prices, including cases where they do not align. Such examples alone do not establish predictive accuracy or causal effects. No train/test split, classifier metrics, calibration or SHAP deliverable is required.
