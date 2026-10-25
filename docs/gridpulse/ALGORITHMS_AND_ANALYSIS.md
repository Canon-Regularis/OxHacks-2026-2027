# Existing algorithms and backend integration

## Scope

Integrate the advanced algorithms already developed by a team member. This project does not require training models. The [P0](#p0-algorithms-issue-6) and [P1](#p1-algorithms-issue-7) statistical methods are implemented and specified below; other algorithms' formulas and capabilities are not yet specified in this repository, so do not replace them with invented rules or advertise unsupported outputs.

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

## P1 algorithms (issue #7)

The P1 families follow the [P0 conventions](#conventions) and add Orchid's `state_space` module (SS), vendored at the same commit. `price_p1_backtest` backtests Theta, AR, the discount DLM, a local-level `StructuralForecaster` and an `Ensemble` of the first three through the P0 fold machinery, then compares each with a daily seasonal naive by `diebold_mariano` at chosen lead steps; see [displaying algorithm results](ALGORITHM_DISPLAY.md#price_p1_backtest-live). `price_decomposition` runs `stl_decomposition`, `classical_decomposition`, `component_strength` and the centred smoothers; `price_stationarity` runs `adf_test`, `kpss_test`, `difference_orders` and `ar_spectrum`. No registered algorithm calls the remaining P1 entry points (the structural filters and smoother, `discount_filter`, VAR and GARCH) yet.

### Additional conventions

- **Availability:** `signals.trailing_savitzky_golay`, `filter_structural`, `discount_filter` and the forecasters use only values up to each period. The filters are causal only when their parameters come from earlier data. Everything else here is retrospective and must not be used as a feature: `decomposition`, `fit_structural`, `smooth_structural`, the unit-root tests, `difference_orders`, `ar_spectrum`, `garch`, `diebold_mariano`, `select_var_lags` and `var_analysis`.
- **Units:** results are in the data's units and do not depend on their scale or, outside multiplicative models, their offset. Inputs are rescaled wherever Orchid's absolute floors would otherwise make them.
- **Gaps:** the smoothers work on each run of present values, and `fit_structural`, `filter_structural`, `smooth_structural` and `discount_filter` accept gaps. Everything else raises `MissingDataError`, as in P0.
- **Structural models:** every state starts from a proper vague prior (variance 10⁴ in standardised units) instead of Orchid's exact diffuse initialisation. Log-likelihoods, AIC and BIC follow Orchid's exact-diffuse convention, with regressors standardised, so compare them only between models with the same state dimension.
- **P-values:** the unit-root tests report critical values and the levels at which they reject, not p-values.

| # | Family | GridPulse entry points | Orchid source |
|---|---|---|---|
| 15 | Theta | `forecasting.Theta` | FB `Theta` |
| 16 | Low-order AR | `forecasting.AR`; `diagnostics.ar_spectrum` | FB `ARForecaster` (Yule–Walker); SM `burg`, SP `ar_psd` |
| 17 | Stationarity | `diagnostics.adf_test`, `kpss_test`, `difference_orders` (used by `AutoARIMA`) | AR `adf_test`, `kpss_test`, `nsdiffs`, `ndiffs` |
| 18 | Seasonal decomposition | `decomposition.classical_decomposition`, `stl_decomposition`, `component_strength` | FB `classical_decomposition`, `trend_strength`, `seasonal_strength`; SM `stl_decomposition` |
| 19 | Robust smoothing | `decomposition.median_filter`, `savitzky_golay`, `lowess` (centred); `signals.trailing_savitzky_golay`, `rolling_median` (causal) | SM `median_filter`, `savitzky_golay`, `lowess`; FB `lstsq` and SP `lfilter` for the trailing filter |
| 20 | Kalman filtering | `state_space.filter_structural` | SS `kalman_filter` |
| 21 | Discount DLM | `state_space.discount_filter`, `DiscountForecaster` | SS `DiscountDLM` |
| 22 | Structural models | `state_space.StructuralModel`, `fit_structural`, `smooth_structural`, `StructuralForecaster` | SS `UnobservedComponents`, `fit_ssm_mle`, `disturbance_smoother`, `forecast` |
| 23 | State-space anomaly signals | `StructuralFilter.standardized_innovation`; `StructuralSmooth.outlier_score`, `level_break_score`, `slope_break_score` | SS filter innovations, `disturbance_smoother` |
| 24 | Forecast combination | `combination.combine_forecasts`, `Ensemble`, `inverse_error_weights` | FB `combine_mean`, `combine_median`, `combine_weighted`, `inverse_error_weights` |
| 25 | Diebold–Mariano | `evaluation.diebold_mariano` | AR `diebold_mariano` |
| 26 | VAR | `multivariate.VAR`, `select_var_lags`, `var_analysis` | AR `VAR` (`fit`, `forecast`, `irf`, `fevd`, `test_causality`) |
| 27 | GARCH | `volatility.garch` | AR `GARCH` |

### P1 forecasters

| Forecaster | Minimum history | Intervals | Runtime, 4 weeks / 92 days |
|---|---|---|---|
| `Theta(season_length=None, theta=2.0, seasonal_model="additive")`, θ ≥ 1 | 3; 2m with a season | `analytic` | < 0.1 s / < 0.1 s |
| `AR(p)`, p ≤ 48 | 2p + 2 | `analytic` | < 0.1 s / < 0.1 s |
| `StructuralForecaster(model)` | max(d + k + 1, 2 × longest season) for d states and k estimated parameters | `analytic` | 2 s / 7 s for a local level |
| `StructuralForecaster(model, parameters=)` | max(d, longest season) + 1 | `analytic` | 0.4 s / 1.7 s for a daily season of 3 harmonics |
| `DiscountForecaster(trend="level", discount=0.95, variance_discount=1.0, prior_periods=48)` | prior_periods + 1 | `student-t` | < 0.1 s / < 0.1 s |
| `VAR(drivers, lags)`, 1 + k × lags ≤ 64 for k series | (k + 1) × lags + k + 1 | `analytic` | 0.3 s / 0.8 s for one driver and 24 lags |
| `Ensemble(members, method="mean", weights=None)` | the members' largest | `quantile-averaging` | the members' total |

Runtimes were measured as for P0, on the same laptop with Python 3.12.

- `Theta` follows R's `thetaf`: simple exponential smoothing plus (θ − 1)/θ of the least-squares slope, applied to the seasonally adjusted history when `season_length` is given. Its intervals treat the drift as known; with a multiplicative season they scale with the seasonal index, which R's do not.
- `AR` is fitted by Yule–Walker, so the model is always stationary.
- `StructuralForecaster` refits to each history unless `parameters` are given. It rejects models with regressors, because their future values are unknown.
- `DiscountForecaster` holds the evolution variance and the Student-t degrees of freedom at their one-step values for every horizon.
- `VAR` forecasts the history jointly with its drivers, using only the drivers' values over the history's periods. Its intervals are the history's.
- `Ensemble` refits its members to every history and combines them with `combine_forecasts`.

### Other P1 entry points

- **`adf_test` / `kpss_test(series, *, lags, regression="c")`** each return a `UnitRootTest`: the statistic, critical values, the levels in `rejected_at`, lags and observations.
  - Both need 30 complete values, and `lags` is fixed.
  - ADF's lags are at most 48 and (n − 5) // 2, or (n − 6) // 2 with a trend. Orchid's AIC lag search is not used, because it compares fits on different samples.
  - KPSS's lags are at most n // 5, because its critical values assume far fewer lags than values. Half-hourly data needs at least 48, so at least 240 values: Orchid's default of 7–10 lags understates the long-run variance.
- **`difference_orders(series, *, season_length=None, seasonal_differencing=None)`** returns the regular and seasonal differences (0 or 1) that `AutoARIMA` applies: Orchid's seasonal-strength test, then KPSS at the 5% level on the seasonally differenced series. It needs 30 values, plus m when a season length is given; testing for a seasonal difference needs a season length of at least 3 and six seasons.
- **`ar_spectrum(series, *, order, nfft=1344)`** fits AR(order) by Burg's method and returns its one-sided spectral density from 0 to 0.5 cycles per half-hour. Any multiple of 336 for `nfft`, such as the default, puts the daily and weekly harmonics on the frequency grid. An exactly predictable series raises `DegenerateDataError`.
- **`classical_decomposition(series, *, period, model="additive")` and `stl_decomposition(series, *, period, seasonal_window=7, trend_window=None)`** return trend, seasonal and remainder series and per-phase seasonal indices.
  - Both need complete data: two cycles for classical, `seasonal_window` cycles for STL.
  - The classical trend and remainder are `None` within period // 2 of the ends.
  - STL is Orchid's simplified robust STL, without the low-pass filter. Its LOWESS spans are set in points, odd and at least 5; the default trend window is 93 at period 48.
  - Phases count elapsed half-hours, so they shift an hour against the clock across a clock change.
- **`component_strength(series, *, period)`** returns trend and seasonal strengths in [0, 1] from the classical decomposition. At short periods its trend follows the noise: white noise scores about 0.63 at period 2 and 0.02 at period 48.
- **`median_filter(series, window)`, `savitzky_golay(series, window, degree)` and `lowess(series, window, *, robustness_iterations=3)`** are centred.
  - Windows are odd: at least 3, or 5 for `lowess`, whose 3-point window returns the raw values. Each run of present values is smoothed separately, and outputs are `None` within window // 2 of each run's ends.
  - `savitzky_golay` and its causal counterpart `signals.trailing_savitzky_golay`, which takes the newest point of each trailing fit, accept degrees 0–3 and windows of at least degree + 2. The causal median is `signals.rolling_median`.
  - Wherever re-weighting leaves a window no weight, LOWESS keeps the raw value, as R's does; that value then fits exactly and regains its weight at the next re-weighting. An outlier tens of standard deviations high can therefore pass through a narrow `lowess` window at any odd `robustness_iterations`, including the default 3, and into the STL seasonal, whose LOWESS re-weights once. Wider windows resist more.
- **Structural models** start from `StructuralModel(trend="level", seasons=(), regressors=(), dynamic_regression=False)` and `StructuralParameters(model, variances, damping=None)`.
  - `trend` is `"level"`, `"linear"` or `"damped"`; `seasons` holds `(period, harmonics)` pairs, no two of which may share a harmonic frequency; a model has at most 13 states.
  - `variances` are in squared data units, except the dynamic `regression` variance, in squared series units per squared regressor unit.
  - `fit_structural(series, model, *, regressors=None)` estimates the parameters by Orchid's Nelder–Mead maximum likelihood.
  - `filter_structural(series, parameters, *, regressors=None, prior_periods=48, levels=(0.8, 0.95))` returns filtered components, one-step predictions with bands, and `standardized_innovation`.
    - The series is standardised by its first `prior_periods` observed values: at least max(2, d) of them, covering the longest season.
    - States start at the last of those values and predictions one period later. Innovations are `None` at gaps.
  - `smooth_structural(series, parameters, *, regressors=None)` returns smoothed components and the auxiliary residuals `outlier_score`, `level_break_score` and `slope_break_score`. A break scores at the first period at its new level.
  - Innovations and scores are standard normal under the model. Regressors are known inputs and must be complete over every period of the series.
- **`discount_filter(series, *, trend="level", discount=0.95, variance_discount=1.0, prior_periods=48, levels=(0.8, 0.95))`** runs West and Harrison's discount DLM with a level (or `"linear"` for a level and slope) and an online observation variance. Its priors come from the first `prior_periods` observed values, and its estimates are Student-t (mean, scale, degrees of freedom).
- **`combine_forecasts(forecasts, *, method="mean", weights=None)`** combines two or more forecasts with the same origin, horizon and levels.
  - `method` is `"mean"`, `"median"` or `"weighted"`.
  - Each interval bound is combined by the same rule (quantile averaging). This keeps nested bands nested but does not recalibrate their coverage.
  - `inverse_error_weights(errors, *, power=1.0)` gives weights proportional to error^−power. Derive the errors from data before the evaluation period.
- **`diebold_mariano(errors_a, errors_b, *, horizon=1, loss="squared")`** tests whether two forecasts have equal expected loss.
  - It needs at least 30 aligned errors, and `loss` is `"squared"` or `"absolute"`. A positive statistic means `errors_b` are more accurate.
  - For backtest errors at lead L from origins `step` apart, pass `horizon=ceil(L / step)`.
  - Where the long-run variance at that horizon is not positive, it uses horizon 1 and adds a warning, as R's `dm.test` does.
- **`select_var_lags(series, *, max_lags)` and `var_analysis(series, *, lags, horizon)`** take complete series covering the same periods; the mapping's order is the Cholesky order.
  - `select_var_lags` returns AIC and BIC for each order, all fitted on a common sample.
  - `var_analysis` returns coefficients, the residual covariance, orthogonalised impulse responses, variance decompositions and pairwise Granger F tests.
  - Granger tests show predictive association, not causation, and their p-values assume stationary series (check with `adf_test` and `kpss_test`). Deseasonalise half-hourly data first.
  - Series whose innovations are linearly dependent, such as residual demand = demand − wind, raise `DegenerateDataError`.
- **`garch(residuals, *, horizon, leverage=False)`** fits GARCH(1, 1), or GJR-GARCH(1, 1) with `leverage`, to at least 100 residuals of a mean model.
  - Do not pass price levels, and test the residuals for ARCH effects first.
  - Orchid's leverage term only increases the response to negative shocks, so negate the residuals to test positive spikes.
  - There is no unconditional variance (`None`, with a warning) when persistence is within 2e-6 of 1 or above, or when omega collapses towards zero, leaving it under 1e-6 of the residuals' variance, as it can without ARCH effects.

| Entry point | Runtime, 4 weeks / 92 days |
|---|---|
| `adf_test(lags=48)` | 0.6 s / 1.5 s |
| `difference_orders(season_length=48)` | 0.4 s / 1.4 s |
| `stl_decomposition(period=48)` | 0.5 s / 1.8 s |
| `savitzky_golay(49, 2)` | 0.1 s / 0.3 s |
| `lowess(49)` | 0.25 s / 0.8 s |
| `fit_structural`, local level | 2 s / 7 s |
| `fit_structural`, linear trend | 6 s / 21 s |
| `fit_structural`, level and daily season of 1 harmonic | 14 s / 40 s |
| `fit_structural`, level and daily season of 3 harmonics | 50 s / 2.6 min |
| `filter_structural`, daily season of 3 harmonics | 0.25 s / 0.9 s |
| `smooth_structural`, daily season of 3 harmonics | 1.2 s / 3.3 s |
| `discount_filter(trend="linear")` | 0.2 s / 0.7 s |
| `select_var_lags(max_lags=12)`, two series | 0.4 s / 1.6 s |
| `var_analysis(lags=4, horizon=48)`, two series | < 0.1 s / 0.2 s |
| `garch()` | 3 s / 10 s |
| `garch(leverage=True)` | 5 s / 21 s |
| Every other P1 entry point | < 0.1 s / < 0.1 s |

Run `fit_structural` and `garch` in batch jobs. For seasonal models, fit on a few weeks, then filter, smooth or forecast with the fitted parameters.

### Orchid behaviours the adapters correct

Each is worth fixing upstream; after a fix, simplify the adapter and re-vendor.

- **SS:**
  - Exact diffuse initialisation misclassifies diffuse updates, because its tests are absolute; the adapters use a vague prior instead.
  - `auxiliary_residuals` divides by √Var(ε | Y) instead of √(H − Var(ε | Y)); the scores are recomputed from the smoother's output.
  - `DiscountDLM.forecast` compounds 1/δ every step, and the DLM's default priors are absolute.
  - `decompose` keys components by class name, so each trigonometric seasonal overwrites the one before it, whatever its period; and `harmonics=0` silently becomes period // 2.
- **AR:**
  - `diebold_mariano` truncates unequal lengths, treats any loss other than `"squared"` as absolute, and floors a non-positive long-run variance at 1e-12.
  - `VAR.forecast` gives intervals for the first variable only.
  - `GARCH` fits on small-scale data stay at their start values; any casing of `"GJR"` other than `"gjr"` fits a symmetric model; `forecast` ignores its result argument.
  - Unit-root p-values are table interpolations clamped to [0.005, 0.55].
- **FB:**
  - `inverse_error_weights` gives a NaN error, or one at or below 1e-12, all of the weight (a raw weight of 1e18); power 0 makes the weights equal and a negative power inverts them.
  - `Theta` silently drops a multiplicative season whose index is at most 1e-10 (the adapter raises), and its automatic seasonality test fires on random walks.

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
