# Displaying algorithm results

How to present each algorithm's outputs and tables. Endpoints, fields and errors are in [FRONTEND_API.md](FRONTEND_API.md); response types are in [`gridpulse/frontend/src/lib/types.ts`](../../gridpulse/frontend/src/lib/types.ts). Visual style, motion and page layout follow [FRONTEND_DESIGN.md](FRONTEND_DESIGN.md).

The two reference algorithms are live. The eight others are planned: build them against [`contracts/examples/algorithms/`](../../gridpulse/contracts/examples/algorithms/), and show an algorithm only when it appears in `/api/catalog`.

## Rules for every algorithm

1. **Show `limitations`** next to the results, in full or behind a clearly visible "What this does not mean" toggle. It is part of the result, not fine print.
2. **Show `warnings`** from `/api/analysis` whenever there are any, above the chart.
3. **Show provenance:** algorithm `version`, `run.finished_at_utc`, and the run's settings. For a default run say "default settings"; otherwise list `run.overrides` ("horizon 1 period"). Use `param_descriptions` for tooltips.
4. **Missing is not zero.** Draw `null` as a gap in lines and an empty cell in tables. Never interpolate across it or fill it with 0.
5. **Nothing is a probability.** Do not label any output a probability, chance, risk or likelihood, and do not render it on a 0–100% "chance" gauge. Prediction intervals are model-based ranges, not guarantees.
6. **Units come from the API:** `outputs[name]` for series, column-name suffixes for tables (below). `""` means unitless.
7. **Correlation is not causation.** Phrase correlograms as "moves with" or "leads by", never "drives" or "causes".
8. **Retrospective or available-at-the-time.** Say which applies, using the wording given per algorithm below.
9. **Clock-change days** have 46 or 50 periods. Label the x-axis by UTC instants converted to London time, so the repeated hour on 25 October 2026 appears twice in order.

## Naming conventions

### Series outputs

| Pattern | Meaning | Examples |
|---|---|---|
| `<model>_forecast` | Point forecast of `<model>` | `ets_forecast`, `seasonal_naive_day_forecast` |
| `<model>_lower_<level>`, `<model>_upper_<level>` | Edges of `<model>`'s central `<level>`% prediction interval | `ets_lower_80`, `ets_upper_95` |
| `price_<statistic>_<window>` | A trailing statistic of the price over `<window>` periods | `price_std_48`, `price_p90_336` |
| `price_change_<lag>` | Price minus the price `<lag>` periods earlier | `price_change_1`, `price_change_48` |
| `daily_<sin\|cos>_<k>`, `weekly_<sin\|cos>_<k>` | Fourier terms of the clock | `daily_sin_1` |

The `<model>` prefix is the same string as the `model` column in that algorithm's tables and the key in its `params.models`, so one colour per model can be used throughout.

Group outputs with:

```ts
const forecast = /^(.*)_forecast$/;
const bound = /^(.*)_(lower|upper)_(\d+)$/;  // groups: model, edge, level
```

### Table columns

Format table cells by column name:

| Column name | Format |
|---|---|
| ends `_utc` | Date and time in Europe/London, e.g. "Thu 22 Oct 17:30" |
| ends `_gbp_mwh` | £/MWh, 1 decimal place |
| `wmape`, `smape` | Fraction: multiply by 100 and show as %, 1 decimal place |
| `coverage_80`, `coverage_95` | Fraction as %, shown against the nominal level ("77% of 80%") |
| `mase`, `rmsse` | 2 decimals. Below 1 beats a seasonal naive forecast; above 1 is worse |
| `interval_score_*`, `msis_*`, `pinball_*` | 1 decimal; lower is better |
| `p_value` | 2 significant figures in scientific notation below 0.001; `0` means below about 1e-16, so show "< 1e-16"; `null` shows "n/a" |
| `lag_periods`, `lag_hours` | Integer; `lag_hours` is `lag_periods / 2` |
| `status` | A chip: `ok` neutral; `insufficient_data`, `degenerate_data`, `failed` muted with `message` as tooltip |
| `n_*` | Integer count |
| anything else | Numbers to 3 significant figures; text as is |

Tables are small (at most 20,000 rows, usually far fewer); sort and filter them in the browser.

## Patterns

These recur across algorithms. Build each once as a component.

**Forecast vs actual.** Fetch the algorithm with `/api/analysis` and the price with `/api/series?ids=price` for the **same** `from`, `to` and the `resolution` returned by the first response. Both responses then have identical `t` arrays. Draw the actual price as the strongest line in a neutral colour, each model's forecast in its own colour, and bands as translucent fills: 95% lighter than 80%. Let the user toggle models from the legend. Forecasts are clearest at `30m`, so default to a range of two to seven days and allow zooming out.

**Scorecard.** The `metrics` table as a table, with the best value per column highlighted, plus a horizontal bar chart of one chosen metric (default MASE) per model. Draw a reference line at MASE = 1 labelled "seasonal naive".

**Accuracy by lead time.** The `by_horizon` table as lines: x = `lead_hours`, y = MAE, one line per model. Real tables have every step from 1 to 48; the examples are truncated.

**Fold timeline.** The `folds` table as a strip of one mark per `origin_utc` per model, coloured by `status`, with `mae_gbp_mwh` on hover. It shows when the models struggled, and which days were excluded and why.

**Correlogram.** A stem (lollipop) chart: x = `lag_hours`, y = `value` from −1 to 1, with a shaded band from −`band` to +`band` at each lag. Stems outside the band are notable; do not call them "significant" for cross-correlations, because that band is optimistic for these series.

## The algorithms

### `price_naive_backtest` (live, reference)

- **Page:** Overview (chart), Explore (tables).
- **Outputs:** `persistence_forecast` and `seasonal_naive_forecast` (GBP/MWh), each day-ahead by default (`horizon_periods` 48).
- **Show:** the forecast-vs-actual pattern, and the `metrics` table (`model`, `horizon_periods`, `n`, `mae_gbp_mwh`, `rmse_gbp_mwh`, `bias_gbp_mwh`) as a scorecard. A positive bias means forecasts were too high on average. `worst_misses` (ten periods with the largest seasonal-naive error) is a list linking each `delivery_start_utc` to the period inspector.
- **Wording:** "What a naive forecaster made a day ahead, using only prices already settled."

### `wind_price_correlation` (live, reference)

- **Page:** Overview, as a small pane under the price chart.
- **Output:** `wind_price_corr`, unitless, −1 to 1: the Pearson correlation between the wind forecast and the price over the previous seven days.
- **Show:** a line on a fixed −1..1 axis with a zero line. Negative values mean higher wind went with lower prices that week.
- **Wording:** "Trailing seven-day correlation; descriptive, not a forecast."

### `price_baseline_backtest` (planned)

- **Page:** Explore.
- **Outputs:** one `<model>_forecast` per baseline: `naive` (last settled price), `seasonal_naive_day` (same period yesterday), `seasonal_naive_week` (same period last week), `naive2`, `drift`, `mean`, `moving_average`, `weighted_moving_average`. Point forecasts only.
- **Tables:** `metrics`, `by_horizon`, `folds`, as described under [patterns](#patterns).
- **Show:** the scorecard first. These are the benchmarks; the message is "how hard is this to beat". Offer the forecast-vs-actual chart with all eight hidden except `seasonal_naive_day`.
- **Wording:** "Each forecast uses only prices settled before the previous day ended."

### `price_forecast_backtest` (planned)

- **Page:** Overview (headline chart), Explore (tables).
- **Outputs:** `seasonal_naive_forecast`, `ets_forecast`, `arima_forecast`, with 80% and 95% bands for ETS and ARIMA.
- **Tables:** `metrics`, `by_horizon`, `folds`, plus `fold_params` (`model`, `origin_utc`, `parameter`, `value`): the fitted parameters at each origin.
- **Show:** the forecast-vs-actual chart with ETS bands on by default, and the scorecard comparing ETS and ARIMA with the seasonal naive benchmark. Use `coverage_80`/`coverage_95` to show whether the bands are honest: near 80%/95% is well calibrated, lower means the bands were too narrow. On Explore, chart `fold_params` as one small line chart per parameter over `origin_utc`; drift there means the fitted model was unstable.
- **Wording:** "Out-of-sample: each day's forecast was made the day before, from a model refitted on the previous 28 days."

### `price_regression_backtest` (planned)

- **Page:** Explore.
- **Outputs:** `trend_forecast`, `seasonal_dummy_forecast`, `fourier_forecast`, `calendar_forecast`. Point forecasts only.
- **Tables:** `metrics`, `by_horizon`, `folds`.
- **Show:** the same components as the other backtests. Point out that `calendar` follows the GB clock and so is the only regression exact on clock-change days.

### `price_day_ahead_forecast` (planned)

- **Page:** Overview: the "tomorrow" panel.
- **Outputs:** `seasonal_naive_forecast`, `auto_ets_forecast` and `auto_arima_forecast`, with 80% and 95% bands for the two automatic models. The values lie on the **forecast day**, after the data used.
- **Finding the day:** take `latest_run` from the catalog. The forecast day is the London date of `range_end_utc` minus 30 minutes. Request `/api/analysis` with `from` and `to` both set to that day, and `/api/series?ids=price` for the same day to draw the actual price once it settles.
- **Tables:** `selected_models` (`model`, `source`, `parameter`, `value`): the model each automatic search chose and its fitted parameters. Show it as "AutoARIMA chose ARIMA(2,0,1) with a daily seasonal difference". `fit_warnings` (`model`, `warning`): show any rows as notes under the chart.
- **Show:** a fan chart: the 95% band, the 80% band inside it, the point forecast, and the seasonal naive as a dashed comparison line.
- **Wording:** "Forecast from prices settled by the end of {last day}. Ranges are model-based, not probabilities." Link to `price_forecast_backtest` for how accurate such forecasts have been.

### `price_rolling_signals` (planned)

- **Page:** Overview, as overlays and a pane under the price chart.
- **Outputs (GBP/MWh unless noted):**
  - levels: `price_mean_48`, `price_weighted_mean_48`, `price_ewma_48`, `price_median_48`, `price_rms_48` (24-hour windows);
  - spread: `price_variance_48` in (GBP/MWh)², `price_std_48`;
  - range: `price_p10_336`, `price_p90_336` (7-day 10th and 90th percentiles);
  - change: `price_change_1` (since the previous period), `price_change_48` (since 48 periods earlier).
- **Show:** on the price chart, the p10–p90 range as a shaded band and one level line (EWMA by default; let the user pick another). In a pane below, `price_std_48` as volatility. Changes fit a bar pane. Prefer `std` to `variance` for display; its unit matches the price.
- **Wording:** "Trailing statistics: each value uses only prices up to that period."

### `calendar_fourier_terms` (planned)

- **Page:** Methods.
- **Outputs:** `daily_sin_1`, `daily_cos_1`, `daily_sin_2`, `daily_cos_2`, `weekly_sin_1`, `weekly_cos_1`, unitless, −1 to 1.
- **Show:** small multiples over one or two days, to explain how the regressions represent the daily and weekly cycle. Fetch 25 October 2026: the values for 01:00 and 01:30 repeat, because the clock repeats that hour.
- **Wording:** "Functions of the clock, not of data."

### `price_driver_correlograms` (planned)

- **Page:** Explore.
- **Outputs:** none. Use `latest_run.run_id` and fetch the tables directly.
- **Tables:** `acf_price` and `pacf_price` (lags 1..96), `ccf_residual_demand` and `ccf_wind` (lags −96..96), each with `lag_periods`, `lag_hours`, `value`, `band`.
- **Show:** four correlogram components. For cross-correlations explain the sign of the lag: at a positive lag k, the driver k half-hours earlier is compared with the price now, so a peak at k > 0 means the driver *leads* the price. The ACF shows how long price shocks persist; a spike at 24 h shows the daily cycle.
- **Wording:** "Association over the whole period, not cause and effect."

### `price_residual_diagnostics` (planned)

- **Page:** Methods.
- **Outputs:** none; fetch the tables by `latest_run.run_id`.
- **Tables:** `ljung_box` (`model`, `lag`, `statistic`, `df`, `p_value`) and `tests` (`model`, `test`, `statistic`, `p_value`, `notes`).
- **Show:** `tests` as a table per model. `ljung_box` as a line of `p_value` against `lag` on a log axis, one line per model. Explain each test in one line:
  - Ljung–Box: is there pattern left in the errors at lags up to k;
  - Durbin–Watson: autocorrelation between consecutive errors, where 2 means none (no p-value);
  - Jarque–Bera: are the errors normally distributed (imbalance prices are not);
  - ARCH-LM: does error size cluster in time.
- **Wording:** "Statistics and p-values only; no pass/fail threshold." Do not show green ticks or red crosses.

## Page map

| Page | Algorithms |
|---|---|
| Overview | `price_forecast_backtest` (forecast vs actual), `price_day_ahead_forecast` (tomorrow), `price_rolling_signals` (overlays), `wind_price_correlation`, `price_naive_backtest` until the planned backtests land |
| Explore | All backtest scorecards, by-horizon charts and fold timelines; `price_driver_correlograms`; `worst_misses` and extremes linking to the period inspector |
| Methods | Every algorithm's `description`, `inputs`, `params` with `param_descriptions`, and `limitations`; `calendar_fourier_terms`; `price_residual_diagnostics`; dataset coverage from the catalog |

## States

| Situation | Show |
|---|---|
| Algorithm not in the catalog | Nothing; it is not deployed yet |
| `latest_run` is `null`, or 404 `no_run` | A "Not computed yet" placeholder with the algorithm's description |
| `warnings` non-empty | A notice above the chart |
| All values in view are `null` | "No results in this range", with the run's range from `run.range_start_utc`/`range_end_utc` |
| `n_excluded_folds` > 0 | "N days excluded", linking to the fold timeline |
| `mode` is not `live` | The "cached data" badge from [FRONTEND_API.md](FRONTEND_API.md#demo-modes) |
