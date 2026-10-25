# Modelling notes

Findings from the first runs of the P0 and P1 algorithms, the modelling choices built into their adapters, and the questions they leave open. Written 24 October 2026 from the Tiger **dev** database: runs 23 to 30, imbalance prices from 1 September to 23 October 2026 (53 days, 2,544 periods, none missing). The algorithms themselves are described in [ALGORITHMS_AND_ANALYSIS.md](ALGORITHMS_AND_ANALYSIS.md#registered-algorithms). These numbers come from one autumn period and one market; treat them as a first look, not a result. The [P1 backtest](#p1-backtest-results) was added on 25 October from the same database.

## Backtest results

Each day from 1 September, every model was refitted on the previous 28 days and forecast the next 48 periods. MAE is in £/MWh. MASE below 1 beats an in-sample daily seasonal naive. Coverage is the share of actual prices inside the 80% and 95% bands; well-calibrated bands score near 0.80 and 0.95.

| Model | Algorithm | MAE | MASE | MAE 30 min ahead | MAE 24 h ahead | 80% coverage | 95% coverage |
|---|---|---|---|---|---|---|---|
| ETS (A,N,A) | forecast | **49.5** | **0.84** | 23.2 | 51.1 | 0.94 | 0.99 |
| Naive2 | baseline | 52.2 | 0.89 | 20.7 | 53.6 | 0.99 | 1.00 |
| Weighted moving average (4) | baseline | 52.9 | 0.91 | 24.2 | 50.2 | 0.99 | 1.00 |
| Fourier regression | regression | 54.8 | 0.93 | 45.0 | 41.9 | 0.99 | 1.00 |
| Seasonal dummy regression | regression | 54.8 | 0.93 | 45.4 | 42.4 | 0.99 | 1.00 |
| Naive (last value) | baseline | 55.1 | 0.94 | **19.2** | 53.6 | 0.99 | 1.00 |
| Calendar regression | regression | 55.2 | 0.94 | 44.7 | 47.6 | 0.99 | 1.00 |
| Drift | baseline | 55.4 | 0.94 | **19.2** | 54.2 | 0.99 | 1.00 |
| Mean of the window | baseline | 57.7 | 0.98 | 44.3 | **39.4** | 0.76 | 0.93 |
| Moving average (48) | baseline | 57.7 | 0.98 | 41.2 | 52.3 | 1.00 | 1.00 |
| Linear trend | regression | 59.3 | 1.01 | 46.6 | 43.2 | 1.00 | 1.00 |
| ARIMA(2,0,1), daily difference | forecast | 61.9 | 1.05 | 35.5 | 54.2 | 0.79 | 0.91 |
| Seasonal naive, daily | forecast, baseline | 65.0 | 1.10 | 59.1 | 53.6 | 0.78 | 0.92 |
| Seasonal naive, weekly | baseline | 72.6 | 1.22 | 69.3 | 59.9 | 0.78 | 0.92 |

### What the results say

1. **The best model depends on the lead time.** Persistence wins half an hour ahead: naive and drift have an MAE of £19.2. A day ahead, the 28-day mean (£39.4) and the seasonal regressions (£41.9–43.2) are best, and ETS (£51.1) is no better than the naive. ETS has the best average over the whole day only because it is good in the first hours. A page that says "ETS is best" should say *for which lead time*; the `by_horizon` table has every step.
2. **Yesterday's shape is a poor guide to today's.** The daily seasonal naive (£65.0) is worse than repeating the last price (£55.1), and the weekly one is worse still (£72.6). The price autocorrelation is 0.81 at 30 minutes but only 0.31 at 24 hours (`price_driver_correlograms`). A regression's *average* daily shape works better than any single previous day.
3. **Seasonal differencing hurts.** ARIMA with a daily difference (£61.9) is barely better than the seasonal naive it is built on, and its residuals keep a large Ljung–Box statistic at lag 48 (728 on 45 df). That is the signature of over-differencing. Its fitted AR terms (ar.L1 around 1.19, ar.L2 around −0.25) are close to a unit root on top of the difference.
4. **ETS's daily season never updates.** The fitted seasonal smoothing weight `gamma` is 0 to four decimal places at all 53 origins, and AutoETS picks the same: ETS(A,N,A) with gamma 0. ETS is therefore a smoothed level (alpha around 0.53) plus a fixed seasonal profile estimated once from the start of the window. This explains its wide bands and its flat evening forecast for 24 October (£58–95, against £228–255 from the seasonal naive). It may be a real property of these prices, given finding 2, or an artefact of Orchid's initial seasonal states or optimiser bounds. **For Matthew:** check against another ETS implementation on the same window.
5. **Most prediction intervals are far too wide.** The sqrt-horizon and random-walk intervals cover 99–100% at the 80% level. Only the seasonal naive, ARIMA and the mean come close to nominal (0.76–0.79 and 0.91–0.93), and they also have the best pinball losses (£10.7–16.1). Wide bands look safe but carry no information; show coverage next to every band.
6. **Errors are heavy-tailed and cluster in time.** The residual diagnostics give excess kurtosis 6.7 (ETS) and 3.0 (ARIMA), Jarque–Bera p-values of 0 and strong ARCH effects (statistics 196 and 170 on 12 lags). Gaussian intervals are therefore misstated: too narrow in volatile spells and too wide in calm ones. Durbin–Watson is near 2 for both models.
7. **The drivers move with the price, mostly at the same moment.** The cross-correlation with residual demand peaks at +0.63 (lag −1, effectively simultaneous) and with the wind forecast at −0.46 (lag 0). The white-noise band (±0.039) is far narrower than these values, but it overstates significance for autocorrelated series. These figures say the drivers matter, not that they cause the price.

## P1 backtest results

`price_p1_backtest` was run on 25 October 2026 against Tiger dev, read-only: the input frame was read and the algorithm run in-process, and no run was stored. The range was the refresh window, 29 August to 23 October 2026 (56 days, 2,688 periods, plus the 28-day warm-up, with no price missing), so there were 56 folds per model and none was excluded. To compare like with like, the three P0 backtests were run on the same frame; their numbers differ slightly from the table above, which starts on 1 September. MAE is in £/MWh.

| Model | Algorithm | MAE | MASE | MAE 30 min ahead | MAE 24 h ahead | 80% coverage | 95% coverage | Pinball q10 / q90 |
|---|---|---|---|---|---|---|---|---|
| Ensemble (Theta, AR, DLM) | P1 | **48.5** | **0.84** | 24.1 | 45.1 | 0.88 | 0.97 | **12.3 / 10.6** |
| Theta (θ = 2, daily season) | P1 | 49.2 | 0.85 | 22.5 | 51.7 | 0.94 | 0.99 | 14.1 / 12.9 |
| ETS (A,N,A) | forecast | 49.2 | 0.85 | 22.5 | 51.7 | 0.94 | 0.99 | 14.3 / 13.1 |
| Naive2 | baseline | 51.7 | 0.89 | 20.0 | 54.2 | 0.99 | 1.00 | 24.8 / 23.9 |
| Local-level structural | P1 | 52.8 | 0.91 | 20.3 | 52.4 | 0.96 | 1.00 | 17.3 / 16.5 |
| Discount DLM (level, δ = 0.95) | P1 | 53.6 | 0.92 | 35.4 | 48.8 | 0.75 | 0.89 | 13.4 / 11.7 |
| Naive (last value) | baseline | 54.1 | 0.93 | **18.5** | 54.2 | 0.99 | 1.00 | 25.3 / 25.0 |
| AR(2) | P1 | 54.3 | 0.94 | 22.5 | **39.0** | 0.78 | 0.93 | 14.0 / 10.1 |
| Mean of the window | baseline | 56.5 | 0.97 | 43.7 | **39.0** | 0.77 | 0.93 | 15.0 / 10.4 |
| Fourier regression | regression | 54.0 | 0.93 | 44.9 | 40.8 | 0.99 | 1.00 | 36.8 / 36.4 |
| ARIMA(2,0,1), daily difference | forecast | 60.8 | 1.04 | 34.8 | 54.8 | 0.80 | 0.92 | 14.3 / 14.8 |
| Seasonal naive, daily | P1, forecast, baseline | 63.9 | 1.10 | 59.4 | 54.2 | 0.78 | 0.92 | 15.2 / 15.6 |

The algorithm's own `dm_tests`, absolute loss, each model against the daily seasonal naive (positive statistic: the model was more accurate; 56 errors per lead):

| Model | 30 min: MAE vs 59.4 | statistic | p-value | 24 h: MAE vs 54.2 | statistic | p-value |
|---|---|---|---|---|---|---|
| Theta | 22.5 | 6.01 | < 1e-5 | 51.7 | 1.19 | 0.24 |
| AR(2) | 22.5 | 6.09 | < 1e-5 | 39.0 | 2.63 | 0.011 |
| Discount DLM | 35.4 | 4.61 | < 1e-4 | 48.8 | 1.22 | 0.23 |
| Local level | 20.3 | 6.24 | < 1e-5 | 52.4 | 1.21 | 0.23 |
| Ensemble | 24.1 | 6.33 | < 1e-5 | 45.1 | 2.74 | 0.008 |

For context, the same test was run outside the algorithm against the stronger P0 benchmarks on the same targets. These comparisons are not stored anywhere.

- **30 minutes ahead,** against the naive (£18.5), every P1 model was less accurate. The local level was closest (£20.3, p = 0.21); for the rest p was 0.0002–0.043.
- **24 hours ahead,** against the window mean (£39.0), AR(2) was level with it (p = 0.76). Theta, the DLM and the local level were worse (p 0.02–0.04), and the ensemble was worse at p = 0.065.
- **24 hours ahead,** against ETS (£51.7), AR(2) and the ensemble were more accurate (p 0.03 and 0.026).

These are ten in-algorithm tests plus thirty more, on 56 days of one autumn, so a p-value near 0.03 is weak evidence on its own.

### What the P1 results say

1. **Theta reproduces ETS.** Their forecasts correlate at 0.99999999 and differ by a median of £0.16. With the season adjusted by a fixed classical decomposition, Theta is simple exponential smoothing (alpha 0.46–0.62 across folds, median 0.53) plus a negligible drift (±£0.02 per half-hour). That is what ETS(A,N,A) with gamma 0 is, and the fitted alpha matches ETS's (around 0.53). Two independent implementations agreeing supports finding 4, that a fixed daily profile is what these prices support, rather than an Orchid artefact. It also means Theta adds nothing to ETS on this data.
2. **The ensemble has the best average, through diversity rather than any good member.** It is not best at either end: 30 minutes ahead it is worse than all of its members except the DLM, and 24 hours ahead it trails AR(2) and the mean. Averaging Theta's daily shape with AR's reversion to the mean still gives the lowest MAE over the day, the lowest pinball losses, and bands closer to nominal (0.88 / 0.97) than any well-scoring P0 model.
3. **AR(2) is the 28-day mean a day ahead.** Its 24-hour MAE (£39.0) equals the window mean's, and the DM test cannot tell them apart. The fitted coefficients (ar.L1 around 0.66, ar.L2 around 0.18) decay to the mean within a few hours. Close in, it matches Theta (£22.5). This confirms finding 1 from the other side: most of the day-ahead skill available here is the level, not the shape.
4. **The local level is the best P1 model close in but never beats persistence.** At 30 minutes it is £20.3 against the naive's £18.5, and its fitted signal-to-noise ratio (level variance 737 against irregular 588, in (£/MWh)²) smooths away part of the last price. A day ahead it is a flat line like the naive. Its bands over-cover (0.96 / 1.00).
5. **The discount DLM is too slow at δ = 0.95.** It has the worst 30-minute error of the P1 models (£35.4) and under-covers (0.75 / 0.89), because its level adapts too slowly for half-hourly prices. Its degrees of freedom reach 1,345, so the Student-t bands are effectively Gaussian. A lower discount, around 0.7–0.8, is worth a backtest; it is one param override.
6. **The Diebold–Mariano tests separate leads clearly.** Every P1 model beats the daily seasonal naive at 30 minutes (p < 1e-4), but only AR(2) and the ensemble do so at 24 hours. Testing against the seasonal naive, which is the weakest benchmark here (finding 2), flatters every model. For the demo, set `dm_benchmark` to a stronger model, or show the naive and mean comparisons above beside the stored table.

## Choices built into the adapters

These are decisions made while wiring Matthew's methods into the API. Each is reasonable but open to review.

| Choice | Where | Why | Consequence |
|---|---|---|---|
| Origins every 48 periods from the start of the range; step equals horizon | `_backtest.py` | One forecast per model per target period, so forecasts chart against actual prices directly | Lead time varies along the day; use `by_horizon` for lead-time comparisons |
| 28-day sliding training window, the same for every model | `_backtest.py` | Comparable models and bounded runtime | Models that need long history (weekly seasonality) see only four weeks |
| One `evaluation.backtest` call per origin | `_backtest.py` | Matthew's `backtest` rejects any series with a gap; per-origin calls let a gap exclude only the folds that contain it (`missing_data`) | The pooled metrics are recomputed in `_backtest._metrics`, mirroring `evaluation._report`; if `_report` changes, update both |
| Pinball loss at the 80% band edges (q10, q90) | `_backtest.py` | Uses the intervals already produced | Not a full quantile forecast |
| Interval levels fixed at 80% and 95% | `_backtest.py` | The output names carry them | Changing levels means new output names and a new version |
| Window lengths fixed in signal names; Fourier harmonics fixed | `price_rolling_signals.py`, `calendar_fourier_terms.py` | Output names carry them | These two algorithms have no parameters |
| Longest gap-free stretch for correlograms and diagnostics | `_frames.longest_complete` | Lags must mean the same elapsed time; dropping missing rows would not keep that | Reported in a `sample` table; currently the whole range |
| Diagnostics fit the latest 56 days; `model_df` 0 for ETS and 3 for ARIMA | `price_residual_diagnostics.py` | Bounded runtime; p + q for ARIMA | ETS's effective degrees of freedom are debatable |
| Day-ahead forecasts the whole next settlement day (46, 48 or 50 periods) | `price_day_ahead_forecast.py` | A "tomorrow" panel should match the settlement day | The day is recorded in the `forecast_day` table |
| AutoARIMA with the daily difference fixed (`seasonal_differencing: 1`) | `price_day_ahead_forecast.py` | Matthew's notes advise fixing it when the cycle is known | **Contradicted by finding 3**; left to its own test, AutoARIMA drops it and picks ARIMA(0,1,1) |
| Seasonal lags count elapsed half-hours | all seasonal models | Orchid works on a regular grid | On and after a clock change, such as **25 October 2026, this weekend**, origins shift by an hour of local time and seasonal lags point an hour off; fits carry a warning |
| Structural model without seasons in the P1 backtest | `price_p1_backtest.py` | A seasonal `fit_structural` takes 14–50 s on four weeks, which at 56 origins is up to 47 minutes of fitting | `structural` is a local level, which at day-ahead leads is a flat forecast like the naive |
| Ensemble of Theta, AR and the DLM, equally weighted | `price_p1_backtest.py` | The cheap members; ETS or the structural model would double the cost of every fold, and weights fitted on the same folds would flatter the ensemble | Its bands average the members' bounds, which keeps them nested but does not recalibrate coverage |
| Diebold–Mariano at leads 1 and 48 only, against the seasonal naive, with absolute loss | `price_p1_backtest.py` | One test per lead keeps errors from separate origins (test horizon 1); absolute loss matches MAE and is less dominated by price spikes | Ten tests per run, so expect some small p-values by chance; other leads or the naive benchmark are param overrides |

## Batch runtime

Almost all compute is Orchid's pure-Python fitting, repeated at every backtest origin: ARIMA's conditional sum of squares, and Nelder–Mead for ETS. Each (model, origin) fit is independent and seeded, so `_parallel.run_all` spreads them over a process pool. The backtests use it, and so do the model fits of `price_day_ahead_forecast` and `price_residual_diagnostics`. `GRIDPULSE_WORKERS` sets the pool size; the default is the CPUs available, at most 4, and 1 runs everything in-process. Workers start from a fork server, or spawn where none exists, so a script that runs algorithms needs an `if __name__ == "__main__":` guard. The `gridpulse` command already has one. Two exact trims also help the cheap models:

- Each fold window is cut once and shared by every model. `HalfHourlySeries.window` no longer re-validates values that were already checked.
- Folds call `evaluation.backtest_folds`, which skips the pooled report that `_backtest` discarded.

Outputs are unchanged, so no algorithm versions moved. On the refresh windows to 23 October 2026 (56 days, inputs from Tiger dev), every series and table was `DataFrame.equals` to the previous code at 1, 2 and 4 workers. These are seconds of `algorithm.run` on a 12-core laptop:

| Algorithm | Before | 1 worker | 2 workers | 4 workers |
|---|---|---|---|---|
| `price_forecast_backtest` | 109.4 | 113.0 | 57.1 | 30.9 |
| `price_regression_backtest` | 35.6 | 30.9 | 16.3 | 10.5 |
| `price_day_ahead_forecast` | 13.7 | 13.8 | 8.0 | 7.9 |
| `price_baseline_backtest` | 6.3 | 4.2 | 2.7 | 1.9 |
| `price_residual_diagnostics` | 3.5 | 3.9 | 2.9 | 2.7 |

The pool only helps with more than one vCPU. A 1-vCPU VPS keeps only the in-process trims, which save about 7 s. Rejected options:

- **Warm-starting each fold from the previous fold's parameters.** `ARIMA.fit` takes no start values; only the lower-level `css_estimate` does. `ETSForecaster` has none. Either change would alter results and need version bumps.

`price_p1_backtest` on the Tiger dev window above (56 folds, 4 workers, the same laptop): 43.0 s for a fresh `algorithm.run`, against 30.0 s for `price_forecast_backtest`, 10.1 s for `price_regression_backtest` and 2.7 s for `price_baseline_backtest` on the same frame. Loading the frame from Tiger took 2.6 s. Almost all of the time is the local-level structural model's maximum-likelihood refit at every origin; Theta, AR, the DLM and the ensemble each take well under a second per fold. Like the other backtests it declares `memo`. After a run over the window one day earlier, the run to 23 October took 5.3 s with that memo, on 4 workers, and returned series and tables identical to the fresh run. Neither run was stored, so `run_algorithm`'s storage time is not included.

### Reusing folds between runs (memo)

After one new day, 55 of a backtest's 56 origins see the same prices as in the previous run. The three backtests declare `memo=True` (see `analysis/contract.py`), so `run` also takes `memo`, a dict in which each fold is stored as JSON. Each fold is keyed by a hash of its model and settings, its prices with their start time, the fold settings and the interval levels. A run fits only the folds it does not find. A revised price changes the key of every fold that contains it, so those folds are refitted.

The algorithm stays pure. `service.run_algorithm` loads the entries before the run and saves new ones in the same transaction as the results. Entries live in `gridpulse.analysis_memo` (migration 005), in a namespace that hashes the algorithm id, its version and every source file of the algorithms package, Orchid included. Any code change therefore starts afresh. Entries unused for 7 days are deleted on save, which keeps about 1 to 3 MB per backtest. `gridpulse refresh --force` and `gridpulse analyse --no-memo` ignore stored entries but still save new ones. Deleting rows only makes the next run slower.

This was checked on Tiger dev in a transaction that was rolled back. Each backtest was run to 22 October, then to 23 October with the memo on one worker, then to 23 October from scratch on four. The two 23 October runs stored identical results and tables. Wall time of `run_algorithm` in seconds, including loading, the memo and storage:

| Algorithm | Fresh, 4 workers | Memo, 1 worker |
|---|---|---|
| `price_forecast_backtest` | 31.0 | 3.1 |
| `price_regression_backtest` | 9.9 | 1.9 |
| `price_baseline_backtest` | 3.1 | 2.2 |

A routine refresh therefore no longer needs more than one vCPU. The pool still matters for the first run after a code change or a version bump.

## Open questions and next steps

1. **ETS gamma (Matthew).** Is a zero seasonal weight real or an Orchid artefact? See finding 4. Theta, a separate Orchid implementation, gives the same forecasts ([P1 finding 1](#what-the-p1-results-say)), which points to a real property of the prices.
2. **AutoARIMA's seasonal difference (team decision).** Keep `seasonal_differencing: 1` or set it to `null` so the test decides? Finding 3 favours dropping it. Settle it with a backtest of both, which costs about 6 minutes for 53 days at 5–7 s per fit. Changing it is one line plus a version bump.
3. **Which lead time matters for the story?** If the dashboard's claim is "day-ahead", compare models at 24 hours ahead, where the mean and the regressions win, rather than averaged over the day.
4. **Calibrated intervals.** Empirical bands from past backtest errors would fix the coverage problem (finding 5). It estimates quantiles from history without training a model, but check that it fits the no-training constraint before building it.
5. **Forecast-error analysis of the official forecasts (the HRT angle).** Score Elexon's WINDFOR and NDF against outturn, by lead time and by period. Caveats: `wind_outturn` (FUELHH) excludes embedded wind, so it is not like-for-like with WINDFOR; NESO demand outturn arrives about 21 days late. The forecast vintages are already in `gridpulse.*_forecast_vintages`.
6. **Drivers as inputs.** None of the P0 forecasters takes exogenous inputs. A regression on residual demand would need a new adapter (P1). It would also need forecasts as published at each origin, not the latest version, to stay out-of-sample (BACKEND_DESIGN open item 1).
7. **Longer evaluations.** A run is capped at about 60 days (`max_folds`). The three backtests take about 3 minutes together for 53 days, so a 92-day evaluation means raising `max_folds` and accepting about 5 minutes.
8. **The reference algorithms.** `price_naive_backtest` duplicates `price_baseline_backtest`, and `wind_price_correlation` overlaps the correlograms. Keep them as simple examples or remove them before the demo.
9. **The Orchid licence.** Upstream has no licence file (AGENTS.md). Confirm the terms before this repository is made public for submission.
