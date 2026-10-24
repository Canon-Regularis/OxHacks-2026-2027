# Spike classifier, experiments and evaluation

## Prediction target

For each settlement period, predict two separate binary outcomes from the retrospective GB imbalance settlement price `p`:

- `positive_extreme = (p > q95_train)`;
- `negative_extreme = (p < q05_train)`.

Compute `q95_train` and `q05_train` **only on the training partition**; freeze the numeric thresholds for validation and test. Report both GBP/MWh values, number of events and event prevalence. Missing/invalid settlement prices are excluded consistently. The two models produce separate probabilities; neither output is a forecast of grid instability or trading profit.

An operational fixed GBP/MWh threshold could be a future experiment but shouldn't be silently substituted midway through evaluation.

## Forecasting protocol

- **Issue time:** 18:00 Europe/London the day before each target settlement date, using IANA timezone conversion to UTC.
- **Horizon:** all settlement periods on the following GB settlement day (46/48/50 depending on DST).
- **Inputs:** forecast information with publication timestamps `<= issue_time` and deterministic target-time calendar features.
- **Outcomes:** later published/finalised system settlement price, for retrospective evaluation.
- **Missing data:** drop with reported coverage or impute using **training-fitted** rules; never fill from post-cutoff data.

This is a **day-ahead probabilistic classification** task; it should not be compared as equivalent to an intraday model that sees later information.

## Model plan

1. **Baseline:** calendar-only event frequency by hour/weekday (smoothed with training data), and/or logistic regression on calendar + demand. Keep a deterministic simple benchmark.
2. **Primary:** two `LGBMClassifier` models (positive and negative extreme), with conservative trees/leaf counts and class weights if appropriate. Use the *same* split and protocol across models.
3. **Three-way ablation:**
   - **A** = calendar + demand forecast;
   - **B** = A + latest wind forecast, residual demand or wind share;
   - **C** = B + wind forecast revisions and revision instability.
4. **Optional calibration:** fit Platt scaling / isotonic calibration on a **separate time-ordered validation** subset if showing probabilities as calibrated percentages; evaluate Brier score and reliability. Do not fit calibration on test.

Suggested features: forecast demand, forecast wind, wind revision over 6h/24h, residual demand, wind share, wind-ramp projection, local hour, day-of-week and month. A/B/C feature differences should be explicit; in particular, derived wind features should not appear in A.

## Chronological evaluation

Use contiguous, nonoverlapping blocks of dates (for example, earliest 60% train / next 20% validation / most recent 20% test, adjusted for adequate samples and seasonal coverage). Choose thresholds, hyperparameters and alert cutoff using **train/validation only**, then evaluate **once** on untouched test. If the sample window is small, report the seasonal limitation rather than random-splitting.

Report, separately for positive and negative spikes:

| Metric | Why |
|---|---|
| Event count and prevalence | Provides a no-skill reference for rare-event PR-AUC. |
| PR-AUC / average precision | Primary ranking metric for rare events. |
| Precision / recall / F1 at stated alert threshold | Judges can see false alarms and misses. |
| Confusion matrix | Actual event-level behaviour. |
| ROC-AUC (optional) | Useful supplementary ranking measure, not the only rare-event score. |
| Brier score + reliability plot (optional) | Checks whether displayed percentages behave as probabilities. |

Also publish ablation results (A, B, C), including **delta vs baseline**. If any model underperforms, show that honestly and discuss limited data, missingness, or market mechanisms that wind alone cannot capture.

## Leakage checklist

- [ ] Target price quantiles computed from training outcomes only.
- [ ] Data split by **settlement date**, not shuffled rows.
- [ ] Forecast publication timestamps verified to precede issue cutoff.
- [ ] Revisions compared for the same future settlement period.
- [ ] No realised wind-generation error or final demand/price in model inputs.
- [ ] Any preprocessing (imputation, encoding, calibration) trained only on its permitted past block.
- [ ] Selection of demo replay day doesn't alter the trained model; disclose if hand-picked.
- [ ] Test results reproducible with pinned inputs and seed.

## Replay result format

For each historical period, save:

```text
settlement_date, settlement_period, prediction_cutoff_utc,
positive_risk, negative_risk, actual_price_gbp_per_mwh,
positive_event, negative_event, model_version
```

A forecast replay must be reconstructed from information that existed as of its declared cutoff; simply rerunning today's `latest` forecasts on a historical date is **not** a valid replay.

## Delivery artifacts

- `models/*.joblib` or LightGBM native model files (if repo allows), with serialized metadata / feature order / training cutoffs;
- held-out metrics JSON/CSV, calibration status, and ablation table;
- a cached set of per-period predictions for offline replay;
- one script/command for a teammate to retrain and reproduce the results.
