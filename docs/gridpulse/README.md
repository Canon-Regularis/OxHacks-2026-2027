# GridPulse — GB electricity imbalance-price risk predictor

**Status:** OxHack 2026 proposal, not an implemented or validated forecasting system.
**Team:** 3 developers. **Build window:** ~48 hours. **Suggested stack:** Python, Elexon Insights API, Parquet/DuckDB, LightGBM, Streamlit, Plotly.

## The idea

Predict **extreme positive and negative GB electricity imbalance settlement prices** using forecasts of wind generation and electricity demand, especially **revisions to wind forecasts**. Demonstrate the model with a settlement-period risk heatmap, a historical replay, and a hypothetical wind-shock slider.

> **Pitch:** “When wind forecasts change, electricity markets must react. GridPulse shows when Britain's imbalance settlement prices are at heightened risk of extremes, what information informed the alert, and how the risk changes under hypothetical wind scenarios.”

**Correct framing:** The target is the *imbalance settlement price*, not a prediction of grid failure or a measure of actual physical grid stability. Risk probabilities are model estimates, not guaranteed events or trading advice.

## Core research question

**Do historical revisions to published wind-generation forecasts improve out-of-sample classification of extreme GB imbalance prices beyond calendar, demand and the latest wind forecast?**

The minimum scientifically useful result is a chronological comparison of:

1. **A:** calendar + point-in-time demand forecast;
2. **B:** A + latest point-in-time wind forecast;
3. **C:** B + point-in-time wind forecast revisions.

It is a valid result if C does *not* outperform B. Never pre-claim an improvement.

## Minimum viable product — ordered

| Order | Feature | Done when |
|---|---|---|
| 1 | Settlement price history | A clean settlement-date/period price-label table exists. |
| 2 | Wind forecast versions | Forecasts retain publication times and support as-of selection. |
| 3 | Forecast demand and residual demand | Matching demand estimates exist or missingness is documented. |
| 4 | Baseline + two LightGBM classifiers | Both positive and negative extreme-event scores run end-to-end. |
| 5 | Time-ordered validation | A held-out result with PR-AUC, precision, recall and event rates is reported. |
| 6 | Dashboard heatmap | A user can view per-period model risk for a selected day. |
| 7 | Historical event replay | Predictions made using earlier-known inputs are compared with actual prices. |
| 8 | Wind forecast vs outturn analysis | Retrospective error chart, never used as forward-looking input. |
| 9 | Offline reproducible demo | Cached data/model and one documented launch path work without a live API. |

**Suggested MVP prediction protocol:** For a local GB settlement day **D**, use only data published by **18:00 Europe/London on D−1** to predict each settlement period on D. Handle the cutoff in UTC internally. This is a project design choice, not a claim about official market deadlines.

## What's in this document set

- [DATA_PIPELINE.md](DATA_PIPELINE.md) — sources, feature schema, point-in-time joins, DST and quality checks.
- [MODEL_AND_EVALUATION.md](MODEL_AND_EVALUATION.md) — labels, baselines, LightGBM, validation and ablations.
- [BUILD_AND_DEMO.md](BUILD_AND_DEMO.md) — developer ownership, 48-hour milestones, dashboard and demo script.
- [RESEARCH_AND_EXTENSIONS.md](RESEARCH_AND_EXTENSIONS.md) — papers, high-impact additions, cuts and caveats.

## Hard guardrails

- No forecast, observation or revised price that was **unavailable at the prediction cutoff** may enter an input row.
- Define extreme-price thresholds using **training data only**, then freeze them for validation/test.
- Keep train/validation/test in chronological order; do not shuffle settlement periods.
- Never present a synthetic slider formula as the trained model's output.
- Do not claim the model is profitable, causal, calibrated or robust until the relevant evaluation supports it.
- For settlement periods, use `(settlement_date, settlement_period)` as canonical identity: Great Britain has **46/48/50 periods** on clock-change/normal days, respectively.

## Go/no-go gate

By the first half of the hackathon, prioritise **cached real data + leakage-safe baseline + real scored predictions visible in the UI**. If blocked on historical version data, reduce scope to a transparent baseline model with fewer features and document limitations; do not silently use later information.
