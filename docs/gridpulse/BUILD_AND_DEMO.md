# Three-developer implementation and demo plan

## Ownership / parallel work

| Developer | Responsibility | Initial deliverable | Interface contract |
|---|---|---|---|
| **Dev 1 — Data** | Elexon client, prices, wind forecasts/versions, demand, as-of feature table, outturn analysis | Cached clean Parquet feature rows + data quality report | `features.parquet` keyed by `(settlement_date, settlement_period, prediction_cutoff_utc)` |
| **Dev 2 — ML** | Labels, baseline, dual LightGBM models, chronological evaluation, ablations, optional SHAP | `predictions.parquet` + `metrics.json` | Each period has `positive_risk`, `negative_risk`, and model metadata |
| **Dev 3 — Product** | Streamlit/Plotly dashboard, heatmap, replay, charts, offline startup, pitch | UI rendering fixture rows before model integration | Reads the same columns Dev 2 will emit; no fabricated evaluation statistics |

**First 30 minutes:** agree on filenames, schema, units, date/cutoff convention, ownership and one integration command. Dev 3 starts from a clearly labelled synthetic fixture; Dev 1/2 replace it with real inputs/outputs.

## Relative ~48-hour milestones

| Window | Deliverable / checkpoint |
|---|---|
| **0–6h** | Confirm the exact Elexon API responses; scaffold/cache a small dataset; agree schema; show a dummy heatmap. |
| **6–18h** | Finish price/wind/demand point-in-time join; train baseline and primary LightGBM classifiers; connect real predictions to UI. |
| **18–30h** | Chronological backtest; validate no leakage; historical event replay and outturn overlay. |
| **30–40h** | If stable, ship wind-shock simulator, A/B/C ablations, then SHAP. |
| **40–48h** | Freeze new features, smoke-test offline demo, ensure sources/limitations visible, rehearse and submit. |

**Go/no-go:** Do not proceed to fancy extensions before one complete prediction → backtest → dashboard loop runs end to end with authentic data.

## Dashboard screen spec

1. **Header:** “GridPulse — GB imbalance-price spike risk”; prediction cutoff and selected settlement day.
2. **Risk overview:** maximum positive risk, maximum negative risk, period of peak risk, current forecast wind/demand, *model version*.
3. **Settlement-period heatmap:** dynamically display 46/48/50 settlement periods (not always 48). Use a diverging or two-track encoding that does not confuse negative-price risk with simply low positive-price risk. Hover/click to inspect the two probabilities.
4. **Price replay:** historical actual prices alongside declared as-of model alerts, including false positives and missed spikes.
5. **Explainer:** per-period wind forecast, recent revisions and residual demand; optional SHAP contributor bar chart.
6. **Wind shock slider:** change a **hypothetical forecast input** by −2 to +2 GW, recompute derived features consistently and rescore with the trained model. Label output *scenario sensitivity, not causal market impact*.
7. **Evidence tab:** baseline vs model metrics and A/B/C ablation results; link to data sources.

Note: for the slider, a changed wind level does **not automatically imply** a changed historical forecast-revision trajectory. Start by changing latest wind and residual demand while holding revision features fixed, and explicitly disclose this assumption. Never substitute a hand-written linear formula as a real prediction.

## Demo sequence (2–3 minutes)

1. **Problem (15s):** “When wind forecasts change, imbalance-price exposure can become volatile.” Distinguish price risk from physical grid reliability.
2. **Forecast (35s):** pick a historical day, show period-by-period positive/negative extreme risk and the issuance cutoff.
3. **Replay (40s):** reveal actual settlement prices, show hits, misses and false alerts.
4. **What-if (30s):** adjust hypothetical wind forecast; watch trained-model risk change, with noncausal disclaimer.
5. **Evidence (30s):** show train/validation/test dates and three-way feature ablation. State honestly whether forecast revisions improved results.
6. **Close (15s):** real public Elexon data, reproducible evaluation, demo works offline.

Use one **real historical event** and a **backup cached date**. If choosing a particularly dramatic historical day, call it an illustrative selected example, not representative test performance.

## Offline smoke test

- [ ] Clone/checkout branch + follow README launch instructions on a clean environment.
- [ ] UI works with network disconnected using an appropriately licensed cached demo dataset.
- [ ] A real model artifact is loaded, not a dummy function, for real-data demo mode.
- [ ] Any fixture/synthetic mode is clearly labelled throughout.
- [ ] Displayed metric values correspond to committed generated evaluation results.
- [ ] Charts handle negative prices, missing inputs, and 46/50-period settlement days.
- [ ] No secrets/API tokens, huge data dumps or confidential files checked in.

## Deliberate cuts

Avoid ERA5/ECMWF weather-model training, bespoke wind-speed neural nets, LSTMs, Transformers, full grid simulation, trading P&L, reinforcement learning, production infra and a second prediction horizon. They're not required to answer the core research question.
