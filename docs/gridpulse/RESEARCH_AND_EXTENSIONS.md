# Research, high-impact extensions, limitations

## Research basis (references supplied in the proposal)

1. **Goodarzi, Perera & Bunn (2019)** — *The impact of renewable energy forecast errors on imbalance volumes and electricity spot prices*, *Energy Policy* 134, 110827. https://doi.org/10.1016/j.enpol.2019.06.035 — Evidence from the German electricity market; motivation for studying renewable forecast errors and price/imbalance behaviour. It does **not** establish that our classifier or hypothetical-slider response is causal.
2. **Browell & Gilbert (2022)** — *Predicting Electricity Imbalance Prices and Volumes: Capabilities and Opportunities*, *Energies* 15(10), 3645. https://doi.org/10.3390/en15103645 — GB-specific forecasting context; compares day-ahead and intraday horizons and includes probabilistic/fan-chart examples. Do not compare their intraday reported performance directly with our day-ahead-only MVP.
3. **Zamudio López, Zareipour & Quashie (2024)** — *Forecasting the Occurrence of Electricity Price Spikes: A Statistical-Economic Investigation Study*, *Forecasting* 6(1), 115–137. https://doi.org/10.3390/forecast6010007 — Tree-based classification, precision/recall and event thresholds. **This study uses Alberta electricity data, not GB**; only the methodology is transferred.
4. **Lundberg & Lee (2017)** — *A Unified Approach to Interpreting Model Predictions*. https://arxiv.org/abs/1705.07874 — SHAP feature attribution, not proof of causal explanation.
5. **Elexon Insights API documentation** — https://bmrs.elexon.co.uk/api-documentation/endpoint — Public forecast/price endpoint catalogue.
6. **Elexon settlement period rules** — https://www.elexon.co.uk/bsc/settlement/ — 46/48/50 settlement periods around daylight-saving transitions.

## Optional extensions ranked by impact / effort

| Priority | Extension | Expected benefit | Rough effort after MVP | Key caveat |
|---|---|---|---|---|
| **P1.1** | Hypothetical **Wind Shock Simulator** | Immediate interactive demo: change wind forecast by ±2 GW and rescore risk. | 1–2h | Sensitivity, **not** a causal market simulator; update derived features consistently. |
| **P1.2** | **SHAP explanation** of selected settlement period | Judges see top feature contributions behind alerts. | 1–3h | Model attribution, not real-world cause. |
| **P1.3** | **A/B/C wind-information ablation** | Strongest empirical research narrative: do revisions add held-out predictive value? | 1–2h | Use one dataset, issue protocol and held-out split; don't cherry-pick. |
| **P1.4** | **Quantile price fan chart** | Visual uncertainty bands for settlement price (£/MWh). | 2–4h | Secondary model; quantiles may cross and need calibration checks. |
| **P1.5** | **Forecast instability alerts** | Show volatile/reversing wind forecast updates. | 2–3h | Derive exclusively from versions published by cutoff; test relationship, don't assume. |

If time only allows one extra: choose the **wind shock slider** after historical replay works. If the judges value a research result, the **three-way ablation** is arguably the most important add-on and can be implemented independently of the slider.

## Methodological / product limitations to show judges

- **Forecast revisions are informative hypotheses, not demonstrated causes.** Wind alone misses outages, generator bids, market positions, interconnector flows, balancing actions and many other drivers.
- **Imbalance prices can be revised** by settlement runs. The chosen DISEBSP endpoint returns latest run information and is suitable as a retrospective outcome, not proof of what a participant saw at an earlier time.
- **Data availability differs by horizon.** A published API timestamp must be checked against the exact issue time; every forecast row must obey the as-of join rule.
- **Rare events are difficult.** Report prevalence and false alarms, not just impressive individual replay days or accuracy.
- **Probability is not automatic calibration.** LightGBM scores need empirical reliability checks before claiming that “80% risk” means 8 in 10 similar cases.
- **Weather ≠ realised wind generation.** Retrospective forecast error is useful for analysis, but actual outturn cannot be used as an ahead-of-time predictor.
- **Local-day length varies.** The Oct 25, 2026 clock change creates a 50-period settlement day; build labels and plots by settlement period, not fixed 48-slot assumptions.

## Future work (out of hackathon scope)

Intraday issue times; additional system margin and interconnector data; calibrated conditional quantiles; conformal coverage evaluation; rolling-origin multi-season backtests; and actual operational decision-value studies with transaction-cost assumptions. These should not be advertised as implemented MVP features.
