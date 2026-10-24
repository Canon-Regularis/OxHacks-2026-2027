# Data pipeline and point-in-time contracts

## Sources (Elexon Insights API)

Base API: `https://data.elexon.co.uk/bmrs/api/v1` (confirm actual request paths/parameters using the linked documentation before implementing). Elexon's developer portal describes its APIs as public without an API key; cache results locally for reliability.

| Purpose | Dataset / documented endpoint | Important handling |
|---|---|---|
| **Target:** GB imbalance system prices | [DISEBSP — `/balancing/settlement/system-prices/{settlementDate}`](https://bmrs.elexon.co.uk/api-documentation/endpoint/balancing/settlement/system-prices/%7BsettlementDate%7D) | Returns the latest available settlement run per period; its historical revision state is *not* necessarily what was available in real time. Treat as retrospective labels. |
| Forecasted wind generation | [WINDFOR `/forecast/generation/wind/history`](https://bmrs.elexon.co.uk/api-documentation/endpoint/forecast/generation/wind/history), [evolution](https://bmrs.elexon.co.uk/api-documentation/endpoint/forecast/generation/wind/evolution) | Preserve each `publishTime` and the forecast delivery interval; don't collapse to latest when training. |
| Day-ahead demand forecasts | [NDF `/datasets/NDF`](https://bmrs.elexon.co.uk/api-documentation/endpoint/datasets/NDF) and [demand history](https://bmrs.elexon.co.uk/api-documentation/endpoint/forecast/demand/day-ahead/history) | Preserve published versions; choose one demand definition consistently. |
| Retrospective wind outturn | [FUELHH `/datasets/FUELHH`](https://bmrs.elexon.co.uk/api-documentation/endpoint/datasets/FUELHH) | For analysis/chart labels only, not future prediction features. |

Alternative forecast-specific endpoints such as `/forecast/generation/wind/latest` have their own filtering semantics: **don't assume `latest` means latest as of a past cutoff without testing**. Elexon's WINDFOR documentation describes multiple forecast updates per day, motivating as-of joins.

## Minimal data model

Retain raw API JSON (or other source format) in a local cache, then normalise into tables:

```text
price_outcomes(
  settlement_date DATE, settlement_period INT,
  price_gbp_per_mwh FLOAT, source_dataset TEXT, extracted_at_utc TIMESTAMP
)
wind_forecasts(
  settlement_date DATE, settlement_period INT,
  delivery_start_utc TIMESTAMP, published_at_utc TIMESTAMP,
  forecast_wind_mw FLOAT, source_dataset TEXT
)
demand_forecasts(
  settlement_date DATE, settlement_period INT,
  delivery_start_utc TIMESTAMP, published_at_utc TIMESTAMP,
  forecast_demand_mw FLOAT, source_dataset TEXT
)
wind_outturn(
  settlement_date DATE, settlement_period INT,
  actual_wind_mw FLOAT
)
features(
  settlement_date, settlement_period, delivery_start_utc,
  prediction_cutoff_utc, forecast_wind_mw, wind_revision_6h_mw,
  wind_revision_24h_mw, forecast_demand_mw, residual_demand_mw,
  wind_share_of_demand, local_hour, weekday, month,
  target_price_gbp_per_mwh
)
```

These are **logical** fields, not a claim of exact API response column names. Inspect samples and map source units, timezone, period numbers and publication time before loading. Deduplicate source records by stable key + publication timestamp, and flag ambiguous duplicates rather than silently averaging.

## Join algorithm

For each target period `(D, period)`:

1. Derive `prediction_cutoff` = **18:00 Europe/London on D−1**, converted to UTC with an IANA timezone library.
2. For that target delivery period, select the **newest wind forecast with `published_at_utc <= prediction_cutoff_utc`**.
3. Similarly select the latest demand forecast known by that cutoff.
4. Calculate a 6h / 24h *revision* only when there is a real older published wind forecast whose release is at/before the relevant historical reference cutoff. Clearly document the meaning (e.g. difference between latest forecast as of cutoff and latest forecast as of cutoff−6h for the **same delivery period**). Missing revisions remain missing, not zero by default.
5. Derive `residual_demand_mw = forecast_demand_mw - forecast_wind_mw` and wind fraction, guarding against zero/missing demand.
6. Add calendar features derived from the **target** settlement period without consulting future outcomes.
7. Attach the latest available *retrospective* system price as the training label, never as a prediction-time input.

**Do not** use final actual wind, final imbalance price, or a forecast revised after cutoff as an input. Avoid lagged settlement prices in the MVP unless their exact publication availability can be proven.

## GB settlement periods and DST

The identity of a GB settlement period is the **settlement date + numbered settlement period**, not simply a naive local clock timestamp. Official Elexon guidance specifies 46 periods on the spring transition, 50 on the autumn transition, and 48 on ordinary days. The hackathon weekend includes the autumn transition on **25 October 2026**; this is directly relevant to demo dates. Store UTC instants, zone-aware local labels, and preserve repeated local times. Do **not** assume 48 rows per day or use a naive `(local date, HH:MM)` key. [Elexon settlement guidance](https://www.elexon.co.uk/bsc/settlement/).

## Quality gates

- [ ] Confirm chosen endpoint schemas with at least one sampled response per dataset; document field/unit mapping.
- [ ] One label per valid `(settlement_date, settlement_period)`; no duplicated scoring rows.
- [ ] Unit conversions to MW and GBP/MWh are explicit and tested.
- [ ] No input `published_at_utc` exceeds its row's prediction cutoff.
- [ ] Revisions compare forecasts **for the same target period**, not different deliveries.
- [ ] Record missingness by feature/date; do not silently backfill with future revisions.
- [ ] Test normal (48) and DST-short/long (46/50) settlement days.
- [ ] Write deterministic Parquet outputs + source-fetch cache; demo can run offline.

## Suggested repository paths

```text
data/raw/             # ignored download cache
data/processed/      # ignored normalized tables
data/demo/           # small redistribution-safe, versioned fixture if permitted
src/data/            # API client and normalization
src/features/        # as-of joins and leakage tests
```

Only commit example/sample data if its terms and size are suitable; do not commit secrets or huge API extracts.
