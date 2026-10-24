# Existing algorithms and backend integration

## Scope

Integrate the advanced algorithms already developed by a team member. This project does not require training models. Their formulas and capabilities are not yet specified in this repository; do not replace them with invented rules or advertise unsupported outputs.

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
