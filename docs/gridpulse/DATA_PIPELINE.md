# CSV ingestion and Tiger Data pipeline

## Source and handoff

Reuse the Elexon API exporter already being implemented by the team. CSV files are the initial handoff into the backend; the dashboard reads through the backend API rather than opening CSVs directly.

Expected datasets are imbalance settlement prices, wind forecasts and demand forecasts. Wind outturn is optional for retrospective charts. The exporter owner must provide one real sample per dataset and document actual column names, units, timezone, publication timestamps, identifiers and revision semantics before ingestion is finalised.

Source catalogue: [Elexon Insights API](https://bmrs.elexon.co.uk/api-documentation/endpoint). Existing candidate datasets are DISEBSP (system prices), WINDFOR (wind forecasts), NDF (demand) and optionally FUELHH (outturn). These are source candidates; verify mappings against actual exports during implementation.

## Weather and regional data for the map extension

The confirmed extension calls for synoptic weather layers plus a separately selectable electricity-use/forecast heatmap. Its external providers and endpoints have not been selected. The Discord mention of the Royal Meteorological Society is a source lead, not a verified API or streaming integration.

Before implementing the layers, establish:

- A spatial pressure field, including coordinates, valid time and units, from which meaningful isobars can be rendered. Isolated city samples alone do not define a synoptic chart.
- Wind speed/direction and relevant height, spatial resolution, forecast valid time and run/publication metadata.
- A source of meteorological front geometry, or a separately validated derivation. Wind arrows and pressure contours are not themselves fronts.
- Genuine regional electricity-use and forecast data with region boundaries/identifiers and coverage. Do not present the existing national series as measured regional demand.
- A common time selection and explicit handling of differing spatial and temporal resolutions. Record source, provenance and freshness for each layer.

Weather forecasts must remain distinguishable from observations/reanalysis. For historical investigations, establish what was available at the claimed cutoff; weather-model initialisation time alone does not prove publication availability. Wind speed is not electrical generation and must not be labelled MW without an explicit supported conversion.

No new weather table schema, provider dependency or regional-demand endpoint is approved by this brief. Agree those contracts after verifying source data. See the [frontend brief](FRONTEND_DESIGN.md) for the display requirements.

## Ingestion flow

1. Read a completed CSV file and validate its required columns and types.
2. Normalise names, explicit units and timestamps; reject or report invalid rows.
3. Record an import batch with filename, content hash, source, extraction time and row counts.
4. Insert/upsert records into Tiger Data in a transaction using dataset-specific identities.
5. Record accepted/rejected counts, coverage and completion status; run analysis only after a successful import.

Repeated imports must be safe. Preserve different forecast publication versions rather than overwriting them with the newest forecast. For revised settlement prices, retain settlement-run/version information when supplied; otherwise explicitly document that the export contains only the latest retrospective outcome.

## Proposed logical tables

> **Superseded:** the data is now loaded into Tiger Data's `raw` schema, catalogued in `raw.dataset_catalog`, and the backend reads it through `gridpulse.*` views. See [BACKEND_DESIGN.md](BACKEND_DESIGN.md#data-in-tiger-data). The table below is the original proposal.

These fields describe the application contract, not exact CSV column names. Final database migrations depend on real samples.

| Table | Important fields | Identity / handling |
|---|---|---|
| `import_batches` | batch ID, dataset, filename/hash, extracted/imported times, status, counts | Identify already imported content; track failures |
| `wind_forecasts` | settlement date/period, delivery start UTC, publication UTC, wind MW, source, batch ID | Delivery + publication + source and any source record identifier |
| `demand_forecasts` | settlement date/period, delivery start UTC, publication UTC, demand MW, source, batch ID | Preserve versions and demand definition |
| `settlement_prices` | settlement date/period, delivery start UTC, price GBP/MWh, settlement run/version if available, source, batch ID | Explicit rule for versioned records or latest-only upserts |
| `analysis_runs` | run ID, algorithm ID/version, parameters, input batches, time range, cutoff/mode, completion time | Makes results reproducible |
| `analysis_results` | run ID, settlement date/period, delivery start UTC, indicator name, value, unit/status | Run + period + indicator; missing differs from zero |

Use PostgreSQL `TIMESTAMPTZ` for instants. Index delivery times and common date-range filters. Configure Tiger Data time-series tables for the main time-series queries once the schema is confirmed; unique indexes must include any time partition column. Show a useful SQL time-window query or aggregation in the demo. Continuous aggregates and compression are optional, not prerequisites.

## Time and historical availability

Use `(settlement_date, settlement_period)` as the business identity and UTC for instants. GB settlement days contain 46, 48 or 50 periods; avoid fixed 48-slot arrays and naive local-time joins. The hackathon's 25 October 2026 clock-change day is a relevant edge case. See [Elexon settlement guidance](https://www.elexon.co.uk/bsc/settlement/).

Retain publication timestamps separately from delivery and ingestion timestamps. For a declared historical cutoff, select the newest forecast published by that cutoff for the same delivery period. Do not substitute ingestion time for publication time. If versions or timestamps are unavailable, describe the view as retrospective and disclose the limitation.

## Quality and configuration

- Check duplicate identities, date/period validity, units and missing-data coverage.
- Preserve negative prices and missing values; do not silently replace either with zero.
- Confirm revisions compare forecasts for the same delivery interval.
- Keep raw CSV exports available locally for debugging; ignore large exports and credentials in Git.
- Commit only a small redistribution-permitted demo fixture with source and extraction notes.
- Use server-side database configuration, for example `DATABASE_URL`; commit placeholders only.
- Import at startup or via an explicit command initially. Scheduling and direct API ingestion can wait.

## Cached demo fallback

Export a small set of source series and real analysis results from one completed run. The backend should be able to serve this cache if the hosted database is unavailable, and the UI must label cached mode and data freshness. Tiger Data remains the normal runtime store; the cache is a demo fallback.
