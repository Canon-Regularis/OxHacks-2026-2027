# GridPulse API for the frontend

Everything the frontend needs to call the backend, without reading backend code. How to *display* algorithm results is in [ALGORITHM_DISPLAY.md](ALGORITHM_DISPLAY.md). TypeScript types for every response are in [`gridpulse/frontend/src/lib/types.ts`](../../gridpulse/frontend/src/lib/types.ts), and a synthetic example of every response is in [`gridpulse/contracts/examples/`](../../gridpulse/contracts/examples/).

## What is live and what is planned

- **Live:** every endpoint below, the two reference algorithms (`wind_price_correlation`, `price_naive_backtest`) and eight algorithms built on the Orchid methods: three backtests, a day-ahead forecast, rolling signals, correlograms, residual diagnostics and Fourier terms. Example responses are in [`contracts/examples/algorithms/`](../../gridpulse/contracts/examples/algorithms/); tests keep their metadata and table columns identical to the real algorithms.
- **Computed per environment:** each algorithm shows results only after someone runs it against that environment's database (`latest_run` is `null` until then).

Build every algorithm panel from `/api/catalog` rather than hard-coding ids. New algorithms then appear without frontend changes.

## Basics

- Every endpoint is `GET` and returns JSON. Call them with **relative** URLs (`/api/...`): the production build is served from the same origin as the API.
- No authentication. Never put credentials in frontend code.
- Interactive documentation of the running server is at `/docs`.

### Time and settlement periods

- Every instant is **UTC**, written as an ISO string ending in `Z`, e.g. `2026-10-24T16:30:00Z`. Convert to Europe/London only for display.
- GB electricity is settled in half-hour **settlement periods**, identified by `(settlement_date, settlement_period)`. A settlement date is a Europe/London calendar day. Period 1 starts at local midnight.
- **Days do not always have 48 periods.** The day the clocks go forward has 46 and the day they go back has 50. **25 October 2026, the hackathon weekend, has 50 periods**: the local hours 01:00 and 01:30 occur twice. Never assume 48 periods per day, and never key data by local time.
- `from` and `to` query parameters are **inclusive settlement dates** (`YYYY-MM-DD`). `from=2026-10-24&to=2026-10-24` covers one whole day: `2026-10-23T23:00Z` to `2026-10-24T23:00Z`.
- Parameters that take an instant (`at`, `cutoff`) need an explicit offset, e.g. `2026-10-24T12:00Z`.

### Columnar series

Time series come as one shared `t` array of UTC bucket starts, plus one `values` array per series of the same length:

```json
{ "t": ["2026-10-23T23:00:00Z", "2026-10-23T23:30:00Z"],
  "series": { "price": { "label": "Imbalance price", "unit": "GBP/MWh", "values": [71.2, null] } } }
```

`null` means **missing**, never zero: draw a gap. This shape maps straight onto uPlot (`[t.map(Date.parse), ...values]`) and converts easily for other chart libraries.

### Resolution

`resolution` is `auto` (the default), `30m`, `1h` or `1d`. `auto` chooses the finest resolution that fits within 5,000 points per series:

| Range | `auto` gives |
|---|---|
| up to 104 days | `30m` |
| up to 208 days | `1h` |
| longer | `1d` |

Coarser buckets aggregate the 30-minute values: the mean for most series, `min`/`max` for `price_min`/`price_max`, and the sum for `niv`. `1d` buckets are London settlement days. Algorithm outputs are averaged the same way, so forecasts and their bands are clearest at `30m`. Refetch with the new range when the user zooms. Ranges are limited to 3,660 days (`limits.max_range_days` in the catalog).

## Start-up sequence

1. `GET /api/health`: always answers, even without a database. Show `mode` and `database` in a status bar.
2. `GET /api/catalog`: series, datasets, algorithms and limits. Build menus, legends and panels from it.
3. Then fetch what the page shows: `/api/snapshot` for a ticker, `/api/series` and `/api/analysis` for charts.

## Endpoints

### `GET /api/health`

Example: [`health.json`](../../gridpulse/contracts/examples/health.json).

| Field | Meaning |
|---|---|
| `status` | `ok` or `degraded` (the database is unavailable or not configured) |
| `mode` | `live`, `record` or `replay`; show a "cached" badge whenever it is not `live` (see [demo modes](#demo-modes)) |
| `database` | `ok`, `unavailable`, `not_configured`, or `not_used` (replay mode) |
| `version` | Backend version string |

### `GET /api/catalog`

Example: [`catalog.json`](../../gridpulse/contracts/examples/catalog.json).

| Field | Meaning |
|---|---|
| `series[]` | `{id, label, unit, kind, datasets}` for every series `/api/series` serves; see [datasets and lineage](#datasets-and-lineage) |
| `datasets[]` | Raw datasets loaded: `dataset` (its name), `source`, `notes`, `first_utc`, `last_utc`, `loaded_at_utc`, `approx_rows`. Use for a "data coverage" panel on Terminal or Methods, and to look up the names in `series[].datasets`, `algorithms[].datasets` and `RunInfo.datasets` |
| `algorithms[]` | One `AlgorithmInfo` per algorithm, described [below](#algorithminfo) |
| `limits` | `max_points` (5,000) and `max_range_days` (3,660) |
| `data_updated_at_utc` | When data was last loaded; show it as "data as of …" |

Series ids:

| id | Label | Unit | `kind` | `datasets` |
|---|---|---|---|---|
| `price` | Imbalance price | GBP/MWh (can be negative) | settlement | `elexon_system_prices` |
| `price_min`, `price_max` | Lowest and highest price in each bucket | GBP/MWh | settlement | `elexon_system_prices` |
| `niv` | Net imbalance volume | MWh | settlement | `elexon_system_prices` |
| `wind` | Wind forecast (latest version published before delivery; hourly, so both halves of an hour match) | MW | forecast | `elexon_windfor` |
| `wind_outturn` | Wind outturn (excludes embedded wind, so not like-for-like with `wind`) | MW | outturn | `elexon_fuelhh` |
| `demand` | National demand forecast (latest version before delivery) | MW | forecast | `elexon_ndf` |
| `demand_outturn` | National demand outturn (about 21 days in arrears) | MW | outturn | `neso_historic_demand` |
| `residual_demand` | Demand forecast minus wind forecast | MW | forecast | `elexon_ndf`, `elexon_windfor` |

#### AlgorithmInfo

| Field | Meaning |
|---|---|
| `id`, `version` | Stable id for requests; version string |
| `description` | One line for headings and tooltips |
| `inputs` | Input columns the algorithm reads; show on Methods |
| `datasets` | Raw datasets behind those inputs, e.g. `["elexon_system_prices", "elexon_windfor"]`; empty if it only uses the clock |
| `outputs` | Output name → unit. Each output is a time series in `/api/analysis`. Empty for tables-only algorithms |
| `tables` | Table name → description, fetched from the tables endpoint |
| `params` | Default parameters ([settings](#algorithm-settings-runs-and-variants)) |
| `param_descriptions` | Parameter name → what it means; same keys as `params` |
| `limitations` | What the numbers do **not** mean. **Always show it with the results** |
| `latest_run` | The run `/api/analysis` serves by default, or `null` if it has not been run. Its `run_id` gives direct access to the tables |

### `GET /api/snapshot`

The latest settled period, for a ticker strip. Poll every 60 s. Example: [`snapshot.json`](../../gridpulse/contracts/examples/snapshot.json).

| Query | Meaning |
|---|---|
| `at` | Optional instant with an offset: show the ticker as it was then ("replay the day") |

| Field | Meaning |
|---|---|
| `as_of_utc`, `settlement_date`, `settlement_period` | The latest period with a settled price |
| `items[]` | `{id, label, unit, value, previous, change}` for `price`, `niv`, `wind`, `wind_outturn`, `demand` and `residual_demand`. `change = value − previous`; any of the three may be `null` |
| `data_updated_at_utc` | As in the catalog |

### `GET /api/series`

Several series on one time axis. Example: [`series.json`](../../gridpulse/contracts/examples/series.json).

| Query | Meaning |
|---|---|
| `from`, `to` | Required inclusive settlement dates |
| `ids` | Comma-separated series ids; default `price,wind,demand` |
| `resolution` | `auto` (default), `30m`, `1h`, `1d` |
| `cutoff` | Optional instant: use only forecast versions published by then, and prices and outturns only for periods ended by then. `30m` only, so keep the range within 104 days |

| Field | Meaning |
|---|---|
| `from`, `to`, `resolution`, `cutoff_utc` | Echo of what was served |
| `t` | Bucket starts |
| `settlement_date`, `settlement_period` | Arrays aligned with `t`, **only at `30m`** (otherwise `null`). Use them for labels and for drill-down links |
| `series` | Series id → `{label, unit, values}` |

Without `cutoff` a chart is **retrospective** (it uses the latest data), and should say so where it matters. With `cutoff` it shows what was known at that instant.

### `GET /api/extremes`

Highest or lowest periods, for a "top movers" list. Example: [`extremes.json`](../../gridpulse/contracts/examples/extremes.json).

| Query | Meaning |
|---|---|
| `from`, `to` | Required |
| `series` | Series id; default `price` |
| `order` | `desc` (highest first, default) or `asc` |
| `limit` | 1–100, default 10 |

Response: `series` (`{id, label, unit}`), `from`, `to`, `order`, and `rows[]` of `{delivery_start_utc, settlement_date, settlement_period, value}`. Link each row to the period detail.

### `GET /api/periods/{settlement_date}/{settlement_period}`

Everything about one period, for a click-through inspector. Example: [`period.json`](../../gridpulse/contracts/examples/period.json).

| Field | Meaning |
|---|---|
| `settlement_date`, `settlement_period`, `delivery_start_utc`, `delivery_end_utc` | The period |
| `values` | Series id → value (look up labels and units in the catalog) |
| `vintages` | `wind` and `demand` → every published forecast version for this period, oldest first, as `{published_at_utc, value}`. Chart it as a step line to show how the forecast moved before delivery |
| `analysis[]` | Every algorithm output for this period from each algorithm's default run: `{algorithm_id, algorithm_version, run_id, indicator, unit, value}` |

A period number that does not exist on that day (e.g. 49 on a 48-period day) returns 404 `no_such_period`.

### `GET /api/analysis`

The time-series outputs of one algorithm run. Example: [`analysis.json`](../../gridpulse/contracts/examples/analysis.json), and one per Orchid algorithm, live or planned, in [`contracts/examples/algorithms/`](../../gridpulse/contracts/examples/algorithms/).

| Query | Meaning |
|---|---|
| `algorithm` | Required algorithm id from the catalog |
| `from`, `to` | Required inclusive settlement dates |
| `resolution` | `auto` (default), `30m`, `1h`, `1d` |
| `params` | Optional JSON object of parameter overrides, e.g. `{"horizon_periods":24}`. Serves a [variant](#algorithm-settings-runs-and-variants) |
| `run_id` | Optional: serve exactly this run. Do not combine with `params` |

| Field | Meaning |
|---|---|
| `algorithm` | `AlgorithmInfo` as in the catalog (its `latest_run` is `null` here) |
| `run` | The run served: `RunInfo`, below |
| `from`, `to`, `resolution`, `t` | As in `/api/series` (no settlement arrays) |
| `indicators` | Output name → `{label, unit, values}`. Empty `{}` for tables-only algorithms; `t` is still filled |
| `tables` | Names of the run's tables |
| `warnings` | Messages to **show on screen** when non-empty: the run covers only part of the range, uses a cutoff, is an older version, uses non-default parameters, or one of its datasets has been loaded again since it ran |

Periods outside the run's range come back as `null`. A run of a forecasting algorithm can extend past the `to` it was run with, by its forecast horizon.

#### RunInfo

| Field | Meaning |
|---|---|
| `run_id` | Use it for the tables endpoint and to pin a run |
| `algorithm_id`, `algorithm_version` | What produced it |
| `params` | Every parameter the run used |
| `overrides` | The parameters that differ from the current defaults; `{}` for a default run |
| `datasets` | Raw dataset → its `loaded_at_utc` when the run started: exactly which data load the run used. `null` load times mean a run made before this was recorded |
| `range_start_utc`, `range_end_utc` | Periods the run covers; the end includes any forecast horizon |
| `cutoff_utc` | Non-null for a run limited to data available at that instant |
| `finished_at_utc` | When it was computed |

### `GET /api/analysis/runs`

Successful runs of one algorithm, newest first, so the UI can offer variants. Example: [`analysis_runs.json`](../../gridpulse/contracts/examples/analysis_runs.json).

| Query | Meaning |
|---|---|
| `algorithm` | Required algorithm id |
| `limit` | 1–100, default 20 |

Response: `algorithm` (`AlgorithmInfo`, with `latest_run` set to the default run) and `runs[]` (`RunInfo`). The list includes every version, parameter set and cutoff run.

### `GET /api/analysis/runs/{run_id}/tables`

Results that are not one value per period: scorecards, correlograms, fitted parameters, diagnostics. Example: [`analysis_tables.json`](../../gridpulse/contracts/examples/analysis_tables.json).

| Query | Meaning |
|---|---|
| `name` | Optional: return only this table |

Response: `run` (`RunInfo`) and `tables`: name → `{description, columns, rows}`. Each row has one cell per column; cells are numbers, strings, booleans or `null`, and datetimes are UTC ISO strings. A table has at most 20,000 rows.

To show an algorithm's tables you do not need `/api/analysis` first: use `algorithm.latest_run.run_id` from the catalog.

## Algorithm settings: runs and variants

Algorithms are computed in batch jobs on the server, never when a page loads. Each computation is a **run**, stored with the **parameters** it used.

- **Defaults.** Every algorithm declares default parameters (`params`), each explained in `param_descriptions`. Show both on the Methods page.
- **Default run.** With no `params` or `run_id`, `/api/analysis` serves the latest run of the current version made with exactly the default parameters, without a cutoff. That is also `latest_run` in the catalog. A run made later with different settings never replaces it.
- **Variants.** A run with other settings is a variant. Its `overrides` field lists what differs, e.g. `{"horizon_periods": 1}`; label it with them ("horizon 1 period"). Fetch a variant either way:
  - by settings: `params` = the overrides as a JSON object. Only the keys you send change; the rest stay at their defaults:

    ```ts
    const params = encodeURIComponent(JSON.stringify({ horizon_periods: 24 }));
    fetch(`/api/analysis?algorithm=price_naive_backtest&from=2026-10-01&to=2026-10-23&params=${params}`);
    ```

    Build the object with keys in the same order every time: the offline demo cache matches requests by their exact text.
  - by run: list the runs with `/api/analysis/runs?algorithm=…` and pass `run_id`.
- **Not computed yet.** A setting nobody has run returns 404 `no_run`. Its message contains the command a backend developer runs to compute it. Show "not computed yet", not an error. Manual chart controls cannot start a computation. The separate [AI endpoint](AI_AGENT.md) can request bounded execution after reuse/confirmation checks.
- **Versions.** A changed algorithm gets a new `version`; runs of older versions are no longer served by default.
- **Cutoff runs** (`cutoff_utc` set) use only data available at that instant. They are reachable only by `run_id` and come with a warning.

An unknown parameter name returns 400 `unknown_param`; a `params` value that is not a JSON object returns 400 `invalid_params`.

## Datasets and lineage

Three levels tell data sets apart:

1. **Raw datasets** are what the loader put in the database, one per source, e.g. `elexon_windfor`. They are listed in `catalog.datasets[]` with their source, coverage and `loaded_at_utc`. Some, such as weather data for the map extension, are loaded but not used by any series yet.
2. **Series** are what the API serves. Each names the raw datasets it is built from (`series[].datasets`) and its `kind`. Forecasts and outturns of the same quantity are always separate series.
3. **Runs** record the raw datasets their algorithm read and each one's load time (`RunInfo.datasets`). If a dataset has been loaded again since, `/api/analysis` adds a warning, because the run may predate new or revised data.

To show sources, join on the dataset name:

```ts
const byName = new Map(catalog.datasets.map((d) => [d.dataset, d]));
const sources = algorithm.datasets.map((name) => byName.get(name)?.source ?? name);
// e.g. ["Elexon BMRS DISEBSP - ...", "Elexon BMRS WINDFOR - /datasets/WINDFOR"]
```

## Errors

Every error except FastAPI's own parameter validation is `{"detail": {"code": "...", "message": "..."}}`. Show `message` to the user; branch on `code`.

| Status | `code` | When | Suggested UI |
|---|---|---|---|
| 400 | `invalid_range` | `to` before `from` | Fix the date picker |
| 400 | `range_too_large` | Over 3,660 days | Limit the picker |
| 400 | `too_many_points` | Explicit `resolution` too fine for the range | Use `auto` |
| 400 | `unknown_series` | Bad series id | Build ids from the catalog |
| 400 | `naive_datetime` | `at`/`cutoff` without an offset | Always send `Z` |
| 400 | `cutoff_needs_30m` | `cutoff` with a range over 104 days | Narrow the range |
| 400 | `cutoff_unsupported` | A series not available in cutoff mode | Drop that series |
| 400 | `invalid_params`, `unknown_param` | Bad `params`, or `params` with `run_id` | Build params from `AlgorithmInfo.params` |
| 404 | `unknown_algorithm` | Algorithm id not registered | Build ids from the catalog |
| 404 | `no_run` | Not computed for that range/settings, or no such run | "Not computed yet" placeholder |
| 404 | `no_table` | The run has no table with that `name` | Use names from `tables` |
| 404 | `no_such_period` | Period number not on that day | — |
| 404 | `no_data` | No prices loaded at all | Empty state |
| 404 | `not_recorded` | Replay mode: request was not recorded | "Not available offline" |
| 422 | (list) | Missing or malformed query parameter; `detail` is a list | Treat as a bug |
| 503 | `database_unavailable` | Database down or not configured | Status bar; retry |
| 504 | `query_timeout` | Query over 15 s | "Narrow the date range" |

## Demo modes

`/api/health` reports `mode`:

- `live`: normal.
- `record`: normal, and every successful response is saved for an offline demo.
- `replay`: served from those saved responses with no database. Only requests made during recording work; others return 404 `not_recorded`. Responses carry the header `X-GridPulse-Mode: replay`.

Show a "cached data" badge whenever `mode` is not `live`. For a reliable offline demo, click through the demo path once in record mode, using the same date ranges and settings the demo will use.

## Request helpers

A minimal typed fetch wrapper that turns API errors into exceptions:

```ts
import type { ApiError } from "./types";

export class GridPulseError extends Error {
  constructor(public status: number, public code: string, message: string) {
    super(message);
  }
}

export async function get<T>(path: string, query: Record<string, string | number | undefined> = {}): Promise<T> {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) if (value !== undefined) search.set(key, String(value));
  const response = await fetch(`/api${path}?${search}`);
  const body = await response.json();
  if (!response.ok) {
    const detail = (body as ApiError).detail;
    const code = typeof detail === "object" && !Array.isArray(detail) ? detail.code : "validation_error";
    const message = typeof detail === "object" && !Array.isArray(detail) ? detail.message : "Invalid request";
    throw new GridPulseError(response.status, code, message);
  }
  return body as T;
}

// get<Catalog>("/catalog");
// get<AnalysisResponse>("/analysis", { algorithm: "price_naive_backtest", from: "2026-10-01", to: "2026-10-23" });
// get<RunTables>(`/analysis/runs/${runId}/tables`);
```

`URLSearchParams` encodes a `params` JSON string correctly, so pass `params: JSON.stringify(overrides)`.

## Terminal AI streaming endpoint

`POST /api/agent/turns/stream` is a same-origin, session-scoped SSE POST through the SvelteKit server. See [AI_AGENT.md](AI_AGENT.md) for UX, authentication, bounds and cancellation. Canonical request and decision schemas: [turn](../../gridpulse/contracts/agent/turn.schema.json), [decision](../../gridpulse/contracts/agent/decision.schema.json). Backend schemas live in `agent/schemas.py`; frontend types/compiler in `features/terminal/agent/`. Update both with contract changes.

The request identifies turn, browser instance, conversation, workspace and opaque revision token, and includes bounded history/workspace metadata. Responses carry progress/evidence/answer/view/confirmation/completed/error events. Complete views are validated and committed locally, never applied from partial streamed JSON. A confirmation token must be explicitly accepted via Run analysis and is returned in the next turn's `approval` field. Provider errors preserve existing workspace content. HTTP 409 indicates duplicate/busy, 413 excessive context, 429 usage cap, and 503 disabled/misconfigured proxy or offline mode; errors after streaming starts arrive as error events.

Agent decisions now include `layout_mode` (`research` default, or `preserve` for explicit placement). Widget specs may select comparison `metric`/`category`; both refer to inspected table columns. Source `outputs` can be empty or a subset of the selected source IDs. Source tables use the existing aligned `/series` frame, preserve null values/units and label their selected range rather than whole-run metrics. Algorithm table names require an actual algorithm/run. `started` includes model identity for the chat label.
