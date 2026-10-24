# Backend design

**Status:** the infrastructure runs end to end against the team's Tiger Data instance. That covers the API, the app-layer views over `raw`, the algorithm runner, result storage and the demo cache. It has 52 unit tests plus 5 opt-in integration tests. **Data modelling is deliberately deferred.** The `gridpulse.*` views are plain views over the raw tables, and the team decides later how to shape them. Two reference algorithms prove the path: `wind_price_correlation` (series) and `price_naive_backtest` (fixed-horizon backtest with tables). The real algorithms are Matthew's.

## Principles

1. **Tiger Data does the heavy lifting.** Bucketing, latest-forecast selection and joins happen in SQL. Python never loads raw history: the raw tables are GBs, but each response is at most a few thousand rows.
2. **Every response is bounded.** Each series returns at most `MAX_POINTS` (5000) points, and `resolution=auto` picks 30m, 1h or 1d to fit.
3. **One seam between raw data and the app.** The API reads only `gridpulse.*` views and never `raw.*` directly. Modelling work replaces those views without touching the backend.
4. **Algorithms are pure functions behind one contract.** They take a DataFrame and return a DataFrame. They run as batch jobs and store their results, and the API only reads stored results.
5. **Boring and readable.** Sync FastAPI with psycopg 3, raw SQL in one file, no ORM, no async, no task queue, no cache server.
6. **Honest data.** NULL is not 0. Instants are UTC and periods are `(settlement_date, settlement_period)` with 46/48/50 per day. "What was known at a cutoff" is separate from retrospective views.

## Architecture

```text
Elexon / NESO / Open-Meteo ─► data loader (team) ─► raw.* hypertables + raw.dataset_catalog
                                                              │  untouched by the backend
                                                              ▼
                                        gridpulse.* views   (db/migrations/002: the modelling seam)
                                        gridpulse.analysis_* tables
                                           ▲                     ▲
           gridpulse analyse ─► analysis/service.py      db/queries.py ◄─ api/*.py ◄─ Svelte terminal UI
                                  ├─ features.py   30-min input frame, built by SQL
                                  └─ algorithms/*  Matthew's pure pandas functions
```

## Module map

All paths are under `gridpulse/backend/src/gridpulse/`.

| File | Responsibility |
|---|---|
| `main.py` | App wiring: routers, CORS, gzip, demo cache, DB error → 503/504 |
| `config.py` | Settings from env, Tiger's `tiger-cloud-*-credentials.env` at the repo root, or `gridpulse/.env` |
| `settlement.py` | Settlement-period ↔ UTC maths, bucket grids (clock-change safe) |
| `catalog.py` | Series ids → `gridpulse` view, column, unit and 1h/1d aggregate |
| `cli.py` | `gridpulse migrate \| analyse \| algorithms` |
| `db/connection.py` | Request pool (autocommit, 15 s statement timeout); CLI connections |
| `db/queries.py` | **All SQL** |
| `db/migrate.py` | Applies `gridpulse/db/migrations/*.sql` (all re-runnable) |
| `analysis/contract.py` | `Algorithm` dataclass and `INPUT_COLUMNS`: **Matthew's interface** |
| `analysis/features.py` | Builds the algorithm input frame |
| `analysis/service.py` | Runs algorithms, enforces the output contract, stores and reads results |
| `analysis/algorithms/` | One module per algorithm, registered in `__init__.py` |
| `api/deps.py` | Connection dependency, date range, resolution choice, errors |
| `api/system.py`, `market.py`, `analysis.py` | Routes (thin) |
| `api/schemas.py` | Response models, kept in sync with `gridpulse/contracts/examples/` by a test |
| `api/demo_cache.py` | Record/replay of API responses for the offline demo |
| `ingestion/` | Placeholder. Loading is done by the team's loader straight into `raw` |

## Data in Tiger Data

**`raw` schema (owned by the data loader, read-only to the backend).** Each dataset is a hypertable on `start_time` (or `valid_time` for Open-Meteo), keeping every published forecast version. `raw.dataset_catalog` describes each one: source, notes, time range and load time. `/api/catalog` serves it as-is. Approximate sizes on 24 Oct 2026:

| Table | Size | Used by the backend |
|---|---|---|
| `elexon_system_prices` (DISEBSP) | 21 MB | price, NIV |
| `elexon_windfor` (hourly, ~24 vintages per hour) | 104 MB | wind forecast |
| `elexon_ndf` (half-hourly, ~75 vintages per period) | 586 MB | demand forecast (`boundary = 'N'`) |
| `elexon_fuelhh` | 15 MB | wind outturn (`fuel_type = 'WIND'`) |
| `neso_historic_demand` | 13 MB | demand outturn (ND), ~21 days in arrears |
| `elexon_tsdf`, `neso_embedded_forecast_archive`, `openmeteo_*` | 6.6 GB, 1.7 GB, small | not yet: candidates for modelling |

**`gridpulse` schema (owned by the backend, created by `gridpulse migrate`).**

| Relation | Contents |
|---|---|
| `price_30m` | `bucket`, `price_gbp_mwh` (system sell price = the single imbalance price), `niv_mwh` |
| `wind_forecast_vintages`, `demand_forecast_vintages` | Every published version, keyed by `delivery_start_utc`, `published_at_utc`. Each hourly WINDFOR value is applied to both settlement periods in its hour |
| `wind_30m`, `demand_30m` | **Latest version published at or before the period starts** (`last(value, published_at)`). Later versions are excluded: each WINDFOR publish re-issues the previous ~27 h, and a value published after delivery is not a forecast |
| `wind_outturn_30m`, `demand_outturn_30m` | Outturns |
| `residual_30m` | Latest demand forecast − latest wind forecast. This is one aggregate rather than a join, because Postgres does not carry a time filter across a join |
| `analysis_runs`, `analysis_results` (hypertable) | Algorithm runs, and per-period results in long format |
| `analysis_tables` | Non-time-series results per run (scorecards, ACF by lag, parameters, diagnostics) as `{columns, rows}` JSON |

Every `*_30m` view has one row per settlement period, a `bucket` column and named value columns. The API buckets them to 1h/1d on the fly with each series' aggregate (avg/sum/min/max from `catalog.py`). Daily buckets use `time_bucket('1 day', ts, 'Europe/London')`, so a "day" is a settlement day of 23, 24 or 25 hours. Finer buckets stay in UTC so that the repeated local hour in October is not merged.

**Measured latency.** Times are from a laptop to Tiger Cloud, with a 12 ms round trip:

| Request | Time |
|---|---|
| Health, catalog, extremes, stored analysis | 10–150 ms |
| Snapshot, period detail | ~150 ms |
| Up to 1 year of any series | 50–300 ms |
| Full 2.8-year history of 4 forecast series at daily resolution | ~2.4 s |

The slow case comes from re-aggregating every NDF/WINDFOR vintage on each request. **Modelling step:** replace `wind_30m`, `demand_30m` and `residual_30m` with continuous aggregates of the same name and columns (optionally adding 1h/1d hierarchies). Every request then becomes an index read, with no backend change. API connections carry a 15 s `statement_timeout`; a timeout returns 504 "narrow the range".

## Time conventions

- Every instant in the database and the API is UTC (`…Z`). Periods are identified by `(settlement_date, settlement_period)`. All conversions go through `settlement.py`, which has tests for the 46/48/50-period days, including **25 Oct 2026, the hackathon weekend**.
- API `from`/`to` are inclusive settlement dates and become `[London midnight of from, London midnight of to+1)`.
- **Cutoff mode** (`cutoff=` on `/api/series` and `gridpulse analyse`) uses only forecast versions with `published_at_utc <= cutoff`, and, as everywhere, published before the period starts. Prices and outturns are included only for periods that had **ended** by the cutoff. The publication delay of actuals is not modelled; for example, NESO demand arrives about 21 days late. This is the honest "what would we have known" view for the HRT predictions track. Without a cutoff, a view is retrospective and should be labelled as such.

## API

The frontend reads every endpoint below; all are `GET`. Examples of each response are in `gridpulse/contracts/examples/`, and a test keeps them aligned with `api/schemas.py`. Interactive docs are at `/docs` when the server runs.

| Endpoint | Purpose | Terminal panel |
|---|---|---|
| `/api/health` | Mode (`live`/`record`/`replay`), database status. Always answers | Status bar |
| `/api/catalog` | Series (id, label, unit), raw dataset coverage from `raw.dataset_catalog`, algorithms and their latest retrospective run, limits | Boot; command-bar autocomplete |
| `/api/snapshot?at=` | Latest settled period: each ticker series' value, previous value and change. `at` replays history | Ticker strip (poll every 60 s) |
| `/api/series?ids=price,wind&from=&to=&resolution=auto&cutoff=` | Several series on one shared time axis | Main chart, linked crosshair |
| `/api/analysis?algorithm=&from=&to=&resolution=auto&run_id=` | Stored per-period results of one algorithm, with run provenance, warnings and the names of its tables. Defaults to the latest retrospective run; cutoff runs are fetched by `run_id` | Indicator pane under the chart |
| `/api/analysis/runs/{run_id}/tables?name=` | A run's tables, each `{description, columns, rows}` | Scorecard / diagnostics panel |
| `/api/extremes?series=price&from=&to=&order=desc&limit=10` | Highest/lowest periods: a screener for finding events | "Top movers" table |
| `/api/periods/{date}/{period}` | All values for one period, the forecast **vintage history**, and every algorithm output | Click-through inspector |

The series ids are `price`, `price_min`, `price_max`, `niv`, `wind`, `wind_outturn`, `demand`, `demand_outturn` and `residual_demand`.

**Response format.** Series are columnar: one `t` array of UTC bucket starts and one `values` array per series of the same length, with `null` for missing. This maps straight onto uPlot (`[t.map(Date.parse), ...values]`), a good fit for a dense terminal UI. At 30m the response also carries `settlement_date[]` and `settlement_period[]` for labels and drill-down.

**Errors** are always `{"detail": {"code": "...", "message": "..."}}` with a meaningful status: 400 for bad input, 404 for nothing there (e.g. `no_run`, whose message includes the command to run), 503 when the database is down and 504 on a query timeout. FastAPI's own parameter validation returns 422 with a list in `detail`.

**Frontend notes for the Svelte terminal.** Load `/api/health` and `/api/catalog` on boot and build panels and commands from the catalog rather than hard-coding ids. Fetch charts with `resolution=auto` and refetch when the user zooms. Show `mode`, `data_updated_at_utc`, run warnings and algorithm `limitations` on screen. Use `/api/snapshot?at=` plus `/api/series?cutoff=` for a "replay the day" mode.

## Algorithm contract (Matthew)

Defined in `analysis/contract.py`. There are two complete examples in `analysis/algorithms/`: `example_correlation.py` (series only) and `example_backtest.py` (series and tables).

```python
def run(frame: pd.DataFrame, *, window_periods: int) -> pd.DataFrame | AlgorithmResult:
    ...  # pure pandas/numpy: no database, HTTP or files
    return AlgorithmResult(
        series=per_period,                        # index delivery_start_utc, columns = declared outputs
        tables={"metrics": scorecard},            # optional; any small DataFrame
    )                                             # or return just the per-period DataFrame

ALGORITHM = Algorithm(
    id="my_signal", version="0.1.0", description="One line for the UI",
    run=run,
    inputs=("wind_mw", "price_gbp_mwh"),          # chosen from INPUT_COLUMNS
    outputs={"my_signal": "GBP/MWh"},             # name -> unit ("" if unitless); may be {} if tables only
    tables={"metrics": "MAE/RMSE per model"},     # name -> description; may be omitted
    params={"window_periods": 96},                # defaults; CLI can override
    warmup=timedelta(days=2),                     # history needed before the range
    horizon=timedelta(0),                         # >0 if it predicts past the range
    limitations="What these numbers do NOT mean",
)
```

**Which output kind to use.** If a result has one value per settlement period, it's a **series**: features (EWMA, rolling std, differences, Fourier terms), forecasts, interval bands and scores. Everything else is a **table**: ACF/PACF and CCF by lag, accuracy metrics, interval coverage, fitted parameters/AIC and residual tests. A table is any DataFrame of up to 20,000 rows. A meaningful index (e.g. `lag`) is kept as a column, datetimes must be tz-aware and NaN/inf become `null`. Tables are stored as returned, so compute them over the requested range only, i.e. rows from `frame.index[0] + warmup` onwards.

- **Input:** indexed by `delivery_start_utc` (UTC), one row per 30-minute period, ascending, with no gaps (missing data is NaN). Only the declared `inputs` are passed. Available columns are the settlement date/period, `price_gbp_mwh`, `niv_mwh`, `wind_mw`, `wind_outturn_mw`, `demand_mw`, `demand_outturn_mw` and `residual_demand_mw`. Adding a column means adding it to a `gridpulse` view and to `INPUT_COLUMNS`.
- **Output:** a DataFrame with a tz-aware index and exactly the declared output columns. NaN is stored as NULL. Rows outside `[from, to + horizon)` are dropped. A contract violation fails the run with a clear error, and the run's status is set to `failed`.
- **To add one:** create `analysis/algorithms/<name>.py`, add it to the list in `analysis/algorithms/__init__.py`, then run `pytest`. The contract tests automatically check every registered algorithm against synthetic data, including gaps, input mutation and determinism. Run `gridpulse analyse <id> --from … --to …` on real data.
- **Versioning:** bump `version` whenever results change. The API serves only runs of the current version, so stale results never mix with new code.
- **Orchid templates** go here as adapters. Wrap their API inside `run()` and keep their files and licence notice in this folder.
- **No training in the MVP.** Statistical methods that estimate parameters inside `run()` from the data they are given (rolling, AR or state-space fits) are fine. Say so in `limitations`, and use `cutoff` runs for anything presented as a prediction.

## Rolling-origin backtests: the fixed-horizon pattern

A backtest makes forecasts from many origins, so one target period receives one forecast per origin. Rather than storing every (origin, target) pair as a series, **fix the horizon**:

1. **Choose the horizons** that matter, in settlement periods, e.g. `h = 2` (one hour ahead) and `h = 48` (day-ahead).
2. **For each target period `t`**, the forecast comes from origin `o = t − h` and may use only data available at `o`. Inside `run()`, that means `frame.loc[:o]`, or a vectorised equivalent such as `shift(h + 1)` for a settled price. Fit any parameters on that history only (Orchid's `RollingOriginSplitter`/`backtest` produce exactly these folds).
3. **Series outputs:** one column per model and horizon, e.g. `arima_forecast_h48` (and `_lower`/`_upper` for intervals), indexed by the **target** period. It shares the time axis with `price`, so the frontend overlays forecast and actual directly, and `/api/analysis` downsamples it like any series.
4. **Tables:** the scorecard (`model`, `horizon_periods`, `n`, `mae`, `rmse`, `mase`, coverage, …) and optionally fold detail (`origin_utc`, `target_utc`, `horizon`, `forecast`, `actual`) if it fits in 20,000 rows. Otherwise aggregate it, e.g. errors by settlement period or by month.
5. **Declare `warmup`** as the minimum training history; the first target is `from` itself. Use `horizon` only when forecasting beyond `to`.

Mark backtests that use **forecast inputs** (wind/demand) as provisional. The input frame carries the latest forecast for each period, not the version published by each origin, so such backtests see a later forecast than was available. Publication-time inputs per origin are part of the data-modelling decision (open item 1). Backtests on settled prices alone, like `example_backtest.py`, are unaffected.

For a single live-style forecast from one origin, run with `--cutoff <origin>` and a `horizon`. Prices and outturns are then limited to periods ended by the cutoff, and forecasts to versions published by then.

## Running and serving analyses

`gridpulse analyse` creates an `analysis_runs` row (`running`), builds the input frame for `[from − warmup, to)`, calls `run()`, checks the series and tables, then writes both with status `succeeded` **in one transaction**. A failure stores the error and marks the run `failed`. A 3.5-month run of the reference algorithm takes about 2 s. `/api/analysis` serves the newest succeeded **retrospective** run of the current version that overlaps the request, preferring one that covers the whole range. Partial coverage, cutoffs and version mismatches appear in `warnings`. Results are averaged into 1h/1d buckets in SQL to respect the point budget.

There is deliberately no "run" HTTP endpoint for the MVP. If the demo needs a button, add a `POST` guarded by a server-side token that calls `run_algorithm` on a small range.

## Offline demo fallback

`GRIDPULSE_MODE=record` serves normally and saves every successful `GET /api/*` body under `gridpulse/data/demo/responses/`. `GRIDPULSE_MODE=replay` serves only those files, needs no database, and sets `X-GridPulse-Mode: replay`; `/api/health` reports `mode: "replay"`. Click through the demo path once in record mode the evening before. The UI must show a "cached" badge whenever `mode` is not `live`.

## Gemini agent (after the MVP)

`POST /api/agent/ask {question, from, to}` returns `{answer, evidence[]}`. It lives in `agent/service.py` with tools in `agent/tools.py`. Each tool wraps an existing query or service, with no new calculations:

| Tool | Wraps | Bound |
|---|---|---|
| `get_catalog()` | `/api/catalog` logic | — |
| `get_series(ids, from, to)` | `queries.series_values` | ≤ 200 points plus min/mean/max summary |
| `find_extremes(series, from, to, order)` | `queries.extremes` | ≤ 20 rows |
| `inspect_period(date, period)` | period detail | 1 period |
| `get_analysis(algorithm, from, to)` | `service.get_results` | ≤ 200 points |

Guardrails: at most 6 tool calls and a 30 s total timeout. Tool output is passed as data, never as instructions. Every claim must cite periods from `evidence`, and correlation must not be narrated as causation. The dashboard must work with `GEMINI_API_KEY` unset.

## Deployment (Vultr)

Use one VM in London with Tiger Cloud in the closest region. On the VM:

- Caddy for HTTPS. It serves the Svelte static build and proxies `/api/*` to `127.0.0.1:8000`, so the frontend and API share an origin and CORS isn't needed.
- systemd running `uvicorn gridpulse.main:app --host 127.0.0.1 --port 8000 --workers 2` from `gridpulse/backend` (editable install), with `gridpulse/.env` holding `DATABASE_URL` and the keys.
- Health check: `curl https://<host>/api/health`.
- Updates: `git pull && .venv/bin/pip install -e . && gridpulse migrate && systemctl restart gridpulse`.

## Open items

1. **Data modelling** (team decision). Choose which views become continuous aggregates. Decide whether to use TSDF, embedded wind and Open-Meteo, and how to treat hourly WINDFOR. Then decide how algorithms get **publication-time inputs**: forecast versions as published by each origin (P0 layer 2), and a revision definition such as "latest − day-ahead version" (P0 items 3–5). Until then, backtests with forecast inputs are provisional.
2. **Price revisions.** The loader stores prices "as published at fetch time" (see its catalog note), so later settlement runs are not distinguished.
3. **The algorithm list** and each algorithm's inputs, outputs and limitations, from Matthew.
