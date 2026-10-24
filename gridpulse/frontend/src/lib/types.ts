/**
 * Response types of the GridPulse API, mirroring gridpulse/backend/src/gridpulse/api/schemas.py.
 * Field meanings, query parameters and errors: docs/gridpulse/FRONTEND_API.md.
 * Update this file together with schemas.py and gridpulse/contracts/examples/.
 *
 * Conventions: instants are UTC ISO strings ending in "Z"; dates are "YYYY-MM-DD" settlement dates;
 * every `values` array is aligned with the response's `t` array, and null means missing (never zero).
 */

export type Resolution = "30m" | "1h" | "1d";
export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };
export type Cell = string | number | boolean | null;

/** Body of every non-2xx response except FastAPI's own 422 validation errors. */
export interface ApiError {
  detail: { code: string; message: string };
}

export interface Health {
  status: "ok" | "degraded";
  mode: "live" | "record" | "replay";
  database: "ok" | "unavailable" | "not_configured" | "not_used";
  version: string;
}

export interface SeriesInfo {
  id: string;
  label: string;
  unit: string;
  /** settlement: settled market values; forecast: latest version before delivery; outturn: measured. */
  kind: "settlement" | "forecast" | "outturn";
  /** Raw datasets it is built from; look each up in Catalog.datasets by `dataset`. */
  datasets: string[];
}

export interface DatasetCoverage {
  dataset: string;
  source: string | null;
  notes: string | null;
  first_utc: string | null;
  last_utc: string | null;
  loaded_at_utc: string | null;
  approx_rows: number | null;
}

export interface RunInfo {
  run_id: number;
  algorithm_id: string;
  algorithm_version: string;
  /** Every parameter the run used. */
  params: Record<string, JsonValue>;
  /** The params that differ from the algorithm's current defaults; {} for a default run. */
  overrides: Record<string, JsonValue>;
  /** Raw dataset read -> its load time when the run started; null if not recorded (older runs). */
  datasets: Record<string, string | null>;
  range_start_utc: string;
  /** Includes the algorithm's forecast horizon, so it can be later than the requested `to`. */
  range_end_utc: string;
  /** Non-null for a "what was known at this instant" run. */
  cutoff_utc: string | null;
  finished_at_utc: string | null;
}

export interface AlgorithmInfo {
  id: string;
  version: string;
  description: string;
  inputs: string[];
  /** Raw datasets behind the inputs; look each up in Catalog.datasets. Empty for clock-only inputs. */
  datasets: string[];
  /** Output (indicator) name -> unit; "" means unitless. Empty for tables-only algorithms. */
  outputs: Record<string, string>;
  /** Table name -> one-line description. */
  tables: Record<string, string>;
  /** Default parameters. */
  params: Record<string, JsonValue>;
  /** Parameter name -> meaning; same keys as `params`. */
  param_descriptions: Record<string, string>;
  /** Must be shown next to the results. */
  limitations: string;
  /** The run /api/analysis serves by default (current version, default params, no cutoff). */
  latest_run: RunInfo | null;
}

export interface Catalog {
  series: SeriesInfo[];
  datasets: DatasetCoverage[];
  algorithms: AlgorithmInfo[];
  limits: { max_points: number; max_range_days: number };
  data_updated_at_utc: string | null;
}

export interface Column {
  label: string;
  unit: string;
  values: (number | null)[];
}

export interface SeriesResponse {
  from: string;
  to: string;
  resolution: Resolution;
  cutoff_utc: string | null;
  t: string[];
  /** Only at 30m resolution. */
  settlement_date: string[] | null;
  settlement_period: number[] | null;
  series: Record<string, Column>;
}

export interface TickerItem {
  id: string;
  label: string;
  unit: string;
  value: number | null;
  previous: number | null;
  change: number | null;
}

export interface Snapshot {
  as_of_utc: string;
  settlement_date: string;
  settlement_period: number;
  items: TickerItem[];
  data_updated_at_utc: string | null;
}

export interface Extreme {
  delivery_start_utc: string;
  settlement_date: string;
  settlement_period: number;
  value: number;
}

export interface ExtremesResponse {
  series: SeriesInfo;
  from: string;
  to: string;
  order: "desc" | "asc";
  rows: Extreme[];
}

export interface Vintage {
  published_at_utc: string;
  value: number | null;
}

export interface PeriodIndicator {
  algorithm_id: string;
  algorithm_version: string;
  run_id: number;
  indicator: string;
  unit: string | null;
  value: number | null;
}

export interface PeriodDetail {
  settlement_date: string;
  settlement_period: number;
  delivery_start_utc: string;
  delivery_end_utc: string;
  /** Series id -> value; labels and units are in the catalog. */
  values: Record<string, number | null>;
  /** Forecast id ("wind", "demand") -> every published version, oldest first. */
  vintages: Record<string, Vintage[]>;
  analysis: PeriodIndicator[];
}

export interface AnalysisResponse {
  algorithm: AlgorithmInfo;
  run: RunInfo;
  from: string;
  to: string;
  resolution: Resolution;
  t: string[];
  /** Output name -> values aligned with `t`. Empty for tables-only algorithms. */
  indicators: Record<string, Column>;
  /** Names of this run's tables; fetch them from /api/analysis/runs/{run_id}/tables. */
  tables: string[];
  /** Must be shown when non-empty. */
  warnings: string[];
}

export interface RunList {
  algorithm: AlgorithmInfo;
  /** Newest first: every version, parameter set and cutoff. */
  runs: RunInfo[];
}

export interface Table {
  description: string;
  columns: string[];
  /** Each row has one cell per column; datetimes are UTC ISO strings. */
  rows: Cell[][];
}

export interface RunTables {
  run: RunInfo;
  tables: Record<string, Table>;
}
