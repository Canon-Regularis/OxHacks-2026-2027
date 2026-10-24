import type {
  Catalog,
  Health,
  AnalysisResponse,
  SeriesResponse,
  RunTables,
  RunList,
  PeriodDetail,
  RunInfo,
  Column,
  Table,
} from '$lib/types';
import type { Binding, LinkContext, Widget } from '../model/types';
export class TerminalError extends Error {
  constructor(
    public code: string,
    message: string,
  ) {
    super(message);
  }
}
export type Lease<T> = { promise: Promise<T>; release: () => void };
export class RequestPool {
  private entries = new Map<
    string,
    {
      promise: Promise<unknown>;
      controller: AbortController;
      users: number;
      done: boolean;
      expires: number;
    }
  >();
  constructor(private transport: typeof fetch = fetch) {}
  acquire<T>(
    path: string,
    query: Record<string, string | number | undefined> = {},
  ): Lease<T> {
    const params = new URLSearchParams();
    for (const key of Object.keys(query).sort())
      if (query[key] !== undefined) params.set(key, String(query[key]));
    const key = `/api${path}${params.size ? '?' + params : ''}`;
    let e = this.entries.get(key);
    if (e && e.done && e.expires < Date.now()) {
      this.entries.delete(key);
      e = undefined;
    }
    if (!e) {
      const controller = new AbortController();
      const entry = {
        controller,
        users: 0,
        done: false,
        expires: 0,
        promise: Promise.resolve(null) as Promise<unknown>,
      };
      entry.promise = this.transport(key, { signal: controller.signal })
        .then(async (response) => {
          const body = await response.json().catch(() => null);
          if (!response.ok) {
            const detail = body?.detail;
            throw new TerminalError(
              detail?.code ?? 'request_failed',
              detail?.message ?? `Request failed (${response.status})`,
            );
          }
          if (!body || typeof body !== 'object')
            throw new TerminalError(
              'invalid_contract',
              'The server returned an invalid response.',
            );
          entry.done = true;
          entry.expires = Date.now() + 30000;
          return body;
        })
        .catch((error) => {
          this.entries.delete(key);
          throw error;
        });
      this.entries.set(key, entry);
      e = entry;
      if (this.entries.size > 80)
        for (const [k, v] of this.entries) {
          if (v.done && v.users === 0) {
            this.entries.delete(k);
            if (this.entries.size <= 60) break;
          }
        }
    }
    e.users++;
    let released = false;
    const current = e;
    return {
      promise: current.promise as Promise<T>,
      release: () => {
        if (released) return;
        released = true;
        current.users--;
        queueMicrotask(() => {
          if (!current.users && !current.done) {
            current.controller.abort();
            if (this.entries.get(key) === current) this.entries.delete(key);
          }
        });
      },
    };
  }
  invalidate() {
    for (const [key, e] of this.entries) if (e.done) this.entries.delete(key);
  }
}
export const pool = new RequestPool();
export function dateOf(t: number | string) {
  return new Intl.DateTimeFormat('en-CA', {
    timeZone: 'Europe/London',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).format(new Date(t));
}
export function queryRange(ctx: LinkContext) {
  return ctx.zoom
    ? { from: dateOf(ctx.zoom[0]), to: dateOf(ctx.zoom[1]) }
    : { from: ctx.from, to: ctx.to };
}
export function validateCatalog(value: Catalog): Catalog {
  if (
    !Array.isArray(value.series) ||
    !Array.isArray(value.algorithms) ||
    !Array.isArray(value.datasets) ||
    !value.limits ||
    !Number.isFinite(value.limits.max_points) ||
    !Number.isFinite(value.limits.max_range_days)
  )
    throw new TerminalError('invalid_contract', 'Invalid catalogue response.');
  const ids = new Set<string>();
  for (const s of value.series) {
    if (
      typeof s.id !== 'string' ||
      ids.has(s.id) ||
      typeof s.unit !== 'string' ||
      typeof s.label !== 'string'
    )
      throw new TerminalError('invalid_contract', 'Invalid catalogue series.');
    ids.add(s.id);
  }
  for (const a of value.algorithms)
    if (
      typeof a.id !== 'string' ||
      !a.outputs ||
      !a.tables ||
      typeof a.limitations !== 'string'
    )
      throw new TerminalError(
        'invalid_contract',
        'Invalid algorithm catalogue.',
      );
  return value;
}
function validateSeries(t: string[], columns: Record<string, Column>) {
  if (
    !Array.isArray(t) ||
    t.length > 5000 ||
    !t.every(
      (v, i) =>
        typeof v === 'string' &&
        Number.isFinite(Date.parse(v)) &&
        (i === 0 || Date.parse(v) > Date.parse(t[i - 1])),
    )
  )
    throw new TerminalError('invalid_contract', 'Invalid chart timestamps.');
  if (!columns || typeof columns !== 'object')
    throw new TerminalError('invalid_contract', 'Missing chart outputs.');
  for (const c of Object.values(columns))
    if (
      !c ||
      typeof c.unit !== 'string' ||
      typeof c.label !== 'string' ||
      !Array.isArray(c.values) ||
      c.values.length !== t.length ||
      !c.values.every(
        (v) => v === null || (typeof v === 'number' && Number.isFinite(v)),
      )
    )
      throw new TerminalError(
        'invalid_contract',
        'Chart values do not match the shared time axis.',
      );
}
function validateTables(tables: Record<string, Table>) {
  if (!tables || typeof tables !== 'object')
    throw new TerminalError('invalid_contract', 'Invalid result tables.');
  for (const t of Object.values(tables))
    if (
      !Array.isArray(t.columns) ||
      !t.columns.every((c) => typeof c === 'string') ||
      !Array.isArray(t.rows) ||
      t.rows.length > 20000 ||
      !t.rows.every(
        (row) =>
          Array.isArray(row) &&
          row.length === t.columns.length &&
          row.every(
            (c) =>
              c === null || ['number', 'string', 'boolean'].includes(typeof c),
          ),
      )
    )
      throw new TerminalError('invalid_contract', 'Invalid result table rows.');
}
export interface WidgetData {
  t: string[];
  columns: Record<string, Column>;
  run: RunInfo | null;
  tables: Record<string, Table>;
  warnings: string[];
  limitations: string;
  period: PeriodDetail | null;
}
export function loadWidget(
  widget: Widget,
  binding: Binding | undefined,
  ctx: LinkContext,
  catalog: Catalog,
  onRun: (id: number) => void,
): { promise: Promise<WidgetData>; release: () => void } {
  const leases: Lease<unknown>[] = [];
  let disposed = false;
  async function get<T>(
    path: string,
    query: Record<string, string | number | undefined> = {},
  ) {
    if (disposed) throw new DOMException('Aborted', 'AbortError');
    const l = pool.acquire<T>(path, query);
    leases.push(l);
    return l.promise;
  }
  const promise = (async () => {
    const result: WidgetData = {
      t: [],
      columns: {},
      run: null,
      tables: {},
      warnings: [],
      limitations: '',
      period: null,
    };
    if (widget.type === 'inspector') {
      const frame = await get<SeriesResponse>('/series', {
        ...queryRange(ctx),
        ids: catalog.series.map((s) => s.id).join(','),
        resolution: 'auto',
      });
      validateSeries(frame.t, frame.series);
      result.t = frame.t;
      result.columns = frame.series;
      if (!ctx.selected) return result;
      // UTC instants, not repeated local clock labels, identify the inspector period.
      const date = dateOf(ctx.selected);
      const grid = await get<SeriesResponse>('/series', {
        ids: catalog.series[0]?.id ?? 'price',
        from: date,
        to: date,
        resolution: '30m',
      });
      const i = grid.t.findIndex(
        (t) => Date.parse(t) === Date.parse(ctx.selected!),
      );
      if (i < 0 || !grid.settlement_period?.[i])
        throw new TerminalError(
          'no_data',
          'Select a half-hour settlement period to inspect.',
        );
      result.period = await get<PeriodDetail>(
        `/periods/${date}/${grid.settlement_period[i]}`,
      );
      return result;
    }
    if (!binding) return result;
    const range = queryRange(ctx);
    if (binding.kind === 'source') {
      if (binding.ids.some((id) => !catalog.series.some((s) => s.id === id)))
        throw new TerminalError(
          'unknown_series',
          'A saved series is no longer in the catalogue. Rebind this widget.',
        );
      const response = await get<SeriesResponse>('/series', {
        ...range,
        ids: binding.ids.join(','),
        resolution: 'auto',
      });
      validateSeries(response.t, response.series);
      result.t = response.t;
      result.columns = response.series;
      if (response.cutoff_utc)
        result.warnings.push(`Available at ${response.cutoff_utc}`);
      return result;
    }
    const algo = catalog.algorithms.find((a) => a.id === binding.algorithm);
    if (!algo)
      throw new TerminalError(
        'unknown_algorithm',
        'This algorithm is no longer in the catalogue. Rebind or remove this widget.',
      );
    const response = await get<AnalysisResponse>('/analysis', {
      ...range,
      algorithm: binding.algorithm,
      run_id: binding.runId ?? undefined,
      resolution: 'auto',
    });
    validateSeries(response.t, response.indicators);
    if (
      !response.run ||
      !Number.isSafeInteger(response.run.run_id) ||
      !Array.isArray(response.warnings)
    )
      throw new TerminalError(
        'invalid_contract',
        'Invalid stored-run response.',
      );
    if (response.run.algorithm_version !== algo.version)
      throw new TerminalError(
        'legacy_schema',
        'This stored run uses an older algorithm version. Its output schema cannot be verified; choose a current run.',
      );
    if (!disposed && binding.runId === null) onRun(response.run.run_id);
    result.t = response.t;
    result.columns = response.indicators;
    result.run = response.run;
    result.warnings = response.warnings;
    result.limitations = response.algorithm.limitations;
    if (widget.type === 'table' || widget.type === 'comparison') {
      const responseTables = await get<RunTables>(
        `/analysis/runs/${response.run.run_id}/tables`,
      );
      validateTables(responseTables.tables);
      if (responseTables.run.run_id !== response.run.run_id)
        throw new TerminalError(
          'invalid_contract',
          'Chart and table run IDs do not match.',
        );
      result.tables = responseTables.tables;
    }
    return result;
  })();
  return {
    promise,
    release: () => {
      disposed = true;
      for (const l of leases) l.release();
    },
  };
}
export async function bootstrap() {
  const leases = [
    pool.acquire<Catalog>('/catalog'),
    pool.acquire<Health>('/health'),
    pool.acquire<{ settlement_date: string }>('/snapshot'),
  ];
  try {
    const [catalog, health, snapshot] = await Promise.all(
      leases.map((l) => l.promise),
    );
    return {
      catalog: validateCatalog(catalog as Catalog),
      health: health as Health,
      date: (snapshot as { settlement_date: string }).settlement_date,
    };
  } finally {
    leases.forEach((l) => l.release());
  }
}
export function runs(algorithm: string) {
  return pool.acquire<RunList>('/analysis/runs', { algorithm, limit: 100 });
}
