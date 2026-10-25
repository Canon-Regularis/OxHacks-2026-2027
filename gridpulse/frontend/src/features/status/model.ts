import type {
  AlgorithmInfo,
  Catalog,
  DatasetCoverage,
  Health,
} from '$lib/types';

export type Level = 'ok' | 'warn' | 'bad' | 'unknown';

/** A dataset loaded within this long is current; within STALE_HOURS it is ageing. Heuristic, not a source SLA. */
export const AGEING_HOURS = 6;
export const STALE_HOURS = 24;

export function ageHours(iso: string | null, now: number): number | null {
  if (!iso) return null;
  const t = Date.parse(iso);
  return Number.isFinite(t) ? Math.max(0, (now - t) / 3_600_000) : null;
}

export function ageLabel(hours: number | null): string {
  if (hours === null) return 'unknown';
  if (hours < 1 / 60) return 'just now';
  if (hours < 1) return `${Math.round(hours * 60)} min ago`;
  if (hours < 48) return `${Math.round(hours)} h ago`;
  return `${Math.round(hours / 24)} d ago`;
}

export function freshness(hours: number | null): Level {
  if (hours === null) return 'unknown';
  if (hours <= AGEING_HOURS) return 'ok';
  if (hours <= STALE_HOURS) return 'warn';
  return 'bad';
}

export interface DatasetRow {
  dataset: string;
  source: string | null;
  firstUtc: string | null;
  lastUtc: string | null;
  loadedUtc: string | null;
  rows: number | null;
  loadedAgeHours: number | null;
  level: Level;
  note: string;
}

export function datasetRow(d: DatasetCoverage, now: number): DatasetRow {
  const age = ageHours(d.loaded_at_utc, now);
  let level = freshness(age);
  let note = age === null ? 'Load time not recorded' : '';
  if (d.approx_rows === 0 || (!d.first_utc && !d.last_utc)) {
    level = 'bad';
    note = 'No data loaded';
  }
  return {
    dataset: d.dataset,
    source: d.source,
    firstUtc: d.first_utc,
    lastUtc: d.last_utc,
    loadedUtc: d.loaded_at_utc,
    rows: d.approx_rows,
    loadedAgeHours: age,
    level,
    note,
  };
}

export interface AlgorithmRow {
  id: string;
  version: string;
  finishedUtc: string | null;
  ageHours: number | null;
  level: Level;
  note: string;
}

/**
 * An algorithm with no served run is "bad"; one whose inputs were reloaded after it ran is "warn"
 * (its stored result predates the data now in the database).
 */
export function algorithmRow(
  a: AlgorithmInfo,
  datasets: DatasetCoverage[],
  now: number,
): AlgorithmRow {
  const run = a.latest_run;
  if (!run)
    return {
      id: a.id,
      version: a.version,
      finishedUtc: null,
      ageHours: null,
      level: 'bad',
      note: 'This algorithm has not stored a result yet',
    };
  const loaded = new Map(datasets.map((d) => [d.dataset, d.loaded_at_utc]));
  const reloaded = Object.entries(run.datasets)
    .filter(([name, at]) => {
      const current = loaded.get(name);
      return at && current && Date.parse(current) > Date.parse(at);
    })
    .map(([name]) => name);
  return {
    id: a.id,
    version: a.version,
    finishedUtc: run.finished_at_utc,
    ageHours: ageHours(run.finished_at_utc, now),
    level: reloaded.length ? 'warn' : 'ok',
    note: reloaded.length
      ? `Calculated before ${reloaded.join(', ')} was reloaded`
      : '',
  };
}

const RANK: Record<Level, number> = { ok: 0, unknown: 1, warn: 2, bad: 3 };
export function worst(levels: Level[]): Level {
  return levels.reduce<Level>((a, b) => (RANK[b] > RANK[a] ? b : a), 'ok');
}

export interface Summary {
  level: Level;
  headline: string;
  api: Level;
  database: Level;
  data: Level;
  algorithms: Level;
  dataUpdatedHours: number | null;
  algorithmsServed: number;
  algorithmsTotal: number;
  datasetsCurrent: number;
  datasetsTotal: number;
}

/** health null = API unreachable; catalog null = unavailable (usually because the database is). */
export function summarise(
  health: Health | null,
  catalog: Catalog | null,
  now: number,
): Summary {
  const api: Level = health ? 'ok' : 'bad';
  const database: Level = !health
    ? 'unknown'
    : health.database === 'ok' || health.database === 'not_used'
      ? 'ok'
      : 'bad';
  const datasets = (catalog?.datasets ?? []).map((d) => datasetRow(d, now));
  const algos = (catalog?.algorithms ?? []).map((a) =>
    algorithmRow(a, catalog?.datasets ?? [], now),
  );
  const dataUpdatedHours = ageHours(catalog?.data_updated_at_utc ?? null, now);
  const data: Level = catalog
    ? worst([freshness(dataUpdatedHours), ...datasets.map((d) => d.level)])
    : 'unknown';
  const algorithms: Level = catalog
    ? worst(algos.map((a) => a.level))
    : 'unknown';
  const level = worst([api, database, data, algorithms]);
  const headline = !health
    ? 'API unreachable'
    : database === 'bad'
      ? 'Database unavailable'
      : level === 'ok'
        ? 'All systems operational'
        : level === 'unknown'
          ? 'Some checks unavailable'
          : 'Operational, with warnings';
  return {
    level,
    headline,
    api,
    database,
    data,
    algorithms,
    dataUpdatedHours,
    algorithmsServed: algos.filter((a) => a.finishedUtc !== null).length,
    algorithmsTotal: algos.length,
    datasetsCurrent: datasets.filter((d) => d.level === 'ok').length,
    datasetsTotal: datasets.length,
  };
}
