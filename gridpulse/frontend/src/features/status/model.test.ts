import { describe, expect, it } from 'vitest';
import type { AlgorithmInfo, Catalog, Health } from '$lib/types';
import {
  ageLabel,
  algorithmRow,
  datasetRow,
  freshness,
  summarise,
} from './model';

const NOW = Date.parse('2026-10-25T12:00:00Z');
const hoursAgo = (h: number) => new Date(NOW - h * 3_600_000).toISOString();
const health: Health = {
  status: 'ok',
  mode: 'live',
  database: 'ok',
  version: '0.1.0',
};
const dataset = (loaded: string | null, rows: number | null = 10) => ({
  dataset: 'elexon_ndf',
  source: 'Elexon',
  notes: null,
  first_utc: '2026-10-01T00:00:00Z',
  last_utc: '2026-10-26T00:00:00Z',
  loaded_at_utc: loaded,
  approx_rows: rows,
});
const algo = (
  runDatasets: Record<string, string | null> | null,
): AlgorithmInfo =>
  ({
    id: 'a',
    version: '1',
    latest_run: runDatasets
      ? { datasets: runDatasets, finished_at_utc: hoursAgo(1) }
      : null,
  }) as unknown as AlgorithmInfo;
const catalog = (
  loaded: string | null,
  algorithms: AlgorithmInfo[],
): Catalog => ({
  series: [],
  datasets: [dataset(loaded)],
  algorithms,
  limits: { max_points: 1, max_range_days: 1 },
  data_updated_at_utc: loaded,
});

describe('status model', () => {
  it('labels ages and freshness thresholds', () => {
    expect(ageLabel(null)).toBe('unknown');
    expect(ageLabel(0.5)).toBe('30 min ago');
    expect(ageLabel(72)).toBe('3 d ago');
    expect(freshness(1)).toBe('ok');
    expect(freshness(10)).toBe('warn');
    expect(freshness(30)).toBe('bad');
    expect(freshness(null)).toBe('unknown');
  });
  it('flags empty datasets', () => {
    expect(datasetRow(dataset(hoursAgo(1), 0), NOW).level).toBe('bad');
    expect(datasetRow(dataset(null), NOW).note).toMatch(/not recorded/);
  });
  it('warns when inputs were reloaded after the run', () => {
    const ds = [dataset(hoursAgo(1))];
    expect(algorithmRow(algo({ elexon_ndf: hoursAgo(5) }), ds, NOW).level).toBe(
      'warn',
    );
    expect(
      algorithmRow(algo({ elexon_ndf: hoursAgo(0.5) }), ds, NOW).level,
    ).toBe('ok');
    expect(algorithmRow(algo(null), ds, NOW).level).toBe('bad');
  });
  it('summarises healthy, degraded and unreachable states', () => {
    const ok = summarise(
      health,
      catalog(hoursAgo(1), [algo({ elexon_ndf: hoursAgo(1) })]),
      NOW,
    );
    expect(ok.level).toBe('ok');
    expect(ok.headline).toBe('All systems operational');
    const stale = summarise(health, catalog(hoursAgo(40), []), NOW);
    expect(stale.data).toBe('bad');
    const down = summarise(null, null, NOW);
    expect(down.level).toBe('bad');
    expect(down.headline).toBe('API unreachable');
    const noDb = summarise(
      { ...health, status: 'degraded', database: 'unavailable' },
      null,
      NOW,
    );
    expect(noDb.headline).toBe('Database unavailable');
  });
});
