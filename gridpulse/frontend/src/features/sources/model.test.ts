import { describe, expect, it } from 'vitest';
import type { Catalog, DatasetCoverage } from '$lib/types';
import { sourceRows, coverageExtent } from './model';
const dataset = (name: string): DatasetCoverage => ({
  dataset: name,
  source: 'Fixture source',
  notes: null,
  first_utc: null,
  last_utc: null,
  loaded_at_utc: null,
  approx_rows: 0,
});
const catalog = (): Catalog => ({
  datasets: [dataset('elexon_ndf')],
  series: [
    {
      id: 'demand',
      label: 'Demand forecast',
      unit: 'MW',
      kind: 'forecast',
      datasets: ['elexon_ndf'],
    },
    {
      id: 'residual',
      label: 'Residual demand',
      unit: 'MW',
      kind: 'forecast',
      datasets: ['elexon_ndf'],
    },
  ],
  algorithms: [],
  limits: { max_points: 5000, max_range_days: 3660 },
  data_updated_at_utc: null,
});

describe('Data Sources catalogue model', () => {
  it('joins chart series by dataset, deduplicates units and preserves zero record counts', () => {
    const input = catalog();
    const before = JSON.stringify(input);
    const [row] = sourceRows(input);
    expect(row.units).toEqual(['MW']);
    expect(row.series.map((s) => s.id)).toEqual(['demand', 'residual']);
    expect(row.kind).toBe('Forecast');
    expect(row.dataset.approx_rows).toBe(0);
    expect(JSON.stringify(input)).toBe(before);
  });
  it('retains stored datasets that have no chart series without inventing units or frequency', () => {
    const input = catalog();
    input.datasets.push(dataset('future_dataset'));
    const row = sourceRows(input).find(
      (row) => row.dataset.dataset === 'future_dataset',
    )!;
    expect(row.series).toEqual([]);
    expect(row.units).toEqual([]);
    expect(row.kind).toBe('Other');
    expect(row.cadence).toBe('Not supplied');
    expect(sourceRows(null)).toEqual([]);
  });
  it('uses catalogue kinds for new dataset IDs and keeps unitless series explicit', () => {
    const input = catalog();
    input.datasets = [dataset('new_source')];
    input.series = [
      {
        id: 'index',
        label: 'Index',
        kind: 'settlement',
        unit: '',
        datasets: ['new_source'],
      },
    ];
    expect(sourceRows(input)[0]).toMatchObject({
      kind: 'Settlement',
      units: ['Unitless'],
    });
  });
  it('calculates record bounds by actual instants, ignoring absent or invalid metadata', () => {
    const records = [dataset('one'), dataset('two'), dataset('empty')];
    records[0].first_utc = '2026-10-25T01:30:00+01:00';
    records[0].last_utc = '2026-10-25T01:30:00Z';
    records[1].first_utc = '2026-10-25T01:00:00Z';
    records[1].last_utc = 'not-a-date';
    expect(coverageExtent(records)).toEqual({
      first: records[0].first_utc,
      last: records[0].last_utc,
    });
    expect(coverageExtent([])).toEqual({ first: null, last: null });
  });
});
