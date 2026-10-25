import { expect, it } from 'vitest';
import { sourceTable } from './sourceTable';
it('renders aligned values with units and null gaps, bounded by linked zoom', () => {
  const t = ['2026-02-01T00:00:00Z', '2026-02-01T00:30:00Z'];
  const columns = {
    wind: { label: 'Wind', unit: 'MW', values: [100, null] },
    demand: { label: 'Demand', unit: 'MW', values: [200, 220] },
  };
  expect(sourceTable(t, columns, null).rows).toEqual([
    [t[0], 100, 200],
    [t[1], null, 220],
  ]);
  expect(sourceTable(t, columns, null).columns).toEqual([
    'delivery_start_utc',
    'Wind (MW)',
    'Demand (MW)',
  ]);
  expect(
    sourceTable(t, columns, [Date.parse(t[1]), Date.parse(t[1])]).rows,
  ).toEqual([[t[1], null, 220]]);
});
