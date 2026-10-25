import { expect, it } from 'vitest';
import { csv, seriesCsv, tableCsv } from './export';
import type { WidgetData } from './client';
it('quotes CSV cells, protects formulas and preserves missing and numeric values', () => {
  expect(csv([['a,b', 'say "hi"', 'a\nb', '=1+1', null, 0, -2]])).toBe(
    '"a,b","say ""hi""","a\nb","\'=1+1","","0","-2"\r\n',
  );
});
it('preserves UTC timestamps, units and null gaps', () => {
  const data: WidgetData = {
    t: ['2026-10-25T00:30:00Z', '2026-10-25T01:30:00Z'],
    columns: { price: { label: 'Price', unit: 'GBP/MWh', values: [0, null] } },
    tables: {},
    run: null,
    period: null,
    warnings: [],
    limitations: '',
  };
  expect(seriesCsv(data)).toBe(
    '"timestamp_utc","price (GBP/MWh)"\r\n"2026-10-25T00:30:00Z","0"\r\n"2026-10-25T01:30:00Z",""\r\n',
  );
});
it('exports all table rows including booleans and missing cells', () => {
  expect(
    tableCsv({
      description: 'Test results',
      columns: ['name', 'score'],
      rows: [
        ['baseline', 0],
        ['missing', null],
        ['enabled', true],
      ],
    }),
  ).toContain('"missing",""\r\n"enabled","true"\r\n');
});
