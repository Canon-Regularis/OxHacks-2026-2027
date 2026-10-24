import { describe, it, expect } from 'vitest';
import {
  dayIndices,
  previousDay,
  priceColor,
  METRICS,
  formatValue,
  disabledReason,
  normalizeSelection,
  dailySummary,
  getMetric,
  withWeather,
  seriesColor,
} from './model';
import { readPreference, writePreference } from '$lib/state/view.svelte';
import type { MarketSeries } from '$lib/api';
describe('settlement selection and metric presentation', () => {
  it.each([46, 48, 50])(
    'preserves every period on a %i-period day',
    (count) => {
      const data: MarketSeries = {
        from: '2026-10-25',
        to: '2026-10-25',
        resolution: '30m',
        t: Array.from({ length: count }, (_, i) => String(i)),
        settlement_date: Array(count).fill('2026-10-25'),
        settlement_period: Array.from({ length: count }, (_, i) => i + 1),
        series: {},
      };
      expect(dayIndices(data, '2026-10-25')).toHaveLength(count);
    },
  );
  it('preserves null, zero and negative prices as different values', () => {
    expect(formatValue(null, METRICS[0])).toBe('No data');
    expect(formatValue(0, METRICS[0])).toBe('0.00');
    expect(formatValue(-10, METRICS[0])).toBe('-10.00');
    expect(priceColor(null, 0, 0)).toBe('var(--muted)');
    expect(priceColor(0, 0, 0)).not.toContain('NaN');
  });
  it('uses date arithmetic across year and DST changes', () => {
    expect(previousDay('2026-01-01')).toBe('2025-12-31');
    expect(previousDay('2026-10-26')).toBe('2026-10-25');
  });
});
describe('shared persistent view preference', () => {
  it('roundtrips the selected date and location without storing market values', () => {
    let value = '';
    const storage = {
      getItem: () => value,
      setItem: (_key: string, v: string) => (value = v),
    };
    writePreference(storage, {
      version: 1,
      date: '2026-10-23',
      location: 'dogger_bank',
      metrics: ['price', 'wind', 'demand'],
      scanMetric: 'price',
    });
    expect(readPreference(storage)).toEqual({
      version: 1,
      date: '2026-10-23',
      location: 'dogger_bank',
      metrics: ['price', 'wind', 'demand'],
      scanMetric: 'price',
    });
  });
  it('survives disabled storage and corrupt previous state', () => {
    expect(readPreference({ getItem: () => '{bad' }).date).toBeNull();
    expect(() =>
      writePreference(
        {
          setItem: () => {
            throw new Error('disabled');
          },
        },
        {
          version: 1,
          date: null,
          location: null,
          metrics: ['price'],
          scanMetric: 'price',
        },
      ),
    ).not.toThrow();
  });
});

describe('series selection constraints', () => {
  it('blocks a third unit family but allows five compatible properties', () => {
    expect(disabledReason('niv', ['price', 'wind'])).toContain('third unit');
    expect(
      disabledReason('wind_outturn', ['price', 'wind', 'demand']),
    ).toBeNull();
    expect(disabledReason('niv', ['price'])).toBeNull();
    expect(
      disabledReason('niv', [
        'price',
        'wind',
        'demand',
        'wind_outturn',
        'demand_outturn',
      ]),
    ).toContain('five');
    expect(disabledReason('price', ['price'])).toContain('at least one');
    expect(disabledReason('price', ['price', 'wind'])).toBeNull();
  });
  it('repairs old and invalid saved selections without losing valid date/location', () => {
    const old = readPreference({
      getItem: () =>
        JSON.stringify({
          version: 1,
          date: '2026-10-23',
          location: 'dogger_bank',
        }),
    });
    expect(old.metrics).toEqual(['price', 'wind', 'demand']);
    expect(old.date).toBe('2026-10-23');
    const restored = readPreference({
      getItem: () =>
        JSON.stringify({
          version: 1,
          date: null,
          location: null,
          metrics: ['temperature', 'pressure', 'price', 'bad', 'temperature'],
          scanMetric: 'price',
        }),
    });
    expect(restored.metrics).toEqual(['price']);
    expect(restored.scanMetric).toBe('price');
    expect(normalizeSelection([])).toEqual(['price', 'wind', 'demand']);
  });
  it('excludes missing values from daily averages without dropping real zeroes', () => {
    const data: MarketSeries = {
      from: '2026-10-23',
      to: '2026-10-23',
      resolution: '30m',
      t: [],
      settlement_date: [],
      settlement_period: [],
      series: {
        price: { label: 'Price', unit: 'GBP/MWh', values: [null, 0, -10, 20] },
      },
    };
    expect(dailySummary(data, [0, 1, 2, 3], 'price')).toEqual({
      average: 10 / 3,
      count: 3,
      total: 4,
    });
    expect(dailySummary(data, [0], 'price').average).toBeNull();
  });
  it('uses different property hues with an ordered lightness scale', () => {
    expect(seriesColor(null, 0, 10, getMetric('wind'))).toBe('var(--muted)');
    expect(seriesColor(0, 0, 10, getMetric('wind'))).not.toBe(
      seriesColor(10, 0, 10, getMetric('wind')),
    );
    expect(seriesColor(5, 0, 10, getMetric('wind'))).not.toBe(
      seriesColor(5, 0, 10, getMetric('price')),
    );
  });
});
