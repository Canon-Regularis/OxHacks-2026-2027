import { describe, expect, it } from 'vitest';
import {
  defaultSettings,
  normalizeSettings,
  readSettings,
  formatTime,
  formatDateTime,
} from './settings';
import {
  readPreference,
  initialViewPreference,
  writePreference,
} from './view.svelte';

describe('saved preferences', () => {
  it.each([
    null,
    undefined,
    [],
    'dark',
    3,
    {},
    { version: 2, appearance: 'light' },
  ])('uses defaults for unsupported data %j', (value) => {
    expect(normalizeSettings(value)).toEqual(defaultSettings());
  });
  it('validates fields independently and discards unknown fields', () => {
    expect(
      normalizeSettings({
        version: 1,
        appearance: 'light',
        timeDisplay: 'Mars',
        motion: 'reduce',
        gridlines: 'false',
        lineThickness: 'bold',
        rememberView: false,
        secret: 'discard',
      }),
    ).toEqual({
      ...defaultSettings(),
      appearance: 'light',
      motion: 'reduce',
      lineThickness: 'bold',
      rememberView: false,
    });
  });
  it.each(['dark', 'light', 'system'] as const)(
    'accepts appearance %s',
    (appearance) => {
      expect(
        normalizeSettings({ ...defaultSettings(), appearance }).appearance,
      ).toBe(appearance);
    },
  );
  it.each([false, true])('retains explicit checkbox values %s', (value) => {
    const saved = {
      ...defaultSettings(),
      gridlines: value,
      rememberView: value,
    };
    expect(readSettings({ getItem: () => JSON.stringify(saved) })).toEqual(
      saved,
    );
  });
  it.each(['{broken', 'null', '[]', '"dark"'])(
    'survives invalid saved JSON %s',
    (json) => {
      expect(readSettings({ getItem: () => json })).toEqual(defaultSettings());
    },
  );
  it('survives unavailable storage', () => {
    const storage = {
      getItem: () => {
        throw new Error('denied');
      },
    };
    expect(readSettings(storage)).toEqual(defaultSettings());
    expect(readPreference(storage)).toEqual(initialViewPreference());
    expect(() =>
      writePreference(
        {
          setItem: () => {
            throw new Error('quota');
          },
        },
        initialViewPreference(),
      ),
    ).not.toThrow();
  });
  it('normalizes saved view selections and an unavailable scan metric', () => {
    expect(
      readPreference({
        getItem: () =>
          JSON.stringify({
            version: 1,
            date: '2026-10-25',
            metrics: ['wind', 'invalid', 'wind'],
            scanMetric: 'invalid',
          }),
      }),
    ).toMatchObject({
      date: '2026-10-25',
      metrics: ['wind'],
      scanMetric: 'wind',
      location: null,
    });
  });
  it('creates independent defaults', () => {
    const a = initialViewPreference();
    a.metrics.push('niv');
    expect(initialViewPreference().metrics).not.toContain('niv');
  });
});

describe('UK and UTC display', () => {
  it.each([
    ['2026-10-25T00:00:00Z', '01:00 BST'],
    ['2026-10-25T01:00:00Z', '01:00 GMT'],
    ['2026-03-29T00:30:00Z', '00:30 GMT'],
    ['2026-03-29T01:00:00Z', '02:00 BST'],
    ['2026-07-01T23:30:00Z', '00:30 BST'],
  ])('labels %s as %s', (iso, label) =>
    expect(formatTime(iso, 'local')).toBe(label),
  );
  it('keeps UTC stable across the clock change', () => {
    expect(formatTime('2026-10-25T00:00:00Z', 'utc')).toBe('00:00 UTC');
    expect(formatTime('2026-10-25T01:00:00Z', 'utc')).toBe('01:00 UTC');
  });
  it('uses the correct local date at midnight', () => {
    expect(formatDateTime('2026-07-01T23:30:00Z', 'local')).toContain(
      '02 Jul 2026',
    );
    expect(formatDateTime('2026-07-01T23:30:00Z', 'utc')).toContain(
      '01 Jul 2026',
    );
  });
});
