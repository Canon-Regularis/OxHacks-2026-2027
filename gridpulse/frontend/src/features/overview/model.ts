import type { MarketSeries, WeatherPoint, WeatherResponse } from '$lib/api';
export type UnitGroup =
  'price' | 'power' | 'energy' | 'temperature' | 'pressure' | 'correlation';
export interface MetricDefinition {
  id: string;
  label: string;
  shortLabel: string;
  color: string;
  displayUnit: string;
  unitGroup: UnitGroup;
  divisor: number;
  digits: number;
  description: string;
  source: 'market' | 'weather';
  section: 'Electricity' | 'Weather' | 'Hourly changes' | 'Correlations';
  hue: number;
}
function metric(
  id: string,
  label: string,
  shortLabel: string,
  color: string,
  displayUnit: string,
  unitGroup: UnitGroup,
  divisor: number,
  digits: number,
  description: string,
  hue: number,
  source: 'market' | 'weather' = 'market',
  section: MetricDefinition['section'] = 'Electricity',
): MetricDefinition {
  return {
    id,
    label,
    shortLabel,
    color,
    displayUnit,
    unitGroup,
    divisor,
    digits,
    description,
    hue,
    source,
    section,
  };
}
// One catalogue defines available choices, cards, chart axes, data requests and scan colours.
export const METRICS: MetricDefinition[] = [
  metric(
    'price',
    'Imbalance price',
    'Price',
    '#e6b080',
    '£/MWh',
    'price',
    1,
    2,
    'The price used to settle contracted versus actual electricity volumes. It is not demand minus wind. Retrospective settlement price.',
    28,
  ),
  metric(
    'wind',
    'Forecast wind',
    'Wind forecast',
    '#a2c7b8',
    'GW',
    'power',
    1000,
    2,
    'Latest wind generation forecast published before delivery; not our own forecast.',
    155,
  ),
  metric(
    'wind_outturn',
    'Actual wind',
    'Wind actual',
    '#6faaa1',
    'GW',
    'power',
    1000,
    2,
    'FUELHH wind generation outturn. Coverage can differ from forecast wind; missing records remain missing.',
    170,
  ),
  metric(
    'demand',
    'Forecast demand',
    'Demand forecast',
    '#b8afd0',
    'GW',
    'power',
    1000,
    2,
    'Published GB national electricity demand forecast.',
    265,
  ),
  metric(
    'demand_outturn',
    'Actual demand',
    'Demand actual',
    '#9481b9',
    'GW',
    'power',
    1000,
    2,
    'NESO national demand outturn. Publication can lag forecasts. This is electricity consumption, not total generation.',
    280,
  ),
  metric(
    'residual_demand',
    'Residual demand',
    'Residual',
    '#9cb8cf',
    'GW',
    'power',
    1000,
    2,
    'Forecast demand minus forecast wind. Not a shortage or system imbalance.',
    207,
  ),
  metric(
    'niv',
    'Net imbalance volume',
    'Imbalance volume',
    '#d3b596',
    'MWh',
    'energy',
    1,
    1,
    'Net system balancing energy volume. MWh measures energy, unlike the GW power scale.',
    35,
  ),
];
export const DEFAULT_SELECTION = ['price', 'wind', 'demand'];
export const MAX_SELECTED = 5;
export function getMetric(id: string): MetricDefinition {
  return METRICS.find((m) => m.id === id) ?? METRICS[0];
}
export function disabledReason(id: string, selection: string[]): string | null {
  if (selection.includes(id))
    return selection.length === 1
      ? 'Keep at least one property selected.'
      : null;
  if (selection.length >= MAX_SELECTED)
    return 'Maximum five properties. Deselect one to add another.';
  const groups = new Set(selection.map((id) => getMetric(id).unitGroup));
  groups.add(getMetric(id).unitGroup);
  return groups.size > 2
    ? `This would add a third unit group (${getMetric(id).displayUnit}). Remove a selected unit group first.`
    : null;
}
export function normalizeSelection(value: unknown): string[] {
  if (!Array.isArray(value)) return [...DEFAULT_SELECTION];
  const result: string[] = [];
  for (const id of value) {
    if (
      typeof id !== 'string' ||
      !METRICS.some((m) => m.id === id) ||
      result.includes(id)
    )
      continue;
    if (!disabledReason(id, result)) result.push(id);
  }
  return result.length ? result : [...DEFAULT_SELECTION];
}
export function formatValue(
  value: number | null | undefined,
  metric: MetricDefinition,
): string {
  return value == null || !Number.isFinite(value)
    ? 'No data'
    : (value / metric.divisor).toLocaleString('en-GB', {
        minimumFractionDigits: metric.digits,
        maximumFractionDigits: metric.digits,
      });
}
export function dayIndices(data: MarketSeries | null, day: string): number[] {
  return (
    data?.t.flatMap((_, i) => (data.settlement_date?.[i] === day ? [i] : [])) ??
    []
  );
}
export function previousDay(day: string): string {
  const d = new Date(day + 'T12:00:00Z');
  d.setUTCDate(d.getUTCDate() - 1);
  return d.toISOString().slice(0, 10);
}
export function timeLabel(t: string): string {
  return new Date(t).toLocaleTimeString('en-GB', {
    timeZone: 'UTC',
    hour: '2-digit',
    minute: '2-digit',
  });
}
export function seriesColor(
  value: number | null,
  min: number,
  max: number,
  metric: MetricDefinition,
): string {
  if (value === null || !Number.isFinite(value)) return 'var(--muted)';
  const ratio =
    max === min ? 0.5 : Math.max(0, Math.min(1, (value - min) / (max - min)));
  return `hsl(${metric.hue} ${27 + ratio * 40}% ${25 + ratio * 52}%)`;
}
export function priceColor(
  value: number | null,
  min: number,
  max: number,
): string {
  return seriesColor(value, min, max, getMetric('price'));
}
export function dailySummary(
  data: MarketSeries,
  indices: number[],
  id: string,
): { average: number | null; count: number; total: number } {
  const values = indices
    .map((i) => data.series[id]?.values[i])
    .filter((n): n is number => n != null && Number.isFinite(n));
  return {
    average: values.length
      ? values.reduce((sum, n) => sum + n, 0) / values.length
      : null,
    count: values.length,
    total: indices.length,
  };
}
export function withWeather(
  data: MarketSeries | null,
  weather: WeatherResponse | null,
): MarketSeries | null {
  if (!data) return null;
  const byHour = new Map(weather?.points.map((p) => [Date.parse(p.t), p]));
  const series = { ...data.series };
  for (const m of METRICS.filter((m) => m.source === 'weather')) {
    const field = m.id as Exclude<keyof WeatherPoint, 't'>;
    series[m.id] = {
      label: m.label,
      unit: m.displayUnit,
      values: data.t.map((t) => {
        const point = byHour.get(Math.floor(Date.parse(t) / 3600000) * 3600000);
        return point?.[field] ?? null;
      }),
    };
  }
  return { ...data, series };
}
