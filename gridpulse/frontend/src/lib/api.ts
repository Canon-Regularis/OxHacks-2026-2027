export interface SeriesColumn {
  label: string;
  unit: string;
  values: (number | null)[];
}
export interface MarketSeries {
  from: string;
  to: string;
  resolution: string;
  t: string[];
  settlement_date: string[] | null;
  settlement_period: number[] | null;
  series: Record<string, SeriesColumn>;
}
export interface Snapshot {
  as_of_utc: string;
  settlement_date: string;
  settlement_period: number;
  data_updated_at_utc: string | null;
}
import type { Catalog, Health } from './types';
export interface WeatherLocation {
  id: string;
  label: string;
  latitude: number;
  longitude: number;
}
export interface WeatherPoint {
  t: string;
  temperature: number | null;
  pressure: number | null;
  temperature_change: number | null;
  pressure_change: number | null;
  price_change: number | null;
  temperature_correlation: number | null;
  pressure_correlation: number | null;
  temperature_samples: number;
  pressure_samples: number;
}
export interface WeatherResponse {
  location: WeatherLocation;
  points: WeatherPoint[];
  source: string;
  warnings: string[];
  window_hours: number;
}
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}
async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`/api/${path}`, { signal });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new ApiError(
      response.status,
      error?.detail?.message ?? `Request failed (${response.status})`,
    );
  }
  return response.json();
}
export const api = {
  snapshot: (signal?: AbortSignal) => get<Snapshot>('snapshot', signal),
  health: (signal?: AbortSignal) => get<Health>('health', signal),
  catalog: (signal?: AbortSignal) => get<Catalog>('catalog', signal),
  series: (from: string, to: string, ids: string[], signal?: AbortSignal) =>
    get<MarketSeries>(
      `series?${new URLSearchParams({ from, to, ids: ids.join(','), resolution: '30m' })}`,
      signal,
    ),
  locations: (signal?: AbortSignal) =>
    get<{ locations: WeatherLocation[]; warnings: string[] }>(
      'weather/locations',
      signal,
    ),
  weather: (date: string, location: string, signal?: AbortSignal) =>
    get<WeatherResponse>(
      `weather?${new URLSearchParams({ date, location })}`,
      signal,
    ),
};
