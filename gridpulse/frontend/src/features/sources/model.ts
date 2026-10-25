import type { Catalog, DatasetCoverage, SeriesInfo } from '$lib/types';

const DESCRIPTIONS: Record<
  string,
  { name: string; kind: string; cadence: string }
> = {
  elexon_system_prices: {
    name: 'Imbalance prices & volume',
    kind: 'Settlement',
    cadence: 'Half-hourly',
  },
  elexon_windfor: {
    name: 'Wind generation forecasts',
    kind: 'Forecast',
    cadence: 'Hourly',
  },
  elexon_ndf: {
    name: 'National demand forecasts',
    kind: 'Forecast',
    cadence: 'Half-hourly',
  },
  elexon_tsdf: {
    name: 'Transmission demand forecasts',
    kind: 'Forecast',
    cadence: 'Half-hourly',
  },
  elexon_fuelhh: {
    name: 'Wind generation outturn',
    kind: 'Outturn',
    cadence: 'Half-hourly',
  },
  neso_historic_demand: {
    name: 'Historic electricity demand',
    kind: 'Outturn',
    cadence: 'Half-hourly',
  },
  neso_embedded_forecast_archive: {
    name: 'Embedded wind & solar forecasts',
    kind: 'Forecast',
    cadence: 'Half-hourly',
  },
  openmeteo_era5: {
    name: 'ERA5 weather reanalysis',
    kind: 'Reanalysis',
    cadence: 'Hourly',
  },
  openmeteo_previous_runs: {
    name: 'Archived weather forecasts',
    kind: 'Forecast',
    cadence: 'Hourly',
  },
};
export interface SourceRow {
  dataset: DatasetCoverage;
  name: string;
  provider: string;
  kind: string;
  cadence: string;
  series: SeriesInfo[];
  units: string[];
  searchable: string;
}
export function sourceRows(catalog: Catalog | null): SourceRow[] {
  return (catalog?.datasets ?? [])
    .map((dataset) => {
      const series = (catalog?.series ?? []).filter((item) =>
        item.datasets.includes(dataset.dataset),
      );
      const kinds = [
        ...new Set(
          series.map(
            (item) =>
              ({
                forecast: 'Forecast',
                outturn: 'Outturn',
                settlement: 'Settlement',
              })[item.kind],
          ),
        ),
      ];
      const description = DESCRIPTIONS[dataset.dataset] ?? {
        name: dataset.dataset.replaceAll('_', ' '),
        kind: kinds.join(' / ') || 'Other',
        cadence: 'Not supplied',
      };
      const units = [...new Set(series.map((item) => item.unit || 'Unitless'))];
      return {
        dataset,
        ...description,
        provider: dataset.source?.split(' - ')[0] || 'Source not supplied',
        series,
        units,
        searchable: [
          dataset.dataset,
          dataset.source,
          dataset.notes,
          description.name,
          description.kind,
          ...series.map((item) => item.label),
          ...units,
        ]
          .join(' ')
          .toLowerCase(),
      };
    })
    .sort((a, b) => a.name.localeCompare(b.name));
}
export function validInstant(value: string | null): value is string {
  return value !== null && Number.isFinite(Date.parse(value));
}
export function coverageExtent(datasets: DatasetCoverage[]): {
  first: string | null;
  last: string | null;
} {
  const first =
    datasets
      .map((d) => d.first_utc)
      .filter(validInstant)
      .sort((a, b) => Date.parse(a) - Date.parse(b))[0] ?? null;
  const last =
    datasets
      .map((d) => d.last_utc)
      .filter(validInstant)
      .sort((a, b) => Date.parse(b) - Date.parse(a))[0] ?? null;
  return { first, last };
}
