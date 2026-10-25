import type { Column, Table } from '$lib/types';
/** Same aligned values as the chart, projected into a sortable, virtualised data table. */
export function sourceTable(
  t: string[],
  columns: Record<string, Column>,
  zoom: [number, number] | null,
  outputs: string[] = [],
): Table {
  const selected = Object.entries(columns)
    .filter(([id]) => !outputs.length || outputs.includes(id))
    .map(([, column]) => column);
  return {
    description:
      'Source series over the selected time range. Timestamps are UTC; missing values remain missing.',
    columns: [
      'delivery_start_utc',
      ...selected.map(
        (column) => `${column.label}${column.unit ? ` (${column.unit})` : ''}`,
      ),
    ],
    rows: t.flatMap((instant, index) =>
      zoom && (Date.parse(instant) < zoom[0] || Date.parse(instant) > zoom[1])
        ? []
        : [
            [
              instant,
              ...selected.map((column) => column.values[index] ?? null),
            ],
          ],
    ),
  };
}
