import type { WidgetData } from './client';
import type { Table } from '$lib/types';
export function csv(rows: unknown[][]): string {
  return (
    rows
      .map((row) =>
        row
          .map((value) => {
            let text = value == null ? '' : String(value);
            if (typeof value === 'string' && /^\s*[=+@-]/.test(text))
              text = "'" + text;
            return '"' + text.replaceAll('"', '""') + '"';
          })
          .join(','),
      )
      .join('\r\n') + '\r\n'
  );
}
export function seriesCsv(data: WidgetData): string {
  const columns = Object.entries(data.columns);
  return csv([
    ['timestamp_utc', ...columns.map(([id, c]) => `${id} (${c.unit})`)],
    ...data.t.map((t, i) => [t, ...columns.map(([, c]) => c.values[i])]),
  ]);
}
export function tableCsv(table: Table): string {
  return csv([table.columns, ...table.rows]);
}
export function downloadCsv(contents: string, filename: string) {
  const href = URL.createObjectURL(
    new Blob(['\uFEFF', contents], { type: 'text/csv;charset=utf-8' }),
  );
  const link = document.createElement('a');
  link.href = href;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(href), 1000);
}
