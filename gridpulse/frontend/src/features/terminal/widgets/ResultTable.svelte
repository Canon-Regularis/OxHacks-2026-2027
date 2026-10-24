<script lang="ts">
  import type { Table } from '$lib/types';
  let { table, onselect }: { table: Table; onselect: (time: string) => void } =
    $props();
  let search = $state(''),
    sort = $state(-1),
    ascending = $state(true),
    scroll = $state(0),
    viewport = $state(240);
  let rows = $derived.by(() => {
    const matched = table.rows
      .map((row, index) => ({ row, index }))
      .filter(
        ({ row }) =>
          !search ||
          row.some((c) =>
            String(c ?? '')
              .toLowerCase()
              .includes(search.toLowerCase()),
          ),
      );
    if (sort >= 0)
      matched.sort((a, b) => {
        const x = a.row[sort],
          y = b.row[sort];
        if (x === null) return 1;
        if (y === null) return -1;
        const cmp =
          typeof x === 'number' && typeof y === 'number'
            ? x - y
            : String(x).localeCompare(String(y), undefined, { numeric: true });
        return (ascending ? cmp : -cmp) || a.index - b.index;
      });
    return matched;
  });
  let start = $derived(Math.max(0, Math.floor(scroll / 30) - 4));
  let end = $derived(
    Math.min(rows.length, start + Math.ceil(viewport / 30) + 10),
  );
  let timeColumn = $derived(
    table.columns.findIndex((c) => c === 'delivery_start_utc'),
  );
  function cell(value: unknown, column: string) {
    if (value === null) return '—';
    if (typeof value === 'number') {
      if (['wmape', 'smape', 'coverage_80', 'coverage_95'].includes(column))
        return `${(value * 100).toFixed(1)}%`;
      if (column === 'p_value' && value === 0) return '< 1e-16';
      return value.toLocaleString('en-GB', { maximumSignificantDigits: 5 });
    }
    return String(value);
  }
</script>

<div class="terminal-table-tools">
  <input
    aria-label="Search result rows"
    placeholder="Filter rows…"
    bind:value={search}
    oninput={() => (scroll = 0)}
  /><span>{rows.length} / {table.rows.length} rows · whole run</span>
</div>
<div
  class="terminal-table-scroll"
  bind:clientHeight={viewport}
  onscroll={(e) => (scroll = e.currentTarget.scrollTop)}
>
  <table>
    <thead
      ><tr
        >{#each table.columns as column, i}<th
            aria-sort={sort === i
              ? ascending
                ? 'ascending'
                : 'descending'
              : 'none'}
            ><button
              onclick={() => {
                ascending = sort === i ? !ascending : true;
                sort = i;
              }}
              >{column.replaceAll('_', ' ')}
              {sort === i ? (ascending ? '↑' : '↓') : ''}</button
            ></th
          >{/each}</tr
      ></thead
    ><tbody>
      {#if start > 0}<tr aria-hidden="true"
          ><td
            colspan={table.columns.length}
            style={`height:${start * 30}px;padding:0`}
          ></td></tr
        >{/if}
      {#each rows.slice(start, end) as { row, index } (index)}<tr
          >{#each row as value, i}<td
              >{#if i === timeColumn && typeof value === 'string'}<button
                  class="terminal-cell-link"
                  onclick={() => onselect(value)}>{value}</button
                >{:else}{cell(value, table.columns[i])}{/if}</td
            >{/each}</tr
        >{/each}
      {#if end < rows.length}<tr aria-hidden="true"
          ><td
            colspan={table.columns.length}
            style={`height:${(rows.length - end) * 30}px;padding:0`}
          ></td></tr
        >{/if}
    </tbody>
  </table>
  {#if !rows.length}<p class="terminal-empty">No matching rows.</p>{/if}
</div>
