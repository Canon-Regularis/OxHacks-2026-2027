<script lang="ts">
  import type { Table } from '$lib/types';
  let {
    table,
    metric,
    category,
  }: { table: Table; metric: string; category: string } = $props();
  let numeric = $derived(
    table.columns.filter((_, i) =>
      table.rows.some((r) => typeof r[i] === 'number'),
    ),
  );
  let m = $derived(
    table.columns.indexOf(numeric.includes(metric) ? metric : numeric[0]),
  );
  let c = $derived(
    table.columns.indexOf(category || 'model') >= 0
      ? table.columns.indexOf(category || 'model')
      : 0,
  );
  let entries = $derived(
    table.rows
      .flatMap((row, i) =>
        typeof row[m] === 'number'
          ? [{ label: String(row[c] ?? i + 1), value: row[m] as number }]
          : [],
      )
      .slice(0, 100),
  );
  let max = $derived(Math.max(1, ...entries.map((e) => Math.abs(e.value))));
</script>

<p class="terminal-hint">
  {table.columns[m]?.replaceAll('_', ' ') ?? 'No numeric metric'} · whole-run results
  · no automatic ranking
</p>
<div class="terminal-comparison">
  {#each entries as entry}<div>
      <span title={entry.label}>{entry.label}</span>
      <div class="terminal-bar-track">
        <i
          style={`width:${(Math.abs(entry.value) / max) * 100}%`}
          class:negative={entry.value < 0}
        ></i>
      </div>
      <strong
        >{entry.value.toLocaleString('en-GB', {
          maximumSignificantDigits: 4,
        })}</strong
      >
    </div>{/each}{#if !entries.length}<p class="terminal-empty">
      Choose a table with numeric metrics.
    </p>{/if}
</div>
