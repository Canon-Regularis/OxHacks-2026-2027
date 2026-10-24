<script lang="ts">
  import { untrack } from 'svelte';
  import { browser } from '$app/environment';
  import type { Catalog } from '$lib/types';
  import type { Widget, Binding, LinkContext } from '../model/types';
  import { loadWidget, TerminalError, type WidgetData } from '../data/client';
  import { isWidgetType } from '../model/document';
  import Chart from './Chart.svelte';
  import ResultTable from './ResultTable.svelte';
  import Comparison from './Comparison.svelte';
  let {
    widget,
    binding,
    ctx,
    catalog,
    onselect,
    onpin,
    inspection,
    onzoom,
    onrun,
    onnotes,
    refresh = 0,
  }: {
    widget: Widget;
    binding: Binding | undefined;
    ctx: LinkContext;
    catalog: Catalog;
    onselect: (t: string) => void;
    onpin: (t: string) => void;
    inspection: string | null;
    onzoom: (z: [number, number] | null) => void;
    onrun: (id: number) => void;
    onnotes: (notes: string) => void;
    refresh?: number;
  } = $props();
  let data = $state<WidgetData | null>(null),
    loading = $state(true),
    error = $state(''),
    code = $state(''),
    retry = $state(0);
  let requestKey = $derived(
    JSON.stringify({
      type: widget.type,
      binding,
      from: ctx.from,
      to: ctx.to,
      zoom: ctx.zoom,
      selected: widget.type === 'inspector' ? inspection : null,
      refresh,
      retry,
    }),
  );
  $effect(() => {
    if (!browser) return;
    requestKey;
    return untrack(() => {
      const loader = loadWidget(
        widget,
        binding,
        {
          ...ctx,
          selected: widget.type === 'inspector' ? inspection : ctx.selected,
        },
        catalog,
        onrun,
      );
      let alive = true;
      loading = true;
      error = '';
      data = null;
      loader.promise
        .then((result) => {
          if (alive) {
            data = result;
            loading = false;
          }
        })
        .catch((e) => {
          if (alive && e?.name !== 'AbortError') {
            loading = false;
            error =
              e instanceof Error ? e.message : 'Unable to load this widget';
            code = e instanceof TerminalError ? e.code : 'network';
          }
        });
      return () => {
        alive = false;
        loader.release();
      };
    });
  });
  let tableName = $derived(
    data?.tables[widget.settings.table]
      ? widget.settings.table
      : Object.keys(data?.tables ?? {})[0],
  );
  let table = $derived(data?.tables[tableName]);
</script>

{#if !isWidgetType(widget.type)}<p class="terminal-empty">
    This widget type is not supported by this version. Export it, rebind it or
    remove it.
  </p>
{:else if loading}<div class="terminal-empty" role="status">
    Loading {widget.title.toLowerCase()}…
  </div>
{:else if error}<div class="terminal-widget-error" role="status">
    <strong
      >{code === 'no_run' ? 'Not computed yet' : 'Data unavailable'}</strong
    >
    <p>{error}</p>
    <button onclick={() => retry++}>Retry widget</button>
  </div>
{:else if data}
  {#if data.warnings.length}<details class="terminal-warnings" open>
      <summary
        >{data.warnings.length} data warning{data.warnings.length === 1
          ? ''
          : 's'}</summary
      >{#each data.warnings as warning}<p>{warning}</p>{/each}
    </details>{/if}
  {#if widget.type === 'chart'}<Chart
      t={data.t}
      columns={data.columns}
      settings={widget.settings}
      selected={ctx.selected}
      zoom={ctx.zoom}
      {onselect}
      {onpin}
      {onzoom}
    />
  {:else if widget.type === 'table'}{#if table}<ResultTable
        {table}
        {onselect}
      />{:else}<p class="terminal-empty">
        {binding?.kind === 'algorithm'
          ? 'This run has no result tables.'
          : 'Choose an algorithm binding in widget settings.'}
      </p>{/if}
  {:else if widget.type === 'comparison'}{#if table}<Comparison
        {table}
        metric={widget.settings.metric}
        category={widget.settings.category}
      />{:else}<p class="terminal-empty">
        Choose a stored algorithm with numeric result tables.
      </p>{/if}
  {:else if widget.type === 'inspector'}
    {#if ctx.selected}<div class="terminal-inspector-preview">
        <p>{ctx.selected} · selected</p>
        <dl>
          {#each Object.entries(data.columns) as [id, column]}<div>
              <dt>{column.label}</dt>
              <dd>
                {column.values[data.t.indexOf(ctx.selected)]?.toLocaleString(
                  'en-GB',
                  { maximumFractionDigits: 2 },
                ) ?? 'No data'} <small>{column.unit}</small>
              </dd>
            </div>{/each}
        </dl>
        <p class="terminal-hint">
          Click a chart point for its full publication history.
        </p>
      </div>{/if}
    {#if data.period}<div class="terminal-inspector">
        <p>
          {data.period.settlement_date} · SP {data.period.settlement_period}<br
          />{data.period.delivery_start_utc}
        </p>
        <dl>
          {#each Object.entries(data.period.values) as [id, value]}<div>
              <dt>{catalog.series.find((s) => s.id === id)?.label ?? id}</dt>
              <dd>
                {value?.toLocaleString('en-GB', { maximumFractionDigits: 2 }) ??
                  'No data'}
                <small>{catalog.series.find((s) => s.id === id)?.unit}</small>
              </dd>
            </div>{/each}
        </dl>
        {#each Object.entries(data.period.vintages) as [id, versions]}<details>
            <summary
              >{id} forecast publication history · {versions.length}</summary
            >
            <ul>
              {#each versions as v}<li>
                  {v.published_at_utc} · {v.value ?? 'No data'} MW
                </li>{/each}
            </ul>
          </details>{/each}
        <details>
          <summary>Default-run analysis · {data.period.analysis.length}</summary
          >
          <p class="terminal-hint">
            These are default runs, not a chart's pinned variant.
          </p>
          {#each data.period.analysis as a}<p>
              {a.algorithm_id} · #{a.run_id} · {a.indicator}: {a.value ??
                'No data'}
              {a.unit ?? ''}
            </p>{/each}
        </details>
      </div>{:else}<p class="terminal-empty">
        Click or use arrow keys on a linked chart for full details of a
        settlement period.
      </p>{/if}
  {:else}<div class="terminal-context">
      <label
        >Investigation notes<textarea
          aria-label="Investigation notes"
          value={widget.settings.notes}
          onblur={(e) => onnotes(e.currentTarget.value)}
          maxlength="10000"
          placeholder="Record a finding or a question…"></textarea></label
      >
      <p class="terminal-hint">Notes save when you leave the field.</p>
      {#if binding?.kind === 'source'}<p>Retrospective source data</p>
        {#each binding.ids as id}{@const s = catalog.series.find(
            (s) => s.id === id,
          )}
          <details>
            <summary>{s?.label ?? id}</summary
            >{#each s?.datasets ?? [] as dataset}<p>
                {catalog.datasets.find((d) => d.dataset === dataset)?.source ??
                  dataset}
              </p>{/each}
          </details>{/each}{/if}
    </div>{/if}
  {#if data.run}<details class="terminal-provenance">
      <summary
        >Run #{data.run.run_id} · v{data.run.algorithm_version} · {data.run
          .cutoff_utc
          ? 'historical cutoff'
          : 'retrospective'} · whole-run metadata</summary
      >
      <p>{data.run.range_start_utc} → {data.run.range_end_utc}</p>
      <p>
        Computed {data.run.finished_at_utc ?? 'unknown'} · settings {JSON.stringify(
          data.run.params,
        )}
      </p>
      {#each Object.entries(data.run.datasets) as [name, loaded]}<p>
          {catalog.datasets.find((d) => d.dataset === name)?.source ?? name} · loaded
          {loaded ?? 'unknown'}
        </p>{/each}
    </details>{/if}
  {#if data.limitations}<details class="terminal-limitations">
      <summary>What this does not mean</summary>
      <p>{data.limitations}</p>
    </details>{/if}
{/if}
