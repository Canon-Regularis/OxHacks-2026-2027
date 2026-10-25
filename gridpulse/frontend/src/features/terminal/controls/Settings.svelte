<script lang="ts">
  import type { Catalog, RunInfo, RunTables, Table } from '$lib/types';
  import type {
    Widget,
    Workspace,
    Position,
    Binding,
    NewBinding,
    WidgetType,
  } from '../model/types';
  import { runs, pool } from '../data/client';
  let {
    widget,
    exportCsv,
    workspace,
    catalog,
    onchange,
    onbinding,
    onrun,
    onlink,
    onmove,
    onduplicate,
    onclose,
  }: {
    widget: Widget;
    exportCsv?: () => void;
    workspace: Workspace;
    catalog: Catalog;
    onchange: (widget: Widget) => void;
    onbinding: (binding: NewBinding | null) => void;
    onrun: (id: number | null) => void;
    onlink: (group: string | null) => void;
    onmove: (target: string, position: Position) => void;
    onduplicate: () => void;
    onclose: () => void;
  } = $props();
  let binding = $derived(
    widget.binding ? workspace.bindings[widget.binding] : undefined,
  );
  let algo = $derived(
    binding?.kind === 'algorithm'
      ? catalog.algorithms.find((a) => a.id === binding.algorithm)
      : undefined,
  );
  let stored = $state<RunInfo[]>([]),
    runError = $state(''),
    runLoading = $state(false),
    moveTarget = $state(''),
    movePosition = $state<Position>('tab');
  let outputs = $derived(
    binding?.kind === 'source'
      ? Object.fromEntries(
          binding.ids.map((id) => [
            id,
            catalog.series.find((s) => s.id === id)?.unit ?? '',
          ]),
        )
      : {
          ...(algo?.outputs ?? {}),
          ...Object.fromEntries(
            (binding?.kind === 'algorithm' ? (binding.sources ?? []) : []).map(
              (id) => [
                'source:' + id,
                catalog.series.find((s) => s.id === id)?.unit ?? '',
              ],
            ),
          ),
        },
  );
  $effect(() => {
    const id = algo?.id;
    stored = [];
    runError = '';
    if (!id) return;
    const lease = runs(id);
    let alive = true;
    runLoading = true;
    lease.promise
      .then((r) => {
        if (alive) stored = r.runs;
      })
      .catch((e) => {
        if (alive) runError = e.message;
      })
      .finally(() => {
        if (alive) runLoading = false;
      });
    return () => {
      alive = false;
      lease.release();
    };
  });
  let resultTables = $state<Record<string, Table>>({});
  let selectedTable = $derived(
    resultTables[widget.settings.table] ?? Object.values(resultTables)[0],
  );
  let metricColumns = $derived(
    selectedTable?.columns.filter((_, i) =>
      selectedTable.rows.some((row) => typeof row[i] === 'number'),
    ) ?? [],
  );
  $effect(() => {
    const id = binding?.kind === 'algorithm' ? binding.runId : null;
    if (widget.type !== 'comparison' || !id) {
      resultTables = {};
      return;
    }
    let alive = true;
    const lease = pool.acquire<RunTables>(`/analysis/runs/${id}/tables`);
    lease.promise
      .then((response) => {
        if (alive) resultTables = response.tables;
      })
      .catch(() => {
        if (alive) resultTables = {};
      });
    return () => {
      alive = false;
      lease.release();
    };
  });
  function setting(key: keyof Widget['settings'], value: unknown) {
    onchange({ ...widget, settings: { ...widget.settings, [key]: value } });
  }
  function pickBinding(value: string) {
    if (value === 'none') onbinding(null);
    else if (value.startsWith('algo:'))
      onbinding({
        kind: 'algorithm',
        algorithm: value.slice(5),
        runId: null,
      } as NewBinding);
    else onbinding({ kind: 'source', ids: [value.slice(7)] } as NewBinding);
  }
</script>

<label
  >Widget title<input
    aria-label="Widget title"
    value={widget.title}
    maxlength="100"
    onchange={(e) =>
      onchange({ ...widget, title: e.currentTarget.value || widget.title })}
  /></label
>
<label
  >Linked group<select
    aria-label="Linked group"
    value={widget.group ?? ''}
    onchange={(e) => onlink(e.currentTarget.value || null)}
    ><option value="">Independent</option
    >{#each Object.values(workspace.groups) as group}<option value={group.id}
        >{group.name}</option
      >{/each}<option value="__new">Create linked group…</option></select
  ></label
>
{#if widget.type !== 'inspector'}
  <label
    >Data binding<select
      aria-label="Data binding"
      value={binding?.kind === 'algorithm'
        ? `algo:${binding.algorithm}`
        : binding?.kind === 'source'
          ? `source:${binding.ids[0]}`
          : 'none'}
      onchange={(e) => pickBinding(e.currentTarget.value)}
      ><option value="none">No data binding</option><optgroup
        label="Source series"
        >{#each catalog.series as s}<option value={`source:${s.id}`}
            >{s.label} · {s.unit}</option
          >{/each}</optgroup
      ><optgroup label="Algorithms"
        >{#each catalog.algorithms as a}<option value={`algo:${a.id}`}
            >{a.id}</option
          >{/each}</optgroup
      ></select
    ></label
  >
  {#if binding?.kind === 'source'}<fieldset>
      <legend>Source series</legend>{#each catalog.series as s}<label
          class="terminal-check"
          ><input
            type="checkbox"
            checked={binding.ids.includes(s.id)}
            disabled={binding.ids.includes(s.id) && binding.ids.length === 1}
            onchange={() => {
              if (binding?.kind === 'source')
                onbinding({
                  kind: 'source',
                  ids: binding.ids.includes(s.id)
                    ? binding.ids.filter((id) => id !== s.id)
                    : [...binding.ids, s.id],
                } as NewBinding);
            }}
          />{s.label}<small>{s.unit}</small></label
        >{/each}
    </fieldset>{/if}
  {#if algo && widget.type === 'chart' && binding?.kind === 'algorithm'}<fieldset
    >
      <legend>Overlay source data</legend>{#each catalog.series as source}<label
          class="terminal-check"
          ><input
            type="checkbox"
            checked={binding.sources?.includes(source.id) ?? false}
            onchange={() => {
              if (binding?.kind === 'algorithm') {
                const sources = binding.sources ?? [];
                onbinding({
                  ...binding,
                  sources: sources.includes(source.id)
                    ? sources.filter((id) => id !== source.id)
                    : [...sources, source.id],
                });
              }
            }}
          />{source.label}<small>{source.unit}</small></label
        >{/each}
    </fieldset>{/if}
  {#if algo}<label
      >Stored run<select
        aria-label="Stored run"
        value={binding?.kind === 'algorithm'
          ? (binding.runId ?? 'latest')
          : 'latest'}
        onchange={(e) =>
          onrun(
            e.currentTarget.value === 'latest'
              ? null
              : Number(e.currentTarget.value),
          )}
        ><option value="latest">Resolve latest default run</option
        >{#if binding?.kind === 'algorithm' && binding.runId && !stored.some((r) => r.run_id === binding.runId)}<option
            value={binding.runId}>Pinned #{binding.runId}</option
          >{/if}{#each stored as run}<option value={run.run_id}
            >#{run.run_id} · v{run.algorithm_version} · {run.range_start_utc.slice(
              0,
              10,
            )}–{run.range_end_utc.slice(0, 10)} · {JSON.stringify(
              run.overrides,
            )}{run.cutoff_utc ? ' · cutoff' : ''}</option
          >{/each}</select
      ></label
    >
    <p class="terminal-hint">
      {runLoading
        ? 'Loading runs…'
        : runError ||
          `${stored.length} recent completed runs (up to 100). Settings are read-only; computation submission is not available.`}
    </p>
    <details>
      <summary>Algorithm settings and limitations</summary
      >{#each Object.entries(algo.params) as [key, value]}<p>
          {key}: {JSON.stringify(value)}<br /><small
            >{algo.param_descriptions[key]}</small
          >
        </p>{/each}
      <p>{algo.limitations}</p>
    </details>{/if}
{/if}
{#if widget.type === 'chart'}<label
    >Display<select
      aria-label="Chart display"
      value={widget.settings.display}
      onchange={(e) => setting('display', e.currentTarget.value)}
      ><option value="overlay">Overlay · at most two unit axes</option><option
        value="panes">Separate unit panes</option
      ><option value="indexed">Indexed · first non-zero = 100</option></select
    ></label
  >
  <fieldset>
    <legend>Outputs and styles</legend
    >{#each Object.entries(outputs) as [key, unit]}{@const enabled =
        !widget.settings.outputs.length ||
        widget.settings.outputs.includes(key)}
      <div class="terminal-output-row">
        <label class="terminal-check"
          ><input
            type="checkbox"
            checked={enabled}
            onchange={() => {
              const current = widget.settings.outputs.length
                ? widget.settings.outputs
                : Object.keys(outputs);
              const next = current.includes(key)
                ? current.filter((k) => k !== key)
                : [...current, key];
              if (next.length) setting('outputs', next);
            }}
          />{key}<small>{unit || 'unitless'}</small></label
        ><input
          type="color"
          aria-label={`Colour for ${key}`}
          value={widget.settings.styles[key]?.color ?? '#e6b080'}
          onchange={(e) =>
            setting('styles', {
              ...widget.settings.styles,
              [key]: {
                color: e.currentTarget.value,
                dashed: widget.settings.styles[key]?.dashed ?? false,
              },
            })}
        /><label class="terminal-check"
          ><input
            type="checkbox"
            checked={widget.settings.styles[key]?.dashed ?? false}
            onchange={(e) =>
              setting('styles', {
                ...widget.settings.styles,
                [key]: {
                  color: widget.settings.styles[key]?.color ?? '#e6b080',
                  dashed: e.currentTarget.checked,
                },
              })}
          />Dash</label
        >
      </div>{/each}
  </fieldset>{/if}
{#if (widget.type === 'table' || widget.type === 'comparison') && algo}<label
    >Result table<select
      aria-label="Result table"
      value={widget.settings.table}
      onchange={(e) => setting('table', e.currentTarget.value)}
      ><option value="">First available table</option
      >{#each Object.entries(algo.tables) as [name, description]}<option
          value={name}>{name} · {description}</option
        >{/each}</select
    ></label
  >{/if}
{#if widget.type === 'comparison'}<label
    >Numeric column<select
      aria-label="Comparison metric"
      value={widget.settings.metric}
      onchange={(e) => setting('metric', e.currentTarget.value)}
      ><option value="">First numeric column</option
      >{#each metricColumns as column}<option value={column}
          >{column.replaceAll('_', ' ')}</option
        >{/each}</select
    ></label
  ><label
    >Category column<select
      aria-label="Comparison category"
      value={widget.settings.category}
      onchange={(e) => setting('category', e.currentTarget.value)}
      ><option value="">Model / first column</option
      >{#each selectedTable?.columns ?? [] as column}<option value={column}
          >{column.replaceAll('_', ' ')}</option
        >{/each}</select
    ></label
  >{/if}
{#if widget.type === 'chart' || widget.type === 'table' || widget.type === 'comparison'}
  <fieldset>
    <legend>Data export</legend>
    <button disabled={!exportCsv} onclick={() => exportCsv?.()}
      >{widget.type === 'chart'
        ? 'Export chart CSV'
        : 'Export table CSV'}</button
    >
    <p class="terminal-hint">
      {exportCsv
        ? widget.type === 'chart'
          ? 'Export the loaded chart data with UTC timestamps and units.'
          : 'Export all rows of the selected whole-run table.'
        : 'CSV export is available when this widget has loaded data.'}
    </p>
  </fieldset>
{/if}
<fieldset>
  <legend>Move without dragging</legend><label
    >Destination widget<select
      aria-label="Move destination"
      bind:value={moveTarget}
      ><option value="">Choose destination…</option
      >{#each Object.values(workspace.widgets).filter((w) => w.id !== widget.id) as other}<option
          value={other.id}>{other.title}</option
        >{/each}</select
    ></label
  ><label
    >Position<select aria-label="Move position" bind:value={movePosition}
      >{#each ['tab', 'left', 'right', 'top', 'bottom'] as p}<option value={p}
          >{p}</option
        >{/each}</select
    ></label
  ><button
    disabled={!moveTarget || workspace.locked}
    onclick={() => onmove(moveTarget, movePosition)}>Move widget</button
  >
</fieldset>
<div class="terminal-action-row">
  <button onclick={onduplicate}>Duplicate widget</button><button
    class="danger"
    onclick={onclose}>Close widget</button
  >
</div>
