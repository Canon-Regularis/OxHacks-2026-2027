<script lang="ts">
  import type { Catalog } from '$lib/types';
  import type { WidgetType, Workspace } from '../model/types';
  let {
    catalog,
    savedPresets,
    onsaved,
    onadd,
    onalgorithm,
    onpreset,
  }: {
    catalog: Catalog;
    savedPresets: Workspace[];
    onsaved: (id: string) => void;
    onadd: (type: WidgetType, source?: string) => void;
    onalgorithm: (
      id: string,
      action: 'chart' | 'table' | 'existing' | 'layout',
    ) => void;
    onpreset: (kind: 'market' | 'blank') => void;
  } = $props();
  let search = $state(''),
    section = $state('All');
  const types: WidgetType[] = [
    'chart',
    'table',
    'comparison',
    'inspector',
    'context',
  ];
  const matches = (s: string) => s.toLowerCase().includes(search.toLowerCase());
  let resultCount = $derived(
    (section === 'All' || section === 'Data'
      ? catalog.series.filter((s) => matches(s.label + ' ' + s.id)).length
      : 0) +
      (section === 'All' || section === 'Algorithms'
        ? catalog.algorithms.filter((a) => matches(a.id + ' ' + a.description))
            .length
        : 0) +
      (section === 'All' || section === 'Widgets'
        ? types.filter((t) => matches(t)).length
        : 0) +
      (section === 'All' || section === 'Presets'
        ? [...savedPresets.map((p) => p.name), 'market', 'blank'].filter(
            matches,
          ).length
        : 0),
  );
</script>

<input
  class="terminal-search"
  aria-label="Search data algorithms and widgets"
  placeholder="Search data, algorithms, widgets…"
  bind:value={search}
/>
<div class="terminal-filter-tabs">
  {#each ['All', 'Data', 'Algorithms', 'Widgets', 'Presets'] as tab}<button
      class:active={section === tab}
      onclick={() => (section = tab)}>{tab}</button
    >{/each}
</div>
{#if section === 'All' || section === 'Data'}<h3>Source data</h3>
  {#each catalog.series.filter( (s) => matches(s.label + ' ' + s.id) ) as source}<button
      class="terminal-catalogue-source"
      onclick={() => onadd('chart', source.id)}
      ><strong>{source.label}</strong><span
        >{source.unit} · {source.kind} <b>+</b></span
      ></button
    >{/each}{/if}
{#if section === 'All' || section === 'Algorithms'}<h3>
    Registered algorithms
  </h3>
  {#if !catalog.algorithms.length}<p class="terminal-hint">
      No algorithms are registered.
    </p>{/if}{#each catalog.algorithms.filter( (a) => matches(a.id + ' ' + a.description) ) as algo}<section
      class="terminal-algorithm-card"
    >
      <h4>{algo.id.replaceAll('_', ' ')}</h4>
      <p>{algo.description}</p>
      <span class="terminal-hint"
        >v{algo.version} · {algo.latest_run
          ? `Latest default #${algo.latest_run.run_id}`
          : 'No default run recorded'}</span
      >
      <div class="terminal-algorithm-actions">
        <button
          disabled={!Object.keys(algo.outputs).length}
          onclick={() => onalgorithm(algo.id, 'existing')}
          >Add outputs to chart</button
        ><button
          disabled={!Object.keys(algo.outputs).length}
          onclick={() => onalgorithm(algo.id, 'chart')}>Open new chart</button
        ><button
          disabled={!Object.keys(algo.tables).length}
          onclick={() => onalgorithm(algo.id, 'table')}>Open tables</button
        ><button onclick={() => onalgorithm(algo.id, 'layout')}
          >Open analysis layout</button
        >
      </div>
      <details>
        <summary>Inputs and limitations</summary>
        <p>Inputs: {algo.inputs.join(', ') || 'Clock only'}</p>
        <p>{algo.limitations}</p>
      </details>
    </section>{/each}{/if}
{#if section === 'All' || section === 'Widgets'}<h3>Reusable widgets</h3>
  {#each types.filter((t) => matches(t)) as type}<button
      class="terminal-catalogue-source"
      onclick={() => onadd(type)}
      ><strong
        >{type === 'context'
          ? 'Context & notes'
          : type === 'inspector'
            ? 'Period inspector'
            : type === 'comparison'
              ? 'Metric comparison'
              : type === 'table'
                ? 'Results table'
                : 'Time-series chart'}</strong
      ><span>Add +</span></button
    >{/each}{/if}
{#if section === 'All' || section === 'Presets'}<h3>Starter views</h3>
  {#each savedPresets.filter((p) => matches(p.name)) as saved}<button
      class="terminal-catalogue-source"
      onclick={() => onsaved(saved.id)}
      ><strong>{saved.name}</strong><span>Saved preset ↗</span></button
    >{/each}
  {#each ['market', 'blank'] as kind}{#if matches(kind)}<button
        class="terminal-catalogue-source"
        onclick={() => onpreset(kind as 'market' | 'blank')}
        ><strong
          >{kind === 'market'
            ? 'Market investigation'
            : 'Blank workspace'}</strong
        ><span>New workspace ↗</span></button
      >{/if}{/each}{/if}
{#if search && !resultCount}<p class="terminal-empty">
    No matches. Try a series name, algorithm or widget type.
  </p>{/if}
{#if search}<p class="terminal-hint">
    Only currently registered data and algorithms appear. Clear search to see
    everything.
  </p>{/if}
