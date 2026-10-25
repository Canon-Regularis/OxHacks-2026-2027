<script lang="ts">
  import type { Column } from '$lib/types';
  import { METRICS } from '../../overview/model';
  import type { Widget } from '../model/types';
  let {
    t,
    columns,
    settings,
    selected,
    zoom,
    onselect,
    onpin,
    onzoom,
  }: {
    t: string[];
    columns: Record<string, Column>;
    settings: Widget['settings'];
    selected: string | null;
    zoom: [number, number] | null;
    onselect: (t: string) => void;
    onpin: (t: string) => void;
    onzoom: (zoom: [number, number] | null) => void;
  } = $props();
  const palette = [
    '#e6b080',
    '#a2c7b8',
    '#b8afd0',
    '#9cb8cf',
    '#d2a1a1',
    '#cfc58f',
    '#85c3c9',
    '#d7acd1',
  ];
  function colour(key: string) {
    return (
      settings.styles[key]?.color ??
      METRICS.find((metric) => metric.id === key.replace(/^source:/, ''))
        ?.color ??
      palette[keys.indexOf(key) % palette.length]
    );
  }
  let width = $state(600),
    height = $state(300),
    drag = $state<number | null>(null),
    end = $state<number | null>(null);
  let keys = $derived(
    (settings.outputs.length ? settings.outputs : Object.keys(columns))
      .filter((k) => columns[k])
      .slice(0, 12),
  );
  let indices = $derived(
    t.flatMap((time, i) =>
      !zoom || (Date.parse(time) >= zoom[0] && Date.parse(time) <= zoom[1])
        ? [i]
        : [],
    ),
  );
  let units = $derived([
    ...new Set(keys.map((k) => columns[k].unit || 'unitless')),
  ]);
  let panes = $derived(
    settings.display === 'indexed'
      ? ['all']
      : settings.display === 'panes' || units.length > 2
        ? units
        : ['all'],
  );
  let plotHeight = $derived(Math.max(height, panes.length * 110 + 32));
  let paneHeight = $derived(
    Math.max(90, (Math.max(140, plotHeight) - 32) / Math.max(1, panes.length)),
  );
  let plotRight = $derived(Math.max(210, width) - 58);
  let mark = $derived(
    indices.indexOf(t.findIndex((time) => time === selected)),
  );
  let baselines = $derived(
    Object.fromEntries(
      keys.map((k) => [
        k,
        columns[k].values.find((v) => v !== null && v !== 0),
      ]),
    ),
  );
  function value(k: string, i: number) {
    const v = columns[k].values[i];
    if (v == null) return null;
    if (settings.display !== 'indexed') return v;
    const base = baselines[k];
    return base == null ? null : (v / base) * 100;
  }
  function x(j: number) {
    return 58 + (j * (plotRight - 58)) / Math.max(1, indices.length - 1);
  }
  let scales = $derived.by(() => {
    const bounds = new Map<string, [number, number]>();
    for (const k of keys)
      for (const i of indices) {
        const v = value(k, i);
        if (v === null) continue;
        for (const unit of ['all', columns[k].unit || 'unitless']) {
          const old = bounds.get(unit);
          bounds.set(
            unit,
            old ? [Math.min(old[0], v), Math.max(old[1], v)] : [v, v],
          );
        }
      }
    for (const [unit, [low, high]] of bounds) {
      const pad = (high - low || Math.abs(high) * 0.1 || 1) * 0.08;
      bounds.set(unit, [low - pad, high + pad]);
    }
    return bounds;
  });
  function extent(unit: string) {
    return scales.get(unit) ?? [0, 1];
  }
  function axis(k: string) {
    return settings.display === 'indexed'
      ? 'all'
      : columns[k].unit || 'unitless';
  }
  function y(v: number, k: string, p: number) {
    const [lo, hi] = extent(axis(k));
    return (
      p * paneHeight +
      paneHeight -
      24 -
      ((v - lo) / (hi - lo)) * (paneHeight - 48)
    );
  }
  function path(k: string, p: number) {
    let output = '',
      gap = true;
    for (let j = 0; j < indices.length; j++) {
      const v = value(k, indices[j]);
      if (v == null) {
        gap = true;
        continue;
      }
      output += `${gap ? 'M' : 'L'}${x(j)},${y(v, k, p)} `;
      gap = false;
    }
    return output;
  }
  let paths = $derived(
    panes.map((pane, p) =>
      Object.fromEntries(
        keys
          .filter(
            (k) => pane === 'all' || (columns[k].unit || 'unitless') === pane,
          )
          .map((k) => [k, path(k, p)]),
      ),
    ),
  );
  function indexAt(event: PointerEvent) {
    const box = (event.currentTarget as Element).getBoundingClientRect();
    return Math.max(
      0,
      Math.min(
        indices.length - 1,
        Math.round(
          ((((event.clientX - box.left) / box.width) * width - 58) /
            (plotRight - 58)) *
            Math.max(1, indices.length - 1),
        ),
      ),
    );
  }
  function move(event: PointerEvent) {
    const j = indexAt(event);
    if (t[indices[j]]) onselect(t[indices[j]]);
    if (drag !== null) end = j;
  }
  function finish() {
    if (drag !== null && end !== null && Math.abs(drag - end) > 1) {
      const a = Date.parse(t[indices[Math.min(drag, end)]]),
        b = Date.parse(t[indices[Math.max(drag, end)]]);
      onzoom([a, b]);
    }
    if (
      drag !== null &&
      (end === null || Math.abs(drag - end) <= 1) &&
      t[indices[drag]]
    )
      onpin(t[indices[drag]]);
    drag = null;
    end = null;
  }
  function keyboard(event: KeyboardEvent) {
    let j = mark < 0 ? 0 : mark;
    if (event.key === 'ArrowLeft') j--;
    else if (event.key === 'ArrowRight') j++;
    else if (event.key === 'Home') j = 0;
    else if (event.key === 'End') j = indices.length - 1;
    else return;
    event.preventDefault();
    if (indices.length) {
      const time = t[indices[Math.max(0, Math.min(indices.length - 1, j))]];
      onselect(time);
      onpin(time);
    }
  }
  const time = (s: string) =>
    new Date(s).toLocaleString('en-GB', {
      timeZone: 'Europe/London',
      day: '2-digit',
      month: 'short',
      hour: '2-digit',
      minute: '2-digit',
      timeZoneName: 'short',
    });
</script>

<div class="terminal-chart-legend">
  {#each keys as k, i}<span
      ><i style={`background:${colour(k)}`}></i>{columns[k].label}
      <small
        >{settings.display === 'indexed'
          ? 'index'
          : columns[k].unit || 'unitless'}</small
      ></span
    >{/each}{#if zoom}<button onclick={() => onzoom(null)}>Reset zoom</button
    >{/if}
</div>
{#if settings.display === 'indexed'}<p class="terminal-hint">
    Indexed to first non-zero value = 100 · raw values shown below. All-zero
    series cannot be indexed.
  </p>{/if}
{#if (settings.outputs.length || Object.keys(columns).length) > 12}<p
    class="terminal-hint"
  >
    Showing up to 12 series. Choose visible outputs in settings.
  </p>{/if}
<div class="terminal-plot" bind:clientWidth={width} bind:clientHeight={height}>
  {#if keys.length && indices.length}
    <div
      role="slider"
      tabindex="0"
      aria-label="Chart selected period"
      aria-valuemin={0}
      aria-valuemax={Math.max(0, indices.length - 1)}
      aria-valuenow={Math.max(0, mark)}
      aria-valuetext={selected ? time(selected) : 'No selected period'}
      onkeydown={keyboard}
      class="terminal-chart-interaction"
      style={`height:${plotHeight}px`}
      onpointermove={move}
      onpointerdown={(e) => {
        drag = indexAt(e);
        end = drag;
        e.currentTarget.setPointerCapture(e.pointerId);
        move(e);
      }}
      onpointerup={finish}
      onpointercancel={() => {
        drag = null;
        end = null;
      }}
    >
      <svg
        viewBox={`0 0 ${Math.max(210, width)} ${Math.max(140, plotHeight)}`}
        role="img"
        aria-label="Time series. Drag to zoom; arrow keys select a period."
      >
        {#each panes as pane, p}
          {@const axes =
            settings.display === 'indexed'
              ? ['all']
              : pane === 'all'
                ? units
                : [pane]}
          {#each axes as unit, a}
            {@const limits = extent(unit)}
            <text
              x={a === 0 ? 58 : plotRight}
              y={p * paneHeight + 13}
              text-anchor={a === 0 ? 'start' : 'end'}
              class="terminal-axis"
              >{settings.display === 'indexed' ? 'Index (100)' : unit}</text
            >
            {#each [0, 0.5, 1] as fraction}
              {@const yy = p * paneHeight + 24 + fraction * (paneHeight - 48)}
              {#if a === 0}<line
                  x1="58"
                  x2={plotRight}
                  y1={yy}
                  y2={yy}
                  class="terminal-gridline"
                />{/if}
              <text
                x={a === 0 ? 50 : plotRight + 7}
                y={yy + 3}
                text-anchor={a === 0 ? 'end' : 'start'}
                class="terminal-axis"
                >{(
                  limits[1] -
                  fraction * (limits[1] - limits[0])
                ).toLocaleString('en-GB', {
                  maximumFractionDigits: 1,
                  notation:
                    Math.abs(limits[1]) > 99999 ? 'compact' : 'standard',
                })}</text
              >
            {/each}
          {/each}
          {#each keys.filter((k) => pane === 'all' || (columns[k].unit || 'unitless') === pane) as k}
            {@const color = colour(k)}
            <path
              data-terminal-series={k}
              d={paths[p][k] ?? ''}
              fill="none"
              stroke={color}
              stroke-width="1.8"
              stroke-dasharray={settings.styles[k]?.dashed ? '5 4' : undefined}
            />
            {#if mark >= 0 && value(k, indices[mark]) !== null}<circle
                cx={x(mark)}
                cy={y(value(k, indices[mark])!, k, p)}
                r="3"
                fill={color}
              />{/if}
          {/each}
        {/each}
        {#if mark >= 0}<line
            x1={x(mark)}
            x2={x(mark)}
            y1="20"
            y2={panes.length * paneHeight - 20}
            stroke="var(--orange)"
            stroke-dasharray="3 4"
          />{/if}
        {#if drag !== null && end !== null}<rect
            x={x(Math.min(drag, end))}
            y="20"
            width={Math.abs(x(drag) - x(end))}
            height={Math.max(1, panes.length * paneHeight - 40)}
            fill="#e6b08022"
          />{/if}
        {#each [0, indices.length - 1] as endpoint}
          {@const labelX = endpoint === 0 ? 58 : plotRight}
          {@const instant = t[indices[endpoint]]}
          <text
            text-anchor={endpoint === 0 ? 'start' : 'end'}
            class="terminal-axis"
            ><tspan x={labelX} y={Math.max(140, plotHeight) - 17}
              >{new Date(instant).toLocaleDateString('en-GB', {
                timeZone: 'Europe/London',
                day: '2-digit',
                month: 'short',
              })}</tspan
            ><tspan x={labelX} y={Math.max(140, plotHeight) - 4}
              >{new Date(instant).toLocaleTimeString('en-GB', {
                timeZone: 'Europe/London',
                hour: '2-digit',
                minute: '2-digit',
                timeZoneName: 'short',
              })}</tspan
            ></text
          >
        {/each}
      </svg>
    </div>
  {:else}<div class="terminal-empty">
      {keys.length
        ? 'No readings in the visible range.'
        : 'Choose outputs in widget settings.'}
    </div>{/if}
</div>
{#if selected}<div class="terminal-chart-values">
    {#each keys as k}<span
        >{columns[k].label}:
        <strong
          >{columns[k].values[t.indexOf(selected)]?.toLocaleString('en-GB', {
            maximumFractionDigits: 2,
          }) ?? 'No data'}</strong
        >
        {columns[k].unit}</span
      >{/each}
  </div>{/if}
