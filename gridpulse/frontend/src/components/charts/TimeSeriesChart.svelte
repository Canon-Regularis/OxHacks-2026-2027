<script lang="ts">
  import type { MarketSeries } from '$lib/api';
  import {
    getMetric,
    timeLabel,
    type UnitGroup,
  } from '../../features/overview/model';
  let {
    data,
    indices,
    selected,
    onselect,
    ids,
    title,
  }: {
    data: MarketSeries;
    indices: number[];
    selected: number;
    onselect: (index: number) => void;
    ids: string[];
    title: string;
  } = $props();
  let chartWidth = $state(800);
  let chartHeight = $state(250);
  let width = $derived(Math.max(300, chartWidth));
  let height = $derived(Math.max(160, chartHeight));
  let box = $derived({
    left: 54,
    right: width - 53,
    top: 25,
    bottom: height - 30,
  });
  const order: UnitGroup[] = [
    'price',
    'power',
    'energy',
    'temperature',
    'pressure',
    'correlation',
  ];
  const labels: Record<UnitGroup, string> = {
    price: 'Price',
    power: 'Power',
    energy: 'Energy',
    temperature: 'Temperature',
    pressure: 'Pressure',
    correlation: 'Correlation',
  };
  let axes = $derived(
    order
      .filter((group) => ids.some((id) => getMetric(id).unitGroup === group))
      .map((group) => {
        const metrics = ids.map(getMetric).filter((m) => m.unitGroup === group);
        const values = metrics.flatMap((m) =>
          indices
            .map((i) => data.series[m.id]?.values[i])
            .filter((v): v is number => v != null && Number.isFinite(v))
            .map((v) => v / m.divisor),
        );
        const zero = ['price', 'power', 'energy'].includes(group);
        let low = values.length ? Math.min(...values) : 0;
        let high = values.length ? Math.max(...values) : 1;
        if (zero) {
          low = Math.min(0, low);
          high = Math.max(0, high);
        }
        if (group === 'correlation') {
          low = -1;
          high = 1;
        }
        const padding = (high - low || Math.abs(high) * 0.1 || 1) * 0.08;
        return {
          group,
          unit: metrics[0].displayUnit,
          color: metrics[0].color,
          label: labels[group],
          low: low - padding,
          high: high + padding,
        };
      }),
  );
  function axis(id: string) {
    return axes.find((a) => a.group === getMetric(id).unitGroup)!;
  }
  function x(j: number) {
    return (
      box.left + (j * (box.right - box.left)) / Math.max(1, indices.length - 1)
    );
  }
  function y(value: number, id: string) {
    const a = axis(id);
    return (
      box.bottom -
      ((value / getMetric(id).divisor - a.low) / (a.high - a.low)) *
        (box.bottom - box.top)
    );
  }
  function path(id: string) {
    let p = '';
    let gap = true;
    for (let j = 0; j < indices.length; j++) {
      const v = data.series[id]?.values[indices[j]];
      if (v == null || !Number.isFinite(v)) {
        gap = true;
        continue;
      }
      p += gap
        ? `M${x(j)},${y(v, id)} `
        : getMetric(id).source === 'weather'
          ? `H${x(j)} V${y(v, id)} `
          : `L${x(j)},${y(v, id)} `;
      gap = false;
    }
    return p;
  }
  function pick(event: PointerEvent) {
    const bounds = (event.currentTarget as HTMLElement).getBoundingClientRect();
    const pos =
      (((event.clientX - bounds.left) / bounds.width) * width - box.left) /
      (box.right - box.left);
    onselect(
      indices[
        Math.max(
          0,
          Math.min(indices.length - 1, Math.round(pos * (indices.length - 1))),
        )
      ],
    );
  }
  function keyboard(event: KeyboardEvent) {
    let j = indices.indexOf(selected);
    if (event.key === 'ArrowRight') j++;
    else if (event.key === 'ArrowLeft') j--;
    else if (event.key === 'Home') j = 0;
    else if (event.key === 'End') j = indices.length - 1;
    else return;
    event.preventDefault();
    onselect(indices[Math.max(0, Math.min(indices.length - 1, j))]);
  }
  let position = $derived(indices.indexOf(selected));
  function tickValue(value: number) {
    return value.toLocaleString('en-GB', {
      maximumFractionDigits: Math.abs(value) < 10 ? 1 : 0,
    });
  }
</script>

<div
  class="chart-control combined-chart"
  bind:clientWidth={chartWidth}
  bind:clientHeight={chartHeight}
  role="slider"
  tabindex="0"
  aria-label={`${title}, select a settlement period`}
  aria-valuemin="1"
  aria-valuemax={indices.length || 1}
  aria-valuenow={Math.max(1, position + 1)}
  aria-valuetext={data.t[selected] ? `${timeLabel(data.t[selected])} UTC` : ''}
  onpointermove={pick}
  onpointerdown={pick}
  onkeydown={keyboard}
>
  <svg
    viewBox={`0 0 ${width} ${height}`}
    role="img"
    data-axis-left={axes[0]?.unit ?? ''}
    data-axis-right={axes[1]?.unit ?? ''}
    aria-label={`Selected signals. ${axes.map((a, i) => `${i === 0 ? 'Left' : 'Right'} axis: ${a.label}, ${a.unit}`).join('. ')}. Missing values are gaps.`}
  >
    {#each axes as a, i}<text
        x={i === 0 ? box.left : box.right}
        y="13"
        text-anchor={i === 0 ? 'start' : 'end'}
        class="axis-label"
        fill={a.color}>{a.label} · {a.unit}</text
      >{/each}
    {#each [0, 0.5, 1] as ratio}
      {@const gridY = box.bottom - ratio * (box.bottom - box.top)}
      <line
        x1={box.left}
        x2={box.right}
        y1={gridY}
        y2={gridY}
        stroke="var(--border)"
      />
      {#each axes as a, i}<text
          x={i === 0 ? box.left - 10 : box.right + 10}
          y={gridY + 4}
          text-anchor={i === 0 ? 'end' : 'start'}
          class="axis-label"
          fill={a.color}>{tickValue(a.low + (a.high - a.low) * ratio)}</text
        >{/each}
    {/each}
    {#each ids as id}<path
        data-series={id}
        d={path(id)}
        fill="none"
        stroke={getMetric(id).color}
        stroke-width={id === 'price' ? 2.5 : 1.8}
        vector-effect="non-scaling-stroke"
        stroke-linecap="round"
        stroke-linejoin="round"
        stroke-dasharray={id.endsWith('_outturn')
          ? '6 3'
          : id === 'residual_demand'
            ? '3 3'
            : undefined}
      />{/each}
    {#if position >= 0}
      <line
        x1={x(position)}
        x2={x(position)}
        y1={box.top}
        y2={box.bottom}
        stroke="var(--orange)"
        stroke-dasharray="3 4"
        opacity=".6"
      />
      {#each ids as id}{@const value =
          data.series[id]?.values[selected]}{#if value != null}<circle
            cx={x(position)}
            cy={y(value, id)}
            r="3.5"
            fill={getMetric(id).color}
            stroke="var(--background)"
            stroke-width="1.5"
          />{/if}{/each}
    {/if}
    {#each [0, 0.25, 0.5, 0.75, 1] as ratio}{@const j = Math.round(
        (indices.length - 1) * ratio,
      )}{#if indices[j] != null}<text
          x={x(j)}
          y={height - 8}
          text-anchor={ratio === 0 ? 'start' : ratio === 1 ? 'end' : 'middle'}
          class="axis-label">{timeLabel(data.t[indices[j]])}</text
        >{/if}{/each}
  </svg>
</div>
