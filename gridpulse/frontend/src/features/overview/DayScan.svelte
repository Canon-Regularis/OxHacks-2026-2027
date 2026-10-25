<script lang="ts">
  import type { MarketSeries } from '$lib/api';
  import { seriesColor, formatValue, type MetricDefinition } from './model';
  import InfoTooltip from '../../components/ui/InfoTooltip.svelte';
  import { useSettings } from '$lib/state/settings.svelte';
  import { formatTime } from '$lib/state/settings';
  const settings = useSettings();
  const timeLabel = (t: string) => formatTime(t, settings.timeDisplay);
  let {
    data,
    indices,
    selected,
    onselect,
    metric,
    metrics,
    onmetric,
  }: {
    data: MarketSeries;
    indices: number[];
    selected: number;
    onselect: (i: number) => void;
    metric: MetricDefinition;
    metrics: MetricDefinition[];
    onmetric: (id: string) => void;
  } = $props();
  let values = $derived(
    indices.map((i) => data.series[metric.id]?.values[i] ?? null),
  );
  let valid = $derived(
    values.filter((n): n is number => n !== null && Number.isFinite(n)),
  );
  let min = $derived(valid.length ? Math.min(...valid) : 0);
  let max = $derived(valid.length ? Math.max(...valid) : 0);
  let position = $derived(indices.indexOf(selected));
  let lowColor = $derived(seriesColor(min, min, max, metric));
  let highColor = $derived(seriesColor(max, min, max, metric));
</script>

<div class="section-heading scan-heading">
  <h2>
    Scan the day <InfoTooltip
      label="About the day scan"
      text={`Colours show ${metric.label.toLowerCase()} from this day's minimum to maximum, not a risk score. Units are ${metric.displayUnit}. Grey marks missing values. The dropdown only lists properties selected for the graph. Hover centres the focus; touch and keyboard also work.`}
    />
  </h2>
  <div class="scan-controls">
    <label class="sr-only" for="scan-property">Scan property</label><select
      id="scan-property"
      value={metric.id}
      onchange={(e) => onmetric(e.currentTarget.value)}
      >{#each metrics as m}<option value={m.id}>{m.label}</option
        >{/each}</select
    ><span class="subtle">{indices.length} periods · {metric.displayUnit}</span>
  </div>
</div>
<div class="day-scan" style={`--count:${indices.length};--focus:${position}`}>
  {#each indices as index, j}<button
      class="scan-period"
      class:active={index === selected}
      style={`--heat:${seriesColor(values[j], min, max, metric)};--distance:${Math.min(8, Math.abs(j - position))}`}
      aria-label={`${timeLabel(data.t[index])}, ${metric.label}, ${formatValue(values[j], metric)} ${values[j] === null ? '' : metric.displayUnit}`}
      aria-pressed={index === selected}
      onpointerenter={() => onselect(index)}
      onfocus={() => onselect(index)}
      onclick={() => onselect(index)}><span></span></button
    >{/each}
</div>
<div class="scan-bottom">
  <span
    >{indices[0] != null ? timeLabel(data.t[indices[0]]) : '—'} — {indices.at(
      -1,
    ) != null
      ? timeLabel(data.t[indices.at(-1)!])
      : '—'}</span
  ><strong
    >{data.t[selected] ? timeLabel(data.t[selected]) : '—'} · selected</strong
  >
  <div class="price-legend">
    <span>{valid.length ? formatValue(min, metric) : 'No data'}</span>
    <div
      style={`background:linear-gradient(90deg,${lowColor},${highColor})`}
    ></div>
    <span
      >{valid.length ? formatValue(max, metric) : 'No data'}
      {metric.displayUnit}</span
    ><span class="missing-key">▧ Missing</span>
  </div>
</div>
