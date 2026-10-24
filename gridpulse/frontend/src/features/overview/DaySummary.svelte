<script lang="ts">
  import type { MarketSeries } from '$lib/api';
  import InfoTooltip from '../../components/ui/InfoTooltip.svelte';
  import {
    dailySummary,
    formatValue,
    getMetric,
    timeLabel,
    type MetricDefinition,
  } from './model';
  let {
    data,
    indices,
    metrics,
    loading,
  }: {
    data: MarketSeries | null;
    indices: number[];
    metrics: MetricDefinition[];
    loading: boolean;
  } = $props();
  function readings(id: string) {
    return indices.flatMap((i) => {
      const value = data?.series[id]?.values[i];
      return value != null && Number.isFinite(value) ? [{ value, i }] : [];
    });
  }
  let prices = $derived(readings('price'));
  let minimum = $derived(
    prices.reduce<(typeof prices)[number] | null>(
      (best, p) => (!best || p.value < best.value ? p : best),
      null,
    ),
  );
  let maximum = $derived(
    prices.reduce<(typeof prices)[number] | null>(
      (best, p) => (!best || p.value > best.value ? p : best),
      null,
    ),
  );
  let negative = $derived(prices.filter((p) => p.value < 0).length);
  let demand = $derived(readings('demand'));
  let peak = $derived(
    demand.reduce<(typeof demand)[number] | null>(
      (best, p) => (!best || p.value > best.value ? p : best),
      null,
    ),
  );
  function reading(point: { value: number; i: number } | null, id: string) {
    if (loading) return '—';
    if (!point || !data) return 'No data';
    return `${formatValue(point.value, getMetric(id))} ${getMetric(id).displayUnit} · ${timeLabel(data.t[point.i])} UTC`;
  }
</script>

<section class="panel day-summary" aria-label="Day at a glance">
  <h2>
    Day at a glance <InfoTooltip
      label="About daily statistics"
      text="Statistics cover available records for the selected settlement day, independently of hover. Averages exclude missing values. Price extremes show the first matching UTC time. Each negative-price settlement period lasts 30 minutes, including clock-change days. Peak demand uses forecast demand, not actual consumption. Incomplete days describe available data only."
    />
  </h2>
  <div class="day-summary-body">
    <div class="day-averages">
      {#each metrics as metric}
        {@const daily = data ? dailySummary(data, indices, metric.id) : null}
        <div class="day-average">
          <span>{metric.shortLabel} AVG</span><strong
            >{loading ? '—' : formatValue(daily?.average, metric)}
            <small>{metric.displayUnit}</small></strong
          >
        </div>
      {/each}
    </div>
    <dl class="day-facts">
      <div>
        <dt>Lowest price</dt>
        <dd>{reading(minimum, 'price')}</dd>
      </div>
      <div>
        <dt>Highest price</dt>
        <dd>{reading(maximum, 'price')}</dd>
      </div>
      <div>
        <dt>Negative prices</dt>
        <dd>
          {loading
            ? '—'
            : prices.length
              ? `${negative} periods · ${negative / 2} h`
              : 'No data'}
        </dd>
      </div>
      <div>
        <dt>Peak forecast demand</dt>
        <dd>{reading(peak, 'demand')}</dd>
      </div>
    </dl>
    <p class="day-coverage">
      {loading
        ? 'Loading day…'
        : `${prices.length}/${indices.length} price periods available`}
    </p>
  </div>
</section>
