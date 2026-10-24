<script lang="ts">
  import { onMount, getContext } from 'svelte';
  import {
    api,
    type MarketSeries,
    type WeatherLocation,
    type WeatherResponse,
  } from '$lib/api';
  import {
    readPreference,
    writePreference,
    type ViewPreference,
  } from '$lib/state/view.svelte';
  import Card from '$lib/vendor/kcpc/primitives/card/card.svelte';
  import TimeSeriesChart from '../../components/charts/TimeSeriesChart.svelte';
  import InfoTooltip from '../../components/ui/InfoTooltip.svelte';
  import DayScan from './DayScan.svelte';
  import SeriesSelector from './SeriesSelector.svelte';
  import DaySummary from './DaySummary.svelte';
  import {
    METRICS,
    getMetric,
    disabledReason,
    withWeather,
    dailySummary,
    formatValue,
    dayIndices,
    previousDay,
    timeLabel,
  } from './model';
  const view = getContext<ViewPreference>('gridpulse:view');
  let data = $state<MarketSeries | null>(null);
  let latest = $state('');
  let day = $state('');
  let selected = $state(0);
  let loading = $state(true);
  let error = $state('');
  let updated = $state<string | null>(null);
  let mode = $state('');
  let locations = $state<WeatherLocation[]>([]);
  let weather = $state<WeatherResponse | null>(null);
  let weatherLoading = $state(false);
  let weatherError = $state('');
  let controller: AbortController | null = null;
  let weatherController: AbortController | null = null;
  let indices = $derived(dayIndices(data, day));
  let chartData = $derived(withWeather(data, weather));
  let selectedMetrics = $derived(view.metrics.map(getMetric));
  let hasValues = $derived(
    indices.some((i) =>
      selectedMetrics.some((m) => chartData?.series[m.id]?.values[i] != null),
    ),
  );
  let timestamp = $derived(data?.t[selected] ?? null);
  function persist() {
    writePreference(localStorage, view);
  }
  async function loadWeather() {
    weatherController?.abort();
    const current = new AbortController();
    weatherController = current;
    weather = null;
    weatherError = '';
    if (!view.location || !day) return;
    weatherLoading = true;
    try {
      const response = await api.weather(day, view.location, current.signal);
      if (!current.signal.aborted) weather = response;
    } catch (e) {
      if (current.signal.aborted) return;
      weatherError = e instanceof Error ? e.message : 'Weather is unavailable.';
    } finally {
      if (!current.signal.aborted) weatherLoading = false;
    }
  }
  async function loadDay(date: string) {
    controller?.abort();
    const current = new AbortController();
    controller = current;
    loading = true;
    error = '';
    data = null;
    day = date;
    void loadWeather();
    try {
      const frame = await api.series(
        previousDay(date),
        date,
        METRICS.filter((m) => m.source === 'market').map((m) => m.id),
        current.signal,
      );
      if (current.signal.aborted) return;
      data = frame;
      const candidates = dayIndices(frame, date);
      selected =
        candidates
          .filter((i) => frame.series.price?.values[i] != null)
          .at(-1) ??
        candidates[0] ??
        0;
    } catch (e) {
      if (current.signal.aborted) return;
      error = e instanceof Error ? e.message : 'Could not load this day.';
    } finally {
      if (!current.signal.aborted) loading = false;
    }
  }
  function chooseDay(date: string) {
    if (!date) return;
    view.date = date;
    persist();
    void loadDay(date);
  }
  function chooseLocation(id: string) {
    view.location = id;
    persist();
    void loadWeather();
  }
  async function latestDay() {
    view.date = null;
    persist();
    await initialize();
  }
  async function initialize() {
    loading = true;
    error = '';
    try {
      const snapshot = await api.snapshot();
      latest = snapshot.settlement_date;
      updated = snapshot.data_updated_at_utc;
      await loadDay(view.date ?? latest);
    } catch (e) {
      loading = false;
      error = e instanceof Error ? e.message : 'Market data is unavailable.';
    }
  }
  function toggle(id: string) {
    if (disabledReason(id, view.metrics)) return;
    view.metrics = view.metrics.includes(id)
      ? view.metrics.filter((current) => current !== id)
      : [...view.metrics, id];
    if (!view.metrics.includes(view.scanMetric))
      view.scanMetric = view.metrics[0];
    persist();
  }
  function chooseScan(id: string) {
    if (view.metrics.includes(id)) {
      view.scanMetric = id;
      persist();
    }
  }
  onMount(() => {
    Object.assign(view, readPreference(localStorage));
    void initialize();
    void api
      .health()
      .then((h) => (mode = h.mode))
      .catch(() => {});
    void api
      .locations()
      .then((response) => {
        locations = response.locations;
        if (!locations.some((l) => l.id === view.location))
          view.location = locations[0]?.id ?? null;
        persist();
        if (!locations.length)
          weatherError =
            response.warnings.join(' ') || 'No weather data is loaded.';
        void loadWeather();
      })
      .catch(
        (e) =>
          (weatherError =
            e instanceof Error ? e.message : 'Weather locations unavailable.'),
      );
    const onStorage = (e: StorageEvent) => {
      if (e.key === 'gridpulse:view:v1') {
        const p = readPreference(localStorage);
        Object.assign(view, p);
        if (latest) void loadDay(view.date ?? latest);
      }
    };
    window.addEventListener('storage', onStorage);
    return () => {
      controller?.abort();
      weatherController?.abort();
      window.removeEventListener('storage', onStorage);
    };
  });
</script>

<a class="skip-link" href="#market">Skip to market overview</a>
<main class="page-shell overview-page">
  <section id="market" class="market-section" aria-busy={loading}>
    <div class="market-toolbar">
      <div>
        <p class="eyebrow">GB ELECTRICITY</p>
        <h1>Market overview<span class="heading-dot">.</span></h1>
      </div>
      <div class="date-controls">
        <button
          class="quiet-button"
          onclick={() => chooseDay(previousDay(day))}
          disabled={!day || loading}
          aria-label="Previous settlement day">←</button
        ><label class="date-field"
          ><span class="sr-only">Settlement date</span><input
            type="date"
            value={day}
            max={latest || undefined}
            onchange={(e) => chooseDay(e.currentTarget.value)}
          /></label
        ><button
          class="quiet-button"
          onclick={() => {
            const d = new Date(day + 'T12:00Z');
            d.setUTCDate(d.getUTCDate() + 1);
            chooseDay(d.toISOString().slice(0, 10));
          }}
          disabled={!day || day >= latest || loading}
          aria-label="Next settlement day">→</button
        ><button
          class="latest-button"
          class:chosen={view.date === null}
          onclick={latestDay}>Latest available ↗</button
        >
      </div>
    </div>
    <div class="data-status">
      <span
        >{mode === 'replay'
          ? 'CACHED REPLAY'
          : mode === 'record'
            ? 'LIVE · RECORDING'
            : mode === 'live'
              ? 'LIVE DATABASE'
              : 'CONNECTING'}</span
      ><span>Settlement day: Europe/London · chart times: UTC</span><span
        >{updated
          ? 'Data loaded ' +
            new Date(updated).toLocaleString('en-GB', { timeZone: 'UTC' }) +
            ' UTC'
          : 'Freshness unavailable'}</span
      >
    </div>
    {#if error}<div class="notice error" role="alert">
        <h3>We couldn't load the market data.</h3>
        <p>{error}</p>
        <button class="quiet-button" onclick={initialize}>Try again ↻</button>
      </div>{/if}
    <div class="metric-grid" style={`--metric-count:${view.metrics.length}`}>
      {#each selectedMetrics as metric}
        {@const value = chartData?.series[metric.id]?.values[selected]}
        {@const previous =
          chartData?.series[metric.id]?.values[
            selected - (metric.source === 'weather' ? 2 : 1)
          ]}
        {@const delta =
          value != null && previous != null ? value - previous : null}
        {@const daily = chartData
          ? dailySummary(chartData, indices, metric.id)
          : { average: null, count: 0, total: indices.length }}
        <Card class="metric-card"
          ><div class="metric-label">
            <span class="metric-dot" style={`background:${metric.color}`}
            ></span><span class="metric-name">{metric.label}</span><InfoTooltip
              text={`${metric.description} Daily AVG uses ${daily.count} of ${daily.total} available settlement-period values. Missing values are excluded; hourly weather is carried across its two settlement periods.`}
              label={`About ${metric.label.toLowerCase()}`}
            /><span class="tiny">{metric.displayUnit}</span>
          </div>
          <div class="metric-reading">
            <div
              class="metric-value"
              class:skeleton={loading}
              class:no-data={!loading && value == null}
            >
              {loading ? '—' : formatValue(value, metric)}
            </div>
            <span class="daily-average"
              >{formatValue(daily.average, metric)}
              <small>Daily AVG</small></span
            >
          </div>
          <div class="metric-bottom">
            <span class="metric-delta"
              >{delta === null
                ? '—'
                : `${delta > 0 ? '+' : ''}${formatValue(delta, metric)}`}
              <span class="muted"
                >{metric.source === 'weather'
                  ? 'vs prior hour'
                  : 'vs prior period'}</span
              ></span
            ><span class="tiny muted"
              >{timestamp
                ? timeLabel(
                    metric.source === 'weather'
                      ? new Date(
                          Math.floor(Date.parse(timestamp) / 3600000) * 3600000,
                        ).toISOString()
                      : timestamp,
                  ) + ' UTC'
                : 'No selection'}</span
            >
          </div>
        </Card>
      {/each}
    </div>
    <div class="workspace-grid">
      <div class="market-charts">
        <section class="panel chart-panel">
          <div class="section-heading">
            <div>
              <h2>
                Selected signals <InfoTooltip
                  label="About the combined chart"
                  text="Up to two independent unit scales are shown, matching the selected properties. Line crossings across different unit groups do not imply equal quantities. Hover, tap or use arrow keys to select a period."
                />
              </h2>
            </div>
            <span class="time-badge"
              >{timestamp ? timeLabel(timestamp) + ' UTC' : '—'}</span
            >
          </div>
          {#if loading}<div class="chart-placeholder" role="status">
              <span class="loader"></span>Loading market series…
            </div>
          {:else if chartData && indices.length && hasValues}
            <div class="chart-legend">
              {#each selectedMetrics as metric}<span
                  ><span class="metric-dot" style={`background:${metric.color}`}
                  ></span>{metric.shortLabel}<small>{metric.displayUnit}</small
                  ></span
                >{/each}
            </div>
            <TimeSeriesChart
              data={chartData}
              {indices}
              {selected}
              onselect={(i) => (selected = i)}
              ids={view.metrics}
              title="Selected signals"
            />
          {:else}<div class="chart-placeholder">
              {error
                ? 'Restore the backend connection to see real data.'
                : 'No data for the selected properties on this day.'}
            </div>{/if}
        </section>
        <section class="panel scan-panel">
          {#if chartData && indices.length}<DayScan
              data={chartData}
              {indices}
              {selected}
              onselect={(i) => (selected = i)}
              metric={getMetric(view.scanMetric)}
              metrics={selectedMetrics}
              onmetric={chooseScan}
            />{:else}<h2>Scan the day.</h2>
            <p class="muted small">
              The period scan appears when settlement data is available.
            </p>{/if}
        </section>
      </div>
      <div class="overview-sidebar">
        <SeriesSelector
          selection={view.metrics}
          ontoggle={toggle}
          {locations}
          location={view.location}
          onlocation={chooseLocation}
          {weatherError}
        />
        <DaySummary
          data={chartData}
          {indices}
          metrics={selectedMetrics}
          {loading}
        />
      </div>
    </div>
  </section>
</main>
