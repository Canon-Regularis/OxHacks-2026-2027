<script lang="ts">
  import type { WeatherLocation } from '$lib/api';
  import { METRICS, disabledReason, getMetric } from './model';
  import InfoTooltip from '../../components/ui/InfoTooltip.svelte';
  let {
    selection,
    ontoggle,
    locations,
    location,
    onlocation,
    weatherError,
  }: {
    selection: string[];
    ontoggle: (id: string) => void;
    locations: WeatherLocation[];
    location: string | null;
    onlocation: (id: string) => void;
    weatherError: string;
  } = $props();
  const sections = [...new Set(METRICS.map((metric) => metric.section))];
  let needsWeather = $derived(
    selection.some((id) => getMetric(id).source === 'weather'),
  );
</script>

<aside class="panel series-selector" aria-label="Chart data selector">
  <div class="selector-heading">
    <h2>
      Chart data <InfoTooltip
        label="About selecting chart data"
        text="Select one to five properties across at most two unit groups. Their units are shown next to each option. Choices that exceed a limit are disabled. Each selected property gets a card and chart line; missing values stay missing."
      />
    </h2>
    <span class="selection-count">{selection.length} / 5</span>
  </div>
  <p class="selector-rule">1–5 properties · up to 2 unit groups</p>
  <div class="selector-options">
    {#each sections as section}<fieldset>
        <legend>{section}</legend>
        {#each METRICS.filter((m) => m.section === section) as metric}
          {@const selected = selection.includes(metric.id)}{@const reason =
            disabledReason(metric.id, selection)}
          <div
            class="selection-row"
            class:blocked={!!reason && !selected}
            class:checked={selected}
          >
            <label title={reason ?? metric.label}
              ><input
                type="checkbox"
                value={metric.id}
                checked={selected}
                disabled={!!reason}
                onchange={() => ontoggle(metric.id)}
              /><span class="metric-dot" style={`background:${metric.color}`}
              ></span><span class="selection-label">{metric.label}</span><span
                class="selection-unit">{metric.displayUnit}</span
              ></label
            >
            <InfoTooltip
              label={`About ${metric.label.toLowerCase()} option`}
              text={`${metric.description}${reason ? ' ' + reason : ''}`}
            />
          </div>
        {/each}
      </fieldset>{/each}
  </div>
  {#if needsWeather}<div class="selector-location">
      <label for="weather-location">Weather location</label><select
        id="weather-location"
        value={location ?? ''}
        disabled={!locations.length}
        onchange={(e) => onlocation(e.currentTarget.value)}
        >{#if !locations.length}<option value="">No locations available</option
          >{/if}{#each locations as l}<option value={l.id}>{l.label}</option
          >{/each}</select
      ><span class="tiny muted">Archived forecasts · completed UTC hours</span
      >{#if weatherError}<p class="small error" role="status">
          {weatherError}
        </p>{/if}
    </div>{/if}
</aside>
