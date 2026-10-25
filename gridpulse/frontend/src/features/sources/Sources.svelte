<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '$lib/api';
  import type { Catalog, Health } from '$lib/types';
  import { useSettings } from '$lib/state/settings.svelte';
  import { formatDateTime } from '$lib/state/settings';
  import { sourceRows, coverageExtent, validInstant } from './model';
  const settings = useSettings();
  let catalog = $state<Catalog | null>(null);
  let health = $state<Health | null>(null);
  let loading = $state(true),
    error = $state(''),
    search = $state(''),
    kind = $state('All types');
  let controller: AbortController | undefined;
  const rows = $derived(sourceRows(catalog));
  const kinds = $derived([...new Set(rows.map((row) => row.kind))].sort());
  const filtered = $derived(
    rows.filter(
      (row) =>
        (kind === 'All types' || row.kind === kind) &&
        row.searchable.includes(search.trim().toLowerCase()),
    ),
  );
  const extent = $derived(coverageExtent(catalog?.datasets ?? []));
  const zone = $derived(
    settings.timeDisplay === 'utc' ? 'UTC' : 'UK local time',
  );
  const timestamp = (value: string | null) =>
    validInstant(value)
      ? formatDateTime(value, settings.timeDisplay)
      : 'Not supplied';
  const date = (value: string | null) =>
    validInstant(value)
      ? new Date(value).toLocaleDateString('en-GB', {
          day: '2-digit',
          month: 'short',
          year: 'numeric',
          timeZone: settings.timeDisplay === 'utc' ? 'UTC' : 'Europe/London',
        })
      : 'Not supplied';
  async function refresh() {
    controller?.abort();
    const mine = (controller = new AbortController());
    loading = true;
    error = '';
    const [data, state] = await Promise.allSettled([
      api.catalog(mine.signal),
      api.health(mine.signal),
    ]);
    if (mine.signal.aborted) return;
    if (data.status === 'fulfilled') catalog = data.value;
    else
      error =
        data.reason instanceof Error
          ? data.reason.message
          : 'The data catalogue could not be loaded.';
    health = state.status === 'fulfilled' ? state.value : null;
    loading = false;
  }
  onMount(() => {
    void refresh();
    return () => controller?.abort();
  });
</script>

<a class="skip-link" href="#sources-main">Skip to data sources</a>
<main class="page-shell learn-page sources-page" id="sources-main">
  <header class="learn-heading">
    <div>
      <p class="learn-eyebrow">THE EVIDENCE / DATA CATALOGUE</p>
      <h1>Know your <span>data.</span></h1>
      <p class="learn-intro">
        The electricity and weather records behind Eisengrau. Explore their
        timeframes, what they contain and where they appear in the app.
      </p>
    </div>
    <a class="learn-link" href="/quick-start"
      >New here? Take the quick start <span aria-hidden="true">↗</span></a
    >
  </header>
  <div class="sources-summary" aria-busy={loading}>
    <article class="learn-stat">
      <span>Stored datasets</span><strong
        >{catalog ? rows.length.toLocaleString('en-GB') : '…'}</strong
      ><small>Reported by the connected catalogue</small>
    </article>
    <article class="learn-stat sources-span">
      <span>Catalogue span · {zone}</span><strong
        >{date(extent.first)} <span aria-hidden="true">→</span>
        {date(extent.last)}</strong
      ><small>Earliest to latest record across all datasets</small>
    </article>
    <article class="learn-stat">
      <span>With chart series</span><strong
        >{catalog ? rows.filter((row) => row.series.length).length : '…'}<span>
          / {rows.length}</span
        ></strong
      ><small>Other stored records are listed too</small>
    </article>
  </div>
  <section
    class="sources-catalogue"
    aria-labelledby="sources-table-heading"
    aria-busy={loading}
  >
    <div class="sources-toolbar">
      <div>
        <h2 id="sources-table-heading">Inside the dataset</h2>
        <p>
          {catalog
            ? `${filtered.length} of ${rows.length} datasets`
            : 'Loading the catalogue…'} · timestamps in {zone}
        </p>
      </div>
      <button class="quiet-button" onclick={refresh} disabled={loading}
        >{loading ? 'Refreshing…' : 'Refresh catalogue'}</button
      >
    </div>
    <div class="sources-filters">
      <label
        >Find a dataset<input
          type="search"
          aria-label="Search datasets"
          placeholder="Search wind, price, weather…"
          bind:value={search}
        /></label
      >
      <label
        >Data type<select aria-label="Filter data type" bind:value={kind}
          ><option>All types</option>{#each kinds as type}<option>{type}</option
            >{/each}</select
        ></label
      >
      <span class="sources-mode"
        >{health?.mode === 'replay'
          ? 'Cached replay'
          : health?.mode === 'record'
            ? 'Recording mode'
            : health?.mode === 'live'
              ? 'Connected catalogue'
              : 'Connection mode unavailable'}</span
      >
    </div>
    {#if error}<div class="learn-notice" role="alert">
        <strong>Catalogue unavailable</strong>
        <p>{error}</p>
        {#if catalog}<p>Showing the last successfully loaded catalogue.</p>{/if}
      </div>{/if}
    {#if catalog && filtered.length}
      <!-- svelte-ignore a11y_no_noninteractive_tabindex (The scrollable table region needs keyboard access on narrow screens.) -->
      <div
        class="sources-table-wrap"
        role="region"
        aria-label="Scrollable dataset catalogue"
        tabindex="0"
      >
        <table class="sources-table">
          <caption class="sr-only"
            >Dataset types, coverage, chart availability, load times and
            approximate record counts. All timestamps use {zone}.</caption
          >
          <thead
            ><tr
              ><th scope="col">Dataset &amp; source</th><th scope="col"
                >Type &amp; frequency</th
              ><th scope="col">Available timeframe</th><th scope="col"
                >In the app</th
              ><th scope="col">Last loaded</th><th
                scope="col"
                class="sources-number">Approx. rows</th
              ></tr
            ></thead
          >
          <tbody
            >{#each filtered as row (row.dataset.dataset)}<tr>
                <th scope="row"
                  ><strong>{row.name}</strong><span class="sources-provider"
                    >{row.provider}</span
                  ><code>{row.dataset.dataset}</code>
                  <details>
                    <summary>Source notes</summary>
                    <p>{row.dataset.source ?? 'Source not supplied.'}</p>
                    <p>
                      {row.dataset.notes ?? 'No further source notes supplied.'}
                    </p>
                  </details></th
                >
                <td
                  ><span class="source-kind">{row.kind}</span><small
                    >{row.cadence}</small
                  ><small
                    >{row.units.length
                      ? row.units.join(' · ')
                      : 'Units not in chart catalogue'}</small
                  ></td
                >
                <td class="sources-time"
                  ><time datetime={row.dataset.first_utc ?? undefined}
                    >{timestamp(row.dataset.first_utc)}</time
                  ><span class="sources-until">to</span><time
                    datetime={row.dataset.last_utc ?? undefined}
                    >{timestamp(row.dataset.last_utc)}</time
                  ></td
                >
                <td
                  >{#if row.series.length}<strong
                      >{row.series.length} chart series</strong
                    >
                    <details>
                      <summary>View series</summary
                      >{#each row.series as series}<p>
                          {series.label}
                          <span class="sources-unit">{series.unit}</span>
                        </p>{/each}
                    </details>{:else}<span class="sources-muted"
                      >No catalogue chart series</span
                    >{/if}</td
                >
                <td class="sources-time"
                  ><time datetime={row.dataset.loaded_at_utc ?? undefined}
                    >{timestamp(row.dataset.loaded_at_utc)}</time
                  ></td
                >
                <td class="sources-number"
                  >{row.dataset.approx_rows === null
                    ? 'Not supplied'
                    : row.dataset.approx_rows.toLocaleString('en-GB')}</td
                >
              </tr>{/each}</tbody
          >
        </table>
      </div>
    {:else if !loading && catalog}<div class="learn-empty" role="status">
        <h3>
          {rows.length ? 'No matching datasets.' : 'No datasets reported.'}
        </h3>
        <p>
          {rows.length
            ? 'Try a different search or data type.'
            : 'The catalogue is reachable, but it has no dataset metadata yet.'}
        </p>
        {#if rows.length}<button
            class="quiet-button"
            onclick={() => {
              search = '';
              kind = 'All types';
            }}>Clear filters</button
          >{/if}
      </div>
    {:else if loading && !catalog}<p class="learn-empty" role="status">
        Loading source metadata…
      </p>{/if}
    {#if catalog}<p class="sources-updated">
        Catalogue data updated: {timestamp(catalog.data_updated_at_utc)}. Raw
        row counts can include multiple forecast versions.
      </p>{/if}
  </section>
  <section class="sources-reading" aria-label="Reading the catalogue">
    <article>
      <span class="learn-eyebrow">01 / TIMEFRAMES</span>
      <h3>Coverage is a range.</h3>
      <p>
        Dates show the earliest and latest stored records. They do not guarantee
        an unbroken history. Forecast coverage can extend beyond the latest
        actual reading.
      </p>
    </article>
    <article>
      <span class="learn-eyebrow">02 / INTERPRETATION</span>
      <h3>Keep data types distinct.</h3>
      <p>
        Forecasts, settlement values, actual readings and weather reanalysis
        have different meanings. Open source notes for definitions, publication
        delays and coverage limitations.
      </p>
    </article>
    <article>
      <span class="learn-eyebrow">03 / GEOGRAPHY</span>
      <h3>Understand the scope.</h3>
      <p>
        Electricity series cover Great Britain. Weather records describe
        selected locations; national electricity totals are not regional
        measurements.
      </p>
    </article>
  </section>
</main>
