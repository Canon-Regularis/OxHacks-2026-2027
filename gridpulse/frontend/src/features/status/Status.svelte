<script lang="ts">
  import { onMount } from 'svelte';
  import { useSettings } from '$lib/state/settings.svelte';
  import { formatDateTime } from '$lib/state/settings';
  const settings = useSettings();
  import { api } from '$lib/api';
  import type { Catalog, Health } from '$lib/types';
  import InfoTooltip from '../../components/ui/InfoTooltip.svelte';
  import {
    AGEING_HOURS,
    STALE_HOURS,
    ageHours,
    ageLabel,
    algorithmRow,
    datasetRow,
    summarise,
    type Level,
  } from './model';

  const REFRESH_MS = 30_000;
  const WORDS: Record<Level, string> = {
    ok: 'OK',
    warn: 'Older data',
    bad: 'Problem',
    unknown: 'Unknown',
  };

  let health = $state<Health | null>(null);
  let catalog = $state<Catalog | null>(null);
  let apiError = $state('');
  let catalogError = $state('');
  let checkedAt = $state<number | null>(null);
  let now = $state(Date.now());
  let loading = $state(true);
  let controller: AbortController | undefined;

  const summary = $derived(summarise(health, catalog, now));
  const datasets = $derived(
    (catalog?.datasets ?? []).map((d) => datasetRow(d, now)),
  );
  const algorithms = $derived(
    (catalog?.algorithms ?? []).map((a) =>
      algorithmRow(a, catalog?.datasets ?? [], now),
    ),
  );
  const issues = $derived([
    ...(apiError ? [apiError] : []),
    ...(catalogError ? [catalogError] : []),
    ...datasets
      .filter((d) => d.level === 'bad' || d.level === 'warn')
      .map(
        (d) =>
          `${d.dataset}: ${d.note || 'loaded ' + ageLabel(d.loadedAgeHours)}`,
      ),
    ...algorithms
      .filter((a) => a.level !== 'ok')
      .map((a) => `${a.id}: ${a.note}`),
  ]);

  const timestampLabel = (iso: string | null) =>
    iso ? formatDateTime(iso, settings.timeDisplay) : '—';

  async function refresh() {
    controller?.abort();
    const mine = (controller = new AbortController());
    loading = true;
    const [h, c] = await Promise.allSettled([
      api.health(mine.signal),
      api.catalog(mine.signal),
    ]);
    if (mine.signal.aborted) return;
    health = h.status === 'fulfilled' ? h.value : null;
    apiError =
      h.status === 'rejected'
        ? h.reason instanceof Error
          ? h.reason.message
          : 'Health check failed.'
        : '';
    catalog = c.status === 'fulfilled' ? c.value : null;
    catalogError =
      c.status === 'rejected' && h.status === 'fulfilled'
        ? 'Catalog unavailable: ' +
          (c.reason instanceof Error ? c.reason.message : 'request failed.')
        : '';
    checkedAt = now = Date.now();
    loading = false;
  }

  onMount(() => {
    void refresh();
    const poll = setInterval(() => void refresh(), REFRESH_MS);
    const tick = setInterval(() => (now = Date.now()), 1000);
    return () => {
      clearInterval(poll);
      clearInterval(tick);
      controller?.abort();
    };
  });
</script>

<a class="skip-link" href="#status-main">Skip to system status</a>
<main id="status-main" class="page-shell status-page" aria-busy={loading}>
  <div class="market-toolbar">
    <div>
      <p class="eyebrow">SYSTEM</p>
      <h1>Health &amp; status<span class="heading-dot">.</span></h1>
    </div>
    <div class="date-controls">
      <span class="status-checked"
        >{checkedAt
          ? 'Checked ' +
            ageLabel(ageHours(new Date(checkedAt).toISOString(), now))
          : 'Checking…'} · auto-refresh {REFRESH_MS / 1000}s</span
      ><button class="quiet-button" onclick={refresh} disabled={loading}
        >Refresh ↻</button
      >
    </div>
  </div>

  <section
    class="status-banner"
    data-level={loading && !checkedAt ? 'unknown' : summary.level}
    role="status"
  >
    <span class="status-lamp" aria-hidden="true"></span>
    <div>
      <h2>{loading && !checkedAt ? 'Checking…' : summary.headline}</h2>
      <p>
        {#if issues.length}{issues.length}
          {issues.length === 1 ? 'data notice' : 'data notices'} below.{:else if checkedAt}No
          issues found.{/if}
      </p>
    </div>
  </section>

  {#if datasets.some((d) => d.loadedAgeHours !== null && d.loadedAgeHours > AGEING_HOURS)}
    <p class="status-age-warning" role="note">
      The data is a bit old. Check the load times and coverage below before
      treating it as current market data.
    </p>
  {/if}

  <div class="status-cards">
    <article class="status-card" data-level={summary.api}>
      <h3>
        API <InfoTooltip
          label="About the API check"
          text="Whether the backend answered /api/health."
        />
      </h3>
      <p class="status-value">{health ? 'Reachable' : 'Unreachable'}</p>
      <p class="status-meta">
        {health
          ? `v${health.version ?? '?'} · mode ${health.mode}`
          : apiError || '—'}
      </p>
    </article>
    <article class="status-card" data-level={summary.database}>
      <h3>
        Database <InfoTooltip
          label="About the database check"
          text="The API runs SELECT 1 against Tiger Data. Replay mode serves recorded responses and does not use a database."
        />
      </h3>
      <p class="status-value">
        {health ? health.database.replace('_', ' ') : 'unknown'}
      </p>
      <p class="status-meta">
        {health?.mode === 'replay' ? 'Cached replay' : 'Tiger Data connection'}
      </p>
    </article>
    <article class="status-card" data-level={summary.data}>
      <h3>
        Data freshness <InfoTooltip
          label="How freshness is judged"
          text={`Time since the loader last wrote data. Up to ${AGEING_HOURS} h is OK, up to ${STALE_HOURS} h is ageing, beyond that is stale. These thresholds are a heuristic, not a source guarantee.`}
        />
      </h3>
      <p class="status-value">{ageLabel(summary.dataUpdatedHours)}</p>
      <p class="status-meta">
        {summary.datasetsCurrent}/{summary.datasetsTotal} datasets current
      </p>
    </article>
    <article class="status-card" data-level={summary.algorithms}>
      <h3>
        Analysis runs <InfoTooltip
          label="About analysis runs"
          text="Algorithms with a stored run being served. Inputs updated means a dataset was reloaded after the run, so the stored result predates the data."
        />
      </h3>
      <p class="status-value">
        {summary.algorithmsServed}/{summary.algorithmsTotal} served
      </p>
      <p class="status-meta">
        {algorithms.filter((a) => a.level === 'warn').length} out of date
      </p>
    </article>
  </div>

  {#if catalog}
    <section class="status-section" aria-labelledby="datasets-h">
      <h2 id="datasets-h">Datasets</h2>
      <div class="status-table-wrap">
        <table class="status-table">
          <thead
            ><tr
              ><th>Dataset</th><th>State</th><th>Loaded</th><th
                >Coverage ({settings.timeDisplay === 'utc'
                  ? 'UTC'
                  : 'UK local'})</th
              ><th class="num">Rows</th></tr
            ></thead
          >
          <tbody>
            {#each datasets as d (d.dataset)}
              <tr>
                <th scope="row"
                  >{d.dataset}{#if d.source}<small>{d.source}</small>{/if}</th
                >
                <td
                  ><span
                    class="pill"
                    data-level={!d.note &&
                    d.loadedAgeHours !== null &&
                    d.loadedAgeHours > AGEING_HOURS
                      ? 'warn'
                      : d.level}
                    >{!d.note &&
                    d.loadedAgeHours !== null &&
                    d.loadedAgeHours > AGEING_HOURS
                      ? 'Older data'
                      : WORDS[d.level]}</span
                  >{#if d.note}<small>{d.note}</small>{/if}</td
                >
                <td title={timestampLabel(d.loadedUtc)}
                  >{ageLabel(d.loadedAgeHours)}</td
                >
                <td
                  >{timestampLabel(d.firstUtc)} → {timestampLabel(
                    d.lastUtc,
                  )}</td
                >
                <td class="num"
                  >{d.rows === null ? '—' : d.rows.toLocaleString('en-GB')}</td
                >
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    </section>

    <section class="status-section" aria-labelledby="algos-h">
      <h2 id="algos-h">Algorithms</h2>
      <div class="status-table-wrap">
        <table class="status-table">
          <thead
            ><tr><th>Algorithm</th><th>State</th><th>Last run</th></tr></thead
          >
          <tbody>
            {#each algorithms as a (a.id)}
              <tr>
                <th scope="row">{a.id}<small>v{a.version}</small></th>
                <td
                  ><span class="pill" data-level={a.level}
                    >{a.level === 'warn'
                      ? 'Inputs updated'
                      : WORDS[a.level]}</span
                  >{#if a.note}<small>{a.note}</small>{/if}</td
                >
                <td title={timestampLabel(a.finishedUtc)}
                  >{ageLabel(a.ageHours)}</td
                >
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    </section>
  {:else if checkedAt}
    <div class="notice error" role="alert">
      <h3>Detail unavailable</h3>
      <p>
        {catalogError ||
          apiError ||
          'The catalogue could not be loaded, so dataset and algorithm checks are skipped.'}
      </p>
    </div>
  {/if}
</main>
