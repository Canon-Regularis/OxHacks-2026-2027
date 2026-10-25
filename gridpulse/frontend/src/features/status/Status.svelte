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
  /** Plain-language state names; "Attention" said nothing about what was wrong. */
  const DATA_WORDS: Record<Level, string> = {
    ok: 'Up to date',
    warn: 'Getting old',
    bad: 'Out of date',
    unknown: 'Unknown',
  };
  const RUN_WORDS: Record<Level, string> = {
    ok: 'Current',
    warn: 'Needs re-run',
    bad: 'No result',
    unknown: 'Unknown',
  };
  const LEGEND: { level: Level; word: string; meaning: string }[] = [
    {
      level: 'ok',
      word: 'Up to date / Current',
      meaning: `Data was loaded within the last ${AGEING_HOURS} h, or an analysis result was built from the data now stored.`,
    },
    {
      level: 'warn',
      word: 'Getting old / Needs re-run',
      meaning: `Data was last loaded ${AGEING_HOURS}–${STALE_HOURS} h ago, or it was reloaded after an analysis ran, so that result is based on older data. The dashboard still works.`,
    },
    {
      level: 'bad',
      word: 'Out of date / No result / Unreachable',
      meaning: `Nothing loaded, data older than ${STALE_HOURS} h, no analysis result to show, or a service is down. Dashboard figures may be missing or stale.`,
    },
    {
      level: 'unknown',
      word: 'Unknown',
      meaning:
        'This check could not run, usually because a service it depends on is down.',
    },
  ];

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
  type Issue = { level: Level; tag: string; title: string; text: string };
  const issues = $derived<Issue[]>([
    ...(apiError
      ? [
          {
            level: 'bad' as Level,
            tag: 'API',
            title: 'Backend unreachable.',
            text: apiError,
          },
        ]
      : []),
    ...(catalogError
      ? [
          {
            level: 'unknown' as Level,
            tag: 'Catalogue',
            title: 'Detail unavailable.',
            text: catalogError,
          },
        ]
      : []),
    ...datasets
      .filter((d) => d.level === 'bad' || d.level === 'warn')
      .map((d) => ({
        level: d.level,
        tag: DATA_WORDS[d.level],
        title: `Dataset ${d.dataset}.`,
        text: d.note || `Last loaded ${ageLabel(d.loadedAgeHours)}.`,
      })),
    ...algorithms
      .filter((a) => a.level !== 'ok')
      .map((a) => ({
        level: a.level,
        tag: RUN_WORDS[a.level],
        title: `Algorithm ${a.id}.`,
        text: a.note,
      })),
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
        {#if summary.level === 'ok' && checkedAt}The API is answering, data is
          recent and every analysis result matches the stored data.{:else if issues.length}{issues.length}
          {issues.length === 1 ? 'issue' : 'issues'} found — listed below. The dashboard
          may show older or missing figures until
          {issues.length === 1 ? 'it is' : 'they are'} resolved.{/if}
      </p>
    </div>
  </section>

  {#if issues.length}
    <section class="status-issues" aria-labelledby="issues-h">
      <h2 id="issues-h">What needs looking at</h2>
      <ul>
        {#each issues as item (item.text)}
          <li data-level={item.level}>
            <span class="pill" data-level={item.level}>{item.tag}</span>
            <span><strong>{item.title}</strong> {item.text}</span>
          </li>
        {/each}
      </ul>
    </section>
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
          ? `Backend is answering · v${health.version ?? '?'} · mode ${health.mode}`
          : apiError || 'The backend did not respond.'}
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
        {health?.mode === 'replay'
          ? 'Replay mode: recorded data, no live database'
          : 'Connection to the Tiger Data store'}
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
        Newest load · {summary.datasetsCurrent} of {summary.datasetsTotal} datasets
        up to date
      </p>
    </article>
    <article class="status-card" data-level={summary.algorithms}>
      <h3>
        Analysis results <InfoTooltip
          label="About analysis runs"
          text="Each algorithm's stored result is shown on the dashboard. A result 'needs re-run' when its input data was reloaded after it was calculated, so it reflects older data."
        />
      </h3>
      <p class="status-value">
        {summary.algorithmsServed} of {summary.algorithmsTotal} available
      </p>
      <p class="status-meta">
        {algorithms.filter((a) => a.level === 'warn').length} based on older data
      </p>
    </article>
  </div>

  {#if catalog}
    <section class="status-section" aria-labelledby="datasets-h">
      <h2 id="datasets-h">Datasets</h2>
      <p class="status-intro">
        The market data feeding the dashboard. "Last loaded" is when we last
        fetched it; "Coverage" is the time range it spans.
      </p>
      <div class="status-table-wrap">
        <table class="status-table">
          <thead
            ><tr
              ><th>Dataset</th><th>State</th><th>Last loaded</th><th
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
                  ><span class="pill" data-level={d.level}
                    >{DATA_WORDS[d.level]}</span
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
      <p class="status-intro">
        Each algorithm analyses the datasets above and stores a result. "Needs
        re-run" means newer data has arrived since it was calculated.
      </p>
      <div class="status-table-wrap">
        <table class="status-table">
          <thead
            ><tr><th>Algorithm</th><th>Result</th><th>Calculated</th></tr
            ></thead
          >
          <tbody>
            {#each algorithms as a (a.id)}
              <tr>
                <th scope="row">{a.id}<small>v{a.version}</small></th>
                <td
                  ><span class="pill" data-level={a.level}
                    >{RUN_WORDS[a.level]}</span
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
    <section class="status-section" aria-labelledby="legend-h">
      <h2 id="legend-h">How to read this page</h2>
      <dl class="status-legend">
        {#each LEGEND as l (l.level)}
          <div>
            <dt><span class="pill" data-level={l.level}>{l.word}</span></dt>
            <dd>{l.meaning}</dd>
          </div>
        {/each}
      </dl>
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
