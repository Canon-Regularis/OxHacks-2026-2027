<script lang="ts">
  import { onMount } from 'svelte';
  import Hero from './Hero.svelte';
  import TextShimmer from '$lib/vendor/kcpc/text/text-shimmer/text-shimmer.svelte';
  import Card from '$lib/vendor/kcpc/primitives/card/card.svelte';
  import EvervaultCard from '$lib/vendor/kcpc/cards/evervault-card/evervault-card.svelte';
  let Pixels = $state<
    | typeof import('$lib/vendor/kcpc/backgrounds/pixel-blast/pixel-blast.svelte').default
    | null
  >(null);
  let reduced = $state(true);
  onMount(() => {
    const mq = matchMedia('(prefers-reduced-motion: reduce)');
    reduced = mq.matches;
    const update = () => (reduced = mq.matches);
    mq.addEventListener('change', update);
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2');
    if (gl) {
      gl.getExtension('WEBGL_lose_context')?.loseContext();
      import('$lib/vendor/kcpc/backgrounds/pixel-blast/pixel-blast.svelte')
        .then((m) => (Pixels = m.default))
        .catch(() => {});
    }
    return () => mq.removeEventListener('change', update);
  });
  const sections = [
    {
      number: '01',
      title: 'Observe the market',
      description:
        'Choose a settlement day and compare imbalance price, forecast and actual wind, forecast and actual demand, residual demand and net imbalance volume. Select one to five properties, with up to two unit axes on a shared chart.',
    },
    {
      number: '02',
      title: 'Scan every half-hour',
      description:
        'Hover, tap or use the keyboard to inspect a half-hour across the chart, summary cards and magnifying day scan. The scan follows your chosen property and supports 46, 48 and 50-period days. Your view preferences are remembered.',
    },
    {
      number: '03',
      title: 'See the whole day',
      description:
        'Day at a glance summarises the selected properties’ daily averages, price extremes, negative-price duration and peak forecast demand. These daily statistics stay fixed while you inspect individual periods; missing readings are excluded and price coverage is shown.',
    },
  ];
</script>

<main class="page-shell about-page" id="about-main">
  <Hero />
  <section class="about-intro">
    <p class="eyebrow">ABOUT EISENGRAU</p>
    <h2>The bigger picture.<br /><span>Without losing the detail.</span></h2>
    <p>
      Eisengrau is a dashboard for inspecting Great Britain’s electricity
      market, built for OxHack 2026. Overview brings published forecasts,
      available outturns and settlement prices into one daily view, so you can
      compare what was forecast with what happened and inspect the underlying
      numbers.
    </p>
  </section>
  <div class="about-cards">
    {#each sections as section}<Card class="about-card"
        ><span class="eyebrow">{section.number} / IN OVERVIEW</span>
        <h3>{section.title}</h3>
        <p>{section.description}</p></Card
      >{/each}
  </div>
  <div class="flex flex-row md:h-36 h-28 border-b border-border about-spacer">
    <div class="flex-1 md:pr-12 pr-8"></div>
    <div
      class="md:w-full w-fit max-w-6xl md:border-x border-l border-border"
    ></div>
    <div class="flex-1 md:pr-12 pr-8"></div>
  </div>
  <div
    class="relative flex w-full items-center justify-center overflow-hidden about-pixel-section"
  >
    <div class="absolute inset-0" aria-hidden="true">
      {#if Pixels && !reduced}<Pixels
          color="#e6b080"
          speed={1}
          edgeFade={0.7}
          rippleSpeed={0.2}
        />{/if}
    </div>
    <section
      class="pointer-events-none relative z-10 max-w-4xl mx-auto px-6 about-pixel-copy"
    >
      <p class="eyebrow">FROM CONDITIONS TO CONTEXT</p>
      <h2>
        <TextShimmer
          text="Follow the evidence."
          color="var(--orange)"
          shineColor="rgba(255,235,216,.9)"
          disabled={reduced}
        />
      </h2>
      <p>
        Follow a price spike, a negative-price period or a change in wind and
        demand. Keep the same half-hour in view across every selected property,
        then step back to the day’s summary.
      </p>
      <div class="pointer-events-auto">
        <a class="about-cta" href="/">Open the overview <span>↗</span></a>
      </div>
    </section>
  </div>
  <section class="about-data" aria-labelledby="about-data-heading">
    <div>
      <p class="eyebrow">PUBLIC MARKET DATA</p>
      <h2 id="about-data-heading">
        Published sources.<br /><span>Clear distinctions.</span>
      </h2>
    </div>
    <div class="about-data-copy">
      <p>
        The market series use publicly available Elexon and NESO data, served
        through Eisengrau’s backend from Tiger Data. Wind and demand forecasts
        are published source forecasts, with the latest version issued before
        delivery selected for each period. They are separate from actual
        generation and demand readings.
      </p>
      <p>
        Prices are retrospective settlement values. Actual demand can arrive
        later, and wind forecast and outturn coverage can differ. Missing
        readings stay missing; units, source notes and data freshness help you
        interpret what is available. This is an inspection view, not a claim
        that every value was known at the time.
      </p>
      <div class="about-source-links">
        <a
          class="text-link"
          href="https://bmrs.elexon.co.uk/"
          target="_blank"
          rel="noreferrer">Elexon Insights ↗</a
        >
        <a
          class="text-link"
          href="https://www.neso.energy/data-portal"
          target="_blank"
          rel="noreferrer">NESO Data Portal ↗</a
        >
      </div>
    </div>
  </section>
  <section class="about-definition">
    <div class="about-definition-copy">
      <p class="eyebrow">KNOW THE DIFFERENCE</p>
      <h2>What is imbalance?</h2>
      <p>
        A participant's imbalance is the difference between the electricity it
        contracted to generate or use and what it actually generated or used,
        after settlement adjustments. The imbalance price is used to settle that
        surplus or shortfall.
      </p>
      <p>
        Residual demand is a different metric: our forecast demand minus
        forecast wind. It is not a measure of system imbalance.
      </p>
      <a
        href="https://www.elexon.co.uk/bsc/settlement/imbalance-pricing/"
        target="_blank"
        rel="noreferrer"
        class="text-link">Read Elexon's explanation ↗</a
      >
    </div>
    <div class="about-decoration" aria-hidden="true">
      <EvervaultCard text="CONTEXT" />
    </div>
  </section>
  <section class="about-roadmap" aria-labelledby="about-roadmap-heading">
    <p class="eyebrow">WHAT COMES NEXT</p>
    <h2 id="about-roadmap-heading">More ways to investigate.</h2>
    <p>
      Overview, Status and this About page are available today. Status shows
      service health, data freshness and analysis-run availability. Explore and
      Methods are planned for deeper data inspection and explanations of
      analysis. The backend now computes statistical forecasts, backtests,
      rolling signals and diagnostics in batch jobs; their results are available
      through the API, with dedicated dashboard views still to come. Weather
      analysis is available in the backend, while weather maps and a Gemini
      investigation assistant remain future extensions.
    </p>
  </section>
  <p class="about-limitations">
    Retrospective market analysis · missing data stays missing · comparisons do
    not establish cause and effect.
  </p>
</main>
