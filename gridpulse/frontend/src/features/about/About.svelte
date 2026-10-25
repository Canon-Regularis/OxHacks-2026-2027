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
        'Explore GB imbalance settlement prices alongside forecasts of wind and electricity demand. Select a day, then move across the timeline to inspect its settlement periods.',
    },
    {
      number: '02',
      title: 'Put changes in context',
      description:
        'Compare hourly temperature and pressure changes with price changes. Seven-day correlations show descriptive relationships, with source, sample counts and limitations available beside the numbers.',
    },
    {
      number: '03',
      title: 'Keep the evidence visible',
      description:
        'Data comes through a backend connected to Tiger Data. Missing records remain missing, forecasts are labelled, and each chart keeps its units and timeframe explicit.',
    },
  ];
</script>

<main class="page-shell about-page" id="about-main">
  <Hero />
  <section class="about-intro">
    <p class="eyebrow">ABOUT EISENGRAU</p>
    <h2>The bigger picture.<br /><span>Without losing the detail.</span></h2>
    <p>
      Eisengrau is an electricity-market investigation dashboard built for
      OxHack 2026. It brings prices, generation forecasts and weather context
      into one place so you can inspect changing conditions.
    </p>
  </section>
  <div class="about-cards">
    {#each sections as section}<Card class="about-card"
        ><span class="eyebrow">{section.number} / THE APPROACH</span>
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
          color="#a2c7b8"
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
        Electricity demand and generation forecasts change. Weather adds another
        layer of context. Eisengrau lets you explore these together—without
        mistaking correlation for a cause.
      </p>
      <div class="pointer-events-auto">
        <a class="about-cta" href="/">Open the overview <span>↗</span></a>
      </div>
    </section>
  </div>
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
  <p class="about-limitations">
    Retrospective analysis · archived weather forecasts · descriptive
    correlations · no trading advice.
  </p>
</main>
