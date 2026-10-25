<script lang="ts">
  import { onMount } from 'svelte';
  import { useSettings, reducedMotion } from '$lib/state/settings.svelte';
  const settings = useSettings();
  import Hero from './Hero.svelte';
  import DotBackground from '$lib/vendor/kcpc/backgrounds/dot-background/dot-background.svelte';
  import TextShimmer from '$lib/vendor/kcpc/text/text-shimmer/text-shimmer.svelte';
  import Card from '$lib/vendor/kcpc/primitives/card/card.svelte';
  let Pixels = $state<
    | typeof import('$lib/vendor/kcpc/backgrounds/pixel-blast/pixel-blast.svelte').default
    | null
  >(null);
  let reduced = $derived(reducedMotion(settings));
  onMount(() => {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2');
    if (gl) {
      gl.getExtension('WEBGL_lose_context')?.loseContext();
      import('$lib/vendor/kcpc/backgrounds/pixel-blast/pixel-blast.svelte')
        .then((m) => (Pixels = m.default))
        .catch(() => {});
    }
  });
  const sections = [
    {
      number: '01',
      label: 'THE ALGORITHMS',
      title: 'Explore the methods',
      description:
        'Compare ARIMA, exponential smoothing, regression and baseline forecasts through stored backtest results. Examine forecast errors, rolling price signals and correlations to see where patterns hold and where models struggle.',
    },
    {
      number: '02',
      label: 'THE WORKSPACE',
      title: 'Build your investigation',
      description:
        'Use Terminal to bring source data and algorithm outputs into linked charts, tables and comparisons. Arrange your workspace, inspect the same period across views, save your findings and export chart or table data as CSV.',
    },
    {
      number: '03',
      label: 'THE AI AGENT',
      title: 'From questions to evidence',
      description:
        'Ask a question in plain language. The AI agent helps you choose suitable algorithms and builds the charts and comparisons you need, using the app’s data and analysis tools to keep the supporting evidence in view.',
    },
  ];
</script>

<main class="page-shell about-page" id="about-main">
  <Hero />
  <div class="about-cards">
    {#each sections as section}<Card class="about-card"
        ><span class="eyebrow">{section.number} / {section.label}</span>
        <h3>{section.title}</h3>
        <p>{section.description}</p></Card
      >{/each}
  </div>
  <div
    class="flex flex-row md:h-36 h-28 border-b border-border about-spacer"
    aria-hidden="true"
  >
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
          speed={0.35}
          patternDensity={1.8}
          edgeFade={0.7}
          rippleSpeed={0.2}
        />{/if}
    </div>
    <section
      class="pointer-events-none relative z-10 max-w-4xl mx-auto px-6 about-pixel-copy"
    >
      <p class="eyebrow">FROM DATA TO ANALYSIS</p>
      <h2>
        <TextShimmer
          text="Find patterns. Test ideas."
          color="var(--orange)"
          shineColor="rgba(255,235,216,.9)"
          disabled={reduced}
        />
      </h2>
      <p>
        Compare forecasts with settled prices, explore how wind and demand move
        with the market, and inspect the errors behind each model. Bring the
        evidence together in linked charts and tables.
      </p>
      <div class="pointer-events-auto">
        <a class="about-cta" href="/terminal">Explore Terminal <span>↗</span></a
        >
      </div>
    </section>
  </div>
  <section class="about-story" aria-labelledby="about-models-heading">
    <DotBackground
      class="about-story-dots absolute inset-0 -z-10 pointer-events-none"
      mask={20}
      dotsClassName="opacity-30"
    />
    <div>
      <p class="eyebrow">LOOK BEYOND THE FORECAST</p>
      <h2 id="about-models-heading">Put every model to the test.</h2>
      <p>
        A forecast is more useful when you can see how it performs. Compare
        statistical models with simple baselines, explore their errors at
        different lead times and find the periods where their predictions
        diverge from actual prices.
      </p>
      <p>
        Rolling signals reveal changing levels and volatility. Correlation
        analysis explores how wind, demand and price move together, while
        residual diagnostics show the patterns a model leaves unexplained. Use
        these perspectives together to decide which methods suit your question.
      </p>
      <a class="text-link" href="/terminal">Compare the algorithms ↗</a>
    </div>
  </section>
  <section class="about-data about-story" aria-labelledby="about-data-heading">
    <DotBackground
      class="about-story-dots absolute inset-0 -z-10 pointer-events-none"
      mask={20}
      dotsClassName="opacity-30"
    />
    <div class="about-data-copy">
      <p class="eyebrow">DATA BEHIND THE ANALYSIS</p>
      <h2 id="about-data-heading">Trace every result to its source.</h2>
      <p>
        Elexon and NESO provide the GB electricity series used in Eisengrau:
        imbalance prices, wind and demand forecasts, available actual readings
        and net imbalance volume. Overview gives you a daily starting point;
        Terminal brings those series together with the statistical analysis.
      </p>
      <p>
        Algorithms are computed on the backend and their completed results are
        stored in Tiger Data. In Terminal, you can inspect each run’s version,
        settings, coverage and source data, then compare its outputs without
        rerunning a calculation every time you change a chart.
      </p>
      <p>
        Forecasts and actual readings stay distinct, missing data remains
        visible, and each method carries its assumptions and limitations.
        Backtests show how forecasts performed on historical data; correlations
        describe relationships between series. Both help you assess the evidence
        behind an analysis.
      </p>
      <div class="about-source-links">
        <a
          class="text-link"
          href="https://bmrs.elexon.co.uk/"
          target="_blank"
          rel="noreferrer">Elexon electricity data ↗</a
        >
        <a
          class="text-link"
          href="https://www.neso.energy/data-portal"
          target="_blank"
          rel="noreferrer">NESO electricity data ↗</a
        >
      </div>
    </div>
  </section>
  <section class="about-story" aria-labelledby="about-workspace-heading">
    <DotBackground
      class="about-story-dots absolute inset-0 -z-10 pointer-events-none"
      mask={20}
      dotsClassName="opacity-30"
    />
    <div>
      <p class="eyebrow">CONTINUE THE INVESTIGATION</p>
      <h2 id="about-workspace-heading">Pick up where you left off.</h2>
      <p>
        Keep the charts, comparisons and notes that make an investigation yours.
        Save named workspaces, reuse a useful layout as a preset and return to
        the same linked views when a new question comes up. Work independently
        or ask the AI agent to help you take the next step.
      </p>
      <p>
        Export chart data and result tables as CSV for further analysis, or save
        your workspace configuration as JSON to restore your setup. Your
        findings stay connected to the methods and data behind them.
      </p>
      <a class="text-link" href="/terminal">Build your workspace ↗</a>
    </div>
  </section>
</main>
