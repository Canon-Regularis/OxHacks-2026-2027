<script lang="ts">
  import { onMount } from 'svelte';
  import { useSettings, reducedMotion } from '$lib/state/settings.svelte';
  const settings = useSettings();
  let Globe = $state<
    | typeof import('$lib/vendor/kcpc/animations/rotating-earth/rotating-earth.svelte').default
    | null
  >(null);
  let reduced = $derived(reducedMotion(settings));
  let ready = $state(false);
  const globeColor = '#fafafa';
  onMount(() => {
    import('$lib/vendor/kcpc/animations/rotating-earth/rotating-earth.svelte')
      .then((m) => {
        Globe = m.default;
        ready = true;
      })
      .catch(() => {});
  });
</script>

<section class="hero">
  <div class="hero-copy">
    <p class="eyebrow">UK ENERGY / ALGORITHMS &amp; AI</p>
    <h1>Energy data.<br /><span>Deeper analysis.</span></h1>
    <p class="hero-description">
      Explore Great Britain’s electricity market with algorithms for
      forecasting, backtesting and statistical analysis. Bring wind, demand and
      imbalance prices together in one research workspace.
    </p>
    <a class="text-link" href="/terminal">Open Terminal <span>↗</span></a>
  </div>
  <div class="hero-globe" aria-hidden="true">
    {#if Globe && ready}<Globe
        width={420}
        height={310}
        scale={125}
        rotationSpeed={reduced ? 0 : 0.2}
        autoRotate={!reduced}
        scrollable={false}
        landOutlineColor={globeColor}
        graticuleColor="#a1a1a1"
        coastColor={globeColor}
        halftoneDotColor="#a1a1a1"
      />{/if}
    <span class="globe-caption">GREAT BRITAIN / ELECTRICITY ANALYSIS</span>
  </div>
</section>
