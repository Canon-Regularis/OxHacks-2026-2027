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
    <p class="eyebrow">GB ELECTRICITY / AI-POWERED ANALYSIS</p>
    <h1>Follow the market.<br /><span>Inspect the detail.</span></h1>
    <p class="hero-description">
      AI-powered investigation of Britain’s electricity market. An AI agent
      helps you choose the right algorithms and visualise the data you need,
      from wind and demand to imbalance prices.
    </p>
    <a class="text-link" href="/">Open the overview <span>↗</span></a>
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
    <span class="globe-caption">GREAT BRITAIN / CONNECTED ENERGY</span>
  </div>
</section>
