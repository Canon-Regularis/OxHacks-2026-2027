<script lang="ts">
  import { onMount } from 'svelte';
  let Globe = $state<
    | typeof import('$lib/vendor/kcpc/animations/rotating-earth/rotating-earth.svelte').default
    | null
  >(null);
  let reduced = $state(false);
  let globeWidth = $state(360);
  let ready = $state(false);
  onMount(() => {
    const mq = matchMedia('(prefers-reduced-motion: reduce)');
    reduced = mq.matches;
    const listener = () => (reduced = mq.matches);
    mq.addEventListener('change', listener);
    import('$lib/vendor/kcpc/animations/rotating-earth/rotating-earth.svelte')
      .then((m) => {
        Globe = m.default;
        ready = true;
      })
      .catch(() => {});
    return () => mq.removeEventListener('change', listener);
  });
</script>

<section class="hero">
  <div class="hero-copy">
    <p class="eyebrow">GB ELECTRICITY / MARKET EXPLORER</p>
    <h1>Follow the market.<br /><span>Inspect the detail.</span></h1>
    <p class="hero-description">
      Published forecasts, actual readings and imbalance prices.<br />One
      connected view of Britain’s electricity market.
    </p>
    <a class="text-link" href="/">Open the overview <span>↗</span></a>
  </div>
  <div class="hero-globe" bind:clientWidth={globeWidth} aria-hidden="true">
    {#if Globe && ready}{#key `${Math.round(globeWidth)}-${reduced}`}<Globe
          width={Math.max(240, Math.round(globeWidth))}
          height={310}
          scale={125}
          rotationSpeed={reduced ? 0 : 0.2}
          autoRotate={!reduced}
          scrollable={false}
          landOutlineColor="#fafafa"
          graticuleColor="#a1a1a1"
          coastColor="#fafafa"
          halftoneDotColor="#a1a1a1"
        />{/key}{/if}
    <span class="globe-caption">GREAT BRITAIN / CONNECTED ENERGY</span>
  </div>
</section>
