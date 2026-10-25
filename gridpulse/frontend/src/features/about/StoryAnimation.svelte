<script lang="ts">
  import { onMount } from 'svelte';
  import { ACCENTS } from '$lib/state/settings';
  import { useSettings, reducedMotion } from '$lib/state/settings.svelte';
  let { kind }: { kind: 'square' | 'cubes' | 'fluid' } = $props();
  const settings = useSettings();
  let host: HTMLDivElement;
  let visible = $state(false);
  let webgl = $state(false);
  let reduced = $derived(reducedMotion(settings));
  let Square = $state<
    | typeof import('$lib/vendor/kcpc/animations/shape-blur/shape-blur.svelte').default
    | null
  >(null);
  let Cubes = $state<
    | typeof import('$lib/vendor/kcpc/animations/cubes/cubes.svelte').default
    | null
  >(null);
  let Fluid = $state<
    | typeof import('$lib/vendor/kcpc/backgrounds/liquid-ether/liquid-ether.svelte').default
    | null
  >(null);
  onMount(() => {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2');
    webgl = !!gl;
    gl?.getExtension('WEBGL_lose_context')?.loseContext();
    const observer = new IntersectionObserver(
      ([entry]) => (visible = entry.isIntersecting),
      { rootMargin: '100px' },
    );
    observer.observe(host);
    return () => observer.disconnect();
  });
  $effect(() => {
    if (!visible || reduced) return;
    let active = true;
    const load = async () => {
      try {
        if (kind === 'square' && webgl) {
          const module =
            await import('$lib/vendor/kcpc/animations/shape-blur/shape-blur.svelte');
          if (active) Square = module.default;
        } else if (kind === 'cubes') {
          const module =
            await import('$lib/vendor/kcpc/animations/cubes/cubes.svelte');
          if (active) Cubes = module.default;
        } else if (kind === 'fluid' && webgl) {
          const module =
            await import('$lib/vendor/kcpc/backgrounds/liquid-ether/liquid-ether.svelte');
          if (active) Fluid = module.default;
        }
      } catch {
        /* Keep the decorative fallback if loading fails. */
      }
    };
    void load();
    return () => {
      active = false;
    };
  });
</script>

<div
  bind:this={host}
  class="about-story-animation"
  data-animation={kind}
  class:animation-ready={visible &&
    !reduced &&
    ((kind === 'square' && Square) ||
      (kind === 'cubes' && Cubes) ||
      (kind === 'fluid' && Fluid))}
  aria-hidden="true"
>
  <div
    class="story-animation-fallback"
    class:story-square={kind === 'square'}
    class:story-cubes={kind === 'cubes'}
    class:story-fluid={kind === 'fluid'}
  ></div>
  {#if visible && !reduced}
    {#if kind === 'square' && Square}<Square
        shapeSize={0.85}
        roundness={0.2}
        borderSize={0.035}
      />
    {:else if kind === 'cubes' && Cubes}<Cubes
        class="story-cube-grid"
        gridSize={6}
        borderStyle="1px solid var(--orange)"
        faceColor="var(--background)"
        rippleOnClick={false}
      />
    {:else if kind === 'fluid' && Fluid}<Fluid
        colors={[ACCENTS[settings.accent].color, '#afcabc', '#eac6a5']}
        resolution={0.35}
        iterationsPoisson={12}
        autoSpeed={0.3}
      />{/if}
  {/if}
</div>
