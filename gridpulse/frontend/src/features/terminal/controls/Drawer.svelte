<script lang="ts">
  import { onMount } from 'svelte';
  import type { Snippet } from 'svelte';
  let {
    title,
    onclose,
    children,
  }: { title: string; onclose: () => void; children: Snippet } = $props();
  let panel: HTMLElement;
  onMount(() => {
    const previous = document.activeElement as HTMLElement;
    (panel.querySelector<HTMLElement>('input,button') ?? panel).focus();
    return () => previous?.focus?.();
  });
  function keyboard(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      e.stopPropagation();
      onclose();
    }
    if (e.key === 'Tab') {
      const nodes = Array.from(
        panel.querySelectorAll<HTMLElement>(
          'button:not(:disabled), input, select, textarea, a[href], [tabindex="0"]',
        ),
      ).filter((el) => el.getClientRects().length);
      if (!nodes.length) {
        e.preventDefault();
        return;
      }
      if (e.shiftKey && document.activeElement === nodes[0]) {
        e.preventDefault();
        nodes.at(-1)!.focus();
      } else if (!e.shiftKey && document.activeElement === nodes.at(-1)) {
        e.preventDefault();
        nodes[0].focus();
      }
    }
  }
</script>

<div
  class="terminal-drawer-backdrop"
  role="presentation"
  onclick={(e) => {
    if (e.target === e.currentTarget) onclose();
  }}
>
  <div
    class="terminal-drawer"
    role="dialog"
    aria-modal="true"
    aria-label={title}
    tabindex="-1"
    bind:this={panel}
    onkeydown={keyboard}
  >
    <header>
      <h2>{title}</h2>
      <button aria-label="Close drawer" onclick={onclose}>×</button>
    </header>
    {@render children()}
  </div>
</div>
