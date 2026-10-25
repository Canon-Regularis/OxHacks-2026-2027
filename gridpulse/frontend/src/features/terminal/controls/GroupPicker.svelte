<script lang="ts">
  import { tick } from 'svelte';
  import type { Group } from '../model/types';
  let {
    groups,
    group,
    title,
    onchange,
    onfocus,
  }: {
    groups: Record<string, Group>;
    group: string | null;
    title: string;
    onchange: (group: string | null) => void;
    onfocus: () => void;
  } = $props();
  let open = $state(false),
    left = $state(0),
    top = $state(0);
  let trigger = $state<HTMLButtonElement>(null!),
    menu = $state<HTMLDivElement>(null!);
  function close(restore = false) {
    open = false;
    if (restore) trigger.focus();
  }
  async function toggle() {
    if (open) {
      close();
      return;
    }
    onfocus();
    const box = trigger.getBoundingClientRect();
    left = Math.max(8, Math.min(innerWidth - 228, box.right - 220));
    top = box.bottom + 6;
    open = true;
    await tick();
    if (!open || !menu) return;
    top = Math.max(8, Math.min(top, innerHeight - menu.offsetHeight - 8));
    (
      menu.querySelector<HTMLButtonElement>('[aria-checked="true"]') ??
      menu.querySelector('button')
    )?.focus();
  }
  function choose(value: string | null) {
    if (value !== group) onchange(value);
    close(true);
  }
  function keyboard(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      close(true);
    } else if (e.key === 'Tab') close(true);
    else if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(e.key)) {
      e.preventDefault();
      const buttons = Array.from(
        menu.querySelectorAll<HTMLButtonElement>('button:not(:disabled)'),
      );
      let index = buttons.indexOf(document.activeElement as HTMLButtonElement);
      index =
        e.key === 'Home'
          ? 0
          : e.key === 'End'
            ? buttons.length - 1
            : (index + (e.key === 'ArrowDown' ? 1 : -1) + buttons.length) %
              buttons.length;
      buttons[index]?.focus();
    }
  }
</script>

<svelte:window
  onpointerdown={(e) => {
    if (
      open &&
      !menu?.contains(e.target as Node) &&
      !trigger.contains(e.target as Node)
    )
      close();
  }}
  onresize={() => close()}
/>
<button
  bind:this={trigger}
  class="terminal-group-badge"
  style={`--group:${groups[group ?? '']?.color ?? '#888888'}`}
  aria-label={`Change group for ${title}`}
  aria-haspopup="menu"
  aria-expanded={open}
  title={`Linked group: ${group ? (groups[group]?.name ?? group) : 'Independent'}. Click to change.`}
  onclick={toggle}
  onkeydown={(e) => {
    if (e.key === 'ArrowDown' && !open) {
      e.preventDefault();
      void toggle();
    }
  }}>{group ?? '—'}<span aria-hidden="true">⌄</span></button
>
{#if open}<div
    bind:this={menu}
    class="terminal-group-menu"
    role="menu"
    tabindex="-1"
    aria-label={`Link group for ${title}`}
    style={`left:${left}px;top:${top}px`}
    onkeydown={keyboard}
  >
    <p>Link this widget</p>
    <button
      role="menuitemradio"
      aria-checked={group === null}
      onclick={() => choose(null)}
      ><i style="background:#888888"></i>Independent<span aria-hidden="true"
        >{group === null ? '✓' : ''}</span
      ></button
    >
    {#each Object.values(groups) as g}<button
        role="menuitemradio"
        aria-checked={g.id === group}
        onclick={() => choose(g.id)}
        ><i style={`background:${g.color}`}></i>{g.name}<span aria-hidden="true"
          >{g.id === group ? '✓' : ''}</span
        ></button
      >{/each}
    <button
      role="menuitem"
      class="terminal-new-group"
      disabled={Object.keys(groups).length >= 12}
      onclick={() => choose('__new')}
      >＋ {Object.keys(groups).length >= 12
        ? 'Group limit reached'
        : 'Create linked group…'}</button
    >
    <small>Shares dates, period selection and compatible chart zoom.</small>
  </div>{/if}
