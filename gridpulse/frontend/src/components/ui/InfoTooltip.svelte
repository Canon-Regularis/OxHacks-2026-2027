<script lang="ts">
  import { tick } from 'svelte';
  let { text, label = 'More information' }: { text: string; label?: string } =
    $props();
  const id = $props.id();
  let open = $state(false);
  let pinned = $state(false);
  let left = $state(0);
  let top = $state(0);
  let trigger: HTMLButtonElement;
  async function show() {
    const box = trigger.getBoundingClientRect();
    left = Math.max(12, Math.min(window.innerWidth - 284, box.left - 20));
    top = box.bottom + 8;
    open = true;
    await tick();
    const popup = document.getElementById(id);
    if (open && popup)
      top = Math.max(
        12,
        Math.min(
          top,
          window.innerHeight - popup.getBoundingClientRect().height - 12,
        ),
      );
  }
  function dismiss() {
    open = false;
    pinned = false;
  }
</script>

<span
  class="info-tooltip"
  role="group"
  aria-label={label}
  onpointerenter={show}
  onpointerleave={() => {
    if (!pinned && document.activeElement !== trigger) open = false;
  }}
>
  <button
    bind:this={trigger}
    type="button"
    class="info-button"
    aria-label={label}
    aria-describedby={open ? id : undefined}
    aria-expanded={open}
    onfocus={show}
    onblur={dismiss}
    onclick={() => {
      pinned = !pinned;
      if (pinned) show();
      else {
        dismiss();
        trigger.blur();
      }
    }}
    onkeydown={(event) => {
      if (event.key === 'Escape') {
        dismiss();
        trigger.blur();
      }
    }}>i</button
  >
  {#if open}<span
      {id}
      role="tooltip"
      class="info-popup"
      style={`left:${left}px;top:${top}px`}>{text}</span
    >{/if}
</span>
