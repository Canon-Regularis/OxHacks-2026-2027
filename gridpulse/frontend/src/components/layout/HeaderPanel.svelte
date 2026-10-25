<script lang="ts">
  import { onMount, type Snippet } from 'svelte';
  let {
    id,
    title,
    eyebrow,
    description,
    openLabel,
    closeLabel,
    icon,
    children,
  }: {
    id: string;
    title: string;
    eyebrow: string;
    description: string;
    openLabel: string;
    closeLabel: string;
    icon: Snippet;
    children: Snippet;
  } = $props();
  let mounted = $state(false);
  let dialog: HTMLDialogElement;
  let trigger: HTMLButtonElement;
  onMount(() => {
    mounted = true;
  });
  function trapFocus(event: KeyboardEvent) {
    if (event.key !== 'Tab') return;
    const elements = Array.from(
      dialog.querySelectorAll<HTMLElement>('button, select, input, a[href]'),
    ).filter(
      (element) =>
        !element.hasAttribute('disabled') && element.getClientRects().length,
    );
    const first = elements[0],
      last = elements.at(-1);
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last?.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first?.focus();
    }
  }
  function dismissBackdrop(event: MouseEvent) {
    if (event.target !== dialog) return;
    const bounds = dialog.getBoundingClientRect();
    if (
      event.clientX < bounds.left ||
      event.clientX > bounds.right ||
      event.clientY < bounds.top ||
      event.clientY > bounds.bottom
    )
      dialog.close();
  }
</script>

<button
  class="settings-trigger"
  bind:this={trigger}
  disabled={!mounted}
  aria-label={openLabel}
  aria-haspopup="dialog"
  {title}
  onclick={() => dialog.showModal()}>{@render icon()}</button
>
<dialog
  bind:this={dialog}
  class="settings-drawer"
  aria-labelledby={`${id}-title`}
  aria-describedby={`${id}-description`}
  onclose={() => trigger.focus()}
  onclick={dismissBackdrop}
  onkeydown={trapFocus}
>
  <div class="settings-heading">
    <div>
      <p class="eyebrow">{eyebrow}</p>
      <h2 id={`${id}-title`}>{title}</h2>
    </div>
    <button
      class="settings-close"
      aria-label={closeLabel}
      onclick={() => dialog.close()}>✕</button
    >
  </div>
  <p class="settings-description" id={`${id}-description`}>{description}</p>
  {@render children()}
</dialog>
