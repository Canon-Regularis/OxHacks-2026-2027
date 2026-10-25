<script lang="ts">
  import { getContext, onMount } from 'svelte';
  import TerminalShortcuts from '../help/TerminalShortcuts.svelte';
  import { useSettings } from '$lib/state/settings.svelte';
  import { ACCENTS, type Accent } from '$lib/state/settings';
  const settings = useSettings();
  const reset = getContext<() => void>('gridpulse:reset');
  let mounted = $state(false);
  let showShortcuts = $state(false);
  onMount(() => {
    mounted = true;
  });
  let dialog: HTMLDialogElement;
  let trigger: HTMLButtonElement;
  function trapFocus(event: KeyboardEvent) {
    if (event.key !== 'Tab') return;
    const elements = Array.from(
      dialog.querySelectorAll<HTMLElement>('button, select, input, a[href]'),
    ).filter(
      (el) => !el.hasAttribute('disabled') && el.getClientRects().length,
    );
    const first = elements[0];
    const last = elements.at(-1);
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
    const r = dialog.getBoundingClientRect();
    if (
      event.clientX < r.left ||
      event.clientX > r.right ||
      event.clientY < r.top ||
      event.clientY > r.bottom
    )
      dialog.close();
  }
</script>

<button
  class="settings-trigger"
  bind:this={trigger}
  disabled={!mounted}
  aria-label="Open settings"
  aria-haspopup="dialog"
  onclick={() => dialog.showModal()}
>
  <svg
    width="20"
    height="20"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="1.6"
    aria-hidden="true"
  >
    <path
      d="m9.3 3-.6 2.3-1.6.9-2.3-.6-2.7 4.7 1.7 1.7v1.9l-1.7 1.7 2.7 4.7 2.3-.6 1.6.9.6 2.4h5.4l.6-2.4 1.6-.9 2.3.6 2.7-4.7-1.7-1.7V12l1.7-1.7-2.7-4.7-2.3.6-1.6-.9-.6-2.3z"
      transform="translate(0 -1)"
    />
    <circle cx="12" cy="12" r="3.2" />
  </svg>
</button>
<dialog
  bind:this={dialog}
  class="settings-drawer"
  aria-labelledby="settings-title"
  aria-describedby="settings-description"
  onclose={() => {
    showShortcuts = false;
    trigger.focus();
  }}
  onclick={dismissBackdrop}
  onkeydown={trapFocus}
>
  <div class="settings-heading">
    <div>
      <p class="eyebrow">YOUR WORKSPACE</p>
      <h2 id="settings-title">Settings</h2>
    </div>
    <button
      class="settings-close"
      aria-label="Close settings"
      onclick={() => dialog.close()}>✕</button
    >
  </div>
  <p class="settings-description" id="settings-description">
    Make the view yours. Changes apply immediately.
  </p>
  <div class="settings-fields">
    <label class="settings-field"
      ><span>Time display</span><select
        aria-label="Time display"
        aria-describedby="settings-time-help"
        bind:value={settings.timeDisplay}
        ><option value="utc">UTC</option><option value="local"
          >UK local time (BST / GMT)</option
        ></select
      ><small id="settings-time-help"
        >Settlement dates always follow the UK calendar.</small
      ></label
    >
    <label class="settings-field"
      ><span>Motion</span><select
        aria-label="Motion"
        bind:value={settings.motion}
        ><option value="system">Follow device</option><option value="reduce"
          >Reduce animations</option
        ></select
      ></label
    >
    <fieldset class="settings-accents">
      <legend>Accent colour</legend>
      <div class="settings-accent-options">
        {#each Object.entries(ACCENTS) as [id, accent]}
          <label
            class="settings-accent-choice"
            class:active={settings.accent === id}
          >
            <input
              type="radio"
              name="accent-colour"
              value={id as Accent}
              bind:group={settings.accent}
              aria-label={accent.label}
            />
            <span
              class="settings-accent-swatch"
              style:background={accent.color}
              aria-hidden="true"
            ></span>
            <span>{accent.label.replace('Pastel ', '')}</span>
          </label>
        {/each}
      </div>
      <p>Chart series keep consistent colours.</p>
    </fieldset>
    <label class="settings-toggle"
      ><span>Chart gridlines</span><input
        type="checkbox"
        aria-label="Chart gridlines"
        bind:checked={settings.gridlines}
      /></label
    >
    <label class="settings-field"
      ><span>Chart line thickness</span><select
        aria-label="Chart line thickness"
        bind:value={settings.lineThickness}
        ><option value="standard">Standard</option><option value="bold"
          >Bold</option
        ></select
      ></label
    >
    <label class="settings-toggle"
      ><span
        >Remember my view<small id="settings-remember-help"
          >Save the date, selected properties and scan property on this browser.
          Turning this off clears the saved view.</small
        ></span
      ><input
        type="checkbox"
        aria-label="Remember my view"
        aria-describedby="settings-remember-help"
        bind:checked={settings.rememberView}
      /></label
    >
  </div>
  <div class="settings-help">
    <button
      class="quiet-button"
      aria-expanded={showShortcuts}
      aria-controls="settings-shortcuts"
      onclick={() => (showShortcuts = !showShortcuts)}
      >Keyboard shortcuts</button
    >
    <section
      id="settings-shortcuts"
      class="settings-shortcuts"
      hidden={!showShortcuts}
      aria-labelledby="settings-shortcuts-heading"
    >
      <h3 id="settings-shortcuts-heading">Terminal keyboard shortcuts</h3>
      <p>
        Use these on the Terminal page. Tab moves between controls; Shift + Tab
        moves back.
      </p>
      <TerminalShortcuts />
    </section>
  </div>
  <div class="settings-reset">
    <button class="quiet-button" onclick={reset}>Reset settings</button>
  </div>
</dialog>
