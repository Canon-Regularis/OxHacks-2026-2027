<script lang="ts">
  import { getContext } from 'svelte';
  import HeaderPanel from './HeaderPanel.svelte';
  import { useSettings } from '$lib/state/settings.svelte';
  import { ACCENTS, type Accent } from '$lib/state/settings';
  const settings = useSettings();
  const reset = getContext<() => void>('gridpulse:reset');
</script>

<HeaderPanel
  id="settings"
  title="Settings"
  eyebrow="YOUR WORKSPACE"
  description="Make the view yours. Changes apply immediately."
  openLabel="Open settings"
  closeLabel="Close settings"
>
  {#snippet icon()}
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
  {/snippet}
  <div class="settings-fields">
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
    <div class="settings-dropdowns">
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
      <label class="settings-field"
        ><span>Chart line thickness</span><select
          aria-label="Chart line thickness"
          bind:value={settings.lineThickness}
          ><option value="standard">Standard</option><option value="bold"
            >Bold</option
          ></select
        ></label
      >
    </div>
  </div>
  <div class="settings-reset">
    <button class="quiet-button" onclick={reset}>Reset settings</button>
  </div>
</HeaderPanel>
