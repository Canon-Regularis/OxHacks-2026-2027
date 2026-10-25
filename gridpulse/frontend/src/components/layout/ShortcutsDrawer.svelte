<script lang="ts">
  import HeaderPanel from './HeaderPanel.svelte';
  import TerminalShortcuts from '../help/TerminalShortcuts.svelte';
</script>

<HeaderPanel
  id="shortcuts"
  title="Keyboard shortcuts"
  eyebrow="TERMINAL / KEYBOARD"
  description="Keep your investigation moving. These shortcuts apply on the Terminal page; Tab moves between controls and Shift + Tab moves back."
  openLabel="Open keyboard shortcuts"
  closeLabel="Close keyboard shortcuts"
>
  {#snippet icon()}
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="1.6"
      stroke-linecap="round"
      aria-hidden="true"
      ><rect x="2" y="5" width="20" height="14" rx="2" /><path
        d="M6 9h.01M10 9h.01M14 9h.01M18 9h.01M6 12h.01M10 12h.01M14 12h.01M18 12h.01M7 15h10"
      /></svg
    >
  {/snippet}
  <div class="shortcuts-panel-content"><TerminalShortcuts /></div>
</HeaderPanel>
