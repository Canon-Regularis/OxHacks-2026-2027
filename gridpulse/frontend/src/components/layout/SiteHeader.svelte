<script lang="ts">
  import { page } from '$app/state';
  import SettingsDrawer from './SettingsDrawer.svelte';
</script>

<header class="site-header">
  <a class="brand" href="/" aria-label="Eisengrau overview"
    ><svg width="30" height="30" viewBox="0 0 30 30" aria-hidden="true"
      ><circle cx="15" cy="15" r="13" fill="none" stroke="currentColor" /><path
        d="M5 16h5l3-8 5 15 3-9h4"
        fill="none"
        stroke="currentColor"
        stroke-width="1.5"
      /></svg
    >Eisengrau<span class="brand-beta">GB</span></a
  >
  <nav aria-label="Primary navigation">
    <a href="/" aria-current={page.url.pathname === '/' ? 'page' : undefined}
      >Overview</a
    ><a
      href="/terminal"
      aria-current={page.url.pathname === '/terminal' ? 'page' : undefined}
      >Terminal</a
    ><a
      href="/about"
      aria-current={page.url.pathname === '/about' ? 'page' : undefined}
      >About</a
    >
  </nav>
  <SettingsDrawer />
</header>
