<script lang="ts">
  import { page } from '$app/state';
  import SettingsDrawer from './SettingsDrawer.svelte';
  import ShortcutsDrawer from './ShortcutsDrawer.svelte';
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
    ><span class="brand-name">Eisengrau</span><span class="brand-beta">GB</span
    ></a
  >
  <nav aria-label="Primary navigation">
    <div
      class="nav-group nav-workspace"
      role="group"
      aria-label="Workspace pages"
    >
      <a href="/" aria-current={page.url.pathname === '/' ? 'page' : undefined}
        >Overview</a
      ><a
        href="/terminal"
        aria-current={page.url.pathname === '/terminal' ? 'page' : undefined}
        >Terminal</a
      >
    </div>
    <div
      class="nav-group nav-information"
      role="group"
      aria-label="Information pages"
    >
      <a
        href="/sources"
        aria-label="Data sources"
        aria-current={page.url.pathname === '/sources' ? 'page' : undefined}
        ><span class="nav-long">Data sources</span><span class="nav-short"
          >Data</span
        ></a
      ><a
        href="/quick-start"
        aria-label="Quick start"
        aria-current={page.url.pathname === '/quick-start' ? 'page' : undefined}
        ><span class="nav-long">Quick start</span><span class="nav-short"
          >Guide</span
        ></a
      ><a
        href="/status"
        aria-current={page.url.pathname === '/status' ? 'page' : undefined}
        >Status</a
      ><a
        href="/about"
        aria-current={page.url.pathname === '/about' ? 'page' : undefined}
        >About</a
      >
    </div>
  </nav>
  <div class="header-tools"><ShortcutsDrawer /><SettingsDrawer /></div>
</header>
