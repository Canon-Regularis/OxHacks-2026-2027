<script lang="ts">
  import { page } from '$app/state';
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
    ><button disabled title="Explore is planned for the MVP"
      >Explore <span>Soon</span></button
    ><button disabled title="Methods is planned for the MVP"
      >Methods <span>Soon</span></button
    ><a
      href="/status"
      aria-current={page.url.pathname === '/status' ? 'page' : undefined}
      >Status</a
    ><a
      href="/about"
      aria-current={page.url.pathname === '/about' ? 'page' : undefined}
      >About</a
    >
  </nav>
  <div class="header-status">
    <span class="status-dot"></span>PUBLIC MARKET DATA
  </div>
</header>
