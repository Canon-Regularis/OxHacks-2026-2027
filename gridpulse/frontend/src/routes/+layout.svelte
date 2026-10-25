<script lang="ts">
  import '../styles/app.css';
  import '../styles/overview.css';
  import '../styles/settings.css';
  import { onMount, setContext } from 'svelte';
  import { page } from '$app/state';
  import SiteHeader from '../components/layout/SiteHeader.svelte';
  import SiteFooter from '../components/layout/SiteFooter.svelte';
  import {
    createViewState,
    initialViewPreference,
    readPreference,
    writePreference,
  } from '$lib/state/view.svelte';
  import {
    createSettingsState,
    reducedMotion,
  } from '$lib/state/settings.svelte';
  import {
    defaultSettings,
    normalizeSettings,
    readSettings,
    SETTINGS_KEY,
  } from '$lib/state/settings';
  let { children } = $props();
  const view = createViewState();
  const settings = createSettingsState();
  let ready = $state(false);
  let storage: Storage | undefined;
  setContext('gridpulse:view', view);
  setContext('gridpulse:settings', settings);
  setContext('gridpulse:reset', () => {
    Object.assign(settings, defaultSettings());
    Object.assign(view, initialViewPreference());
    try {
      storage?.removeItem('gridpulse:view:v1');
    } catch {
      /* Storage is optional. */
    }
  });
  onMount(() => {
    try {
      storage = localStorage;
    } catch {
      /* Storage may be disabled. */
    }
    if (storage) {
      Object.assign(settings, readSettings(storage));
      Object.assign(
        view,
        settings.rememberView
          ? readPreference(storage)
          : initialViewPreference(),
      );
    }
    const motion = matchMedia('(prefers-reduced-motion: reduce)');
    const syncDevice = () => {
      settings.systemReduced = motion.matches;
    };
    syncDevice();
    motion.addEventListener('change', syncDevice);
    const syncStorage = (event: StorageEvent) => {
      if (!storage || (event.storageArea && event.storageArea !== storage))
        return;
      if (event.key === SETTINGS_KEY || event.key === null)
        Object.assign(settings, readSettings(storage));
      if (
        (event.key === 'gridpulse:view:v1' || event.key === null) &&
        settings.rememberView
      )
        Object.assign(view, readPreference(storage));
    };
    window.addEventListener('storage', syncStorage);
    ready = true;
    return () => {
      motion.removeEventListener('change', syncDevice);
      window.removeEventListener('storage', syncStorage);
    };
  });
  $effect(() => {
    if (!ready) return;
    document.documentElement.classList.add('dark');
    document.documentElement.dataset.theme = 'dark';
    document.documentElement.dataset.motion = reducedMotion(settings)
      ? 'reduce'
      : 'full';
    try {
      storage?.setItem(
        SETTINGS_KEY,
        JSON.stringify(normalizeSettings(settings)),
      );
    } catch {
      /* Keep the current session usable. */
    }
  });
  $effect(() => {
    if (!ready || !storage) return;
    if (settings.rememberView) writePreference(storage, view);
    else {
      try {
        storage.removeItem('gridpulse:view:v1');
      } catch {
        /* Storage is optional. */
      }
    }
  });
</script>

<div class="app-frame">
  <SiteHeader
  />{@render children()}{#if page.url.pathname !== '/terminal' && !page.url.pathname.startsWith('/terminal/')}<SiteFooter
    />{/if}
</div>
