<script lang="ts">
  import '../styles/app.css';
  import '../styles/overview.css';
  import { setContext } from 'svelte';
  import SiteHeader from '../components/layout/SiteHeader.svelte';
  import SiteFooter from '../components/layout/SiteFooter.svelte';
  import { createViewState } from '$lib/state/view.svelte';
  let { children } = $props();
  setContext('gridpulse:view', createViewState());
</script>

<div class="app-frame"><SiteHeader />{@render children()}<SiteFooter /></div>
