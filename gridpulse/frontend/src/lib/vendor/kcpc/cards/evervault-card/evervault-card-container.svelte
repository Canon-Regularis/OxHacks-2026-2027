<script lang="ts">
import { cn } from "$lib/vendor/kcpc/utils";
import type { Snippet } from "svelte";
import EvervaultCard from "./evervault-card.svelte";

let {
  text = "",
  children,
  class: className = "",
}: {
  text?: string;
  children?: Snippet;
  class?: string;
} = $props();
</script>

<div
  class={cn(
    "border border-border flex flex-col items-start md:max-w-sm max-w-xs p-6 gap-6 relative bg-card shadow-sm",
    className
  )}
>
  <!-- Plus Icon Corners -->
  <svg
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 23 23"
    stroke-width="1"
    stroke="currentColor"
    class="absolute h-6 w-6 md:-top-[13.25px] -top-3.25 -left-[13.25px] text-border pointer-events-none"
  >
    <path d="M12 6v12m6-6H6" />
  </svg>
  <svg
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 23 23"
    stroke-width="1"
    stroke="currentColor"
    class="absolute h-6 w-6 -bottom-3 -left-[13.25px] text-border pointer-events-none"
  >
    <path d="M12 6v12m6-6H6" />
  </svg>
  <svg
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 23 23"
    stroke-width="1"
    stroke="currentColor"
    class="absolute h-6 w-6 md:-top-[13.25px] -top-3.25 -right-3 text-border pointer-events-none"
  >
    <path d="M12 6v12m6-6H6" />
  </svg>
  <svg
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 23 23"
    stroke-width="1"
    stroke="currentColor"
    class="absolute h-6 w-6 -bottom-3 -right-3 text-border pointer-events-none"
  >
    <path d="M12 6v12m6-6H6" />
  </svg>

  <EvervaultCard {text} />

  {#if children}
    <div class="w-full">
      {@render children()}
    </div>
  {/if}
</div>
