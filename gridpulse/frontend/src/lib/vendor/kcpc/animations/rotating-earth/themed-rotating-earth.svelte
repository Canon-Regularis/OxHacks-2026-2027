<script lang="ts">
import RotatingEarth from "./rotating-earth.svelte";
import { mode } from "mode-watcher";

let {
  width = 800,
  height = 600,
  scale,
  class: className = "",
}: {
  width?: number;
  height?: number;
  scale?: number;
  class?: string;
} = $props();

let isDark = $derived(mode.current === "dark");
</script>

<RotatingEarth
  landOutlineColor={isDark ? "#fafafa" : "#0a0a0a"}
  graticuleColor={isDark ? "#fafafa" : "#0a0a0a"}
  coastColor={isDark ? "#fafafa" : "#0a0a0a"}
  halftoneDotColor={isDark ? "#a1a1a1" : "#262626"}
  rotationSpeed={0.2}
  scrollable={false}
  {scale}
  {height}
  {width}
  class={className}
/>
