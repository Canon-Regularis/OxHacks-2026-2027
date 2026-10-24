<script lang="ts" module>
const characters =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

export const generateRandomString = (length: number) => {
  let result = "";
  for (let i = 0; i < length; i++) {
    result += characters.charAt(Math.floor(Math.random() * characters.length));
  }
  return result;
};
</script>

<script lang="ts">
import { cn } from "$lib/vendor/kcpc/utils";
import { onMount } from "svelte";
import type { Snippet } from "svelte";

let {
  text = "",
  children,
  contentClass,
  class: className = "",
}: {
  text?: string;
  children?: Snippet;
  contentClass?: string;
  class?: string;
} = $props();

let mouseX = $state(0);
let mouseY = $state(0);
let randomString = $state("");

onMount(() => {
  randomString = generateRandomString(1500);
  mouseX = Math.random() * 150;
  mouseY = Math.random() * 300;
});

function onMouseMove(e: MouseEvent) {
  const currentTarget = e.currentTarget as HTMLElement;
  const { left, top } = currentTarget.getBoundingClientRect();
  mouseX = e.clientX - left;
  mouseY = e.clientY - top;
  randomString = generateRandomString(1500);
}
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
  class={cn(
    "p-0.5 bg-transparent aspect-square flex items-center justify-center w-full h-full relative",
    className
  )}
>
  <div
    onmousemove={onMouseMove}
    class="group/card rounded-3xl w-full relative overflow-hidden bg-transparent flex items-center justify-center h-full"
  >
    <!-- Pattern -->
    <div class="pointer-events-none">
      <div
        class="absolute inset-0 rounded-2xl mask-[linear-gradient(white,transparent)] md:group-hover/card:opacity-50 md:opacity-100 opacity-50"
      ></div>
      <div
        class="absolute inset-0 rounded-2xl bg-linear-to-r from-background to-foreground md:opacity-0 md:group-hover/card:opacity-100 opacity-100 backdrop-blur-xl transition duration-500"
        style="mask-image: radial-gradient(500px at {mouseX}px {mouseY}px, white, transparent); -webkit-mask-image: radial-gradient(500px at {mouseX}px {mouseY}px, white, transparent);"
      ></div>
      <div
        class="absolute inset-0 rounded-2xl md:opacity-0 mix-blend-overlay md:group-hover/card:opacity-100 opacity-100"
        style="mask-image: radial-gradient(500px at {mouseX}px {mouseY}px, white, transparent); -webkit-mask-image: radial-gradient(500px at {mouseX}px {mouseY}px, white, transparent);"
      >
        <p
          class="absolute inset-x-0 text-xs h-full break-words whitespace-pre-wrap text-foreground font-mono font-bold transition duration-500 select-none"
        >
          {randomString}
        </p>
      </div>
    </div>

    <!-- Center Content -->
    <div class="relative z-10 flex items-center justify-center w-full h-full">
      <div
        class={contentClass ?? "relative size-4/5 max-h-56 max-w-56 rounded-full flex items-center justify-center text-foreground font-bold text-2xl md:text-4xl"}
      >
        <div
          class="absolute w-full h-full bg-background/80 blur-lg rounded-full"
        ></div>
        {#if text}
          <span class="text-foreground z-20 text-center font-mono">{text}</span>
        {/if}
        {#if children}
          <div class="z-20 w-full text-center">
            {@render children()}
          </div>
        {/if}
      </div>
    </div>
  </div>
</div>
