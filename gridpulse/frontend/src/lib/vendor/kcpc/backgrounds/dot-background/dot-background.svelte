<script lang="ts">
import { cn } from "$lib/vendor/kcpc/utils";
import type { Snippet } from "svelte";

let {
  children,
  class: className,
  dotsClassName,
  mask = 1,
}: {
  children?: Snippet;
  class?: string;
  dotsClassName?: string;
  mask?: number;
} = $props();
</script>

<div
  class={cn("relative flex w-full items-center justify-center bg-background", className)}
>
  <div
    aria-hidden="true"
    class={cn(
          "absolute inset-0",
          "bg-size-[23px_23px]",
          "bg-[radial-gradient(var(--muted-foreground)_1px,transparent_1px)]",
          dotsClassName,
        )}
  ></div>
  <!-- Radial gradient for the container to give a faded look -->
  <div
    aria-hidden="true"
    class="z-0 absolute inset-0 flex items-center justify-center bg-background mask-[radial-gradient(ellipse_at_center,transparent_var(--mask-number),black)]"
    style:--mask-number={`${mask}%`}
  ></div>
  {@render children?.()}
</div>
