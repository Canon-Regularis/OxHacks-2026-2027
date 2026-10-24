<script lang="ts">
import { cn } from "$lib/vendor/kcpc/utils";
import type { Snippet } from "svelte";
import type { HTMLAttributes } from "svelte/elements";

type Props = HTMLAttributes<HTMLElement> & {
  text?: string;
  children?: Snippet;
  disabled?: boolean;
  speed?: number;
  spread?: number;
  class?: string;
  className?: string;
  as?: "div" | "span";
  color?: string;
  shineColor?: string;
  shimmerWidth?: number;
};

let {
  text = "",
  children,
  disabled = false,
  speed = 5,
  spread = 120,
  class: className = "",
  className: classNameProp = "",
  as = "div",
  color,
  shineColor = "rgba(255, 255, 255, 0.8)",
  shimmerWidth,
  style: styleProp = "",
  ...restProps
}: Props = $props();

const animationDuration = $derived(`${speed}s`);
const backgroundImage = $derived(
  `linear-gradient(${spread}deg, rgba(255, 255, 255, 0) 40%, ${shineColor} 50%, rgba(255, 255, 255, 0) 60%)`,
);
</script>

<svelte:element
  this={as}
  class={cn(
    "text-foreground/60 bg-clip-text inline-block",
    !disabled && "animate-shine",
    className,
    classNameProp,
  )}
  style:background-image={backgroundImage}
  style:background-size="200% 100%"
  style:-webkit-background-clip="text"
  style:background-clip="text"
  style:animation-duration={animationDuration}
  style={color ? `color: ${color}; ${styleProp}` : styleProp}
  {...restProps}
>
  {#if children}
    {@render children()}
  {:else}
    {text}
  {/if}
</svelte:element>

<style>
@keyframes shine {
  0% {
    background-position: 100%;
  }
  100% {
    background-position: -100%;
  }
}

.animate-shine {
  animation: shine 5s linear infinite;
}
</style>
