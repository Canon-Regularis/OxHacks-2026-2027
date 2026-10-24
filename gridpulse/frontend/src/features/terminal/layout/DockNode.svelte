<script lang="ts">
  import type { Snippet } from 'svelte';
  import type { Layout, Workspace, Position } from '../model/types';
  import DockNode from './DockNode.svelte';
  let {
    node,
    workspace,
    focused,
    dragging,
    renderWidget,
    onfocus,
    onactivate,
    ondrag,
    onmove,
    onresize,
    onsettings,
    onmaximise,
    onclose,
    canPlace,
    canResize,
  }: {
    node: Layout;
    workspace: Workspace;
    focused: string | null;
    dragging: string | null;
    renderWidget: Snippet<[string]>;
    onfocus: (id: string) => void;
    onactivate: (id: string) => void;
    ondrag: (id: string | null) => void;
    onmove: (id: string, target: string, position: Position) => void;
    onresize: (id: string, step: number) => void;
    onsettings: (id: string) => void;
    onmaximise: (id: string) => void;
    onclose: (id: string) => void;
    canPlace: (id: string, target: string, position: Position) => boolean;
    canResize: (id: string, step: number) => boolean;
  } = $props();
  let element = $state<HTMLDivElement>(null!);
  let preview = $state<Position | null>(null),
    allowed = $state(false),
    proposed = $state<number | null>(null),
    resizing = $state(false);
  function position(event: DragEvent): Position {
    const r = element.getBoundingClientRect(),
      x = (event.clientX - r.left) / r.width,
      y = (event.clientY - r.top) / r.height;
    return x < 0.22
      ? 'left'
      : x > 0.78
        ? 'right'
        : y < 0.22
          ? 'top'
          : y > 0.78
            ? 'bottom'
            : 'tab';
  }
  function dragover(event: DragEvent) {
    if (!dragging || workspace.locked) return;
    event.preventDefault();
    event.stopPropagation();
    preview = position(event);
    allowed = canPlace(dragging, node.id, preview);
    if (event.dataTransfer)
      event.dataTransfer.dropEffect = allowed ? 'move' : 'none';
  }
  function pointer(event: PointerEvent) {
    if (!resizing || node.kind !== 'split') return;
    const box = element.getBoundingClientRect();
    const part =
      node.axis === 'horizontal'
        ? (event.clientX - box.left) / box.width
        : (event.clientY - box.top) / box.height;
    const step = Math.max(
      1,
      Math.min(
        node.axis === 'horizontal' ? 23 : 15,
        Math.round(part * (node.axis === 'horizontal' ? 24 : 16)),
      ),
    );
    if (canResize(node.id, step)) proposed = step;
  }
  function end() {
    if (resizing && proposed !== null) onresize(node.id, proposed);
    resizing = false;
    proposed = null;
  }
</script>

{#if node.kind === 'split'}
  <div
    bind:this={element}
    class="terminal-split"
    class:horizontal={node.axis === 'horizontal'}
    style={node.axis === 'horizontal'
      ? `grid-template-columns:minmax(0,${proposed ?? node.step}fr) 8px minmax(0,${24 - (proposed ?? node.step)}fr)`
      : `grid-template-rows:minmax(0,${proposed ?? node.step}fr) 8px minmax(0,${16 - (proposed ?? node.step)}fr)`}
  >
    <DockNode
      node={node.children[0]}
      {workspace}
      {focused}
      {dragging}
      {renderWidget}
      {onfocus}
      {onactivate}
      {ondrag}
      {onmove}
      {onresize}
      {onsettings}
      {onmaximise}
      {onclose}
      {canPlace}
      {canResize}
    />
    <div
      role="slider"
      tabindex={workspace.locked ? -1 : 0}
      aria-label="Resize tiles"
      aria-orientation={node.axis === 'horizontal' ? 'vertical' : 'horizontal'}
      aria-valuemin="1"
      aria-valuemax={node.axis === 'horizontal' ? 23 : 15}
      aria-valuenow={proposed ?? node.step}
      class="terminal-divider"
      class:locked={workspace.locked}
      onpointerdown={(e) => {
        if (workspace.locked) return;
        resizing = true;
        e.currentTarget.setPointerCapture(e.pointerId);
      }}
      onpointermove={pointer}
      onpointerup={end}
      onpointercancel={() => {
        resizing = false;
        proposed = null;
      }}
      onkeydown={(e) => {
        if (workspace.locked || node.kind !== 'split') return;
        if (e.key === 'Escape') {
          resizing = false;
          proposed = null;
        }
        if (
          ['ArrowLeft', 'ArrowUp', 'ArrowRight', 'ArrowDown'].includes(e.key)
        ) {
          e.preventDefault();
          const step =
            node.step + (['ArrowLeft', 'ArrowUp'].includes(e.key) ? -1 : 1);
          if (canResize(node.id, step)) onresize(node.id, step);
        }
      }}
    ></div>
    <DockNode
      node={node.children[1]}
      {workspace}
      {focused}
      {dragging}
      {renderWidget}
      {onfocus}
      {onactivate}
      {ondrag}
      {onmove}
      {onresize}
      {onsettings}
      {onmaximise}
      {onclose}
      {canPlace}
      {canResize}
    />
  </div>
{:else}
  {@const widget = workspace.widgets[node.active]}
  <div
    bind:this={element}
    class="terminal-tile"
    class:focused={focused === node.active}
    role="group"
    aria-label={widget.title}
    ondragover={dragover}
    ondragleave={(e) => {
      if (!element.contains(e.relatedTarget as Node)) preview = null;
    }}
    ondrop={(e) => {
      e.preventDefault();
      e.stopPropagation();
      if (dragging && preview && allowed) onmove(dragging, node.id, preview);
      preview = null;
      ondrag(null);
    }}
  >
    <header class="terminal-tile-header">
      <div
        class="terminal-widget-tabs"
        role="tablist"
        aria-label="Tile widgets"
      >
        {#each node.widgets as id}{@const w = workspace.widgets[id]}
          <button
            role="tab"
            aria-selected={id === node.active}
            aria-controls={`widget-${id}`}
            draggable={!workspace.locked}
            ondragstart={(e) => {
              e.dataTransfer?.setData('text/plain', id);
              ondrag(id);
            }}
            ondragend={() => {
              ondrag(null);
              preview = null;
            }}
            onclick={() => {
              onactivate(id);
              onfocus(id);
            }}
            ondblclick={() => onmaximise(id)}
            title="Drag to dock, or use widget settings to move"
            >{w.title}</button
          >
        {/each}
      </div>
      <span
        class="terminal-group-badge"
        style={`--group:${workspace.groups[widget.group ?? '']?.color ?? '#888888'}`}
        >{widget.group ?? '—'}</span
      >
      <button
        aria-label={`Settings for ${widget.title}`}
        onclick={() => onsettings(widget.id)}>⚙</button
      ><button
        aria-label={`Maximise ${widget.title}`}
        onclick={() => onmaximise(widget.id)}>⛶</button
      ><button
        aria-label={`Close ${widget.title}`}
        onclick={() => onclose(widget.id)}>×</button
      >
    </header>
    <div
      class="terminal-widget-content"
      id={`widget-${node.active}`}
      role="tabpanel"
      tabindex="-1"
      onfocusin={() => onfocus(node.active)}
    >
      <svelte:boundary>
        <!---->{@render renderWidget(node.active)}{#snippet failed(
          _error,
          reset,
        )}<div class="terminal-widget-error">
            <p>This widget could not render. Other tiles remain available.</p>
            <button onclick={reset}>Retry rendering</button>
          </div>{/snippet}
      </svelte:boundary>
    </div>
    {#if preview && dragging}<div
        class={`terminal-drop-preview ${preview}`}
        class:invalid={!allowed}
      >
        <span
          >{allowed
            ? preview === 'tab'
              ? 'Stack as tab'
              : `Dock ${preview}`
            : 'No room at minimum size'}</span
        >
      </div>{/if}
  </div>
{/if}
