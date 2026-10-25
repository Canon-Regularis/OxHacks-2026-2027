/** Readable research layouts: content selection belongs to the model, geometry to this planner. */
import type { Layout, Size, Workspace } from '../model/types';
import { uid } from '../model/types';
import { reflow } from '../layout/engine';
const GAP = 8;
const CHART_HEIGHT = 260;
const CHAT_WIDTH = 340;
const CHAT_HEIGHT = 300;

function tabs(ids: string[]): Extract<Layout, { kind: 'stack' }> {
  return { kind: 'stack', id: uid(), widgets: ids, active: ids[0] };
}
function split(
  axis: 'horizontal' | 'vertical',
  step: number,
  first: Layout,
  second: Layout,
): Layout {
  return { kind: 'split', id: uid(), axis, step, children: [first, second] };
}
function rows(nodes: Layout[]): Layout {
  if (nodes.length === 1) return nodes[0];
  return split(
    'vertical',
    Math.max(1, Math.round(16 / nodes.length)),
    nodes[0],
    rows(nodes.slice(1)),
  );
}
export function researchLayout(
  workspace: Workspace,
  size: Size,
  preferred: string[],
  chat: string,
  background: boolean,
): Layout | null {
  const all = Object.keys(workspace.widgets);
  if (!all.length) return null;
  const order = [...new Set([...preferred, ...all])].filter((id) =>
    Object.hasOwn(workspace.widgets, id),
  );
  const chats = [
    ...new Set([
      chat,
      ...all.filter((id) => workspace.widgets[id].type === 'chat'),
    ]),
  ].filter((id) => workspace.widgets[id]?.type === 'chat');
  const analytical = order.filter(
    (id) => workspace.widgets[id].type !== 'chat',
  );
  if (!analytical.length) return tabs(chats);
  const charts = analytical.filter(
    (id) =>
      workspace.widgets[id].type === 'chart' && !!workspace.widgets[id].binding,
  );
  const useful = (id: string) => {
    const w = workspace.widgets[id];
    const binding = w.binding ? workspace.bindings[w.binding] : undefined;
    return (
      (w.type === 'table' && !!binding) ||
      (w.type === 'comparison' &&
        binding?.kind === 'algorithm' &&
        !!w.settings.table) ||
      (w.type === 'context' && !!w.settings.notes.trim()) ||
      (w.type === 'inspector' &&
        !!(workspace.groups[w.group ?? '']?.selected ?? w.context.selected))
    );
  };
  const support = analytical.filter((id) => !charts.includes(id) && useful(id));
  const passive = analytical.filter(
    (id) => !charts.includes(id) && !support.includes(id),
  );
  const showChat = chats.length > 0 && !background;
  // A single chart gets full width above chat; never a tall 50/50 chart/chat column pair.
  const estimatedRows = Math.min(
    3,
    Math.max(1, charts.length + (support.length ? 1 : 0)),
  );
  const sidebar =
    showChat &&
    size.width >= 1040 &&
    (charts.length > 1 || support.length > 0) &&
    (!charts.length ||
      (size.width - CHAT_WIDTH - GAP) / (size.height / estimatedRows) >= 1.5);
  const mainStep = Math.min(
    23,
    Math.max(
      1,
      Math.round(((size.width - CHAT_WIDTH - GAP) / (size.width - GAP)) * 24),
    ),
  );
  const mainSize = {
    width: sidebar ? ((size.width - GAP) * mainStep) / 24 : size.width,
    height:
      showChat && !sidebar ? size.height - CHAT_HEIGHT - GAP : size.height,
  };
  const capacity = Math.max(
    1,
    Math.min(3, Math.floor((mainSize.height + GAP) / (CHART_HEIGHT + GAP))),
  );
  let panels: Extract<Layout, { kind: 'stack' }>[];
  if (!charts.length || capacity === 1) panels = [tabs(analytical)];
  else {
    const reserve = support.length > 0 || charts.length > capacity ? 1 : 0;
    const visible = charts.slice(0, Math.max(1, capacity - reserve));
    panels = visible.map((id) => tabs([id]));
    const remaining = analytical.filter(
      (id) => !visible.includes(id) && !passive.includes(id),
    );
    if (remaining.length) panels.push(tabs(remaining));
    panels[0].widgets.push(...passive);
  }
  if (!showChat) panels[0].widgets.splice(1, 0, ...chats);
  let main: Layout = rows(panels);
  let result: Layout = showChat
    ? sidebar
      ? split('horizontal', mainStep, main, tabs(chats))
      : split(
          'vertical',
          Math.max(
            1,
            Math.min(
              15,
              Math.round((mainSize.height / (size.height - GAP)) * 16),
            ),
          ),
          main,
          tabs(chats),
        )
    : main;
  let fitted = reflow(result, workspace, size);
  if (fitted) return fitted;
  // Preserve every widget as accessible tabs if several visible panels cannot fit.
  main = tabs(analytical);
  if (!showChat)
    (main as Extract<Layout, { kind: 'stack' }>).widgets.splice(1, 0, ...chats);
  result = showChat ? split('vertical', 9, main, tabs(chats)) : main;
  fitted = reflow(result, workspace, size);
  if (!fitted)
    throw new Error(
      'Workspace has no room at readable minimum sizes. Use a larger viewport or another workspace.',
    );
  return fitted;
}

/** Explicit all-panels mode: no analytical tab fallback; only conversations share the primary tile. */
export function separateLayout(
  workspace: Workspace,
  size: Size,
  preferred: string[],
  chat: string,
): Layout | null {
  const all = Object.keys(workspace.widgets);
  if (!all.length) return null;
  const order = [...new Set([...preferred, ...all])].filter((id) =>
    Object.hasOwn(workspace.widgets, id),
  );
  const chats = [
    ...new Set([
      chat,
      ...all.filter((id) => workspace.widgets[id].type === 'chat'),
    ]),
  ].filter((id) => workspace.widgets[id]?.type === 'chat');
  const analytical = order.filter(
    (id) => workspace.widgets[id].type !== 'chat',
  );
  if (!analytical.length) return tabs(chats);
  const leaves = analytical.map((id, i) =>
    tabs(i === 0 ? [id, ...chats] : [id]),
  );
  function columns(nodes: Layout[]): Layout {
    return nodes.length === 1
      ? nodes[0]
      : split(
          'horizontal',
          Math.max(1, Math.round(24 / nodes.length)),
          nodes[0],
          columns(nodes.slice(1)),
        );
  }
  function quality(
    node: Layout,
    area: Size,
  ): { score: number; areas: number[] } {
    if (node.kind === 'stack') {
      const widget = workspace.widgets[node.active];
      const score =
        widget.type === 'chart'
          ? Math.min(area.width / area.height, 2.5) * 3 +
            Math.min(area.height / 260, 1)
          : Math.min(area.width / 420, 1);
      return { score, areas: [area.width * area.height] };
    }
    const horizontal = node.axis === 'horizontal',
      total = (horizontal ? area.width : area.height) - GAP;
    const first = (total * node.step) / (horizontal ? 24 : 16);
    const children = node.children.map((child, i) =>
      quality(
        child,
        horizontal
          ? { width: i === 0 ? first : total - first, height: area.height }
          : { width: area.width, height: i === 0 ? first : total - first },
      ),
    );
    return {
      score: children[0].score + children[1].score,
      areas: children.flatMap((child) => child.areas),
    };
  }
  let best: Layout | null = null,
    bestScore = -Infinity;
  // Search balanced rows AND columns. At most 24 candidates for the 12-widget limit.
  for (let count = 1; count <= leaves.length; count++) {
    const groups: Layout[][] = [];
    let offset = 0;
    for (let i = 0; i < count; i++) {
      const length =
        Math.floor(leaves.length / count) + (i < leaves.length % count ? 1 : 0);
      groups.push(leaves.slice(offset, offset + length));
      offset += length;
    }
    for (const candidate of [
      rows(groups.map(columns)),
      columns(groups.map(rows)),
    ]) {
      const fitted = reflow(candidate, workspace, size);
      if (!fitted) continue;
      const measured = quality(fitted, size);
      const mean =
        measured.areas.reduce((a, b) => a + b, 0) / measured.areas.length;
      const imbalance = measured.areas.reduce(
        (sum, area) => sum + (area / mean - 1) ** 2,
        0,
      );
      const score = measured.score - imbalance * 0.1;
      if (score > bestScore) {
        best = fitted;
        bestScore = score;
      }
    }
  }
  if (!best)
    throw new Error(
      `Cannot show all ${analytical.length} analytical panels separately at readable minimum sizes. Use a larger viewport, fewer panels, or allow tabs.`,
    );
  return best;
}
