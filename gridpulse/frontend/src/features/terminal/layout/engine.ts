import {
  MINIMUM,
  uid,
  type Layout,
  type Workspace,
  type Size,
  type Position,
} from '../model/types';
const DIVIDER = 8;
export function minimum(node: Layout, workspace: Workspace): Size {
  if (node.kind === 'stack')
    return node.widgets.reduce(
      (size, id) => {
        const m = MINIMUM[workspace.widgets[id]?.type] ?? {
          width: 280,
          height: 200,
        };
        return {
          width: Math.max(size.width, m.width),
          height: Math.max(size.height, m.height),
        };
      },
      { width: 0, height: 0 },
    );
  const [a, b] = node.children.map((n) => minimum(n, workspace));
  return node.axis === 'horizontal'
    ? {
        width: a.width + b.width + DIVIDER,
        height: Math.max(a.height, b.height),
      }
    : {
        width: Math.max(a.width, b.width),
        height: a.height + b.height + DIVIDER,
      };
}
export function fits(node: Layout | null, ws: Workspace, size: Size): boolean {
  if (!node) return true;
  const m = minimum(node, ws);
  if (m.width > size.width || m.height > size.height) return false;
  if (node.kind === 'stack') return true;
  const total =
    (node.axis === 'horizontal' ? size.width : size.height) - DIVIDER;
  const first = (total * node.step) / (node.axis === 'horizontal' ? 24 : 16);
  return node.children.every((child, i) =>
    fits(
      child,
      ws,
      node.axis === 'horizontal'
        ? { width: i === 0 ? first : total - first, height: size.height }
        : { width: size.width, height: i === 0 ? first : total - first },
    ),
  );
}
export function stack(widget: string): Layout {
  return { kind: 'stack', id: uid(), widgets: [widget], active: widget };
}
export function findStack(node: Layout | null, widget: string): string | null {
  if (!node) return null;
  return node.kind === 'stack'
    ? node.widgets.includes(widget)
      ? node.id
      : null
    : (findStack(node.children[0], widget) ??
        findStack(node.children[1], widget));
}
export function mapNode(
  node: Layout,
  id: string,
  fn: (n: Layout) => Layout,
): Layout {
  if (node.id === id) return fn(node);
  return node.kind === 'stack'
    ? node
    : {
        ...node,
        children: node.children.map((n) => mapNode(n, id, fn)) as [
          Layout,
          Layout,
        ],
      };
}
export function remove(node: Layout | null, widget: string): Layout | null {
  if (!node) return null;
  if (node.kind === 'stack') {
    const widgets = node.widgets.filter((id) => id !== widget);
    return widgets.length
      ? {
          ...node,
          widgets,
          active: widgets.includes(node.active) ? node.active : widgets[0],
        }
      : null;
  }
  const a = remove(node.children[0], widget),
    b = remove(node.children[1], widget);
  return a && b ? { ...node, children: [a, b] } : (a ?? b);
}
export function place(
  ws: Workspace,
  widget: string,
  target: string | null,
  position: Position,
  size?: Size,
  before?: string,
): Workspace {
  if (ws.locked)
    throw new Error('Unlock the layout before moving or adding widgets.');
  if (!ws.widgets[widget]) throw new Error('Widget no longer exists.');
  const originalStack = findStack(ws.layout, widget);
  if (target === originalStack && position === 'tab' && before === widget)
    return ws;
  if (target === originalStack && position !== 'tab')
    throw new Error('Choose another tile to split.');
  let layout = remove(ws.layout, widget);
  if (!layout) return { ...ws, layout: stack(widget) };
  if (!target) throw new Error('Choose a destination tile.');
  let found = false;
  layout = mapNode(layout, target, (n) => {
    found = true;
    if (n.kind !== 'stack') throw new Error('Choose a tab stack.');
    if (position === 'tab') {
      const widgets = [...n.widgets];
      const index = before ? widgets.indexOf(before) : -1;
      widgets.splice(index < 0 ? widgets.length : index, 0, widget);
      return { ...n, widgets, active: widget };
    }
    const first = position === 'left' || position === 'top';
    return {
      kind: 'split',
      id: uid(),
      axis:
        position === 'left' || position === 'right' ? 'horizontal' : 'vertical',
      step: position === 'left' || position === 'right' ? 12 : 8,
      children: first ? [stack(widget), n] : [n, stack(widget)],
    };
  });
  if (!found) {
    if (target === originalStack && position === 'tab') return ws;
    throw new Error('The destination is no longer available.');
  }
  let result = { ...ws, layout };
  if (size && !fits(layout, result, size)) {
    const adjusted = reflow(layout, result, size);
    if (adjusted) {
      layout = adjusted;
      result = { ...ws, layout };
    }
  }
  if (size && !fits(layout, result, size))
    throw new Error(
      'Workspace has no room at readable minimum sizes. Choose a tab or a new workspace.',
    );
  return result;
}
export function resize(
  ws: Workspace,
  id: string,
  step: number,
  size: Size,
): Workspace {
  if (ws.locked) throw new Error('Layout is locked.');
  const layout =
    ws.layout &&
    mapNode(ws.layout, id, (n) => {
      if (n.kind !== 'split') return n;
      const snapped = Math.round(step),
        max = n.axis === 'horizontal' ? 23 : 15;
      if (snapped < 1 || snapped > max)
        throw new Error('Divider step is outside its grid.');
      return { ...n, step: snapped };
    });
  const result = { ...ws, layout };
  if (!fits(layout, result, size))
    throw new Error('Minimum widget size reached.');
  return result;
}

/** Preserve tile order, changing snapped divider ratios only when needed to fit. */
export function reflow(node: Layout, ws: Workspace, size: Size): Layout | null {
  const m = minimum(node, ws);
  if (m.width > size.width || m.height > size.height) return null;
  if (node.kind === 'stack') return node;
  const steps = node.axis === 'horizontal' ? 24 : 16;
  const total =
    (node.axis === 'horizontal' ? size.width : size.height) - DIVIDER;
  const a = minimum(node.children[0], ws),
    b = minimum(node.children[1], ws);
  const low = Math.max(
    1,
    Math.ceil(
      ((node.axis === 'horizontal' ? a.width : a.height) / total) * steps,
    ),
  );
  const high = Math.min(
    steps - 1,
    Math.floor(
      (1 - (node.axis === 'horizontal' ? b.width : b.height) / total) * steps,
    ),
  );
  const candidates = Array.from(
    { length: Math.max(0, high - low + 1) },
    (_, i) => i + low,
  ).sort((x, y) => Math.abs(x - node.step) - Math.abs(y - node.step));
  for (const step of candidates) {
    const first = (total * step) / steps;
    const left = reflow(
      node.children[0],
      ws,
      node.axis === 'horizontal'
        ? { width: first, height: size.height }
        : { width: size.width, height: first },
    );
    if (!left) continue;
    const right = reflow(
      node.children[1],
      ws,
      node.axis === 'horizontal'
        ? { width: total - first, height: size.height }
        : { width: size.width, height: total - first },
    );
    if (right) return { ...node, step, children: [left, right] };
  }
  return null;
}
export function suggestPlacement(
  ws: Workspace,
  id: string,
  size: Size,
  preferred: string | null = null,
): { widget: string; position: Position } | null {
  if (ws.locked || !ws.layout) return null;
  const leaves: { node: Extract<Layout, { kind: 'stack' }>; area: number }[] =
    [];
  function visit(n: Layout, s: Size) {
    if (n.kind === 'stack') {
      leaves.push({ node: n, area: s.width * s.height });
      return;
    }
    const length = (n.axis === 'horizontal' ? s.width : s.height) - DIVIDER,
      first = (length * n.step) / (n.axis === 'horizontal' ? 24 : 16);
    visit(
      n.children[0],
      n.axis === 'horizontal'
        ? { width: first, height: s.height }
        : { width: s.width, height: first },
    );
    visit(
      n.children[1],
      n.axis === 'horizontal'
        ? { width: length - first, height: s.height }
        : { width: s.width, height: length - first },
    );
  }
  visit(ws.layout, size);
  leaves.sort(
    (a, b) =>
      Number(b.node.widgets.includes(preferred ?? '')) -
        Number(a.node.widgets.includes(preferred ?? '')) || b.area - a.area,
  );
  for (const { node } of leaves)
    for (const position of ['right', 'bottom'] as Position[]) {
      try {
        place(ws, id, node.id, position, size);
        return { widget: node.active, position };
      } catch {
        /* Try the next legal split without changing the workspace. */
      }
    }
  return null;
}
