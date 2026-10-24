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
): Workspace {
  if (ws.locked)
    throw new Error('Unlock the layout before moving or adding widgets.');
  if (!ws.widgets[widget]) throw new Error('Widget no longer exists.');
  const originalStack = findStack(ws.layout, widget);
  if (target === originalStack && position !== 'tab')
    throw new Error('Choose another tile to split.');
  let layout = remove(ws.layout, widget);
  if (!layout) return { ...ws, layout: stack(widget) };
  if (!target) throw new Error('Choose a destination tile.');
  let found = false;
  layout = mapNode(layout, target, (n) => {
    found = true;
    if (n.kind !== 'stack') throw new Error('Choose a tab stack.');
    if (position === 'tab')
      return { ...n, widgets: [...n.widgets, widget], active: widget };
    const before = position === 'left' || position === 'top';
    return {
      kind: 'split',
      id: uid(),
      axis:
        position === 'left' || position === 'right' ? 'horizontal' : 'vertical',
      step: position === 'left' || position === 'right' ? 12 : 8,
      children: before ? [stack(widget), n] : [n, stack(widget)],
    };
  });
  if (!found) {
    if (target === originalStack && position === 'tab') return ws;
    throw new Error('The destination is no longer available.');
  }
  const result = { ...ws, layout };
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
