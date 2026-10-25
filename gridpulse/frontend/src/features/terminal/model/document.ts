import {
  type TerminalDocument,
  type Workspace,
  type Range,
  type Layout,
  type Widget,
  MINIMUM,
  LIMITS,
  context,
  createWidget,
  uid,
} from './types';
import { stack, remove, mapNode, findStack } from '../layout/engine';
export const STORAGE_KEY = 'eisengrau:terminal:v1';
export function emptyWorkspace(
  range: Range,
  name = 'Untitled workspace',
): Workspace {
  return {
    id: uid(),
    name,
    layout: null,
    widgets: {},
    bindings: {},
    groups: {
      A: { id: 'A', name: 'Group A', color: '#e6b080', ...context(range) },
    },
    locked: false,
  };
}
export function preset(
  range: Range,
  kind: 'market' | 'analysis' | 'blank' = 'market',
  algorithm?: string,
): Workspace {
  const ws = emptyWorkspace(
    range,
    kind === 'market'
      ? 'Market investigation'
      : kind === 'analysis'
        ? 'Algorithm research'
        : 'Blank workspace',
  );
  if (kind === 'blank') return ws;
  const binding = uid();
  ws.bindings[binding] =
    kind === 'analysis' && algorithm
      ? { id: binding, kind: 'algorithm', algorithm, runId: null }
      : { id: binding, kind: 'source', ids: ['price', 'wind'] };
  const chart = createWidget('chart', range, binding),
    inspector = createWidget('inspector', range),
    notes = createWidget('context', range, binding);
  if (kind === 'analysis') {
    inspector.type = 'table';
    inspector.title = 'Results table';
    inspector.binding = binding;
  }
  ws.widgets = {
    [chart.id]: chart,
    [inspector.id]: inspector,
    [notes.id]: notes,
  };
  ws.layout = {
    kind: 'split',
    id: uid(),
    axis: 'horizontal',
    step: 16,
    children: [
      stack(chart.id),
      {
        kind: 'split',
        id: uid(),
        axis: 'vertical',
        step: 9,
        children: [stack(inspector.id), stack(notes.id)],
      },
    ],
  };
  return ws;
}
export function createDocument(range: Range): TerminalDocument {
  const ws = preset(range);
  return { version: 1, active: ws.id, workspaces: [ws] };
}
export function duplicateWorkspace(ws: Workspace): Workspace {
  const clone = structuredClone(ws);
  const ids = new Map(Object.keys(ws.widgets).map((id) => [id, uid()]));
  const bindings = new Map(Object.keys(ws.bindings).map((id) => [id, uid()]));
  clone.id = uid();
  clone.name = ws.name + ' copy';
  clone.widgets = {};
  clone.bindings = {};
  for (const b of Object.values(ws.bindings)) {
    const id = bindings.get(b.id)!;
    clone.bindings[id] = { ...structuredClone(b), id };
  }
  for (const w of Object.values(ws.widgets)) {
    const id = ids.get(w.id)!;
    clone.widgets[id] = {
      ...structuredClone(w),
      id,
      binding: w.binding ? bindings.get(w.binding)! : null,
    };
  }
  function copy(n: Layout): Layout {
    return n.kind === 'stack'
      ? {
          ...n,
          id: uid(),
          widgets: n.widgets.map((id) => ids.get(id)!),
          active: ids.get(n.active)!,
        }
      : {
          ...n,
          id: uid(),
          children: [copy(n.children[0]), copy(n.children[1])],
        };
  }
  clone.layout = clone.layout ? copy(clone.layout) : null;
  return clone;
}
export function closeWidget(ws: Workspace, id: string): Workspace {
  const copy = structuredClone(ws);
  delete copy.widgets[id];
  copy.layout = remove(copy.layout, id);
  for (const b of Object.keys(copy.bindings))
    if (!Object.values(copy.widgets).some((w) => w.binding === b))
      delete copy.bindings[b];
  return copy;
}
export function reveal(ws: Workspace, id: string): Workspace {
  const destination = findStack(ws.layout, id);
  return destination && ws.layout
    ? {
        ...ws,
        layout: mapNode(ws.layout, destination, (n) =>
          n.kind === 'stack' ? { ...n, active: id } : n,
        ),
      }
    : ws;
}
function object(x: unknown): x is Record<string, unknown> {
  return !!x && typeof x === 'object' && !Array.isArray(x);
}
const str = (x: unknown): x is string =>
  typeof x === 'string' && x.length <= 10000;
const date = (x: unknown): x is string =>
  typeof x === 'string' &&
  /^\d{4}-\d{2}-\d{2}$/.test(x) &&
  !Number.isNaN(Date.parse(x)) &&
  new Date(x).toISOString().slice(0, 10) === x;
function validContext(x: unknown): boolean {
  return (
    object(x) &&
    date(x.from) &&
    date(x.to) &&
    x.from <= x.to &&
    (x.selected === null ||
      (str(x.selected) && Number.isFinite(Date.parse(x.selected)))) &&
    (x.zoom === null ||
      (Array.isArray(x.zoom) &&
        x.zoom.length === 2 &&
        x.zoom.every((v) => typeof v === 'number' && Number.isFinite(v)) &&
        x.zoom[0] < x.zoom[1]))
  );
}
export function parseDocument(raw: string): TerminalDocument {
  if (new TextEncoder().encode(raw).length > LIMITS.bytes)
    throw new Error('Workspace file exceeds 1 MiB.');
  const d: unknown = JSON.parse(raw);
  if (!object(d) || d.version !== 1)
    throw new Error(
      'Unsupported workspace version. Export or recover the saved file before resetting.',
    );
  if (
    !Array.isArray(d.workspaces) ||
    !d.workspaces.length ||
    d.workspaces.length > LIMITS.workspaces
  )
    throw new Error('Invalid workspace list.');
  const globalIds = new Set<string>();
  const claim = (id: unknown) => {
    if (
      !str(id) ||
      !id ||
      ['__proto__', 'constructor', 'prototype'].includes(id) ||
      globalIds.has(id)
    )
      throw new Error('Duplicate or invalid workspace ID.');
    globalIds.add(id);
  };
  if (
    d.presets !== undefined &&
    (!Array.isArray(d.presets) || d.presets.length > LIMITS.workspaces)
  )
    throw new Error('Invalid preset list.');
  for (const ws of [...d.workspaces, ...((d.presets ?? []) as unknown[])]) {
    if (
      !object(ws) ||
      !str(ws.name) ||
      typeof ws.locked !== 'boolean' ||
      !object(ws.widgets) ||
      !object(ws.bindings) ||
      !object(ws.groups)
    )
      throw new Error('Invalid workspace configuration.');
    claim(ws.id);
    if (
      Object.keys(ws.widgets).length > LIMITS.widgets ||
      Object.keys(ws.groups).length > LIMITS.widgets
    )
      throw new Error('Workspace resource limit exceeded.');
    for (const [id, g] of Object.entries(ws.groups))
      if (
        !object(g) ||
        id !== g.id ||
        !str(g.name) ||
        !/^#[0-9a-f]{6}$/i.test(String(g.color)) ||
        !validContext(g)
      )
        throw new Error('Invalid linked group.');
    for (const [id, b] of Object.entries(ws.bindings)) {
      claim(id);
      if (
        !object(b) ||
        b.id !== id ||
        (b.kind !== 'source' && b.kind !== 'algorithm')
      )
        throw new Error('Invalid data binding.');
      if (
        b.kind === 'source' &&
        (!Array.isArray(b.ids) ||
          !b.ids.length ||
          b.ids.length > 30 ||
          !b.ids.every(str))
      )
        throw new Error('Invalid source selection.');
      if (
        b.kind === 'algorithm' &&
        b.sources !== undefined &&
        (!Array.isArray(b.sources) ||
          b.sources.length > 30 ||
          !b.sources.every(str))
      )
        throw new Error('Invalid source overlay.');
      if (
        b.kind === 'algorithm' &&
        (!str(b.algorithm) ||
          (b.runId !== null &&
            (!Number.isSafeInteger(b.runId) || Number(b.runId) < 1)))
      )
        throw new Error('Invalid stored run.');
    }
    for (const [id, w] of Object.entries(ws.widgets)) {
      claim(id);
      if (
        !object(w) ||
        id !== w.id ||
        !str(w.type) ||
        !str(w.title) ||
        !validContext(w.context) ||
        !object(w.settings)
      )
        throw new Error('Invalid widget.');
      if (
        w.binding !== null &&
        (!str(w.binding) || !Object.hasOwn(ws.bindings, w.binding))
      )
        throw new Error('Missing data binding.');
      if (
        w.group !== null &&
        (!str(w.group) || !Object.hasOwn(ws.groups, w.group))
      )
        throw new Error('Missing linked group.');
      const s = w.settings;
      if (
        !Array.isArray(s.outputs) ||
        s.outputs.length > 30 ||
        !s.outputs.every(str) ||
        !str(s.table) ||
        !str(s.metric) ||
        !str(s.category) ||
        !str(s.notes) ||
        !['overlay', 'panes', 'indexed'].includes(String(s.display)) ||
        !object(s.styles)
      )
        throw new Error('Invalid widget settings.');
      for (const style of Object.values(s.styles))
        if (
          !object(style) ||
          !/^#[0-9a-f]{6}$/i.test(String(style.color)) ||
          typeof style.dashed !== 'boolean'
        )
          throw new Error('Invalid series style.');
    }
    const placed = new Set<string>();
    function visit(n: unknown, depth: number) {
      if (depth > 24 || !object(n))
        throw new Error('Invalid or excessively nested layout.');
      claim(n.id);
      if (n.kind === 'stack') {
        if (
          !Array.isArray(n.widgets) ||
          !n.widgets.length ||
          !n.widgets.includes(n.active)
        )
          throw new Error('Invalid tab stack.');
        for (const id of n.widgets) {
          if (
            !str(id) ||
            placed.has(id) ||
            !Object.hasOwn(ws.widgets as object, id)
          )
            throw new Error('Duplicate or missing widget placement.');
          placed.add(id);
        }
      } else if (n.kind === 'split') {
        if (
          !['horizontal', 'vertical'].includes(String(n.axis)) ||
          !Number.isInteger(n.step) ||
          Number(n.step) < 1 ||
          Number(n.step) > (n.axis === 'horizontal' ? 23 : 15) ||
          !Array.isArray(n.children) ||
          n.children.length !== 2
        )
          throw new Error('Invalid divider.');
        for (const child of n.children) visit(child, depth + 1);
      } else throw new Error('Unknown layout node.');
    }
    if (ws.layout !== null) visit(ws.layout, 0);
    if (placed.size !== Object.keys(ws.widgets).length)
      throw new Error('Unplaced widgets in workspace.');
  }
  if (!d.workspaces.some((ws) => ws.id === d.active))
    throw new Error('Missing active workspace.');
  return d as unknown as TerminalDocument;
}
export class History {
  private past: string[] = [];
  private future: string[] = [];
  get canUndo() {
    return this.past.length > 0;
  }
  get canRedo() {
    return this.future.length > 0;
  }
  record(before: TerminalDocument, after: TerminalDocument) {
    parseDocument(JSON.stringify(after));
    const a = JSON.stringify(before),
      b = JSON.stringify(after);
    if (a === b) return;
    this.past.push(a);
    this.past = this.past.slice(-LIMITS.history);
    this.future = [];
  }
  undo(current: TerminalDocument) {
    const raw = this.past.pop();
    if (!raw) return current;
    this.future.push(JSON.stringify(current));
    return parseDocument(raw);
  }
  redo(current: TerminalDocument) {
    const raw = this.future.pop();
    if (!raw) return current;
    this.past.push(JSON.stringify(current));
    return parseDocument(raw);
  }
}
export function isWidgetType(type: string): type is Widget['type'] {
  return Object.hasOwn(MINIMUM, type);
}
