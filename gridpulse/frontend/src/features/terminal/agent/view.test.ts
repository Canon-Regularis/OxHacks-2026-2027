import { describe, it, expect } from 'vitest';
const catalog: Catalog = {
  series: [
    {
      id: 'price',
      label: 'Price',
      unit: 'GBP/MWh',
      kind: 'settlement',
      datasets: [],
    },
  ],
  algorithms: [],
  datasets: [],
  limits: { max_points: 5000, max_range_days: 3660 },
  data_updated_at_utc: null,
};
import type { Catalog } from '$lib/types';
import { createDocument } from '../model/document';
import { createWidget, type Layout } from '../model/types';
import { compileView, persistentDocument } from './view';
import type { View } from './types';
const range = { from: '2026-10-25', to: '2026-10-25' };
const size = { width: 1400, height: 700 };
const intent = (): View => ({
  version: 1,
  from_date: range.from,
  to_date: range.to,
  widgets: [
    {
      key: 'price',
      existing: null,
      type: 'chart',
      title: 'Price evidence',
      series: ['price'],
      algorithm: null,
      run_id: null,
      outputs: [],
      table: '',
      group: 'A',
      relative_to: null,
      position: 'right',
      notes: '',
    },
  ],
  remove: [],
  background_chat: false,
  new_workspace: false,
  name: 'Investigation',
});
describe('agent view transactions', () => {
  it('adds a real source chart without mutating the original', () => {
    const doc = createDocument(range);
    const before = JSON.stringify(doc);
    const result = compileView(doc, intent(), catalog, size, 'chat');
    expect(
      Object.values(result.workspaces[0].widgets).some(
        (w) => w.title === 'Price evidence',
      ),
    ).toBe(true);
    expect(JSON.stringify(doc)).toBe(before);
  });
  it('rejects unknown data atomically', () => {
    const doc = createDocument(range),
      before = JSON.stringify(doc),
      view = intent();
    view.widgets[0].series = ['invented'];
    expect(() => compileView(doc, view, catalog, size, 'chat')).toThrow();
    expect(JSON.stringify(doc)).toBe(before);
  });
  it('protects chat and excludes it from persistent configurations', () => {
    const doc = createDocument(range),
      ws = doc.workspaces[0];
    const chat = createWidget('chat', range);
    ws.widgets[chat.id] = chat;
    ws.layout = {
      kind: 'stack',
      id: crypto.randomUUID(),
      widgets: Object.keys(ws.widgets),
      active: chat.id,
    };
    const view = intent();
    view.remove = [chat.id];
    expect(() => compileView(doc, view, catalog, size, chat.id)).toThrow(
      /chat/,
    );
    expect(
      Object.values(persistentDocument(doc).workspaces[0].widgets).some(
        (w) => w.type === 'chat',
      ),
    ).toBe(false);
    expect(ws.widgets[chat.id]).toBeDefined();
  });
  it('refuses a locked workspace', () => {
    const doc = createDocument(range);
    doc.workspaces[0].locked = true;
    expect(() => compileView(doc, intent(), catalog, size, 'chat')).toThrow(
      /Unlock/,
    );
  });
});

it('keeps chat exactly second in a setup stack and carries it into an explicit new workspace', () => {
  const doc = createDocument(range),
    ws = doc.workspaces[0];
  const chat = createWidget('chat', range);
  chat.group = null;
  ws.widgets[chat.id] = chat;
  ws.layout = {
    kind: 'stack',
    id: crypto.randomUUID(),
    widgets: Object.keys(ws.widgets),
    active: chat.id,
  };
  const view = intent();
  view.background_chat = true;
  view.widgets[0].position = 'tab';
  const result = compileView(doc, view, catalog, size, chat.id);
  const findChat = (node: Layout): Extract<Layout, { kind: 'stack' }> | null =>
    node.kind === 'stack'
      ? node.widgets.includes(chat.id)
        ? node
        : null
      : (findChat(node.children[0]) ?? findChat(node.children[1]));
  const stack = findChat(result.workspaces[0].layout!)!;
  expect(stack.widgets[1]).toBe(chat.id);
  expect(result.workspaces[0].widgets[stack.active].title).toBe(
    'Price evidence',
  );
  view.new_workspace = true;
  const branched = compileView(doc, view, catalog, size, chat.id);
  expect(branched.workspaces).toHaveLength(2);
  expect(branched.workspaces[0].widgets[chat.id]).toBeUndefined();
  expect(branched.workspaces[1].widgets[chat.id].type).toBe('chat');
});

it('compiles the real Kimi source-view response through the application boundary', async () => {
  const fixture =
    await import('../../../../../contracts/agent/view.example.json');
  const sources: Catalog = {
    ...catalog,
    series: [
      ...catalog.series,
      { id: 'wind', label: 'Wind', unit: 'MW', kind: 'forecast', datasets: [] },
    ],
  };
  const ws = createDocument(range).workspaces[0];
  const chat = createWidget('chat', range);
  chat.group = null;
  ws.widgets = { [chat.id]: chat };
  ws.bindings = {};
  ws.layout = {
    kind: 'stack',
    id: crypto.randomUUID(),
    widgets: [chat.id],
    active: chat.id,
  };
  const doc = { version: 1 as const, active: ws.id, workspaces: [ws] };
  const result = compileView(
    doc,
    JSON.parse(JSON.stringify(fixture.default)),
    sources,
    size,
    chat.id,
  );
  expect(
    Object.values(result.workspaces[0].widgets).filter(
      (w) => w.type === 'chart',
    ),
  ).toHaveLength(1);
  expect(result.workspaces[0].widgets[chat.id]).toBeDefined();
});

it('retains source overlays when the agent selects specific algorithm outputs', () => {
  const algoCatalog: Catalog = {
    ...catalog,
    algorithms: [
      {
        id: 'forecast',
        version: '1',
        description: 'Forecast',
        inputs: [],
        datasets: [],
        outputs: { prediction: 'GBP/MWh' },
        tables: {},
        params: {},
        param_descriptions: {},
        limitations: 'Example',
        latest_run: null,
      },
    ],
  };
  const view = intent();
  Object.assign(view.widgets[0], {
    algorithm: 'forecast',
    run_id: 49,
    outputs: ['prediction'],
  });
  const result = compileView(
    createDocument(range),
    view,
    algoCatalog,
    size,
    'chat',
  );
  const widget = Object.values(result.workspaces[0].widgets).find(
    (w) => w.title === 'Price evidence',
  )!;
  expect(widget.settings.outputs).toEqual(['prediction', 'source:price']);
});

it('places multiple new charts relative to one another in one atomic AI view', () => {
  const doc = createDocument(range);
  const sources: Catalog = {
    ...catalog,
    series: [
      ...catalog.series,
      { id: 'wind', label: 'Wind', unit: 'MW', kind: 'forecast', datasets: [] },
    ],
  };
  const view = intent();
  view.layout_mode = 'preserve';
  view.widgets.push({
    ...view.widgets[0],
    key: 'wind',
    title: 'Wind evidence',
    series: ['wind'],
    relative_to: 'price',
    position: 'bottom',
  });
  const before = JSON.stringify(doc);
  const result = compileView(
    doc,
    view,
    sources,
    { width: 1800, height: 1200 },
    'chat',
  );
  expect(
    Object.values(result.workspaces[0].widgets).filter((w) =>
      ['Price evidence', 'Wind evidence'].includes(w.title),
    ),
  ).toHaveLength(2);
  expect(JSON.stringify(doc)).toBe(before);
});

it('preserves the original workspace if a later chart in an AI batch fails', () => {
  const doc = createDocument(range),
    before = JSON.stringify(doc),
    view = intent();
  view.widgets.push({
    ...view.widgets[0],
    key: 'second',
    series: ['unknown'],
    relative_to: 'price',
  });
  expect(() =>
    compileView(doc, view, catalog, { width: 1800, height: 1200 }, 'chat'),
  ).toThrow(/unavailable data/);
  expect(JSON.stringify(doc)).toBe(before);
});

it('accepts source-series output selections instead of treating them as algorithm keys', () => {
  const view = intent();
  view.widgets[0].outputs = ['price'];
  const result = compileView(
    createDocument(range),
    view,
    catalog,
    size,
    'chat',
  );
  expect(
    Object.values(result.workspaces[0].widgets).find(
      (w) => w.title === 'Price evidence',
    )!.settings.outputs,
  ).toEqual(['price']);
});

it('compiles the captured OpenRouter research view including an unbound explanation widget', async () => {
  const fixture =
    await import('../../../../../contracts/agent/view.openrouter.example.json');
  const sources: Catalog = {
    ...catalog,
    series: [
      ...catalog.series,
      { id: 'wind', label: 'Wind', unit: 'MW', kind: 'forecast', datasets: [] },
      {
        id: 'demand',
        label: 'Demand',
        unit: 'MW',
        kind: 'forecast',
        datasets: [],
      },
    ],
  };
  const result = compileView(
    createDocument(range),
    JSON.parse(JSON.stringify(fixture.default)),
    sources,
    { width: 1400, height: 850 },
    'chat',
  );
  expect(
    Object.values(result.workspaces[0].widgets).some(
      (w) => w.type === 'context' && w.title === 'Notes',
    ),
  ).toBe(true);
  expect(
    Object.values(result.workspaces[0].widgets).some(
      (w) => w.type === 'table' && w.title === 'Wind and demand data table',
    ),
  ).toBe(true);
});

it('rearranges existing panels without rewriting bindings or data-link groups', () => {
  const doc = createDocument(range),
    ws = doc.workspaces[0];
  const before = JSON.stringify({
    widgets: ws.widgets,
    bindings: ws.bindings,
    groups: ws.groups,
  });
  const view = intent();
  view.layout_mode = 'separate';
  view.widgets = [];
  const result = compileView(doc, view, catalog, size, 'chat');
  expect(
    JSON.stringify({
      widgets: result.workspaces[0].widgets,
      bindings: result.workspaces[0].bindings,
      groups: result.workspaces[0].groups,
    }),
  ).toBe(before);
});
