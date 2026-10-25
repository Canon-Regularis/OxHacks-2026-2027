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
import { createWidget } from '../model/types';
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
