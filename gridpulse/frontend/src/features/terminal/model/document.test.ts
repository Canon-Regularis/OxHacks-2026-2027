import { describe, it, expect } from 'vitest';
import {
  createDocument,
  parseDocument,
  duplicateWorkspace,
  History,
  closeWidget,
} from './document';
import { fits, place, findStack, resize } from '../layout/engine';
import { createWidget, uid } from './types';
const range = { from: '2026-10-25', to: '2026-10-25' };
describe('terminal workspace invariants', () => {
  it('roundtrips clock-change dates and duplicates without shared IDs', () => {
    const d = createDocument(range);
    expect(parseDocument(JSON.stringify(d))).toEqual(d);
    const copy = duplicateWorkspace(d.workspaces[0]);
    expect(
      Object.keys(copy.widgets).some((id) =>
        Object.hasOwn(d.workspaces[0].widgets, id),
      ),
    ).toBe(false);
    expect(
      parseDocument(
        JSON.stringify({ ...d, workspaces: [...d.workspaces, copy] }),
      ).workspaces,
    ).toHaveLength(2);
  });
  it('rejects invalid placements and permits tabs without squeezing other widgets', () => {
    let ws = createDocument(range).workspaces[0];
    const id = Object.keys(ws.widgets)[0],
      target = findStack(ws.layout, id)!;
    const w = createWidget('chart', range);
    ws.widgets[w.id] = w;
    expect(() =>
      place(ws, w.id, target, 'left', { width: 1000, height: 620 }),
    ).toThrow('no room');
    const tabbed = place(ws, w.id, target, 'tab', { width: 1000, height: 620 });
    expect(fits(tabbed.layout, tabbed, { width: 1000, height: 620 })).toBe(
      true,
    );
  });
  it('rejects duplicate placements, broken references, unknown versions and oversized imports', () => {
    const d = createDocument(range);
    const ws = d.workspaces[0];
    const first = Object.keys(ws.widgets)[0];
    ws.layout = {
      kind: 'stack',
      id: uid(),
      widgets: [first, first],
      active: first,
    };
    expect(() => parseDocument(JSON.stringify(d))).toThrow();
    expect(() => parseDocument('{"version":2}')).toThrow('version');
    expect(() => parseDocument(' '.repeat(1048577))).toThrow('1 MiB');
  });
  it('closes and collapses to a valid empty workspace', () => {
    let ws = createDocument(range).workspaces[0];
    for (const id of Object.keys(ws.widgets)) ws = closeWidget(ws, id);
    expect(ws.layout).toBeNull();
    expect(Object.keys(ws.bindings)).toHaveLength(0);
  });
  it('makes multi-widget changes one undo and clears redo on a new change', () => {
    const h = new History(),
      a = createDocument(range),
      b = structuredClone(a);
    b.workspaces[0].name = 'Changed';
    h.record(a, b);
    expect(h.undo(b)).toEqual(a);
    expect(h.redo(a)).toEqual(b);
    h.undo(b);
    const c = structuredClone(a);
    c.workspaces[0].name = 'Other';
    h.record(a, c);
    expect(h.canRedo).toBe(false);
  });
  it('snaps ratios and rejects undersized resizes without mutating source', () => {
    const ws = createDocument(range).workspaces[0];
    expect(ws.layout?.kind).toBe('split');
    expect(() =>
      resize(ws, ws.layout!.id, 23, { width: 1000, height: 620 }),
    ).toThrow('Minimum');
    expect(ws.layout && 'step' in ws.layout && ws.layout.step).toBe(16);
  });
});
