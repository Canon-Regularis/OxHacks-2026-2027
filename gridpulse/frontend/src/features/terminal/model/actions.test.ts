import { it, expect } from 'vitest';
import { createDocument } from './document';
import { createWidget, uid } from './types';
import { applyActions } from './actions';
import { findStack } from '../layout/engine';
const range = { from: '2026-10-25', to: '2026-10-25' },
  size = { width: 1400, height: 650 };
it('validates a batch atomically, preserving the original after any rejection', () => {
  const d = createDocument(range),
    ws = d.workspaces[0],
    id = Object.keys(ws.widgets)[0];
  const before = JSON.stringify(d);
  expect(() =>
    applyActions(
      d,
      [
        { type: 'widget.configure', id, title: 'Changed' },
        { type: 'widget.link', id, group: 'missing' },
      ],
      size,
    ),
  ).toThrow('missing');
  expect(JSON.stringify(d)).toBe(before);
});
it('uses the same placement rules for command-driven and manual additions', () => {
  const d = createDocument(range),
    ws = d.workspaces[0],
    target = findStack(ws.layout, Object.keys(ws.widgets)[1])!,
    widget = createWidget('chart', range);
  expect(() =>
    applyActions(
      d,
      [{ type: 'widget.add', widget, target, position: 'left' }],
      size,
    ),
  ).toThrow('no room');
  const added = applyActions(
    d,
    [{ type: 'widget.add', widget, target, position: 'tab' }],
    size,
  );
  expect(Object.keys(added.workspaces[0].widgets)).toHaveLength(4);
});
it('keeps linked charts and tables on one binding in a transaction', () => {
  const d = createDocument(range),
    ids = Object.keys(d.workspaces[0].widgets),
    binding = {
      id: uid(),
      kind: 'algorithm' as const,
      algorithm: 'future_algo',
      runId: 42,
    };
  const next = applyActions(
    d,
    [{ type: 'binding.set', binding, widgetIds: ids.slice(0, 2) }],
    size,
  );
  expect(
    ids.slice(0, 2).map((id) => next.workspaces[0].widgets[id].binding),
  ).toEqual([binding.id, binding.id]);
});
