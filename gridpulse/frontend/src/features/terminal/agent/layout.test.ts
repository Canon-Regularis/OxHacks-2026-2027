import { expect, it } from 'vitest';
import { emptyWorkspace } from '../model/document';
import {
  createWidget,
  type Layout,
  type Size,
  type Workspace,
} from '../model/types';
import { fits } from '../layout/engine';
import { researchLayout } from './layout';
const range = { from: '2026-02-01', to: '2026-03-31' };
function frames(node: Layout, size: Size): { id: string; size: Size }[] {
  if (node.kind === 'stack') return [{ id: node.active, size }];
  const horizontal = node.axis === 'horizontal',
    total = (horizontal ? size.width : size.height) - 8,
    first = (total * node.step) / (horizontal ? 24 : 16);
  return node.children.flatMap((child, i) =>
    frames(
      child,
      horizontal
        ? { width: i === 0 ? first : total - first, height: size.height }
        : { width: size.width, height: i === 0 ? first : total - first },
    ),
  );
}
function setup(count: number) {
  const ws = emptyWorkspace(range);
  const charts = Array.from({ length: count }, () =>
    createWidget('chart', range),
  );
  const chat = createWidget('chat', range);
  for (const w of [...charts, chat]) ws.widgets[w.id] = w;
  for (const chart of charts) {
    chart.binding = chart.id + '-data';
    ws.bindings[chart.binding] = {
      id: chart.binding,
      kind: 'source',
      ids: ['price'],
    };
  }
  return { ws, charts, chat };
}
it('gives a single chart full width above chat rather than a portrait column', () => {
  const { ws, charts, chat } = setup(1),
    size = { width: 1400, height: 850 };
  const tree = researchLayout(ws, size, [charts[0].id], chat.id, false)!;
  const frame = frames(tree, size).find((f) => f.id === charts[0].id)!;
  expect(frame.size.width / frame.size.height).toBeGreaterThan(1.6);
  expect(fits(tree, ws, size)).toBe(true);
});
it('stacks multiple charts in a wide main column with a compact chat sidebar', () => {
  const { ws, charts, chat } = setup(2),
    size = { width: 1400, height: 850 };
  const tree = researchLayout(
    ws,
    size,
    charts.map((w) => w.id),
    chat.id,
    false,
  )!;
  for (const chart of charts) {
    const frame = frames(tree, size).find((f) => f.id === chart.id)!;
    expect(frame.size.width).toBeGreaterThan(900);
    expect(frame.size.width / frame.size.height).toBeGreaterThan(2);
  }
  expect(
    frames(tree, size).find((f) => f.id === chat.id)!.size.width,
  ).toBeLessThan(400);
});
it('keeps crowded content accessible as tabs and never foregrounds an empty inspector', () => {
  const { ws, charts, chat } = setup(6),
    size = { width: 1200, height: 600 };
  const inspector = createWidget('inspector', range);
  ws.widgets[inspector.id] = inspector;
  const tree = researchLayout(
    ws,
    size,
    charts.map((w) => w.id),
    chat.id,
    false,
  )!;
  expect(fits(tree, ws, size)).toBe(true);
  const ids: string[] = [];
  const visit = (node: Layout) => {
    if (node.kind === 'stack') ids.push(...node.widgets);
    else node.children.forEach(visit);
  };
  visit(tree);
  expect(new Set(ids)).toEqual(new Set(Object.keys(ws.widgets)));
  expect(frames(tree, size).some((frame) => frame.id === inspector.id)).toBe(
    false,
  );
});
