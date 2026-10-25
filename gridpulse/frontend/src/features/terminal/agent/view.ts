/** Compile model intent through the same atomic commands and layout engine as manual edits. */
import type { Catalog } from '$lib/types';
import { applyActions, type WorkspaceAction } from '../model/actions';
import { emptyWorkspace, closeWidget, parseDocument } from '../model/document';
import {
  createWidget,
  context,
  uid,
  type Size,
  type TerminalDocument,
  type Workspace,
} from '../model/types';
import { findStack, fits } from '../layout/engine';
import type { View } from './types';

export const revision = (workspace: Workspace) => JSON.stringify(workspace);
export function persistentDocument(doc: TerminalDocument): TerminalDocument {
  const copy = structuredClone(doc);
  const clean = (ws: Workspace) => {
    for (const w of Object.values(ws.widgets))
      if (w.type === 'chat') ws = closeWidget(ws, w.id);
    return ws;
  };
  copy.workspaces = copy.workspaces.map(clean);
  copy.presets = copy.presets?.map(clean);
  return copy;
}
export function compileView(
  document: TerminalDocument,
  view: View,
  catalog: Catalog,
  size: Size,
  chat: string,
): TerminalDocument {
  if (
    !view ||
    view.version !== 1 ||
    !Array.isArray(view.widgets) ||
    view.widgets.length > 12 ||
    !Array.isArray(view.remove) ||
    view.remove.length > 12
  )
    throw new Error('Invalid AI view contract.');
  const range = { from: view.from_date, to: view.to_date };
  if (
    !/^\d{4}-\d{2}-\d{2}$/.test(range.from) ||
    !/^\d{4}-\d{2}-\d{2}$/.test(range.to) ||
    range.from > range.to
  )
    throw new Error('Invalid AI date range.');
  const sourceChat = document.workspaces.find((w) => w.id === document.active)
    ?.widgets[chat];
  let next = structuredClone(document);
  let ws = next.workspaces.find((w) => w.id === next.active)!;
  if (ws.locked)
    throw new Error('Unlock the workspace before AI can rearrange it.');
  if (view.new_workspace) {
    const created = emptyWorkspace(range, view.name);
    if (sourceChat) {
      next.workspaces = next.workspaces.map((w) =>
        w.id === next.active ? closeWidget(w, chat) : w,
      );
      created.widgets[chat] = structuredClone(sourceChat);
      created.widgets[chat].group = null;
      created.layout = {
        kind: 'stack',
        id: uid(),
        widgets: [chat],
        active: chat,
      };
    }
    next = applyActions(
      next,
      [{ type: 'workspace.add', workspace: created }],
      size,
    );
    ws = next.workspaces.find((w) => w.id === next.active)!;
  }
  const commands: WorkspaceAction[] = [];
  const keys = new Map<string, string>();
  const existing = new Set<string>();
  const ids = new Set<string>();
  for (const spec of view.widgets) {
    if (
      !spec ||
      typeof spec.key !== 'string' ||
      !spec.key ||
      keys.has(spec.key) ||
      !['chart', 'table', 'comparison', 'inspector', 'context'].includes(
        spec.type,
      )
    )
      throw new Error('Invalid or duplicate AI widget.');
    if (
      spec.existing &&
      (!Object.hasOwn(ws.widgets, spec.existing) ||
        ws.widgets[spec.existing].type === 'chat' ||
        existing.has(spec.existing))
    )
      throw new Error(
        'AI references a missing, duplicate or protected widget.',
      );
    if (spec.existing) existing.add(spec.existing);
    const id = spec.existing ?? uid();
    keys.set(spec.key, id);
    ids.add(id);
  }
  for (const id of view.remove) {
    if (
      !Object.hasOwn(ws.widgets, id) ||
      ws.widgets[id].type === 'chat' ||
      existing.has(id)
    )
      throw new Error('AI cannot remove chat or a missing/reused widget.');
    commands.push({ type: 'widget.close', id });
  }
  const groups = new Set(Object.keys(ws.groups));
  let target: string | null =
    Object.values(ws.widgets).find(
      (w) => !view.remove.includes(w.id) && w.type !== 'chat',
    )?.id ?? (ws.widgets[chat] ? chat : null);
  for (const spec of view.widgets) {
    const algorithm = spec.algorithm
      ? catalog.algorithms.find((a) => a.id === spec.algorithm)
      : null;
    if (
      !Array.isArray(spec.series) ||
      spec.series.some((id) => !catalog.series.some((s) => s.id === id)) ||
      (spec.algorithm &&
        (!algorithm ||
          !Number.isSafeInteger(spec.run_id) ||
          Number(spec.run_id) < 1))
    )
      throw new Error('AI references unavailable data or an unpinned run.');
    if (
      !Array.isArray(spec.outputs) ||
      spec.outputs.some(
        (id) => !algorithm || !Object.hasOwn(algorithm.outputs, id),
      ) ||
      (spec.table &&
        (!algorithm || !Object.hasOwn(algorithm.tables, spec.table)))
    )
      throw new Error('AI references unknown algorithm results.');
    const units = new Set([
      ...spec.series.map((id) => catalog.series.find((s) => s.id === id)!.unit),
      ...spec.outputs.map((id) => algorithm!.outputs[id]),
    ]);
    if (spec.type === 'chart' && units.size > 2)
      throw new Error('Use separate charts for more than two units.');
    if (spec.group && !groups.has(spec.group)) {
      if (!/^[A-Z]$/.test(spec.group))
        throw new Error('Invalid AI linked group.');
      groups.add(spec.group);
      commands.push({
        type: 'group.create',
        group: {
          id: spec.group,
          name: `Group ${spec.group}`,
          color: '#a2c7b8',
          ...context(range),
        },
      });
    }
    if (spec.group)
      commands.push({
        type: 'context.set',
        group: spec.group,
        context: context(range),
      });
    const id = keys.get(spec.key)!;
    const binding = algorithm
      ? {
          id: uid(),
          kind: 'algorithm' as const,
          algorithm: algorithm.id,
          runId: spec.run_id,
          sources: spec.series,
        }
      : spec.series.length
        ? { id: uid(), kind: 'source' as const, ids: spec.series }
        : undefined;
    const widget = createWidget(spec.type, range, binding?.id ?? null);
    Object.assign(widget, { id, title: spec.title, group: spec.group });
    Object.assign(widget.settings, {
      outputs: spec.outputs,
      table: spec.table,
      notes: spec.notes,
    });
    const reference = spec.relative_to
      ? (keys.get(spec.relative_to) ?? spec.relative_to)
      : target;
    const current = applyActions(next, commands, size);
    const live = current.workspaces.find((w) => w.id === current.active)!;
    const destination = reference ? findStack(live.layout, reference) : null;
    if (reference && !destination)
      throw new Error('AI layout references an unavailable placement target.');
    if (spec.existing) {
      if (ws.widgets[id].type !== spec.type)
        throw new Error('Use a new widget when changing renderer type.');
      if (binding)
        commands.push({ type: 'binding.set', binding, widgetIds: [id] });
      commands.push(
        {
          type: 'widget.configure',
          id,
          title: spec.title,
          settings: widget.settings,
        },
        { type: 'widget.link', id, group: spec.group },
      );
      if (destination && reference !== id)
        commands.push({
          type: 'widget.move',
          id,
          target: destination,
          position: spec.position,
        });
    } else
      commands.push({
        type: 'widget.add',
        widget,
        binding,
        target: destination,
        position: spec.position,
      });
    target = id;
  }
  next = applyActions(next, commands, size);
  ws = next.workspaces.find((w) => w.id === next.active)!;
  if (view.background_chat && ws.widgets[chat] && target && target !== chat) {
    const destination = findStack(ws.layout, target)!;
    next = applyActions(
      next,
      [
        { type: 'widget.move', id: chat, target: destination, position: 'tab' },
        { type: 'widget.activate', id: target },
      ],
      size,
    );
  }
  ws = next.workspaces.find((w) => w.id === next.active)!;
  if (!fits(ws.layout, ws, size) && size.width >= 850)
    throw new Error(
      'Workspace has no room at readable minimum sizes. Remove widgets or request a new workspace.',
    );
  // Drop unused bindings accumulated by rebinding; all remaining references are validated.
  for (const key of Object.keys(ws.bindings))
    if (!Object.values(ws.widgets).some((w) => w.binding === key))
      delete ws.bindings[key];
  return parseDocument(JSON.stringify(next));
}
