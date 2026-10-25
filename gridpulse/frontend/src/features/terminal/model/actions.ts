/** Validated, atomic commands used by UI controls and available to future agent tools. */
import type {
  Binding,
  Group,
  LinkContext,
  Position,
  Size,
  TerminalDocument,
  Widget,
  Workspace,
} from './types';
import { LIMITS } from './types';
import { closeWidget, parseDocument, reveal } from './document';
import { place, resize } from '../layout/engine';
export type WorkspaceAction =
  | {
      type: 'widget.add';
      widget: Widget;
      binding?: Binding;
      target: string | null;
      position: Position;
      before?: string;
    }
  | {
      type: 'widget.move';
      id: string;
      target: string;
      position: Position;
      before?: string;
    }
  | { type: 'widget.close'; id: string }
  | {
      type: 'widget.configure';
      id: string;
      title?: string;
      settings?: Partial<Widget['settings']>;
    }
  | { type: 'widget.link'; id: string; group: string | null }
  | { type: 'widget.activate'; id: string }
  | { type: 'binding.set'; binding: Binding; widgetIds?: string[] }
  | { type: 'layout.resize'; id: string; step: number }
  | { type: 'layout.lock'; locked: boolean }
  | { type: 'group.create'; group: Group }
  | {
      type: 'context.set';
      group?: string;
      widget?: string;
      context: LinkContext;
    }
  | { type: 'workspace.add'; workspace: Workspace }
  | { type: 'workspace.activate'; id: string };
export function applyActions(
  document: TerminalDocument,
  actions: WorkspaceAction[],
  size: Size,
): TerminalDocument {
  if (actions.length > 100)
    throw new Error('Too many workspace actions in one transaction.');
  const result = JSON.parse(JSON.stringify(document)) as TerminalDocument;
  for (const action of actions) {
    if (
      !action ||
      typeof action !== 'object' ||
      typeof action.type !== 'string'
    )
      throw new Error('Invalid workspace action.');
    const identifiers: unknown[] = [];
    if ('id' in action) identifiers.push(action.id);
    if ('target' in action && action.target !== null)
      identifiers.push(action.target);
    if ('before' in action && action.before !== undefined)
      identifiers.push(action.before);
    if ('widget' in action && typeof action.widget === 'object')
      identifiers.push(action.widget?.id);
    if ('binding' in action) identifiers.push(action.binding?.id);
    if ('workspace' in action) identifiers.push(action.workspace?.id);
    if (
      'group' in action &&
      action.group !== null &&
      action.group !== undefined
    )
      identifiers.push(
        typeof action.group === 'string' ? action.group : action.group.id,
      );
    if ('widgetIds' in action) identifiers.push(...(action.widgetIds ?? []));
    if (
      identifiers.some(
        (id) =>
          typeof id !== 'string' ||
          !id ||
          ['__proto__', 'constructor', 'prototype'].includes(id),
      )
    )
      throw new Error('Invalid action identifier.');

    const ws = result.workspaces.find((w) => w.id === result.active);
    if (!ws) throw new Error('Active workspace is missing.');
    switch (action.type) {
      case 'widget.add': {
        if (Object.hasOwn(ws.widgets, action.widget.id))
          throw new Error('Widget ID already exists.');
        if (Object.keys(ws.widgets).length >= LIMITS.widgets)
          throw new Error('Maximum 12 widgets. Create another workspace.');
        ws.widgets[action.widget.id] = structuredClone(action.widget);
        if (action.binding)
          ws.bindings[action.binding.id] = structuredClone(action.binding);
        ws.layout = place(
          ws,
          action.widget.id,
          action.target,
          action.position,
          size,
          action.before,
        ).layout;
        break;
      }
      case 'widget.move':
        ws.layout = place(
          ws,
          action.id,
          action.target,
          action.position,
          size,
          action.before,
        ).layout;
        break;
      case 'widget.close':
        Object.assign(ws, closeWidget(ws, action.id));
        break;
      case 'widget.configure': {
        const w = ws.widgets[action.id];
        if (!Object.hasOwn(ws.widgets, action.id))
          throw new Error('Widget no longer exists.');
        if (action.title !== undefined) w.title = action.title;
        if (action.settings)
          w.settings = { ...w.settings, ...structuredClone(action.settings) };
        break;
      }
      case 'widget.link': {
        const w = ws.widgets[action.id];
        if (!Object.hasOwn(ws.widgets, action.id))
          throw new Error('Widget no longer exists.');
        if (action.group && !Object.hasOwn(ws.groups, action.group))
          throw new Error('Linked group is missing.');
        if (action.group === null && w.group)
          w.context = structuredClone(ws.groups[w.group]);
        w.group = action.group;
        break;
      }
      case 'widget.activate':
        ws.layout = reveal(ws, action.id).layout;
        break;
      case 'binding.set': {
        ws.bindings[action.binding.id] = structuredClone(action.binding);
        for (const id of action.widgetIds ?? []) {
          if (!Object.hasOwn(ws.widgets, id))
            throw new Error('Widget no longer exists.');
          ws.widgets[id].binding = action.binding.id;
        }
        break;
      }
      case 'layout.resize':
        ws.layout = resize(ws, action.id, action.step, size).layout;
        break;
      case 'layout.lock':
        ws.locked = action.locked;
        break;
      case 'group.create':
        if (Object.hasOwn(ws.groups, action.group.id))
          throw new Error('Group already exists.');
        ws.groups[action.group.id] = structuredClone(action.group);
        break;
      case 'context.set': {
        const target =
          action.group && Object.hasOwn(ws.groups, action.group)
            ? ws.groups[action.group]
            : action.widget && Object.hasOwn(ws.widgets, action.widget)
              ? ws.widgets[action.widget].context
              : undefined;
        if (!target) throw new Error('Context target is missing.');
        Object.assign(target, structuredClone(action.context));
        break;
      }
      case 'workspace.add':
        if (result.workspaces.length >= LIMITS.workspaces)
          throw new Error('Maximum 16 workspaces.');
        result.workspaces.push(structuredClone(action.workspace));
        result.active = action.workspace.id;
        break;
      case 'workspace.activate':
        if (!result.workspaces.some((w) => w.id === action.id))
          throw new Error('Workspace no longer exists.');
        result.active = action.id;
        break;
      default:
        throw new Error('Unsupported workspace action.');
    }
  }
  return parseDocument(JSON.stringify(result));
}
