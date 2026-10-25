export type WidgetType =
  'chart' | 'table' | 'comparison' | 'inspector' | 'context' | 'chat';
export type Position = 'left' | 'right' | 'top' | 'bottom' | 'tab';
export type Layout =
  | { kind: 'stack'; id: string; widgets: string[]; active: string }
  | {
      kind: 'split';
      id: string;
      axis: 'horizontal' | 'vertical';
      step: number;
      children: [Layout, Layout];
    };
export interface Range {
  from: string;
  to: string;
}
export interface LinkContext extends Range {
  selected: string | null;
  zoom: [number, number] | null;
}
export interface Group extends LinkContext {
  id: string;
  name: string;
  color: string;
}
export type NewBinding =
  | { kind: 'source'; ids: string[] }
  | {
      kind: 'algorithm';
      algorithm: string;
      runId: number | null;
      sources?: string[];
    };
export type Binding = NewBinding & { id: string };
export interface Widget {
  id: string;
  type: WidgetType;
  title: string;
  binding: string | null;
  group: string | null;
  context: LinkContext;
  settings: {
    outputs: string[];
    table: string;
    metric: string;
    category: string;
    notes: string;
    display: 'overlay' | 'panes' | 'indexed';
    styles: Record<string, { color: string; dashed: boolean }>;
  };
}
export interface Workspace {
  id: string;
  name: string;
  layout: Layout | null;
  widgets: Record<string, Widget>;
  bindings: Record<string, Binding>;
  groups: Record<string, Group>;
  locked: boolean;
}
export interface TerminalDocument {
  version: 1;
  active: string;
  workspaces: Workspace[];
  presets?: Workspace[];
}
export interface Size {
  width: number;
  height: number;
}
export const LIMITS = {
  widgets: 12,
  workspaces: 16,
  bytes: 1048576,
  history: 40,
};
export const MINIMUM: Record<WidgetType, Size> = {
  chat: { width: 300, height: 240 },
  chart: { width: 320, height: 200 },
  table: { width: 280, height: 180 },
  comparison: { width: 280, height: 180 },
  inspector: { width: 240, height: 180 },
  context: { width: 220, height: 140 },
};
export const uid = () => crypto.randomUUID();
export function context(range: Range): LinkContext {
  return { ...range, selected: null, zoom: null };
}
export function createWidget(
  type: WidgetType,
  range: Range,
  binding: string | null = null,
): Widget {
  return {
    id: uid(),
    type,
    title: {
      chat: 'AI chat',
      chart: 'Time series',
      table: 'Results table',
      comparison: 'Metric comparison',
      inspector: 'Period inspector',
      context: 'Context & notes',
    }[type],
    binding,
    group: 'A',
    context: context(range),
    settings: {
      outputs: [],
      table: '',
      metric: '',
      category: '',
      notes: '',
      display: 'overlay',
      styles: {},
    },
  };
}
