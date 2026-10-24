export type WidgetType =
  'chart' | 'table' | 'comparison' | 'inspector' | 'context';
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
export type Binding =
  | { id: string; kind: 'source'; ids: string[] }
  | { id: string; kind: 'algorithm'; algorithm: string; runId: number | null };
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
  chart: { width: 360, height: 220 },
  table: { width: 300, height: 200 },
  comparison: { width: 300, height: 200 },
  inspector: { width: 280, height: 220 },
  context: { width: 240, height: 160 },
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
