import type { Position, WidgetType, Workspace } from '../model/types';
export interface ViewWidget {
  key: string;
  existing: string | null;
  type: Exclude<WidgetType, 'chat'>;
  title: string;
  series: string[];
  algorithm: string | null;
  run_id: number | null;
  outputs: string[];
  table: string;
  group: string | null;
  relative_to: string | null;
  position: Position;
  size_step?: number | null;
  notes: string;
}
export interface View {
  version: 1;
  from_date: string;
  to_date: string;
  widgets: ViewWidget[];
  remove: string[];
  background_chat: boolean;
  new_workspace: boolean;
  name: string;
}
export interface Evidence {
  id: string;
  label: string;
  request: {
    tool: string;
    from_date: string;
    to_date: string;
    algorithm?: string;
    run_id?: number;
    ids?: string[];
    table?: string;
  };
  data: Record<string, unknown>;
}
export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  details?: string;
  evidence?: Evidence[];
  references?: string[];
}
export interface Confirmation {
  approval: string;
  request: Record<string, unknown>;
  message: string;
}
export interface Conversation {
  messages: ChatMessage[];
  summary: string;
  status: string;
  error: string;
  confirmation: Confirmation | null;
  states: Workspace[];
}
export const conversation = (): Conversation => ({
  messages: [],
  summary: '',
  status: '',
  error: '',
  confirmation: null,
  states: [],
});
export interface AgentEvent {
  version: number;
  turn_id: string;
  sequence?: number;
  type: string;
  text?: string;
  details?: string;
  references?: string[];
  evidence?: Evidence;
  message?: string;
  view?: View;
  revision?: string;
  workspace_id?: string;
  approval?: string;
  request?: Record<string, unknown>;
}
