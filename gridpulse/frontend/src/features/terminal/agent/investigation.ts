/** Conversation lifecycle and evidence streaming; has no authority to mutate the workspace. */
import type { Workspace, Size } from '../model/types';
import { uid, MINIMUM, LIMITS } from '../model/types';
import type { Conversation, Evidence, View } from './types';
import { workspaceContext } from './view';
import { streamTurn } from './client';

export async function investigate({
  chat,
  id,
  workspace,
  browserInstance,
  message,
  approval,
  retry,
  signal,
  size,
}: {
  chat: Conversation;
  id: string;
  workspace: Workspace;
  browserInstance: string;
  message: string;
  approval?: string;
  retry: boolean;
  signal: AbortSignal;
  size: Size;
}): Promise<{ view: View | undefined; answerIndex: number }> {
  if (!retry) chat.messages.push({ role: 'user', content: message });
  chat.confirmation = null;
  chat.error = '';
  chat.status = 'Preparing investigation…';
  while (chat.messages.length > 40) {
    const old = chat.messages.splice(0, 2);
    chat.summary = (
      chat.summary +
      '\n' +
      old.map((m) => `${m.role}: ${m.content.slice(0, 500)}`).join('\n')
    ).slice(-6000);
  }
  const token = uid(),
    turnId = uid();
  let view: View | undefined;
  let answerIndex = -1;
  const references: Evidence[] = [];
  const history = retry ? chat.messages : chat.messages.slice(0, -1);
  await streamTurn(
    {
      version: 1,
      turn_id: turnId,
      conversation: id,
      workspace_id: workspace.id,
      tab_id: browserInstance,
      revision: token,
      message,
      history: history
        .slice(-16)
        .map((m) => ({ role: m.role, content: m.content.slice(0, 12000) })),
      summary: chat.summary,
      snapshot: {
        ...workspaceContext(workspace),
        view_rules: {
          viewport: size,
          minimums: MINIMUM,
          max_widgets: LIMITS.widgets,
          max_chart_units: 2,
          horizontal_steps: 24,
          vertical_steps: 16,
        },
      },
      recent_states: chat.states.slice(-3).map(workspaceContext),
      approval: approval ?? null,
    },
    signal,
    (event) => {
      if (event.type === 'progress')
        chat.status = event.message ?? 'Investigating…';
      if (event.type === 'evidence' && event.evidence) {
        const item = event.evidence;
        const run = item.data.run as
          { run_id?: number; algorithm_id?: string } | undefined;
        references.push({
          ...item,
          request: {
            ...item.request,
            algorithm:
              item.request.algorithm ||
              run?.algorithm_id ||
              (item.data.algorithm as string | undefined),
            run_id:
              run?.run_id ??
              item.request.run_id ??
              (item.data.run_id as number | undefined),
          },
          data: {},
        });
      }
      if (event.type === 'answer') {
        answerIndex = chat.messages.length;
        chat.messages.push({
          role: 'assistant',
          content: event.text ?? '',
          details: event.details,
          evidence: references,
          references: event.references,
        });
      }
      if (event.type === 'view') {
        if (event.revision !== token || event.workspace_id !== workspace.id)
          throw new Error('AI view references an unexpected workspace state.');
        view = event.view;
      }
      if (event.type === 'confirmation')
        chat.confirmation = {
          approval: event.approval!,
          request: event.request!,
          message: event.message!,
        };
    },
  );
  return { view, answerIndex };
}
