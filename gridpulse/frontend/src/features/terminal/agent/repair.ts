/** Bounded frontend-layout feedback. Never retry transport failures or repeat approved computations. */
import type { Size, TerminalDocument, Workspace } from '../model/types';
import type { ChatMessage, LayoutFeedback, View } from './types';
import { revision } from './view';
export const MAX_LAYOUT_REPAIRS = 2;
export interface ProposedView {
  view?: View;
  answer?: ChatMessage;
}
interface Current {
  document: TerminalDocument;
  size: Size;
}
export async function applyWithRepairs({
  workspaceId,
  signal,
  current,
  request,
  compile,
  complete,
  status,
}: {
  workspaceId: string;
  signal: AbortSignal;
  current: () => Current | null;
  request: (context: {
    workspace: Workspace;
    size: Size;
    retry: boolean;
    feedback?: LayoutFeedback;
  }) => Promise<ProposedView>;
  compile: (
    document: TerminalDocument,
    view: View,
    size: Size,
  ) => TerminalDocument;
  complete: (
    result: ProposedView & { document?: TerminalDocument; before: Workspace },
  ) => void;
  status: (message: string) => void;
}) {
  let repairs = 0,
    staleRetries = 0,
    requests = 0;
  let feedback: LayoutFeedback | undefined;
  while (!signal.aborted) {
    const initial = current();
    if (!initial || initial.document.active !== workspaceId) return;
    const before = initial.document.workspaces.find(
      (ws) => ws.id === workspaceId,
    )!;
    const proposed = await request({
      workspace: before,
      size: initial.size,
      retry: requests++ > 0,
      feedback,
    });
    if (signal.aborted) return;
    const latest = current();
    if (!latest || latest.document.active !== workspaceId) return;
    const live = latest.document.workspaces.find(
      (ws) => ws.id === workspaceId,
    )!;
    if (!proposed.view) {
      complete({ ...proposed, before: live });
      return;
    }
    if (revision(before) !== revision(live)) {
      if (staleRetries++ >= 1)
        throw new Error(
          'Workspace changed again. Your edits were preserved. Send your request again to retry with the current layout.',
        );
      status('Workspace changed; adapting to your edits…');
      continue;
    }
    let document: TerminalDocument;
    try {
      document = compile(latest.document, proposed.view, latest.size);
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : 'The proposed layout could not be applied.';
      if (repairs >= MAX_LAYOUT_REPAIRS)
        throw new Error(
          `Could not apply the view after ${MAX_LAYOUT_REPAIRS} repair attempts. Your workspace is unchanged. Last error: ${message}`,
        );
      feedback = {
        attempt: ++repairs,
        error: message.slice(0, 1000),
        failed_view: proposed.view,
      };
      status(`Adjusting the layout (${repairs}/${MAX_LAYOUT_REPAIRS})…`);
      continue;
    }
    // One synchronous commit after validation; no await between final revision check and application.
    complete({ ...proposed, document, before: live });
    return;
  }
}
