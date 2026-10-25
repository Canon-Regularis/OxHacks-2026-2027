import { expect, it, vi } from 'vitest';
import { createDocument } from '../model/document';
import type { LayoutFeedback, View } from './types';
import { applyWithRepairs } from './repair';
const doc = createDocument({ from: '2026-02-01', to: '2026-03-31' });
const view: View = {
  version: 1,
  from_date: '2026-02-01',
  to_date: '2026-03-31',
  widgets: [],
  remove: [],
  background_chat: false,
  new_workspace: false,
  name: 'View',
};
const current = () => ({
  document: structuredClone(doc),
  size: { width: 1400, height: 800 },
});
it('returns the actual error and fresh context for two repairs, then commits once', async () => {
  const feedback: (LayoutFeedback | undefined)[] = [];
  const complete = vi.fn(),
    status = vi.fn();
  let attempts = 0;
  await applyWithRepairs({
    workspaceId: doc.active,
    signal: new AbortController().signal,
    current,
    status,
    complete,
    request: async (context) => {
      feedback.push(context.feedback);
      expect(context.workspace.id).toBe(doc.active);
      expect(context.retry).toBe(feedback.length > 1);
      return { view };
    },
    compile: (document) => {
      if (attempts++ < 2) throw new Error('Minimum widget size reached.');
      return document;
    },
  });
  expect(feedback.map((f) => f?.attempt)).toEqual([undefined, 1, 2]);
  expect(feedback[1]?.error).toBe('Minimum widget size reached.');
  expect(feedback[1]?.failed_view).toEqual(view);
  expect(complete).toHaveBeenCalledTimes(1);
});
it('stops after two repairs and never commits a failed proposal', async () => {
  const request = vi.fn(async () => ({ view })),
    complete = vi.fn();
  await expect(
    applyWithRepairs({
      workspaceId: doc.active,
      signal: new AbortController().signal,
      current,
      status: () => {},
      complete,
      request,
      compile: () => {
        throw new Error('Minimum widget size reached.');
      },
    }),
  ).rejects.toThrow('after 2 repair attempts');
  expect(request).toHaveBeenCalledTimes(3);
  expect(complete).not.toHaveBeenCalled();
});
it('does not retry transport failures or continue after cancellation', async () => {
  const complete = vi.fn(),
    request = vi.fn(async () => {
      throw new Error('HTTP 503');
    });
  await expect(
    applyWithRepairs({
      workspaceId: doc.active,
      signal: new AbortController().signal,
      current,
      status: () => {},
      complete,
      request,
      compile: (d) => d,
    }),
  ).rejects.toThrow('503');
  expect(request).toHaveBeenCalledTimes(1);
  const controller = new AbortController();
  const cancelled = vi.fn(async () => {
    controller.abort();
    return { view };
  });
  await applyWithRepairs({
    workspaceId: doc.active,
    signal: controller.signal,
    current,
    status: () => {},
    complete,
    request: cancelled,
    compile: (d) => d,
  });
  expect(cancelled).toHaveBeenCalledTimes(1);
  expect(complete).not.toHaveBeenCalled();
});
