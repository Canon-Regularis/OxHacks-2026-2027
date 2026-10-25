import { afterEach, expect, it, vi } from 'vitest';
import { streamTurn } from './client';
afterEach(() => vi.unstubAllGlobals());
function response(parts: string[]) {
  return new Response(
    new ReadableStream({
      start(controller) {
        for (const part of parts)
          controller.enqueue(new TextEncoder().encode(part));
        controller.close();
      },
    }),
    { headers: { 'Content-Type': 'text/event-stream' } },
  );
}
it('parses fragmented SSE including a CRLF split across chunks', async () => {
  const raw =
    'data: {"version":1,"turn_id":"turn","sequence":1,"type":"answer","text":"Evidence"}\r\n\r\ndata: {"version":1,"turn_id":"turn","sequence":2,"type":"completed"}\r\n\r\n';
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => response(raw.split(''))),
  );
  const events: string[] = [];
  await streamTurn({ turn_id: 'turn' }, new AbortController().signal, (e) =>
    events.push(e.type),
  );
  expect(events).toEqual(['answer', 'completed']);
});
it('rejects cross-turn output and incomplete streams', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () =>
      response([
        'data: {"version":1,"turn_id":"other","type":"completed"}\n\n',
      ]),
    ),
  );
  await expect(
    streamTurn({ turn_id: 'turn' }, new AbortController().signal, () => {}),
  ).rejects.toThrow(/identity/);
  vi.stubGlobal(
    'fetch',
    vi.fn(async () =>
      response(['data: {"version":1,"turn_id":"turn","type":"answer"}\n\n']),
    ),
  );
  await expect(
    streamTurn({ turn_id: 'turn' }, new AbortController().signal, () => {}),
  ).rejects.toThrow(/before completion/);
});
