import type { AgentEvent } from './types';
export async function streamTurn(
  body: unknown,
  signal: AbortSignal,
  onEvent: (event: AgentEvent) => void,
) {
  const response = await fetch('/api/agent/turns/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(
      error?.detail?.message ?? `AI request failed (${response.status}).`,
    );
  }
  if (!response.body) throw new Error('AI stream unavailable.');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '',
    completed = false,
    bytes = 0,
    sequence = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      bytes += value.byteLength;
      if (bytes > 1000000)
        throw new Error('AI response exceeded the session limit.');
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let end: number;
      while ((end = buffer.indexOf('\n\n')) >= 0) {
        const frame = buffer.slice(0, end);
        buffer = buffer.slice(end + 2);
        const data = frame
          .split('\n')
          .filter((line) => line.startsWith('data:'))
          .map((line) => line.slice(5).trimStart())
          .join('\n');
        if (!data) continue;
        const event: AgentEvent = JSON.parse(data);
        if (
          event.version !== 1 ||
          event.turn_id !== (body as { turn_id: string }).turn_id
        )
          throw new Error('Unexpected AI stream identity.');
        if (event.sequence !== undefined) {
          if (event.sequence <= sequence)
            throw new Error('AI stream events are out of order.');
          sequence = event.sequence;
        }
        if (signal.aborted) throw new DOMException('Cancelled', 'AbortError');
        if (event.type === 'error')
          throw new Error(event.message ?? 'AI investigation failed.');
        if (event.type === 'completed') completed = true;
        onEvent(event);
      }
    }
    if (!completed)
      throw new Error(
        'AI connection ended before completion. Existing work is preserved.',
      );
  } finally {
    await reader.cancel().catch(() => {});
  }
}
