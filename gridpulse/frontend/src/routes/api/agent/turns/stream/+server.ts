import { env } from '$env/dynamic/private';
import { randomUUID } from 'node:crypto';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({
  request,
  cookies,
  url,
  fetch,
}) => {
  if (request.headers.get('origin') !== url.origin)
    return Response.json(
      { detail: { message: 'Invalid request origin.' } },
      { status: 403 },
    );
  if (!env.AGENT_ACCESS_TOKEN)
    return Response.json(
      { detail: { message: 'AI proxy authentication is not configured.' } },
      { status: 503 },
    );
  let session = cookies.get('eisengrau-agent');
  if (!session || !/^[a-zA-Z0-9-]{20,100}$/.test(session)) {
    session = randomUUID();
    cookies.set('eisengrau-agent', session, {
      path: '/api/agent',
      httpOnly: true,
      sameSite: 'strict',
      secure: url.protocol === 'https:',
    });
  }
  const reader = request.body?.getReader();
  if (!reader) return new Response('Request body required', { status: 400 });
  const parts: Uint8Array[] = [];
  let length = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      length += value.byteLength;
      if (length > 200000) {
        await reader.cancel();
        return new Response('Context too large', { status: 413 });
      }
      parts.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  const body = new Uint8Array(length);
  let offset = 0;
  for (const part of parts) {
    body.set(part, offset);
    offset += part.byteLength;
  }
  try {
    const response = await fetch(
      new URL(
        '/api/agent/turns/stream',
        env.API_PROXY_TARGET ?? 'http://127.0.0.1:8000',
      ),
      {
        method: 'POST',
        body,
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${env.AGENT_ACCESS_TOKEN}`,
          'X-Agent-Session': session,
        },
        signal: AbortSignal.any([request.signal, AbortSignal.timeout(190000)]),
      },
    );
    return new Response(response.body, {
      status: response.status,
      headers: {
        'Content-Type':
          response.headers.get('Content-Type') ?? 'application/json',
        'Cache-Control': 'no-store',
        'X-Accel-Buffering': 'no',
      },
    });
  } catch {
    return Response.json(
      {
        detail: {
          message: 'AI backend is unreachable or the request was cancelled.',
        },
      },
      { status: 503 },
    );
  }
};
