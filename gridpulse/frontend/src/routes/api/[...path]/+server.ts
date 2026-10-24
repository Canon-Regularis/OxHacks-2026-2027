import { env } from '$env/dynamic/private';
import type { RequestHandler } from './$types';
export const GET: RequestHandler = async ({ params, url, fetch }) => {
  const path = params.path ?? '';
  if (!/^[a-zA-Z0-9_\-/]+$/.test(path) || path.split('/').includes('..'))
    return new Response('Invalid API path', { status: 400 });
  const target = new URL(
    `/api/${path}`,
    env.API_PROXY_TARGET ?? 'http://127.0.0.1:8000',
  );
  target.search = url.search;
  try {
    const response = await fetch(target, {
      signal: AbortSignal.timeout(25000),
    });
    return new Response(response.body, {
      status: response.status,
      headers: {
        'Content-Type':
          response.headers.get('Content-Type') ?? 'application/json',
        'X-GridPulse-Mode': response.headers.get('X-GridPulse-Mode') ?? 'live',
        'Cache-Control': 'no-store',
      },
    });
  } catch {
    return Response.json(
      {
        detail: {
          code: 'backend_unavailable',
          message:
            'The backend is unreachable. Start the API or check the server connection.',
        },
      },
      { status: 503 },
    );
  }
};
