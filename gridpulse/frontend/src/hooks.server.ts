import { env } from '$env/dynamic/private';
import type { Handle } from '@sveltejs/kit';

// Keep FastAPI's documentation available behind the same deployment origin.
export const handle: Handle = async ({ event, resolve }) => {
  const path = event.url.pathname;
  if (
    path !== '/openapi.json' &&
    path !== '/docs' &&
    !path.startsWith('/docs/')
  ) {
    return resolve(event);
  }
  if (event.request.method !== 'GET' && event.request.method !== 'HEAD') {
    return new Response('Method not allowed', { status: 405 });
  }
  const target = new URL(path, env.API_PROXY_TARGET ?? 'http://127.0.0.1:8000');
  target.search = event.url.search;
  try {
    return await fetch(target, {
      method: event.request.method,
      signal: AbortSignal.timeout(25000),
    });
  } catch {
    return new Response('Backend documentation unavailable', { status: 503 });
  }
};
