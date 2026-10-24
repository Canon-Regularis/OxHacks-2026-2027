import { it, expect } from 'vitest';
import { RequestPool, TerminalError } from './client';
it('deduplicates subscriptions and does not abort data still in use', async () => {
  let count = 0;
  let signal: AbortSignal | null = null;
  const transport: typeof fetch = async (_url, init) => {
    count++;
    signal = init?.signal as AbortSignal;
    return Response.json({ ok: true });
  };
  const pool = new RequestPool(transport),
    a = pool.acquire('/catalog', { z: 2, a: 1 }),
    b = pool.acquire('/catalog', { a: 1, z: 2 });
  a.release();
  await Promise.resolve();
  expect(count).toBe(1);
  expect(signal!.aborted).toBe(false);
  expect(await b.promise).toEqual({ ok: true });
  b.release();
});
it('exposes missing computation separately from a network failure', async () => {
  const pool = new RequestPool(async () =>
    Response.json(
      { detail: { code: 'no_run', message: 'Not computed yet' } },
      { status: 404 },
    ),
  );
  const lease = pool.acquire('/analysis');
  await expect(lease.promise).rejects.toMatchObject({ code: 'no_run' });
  lease.release();
});
