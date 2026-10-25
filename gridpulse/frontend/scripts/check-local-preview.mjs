import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import { chromium } from '@playwright/test';

const root = fileURLToPath(new URL('../', import.meta.url));
const browser = await chromium.launch({
  args: ['--enable-unsafe-swiftshader'],
});
try {
  const page = await browser.newPage({ reducedMotion: 'no-preference' });
  let reloadMessages = 0;
  let navigations = 0;
  page.on('websocket', (socket) =>
    socket.on('framereceived', ({ payload }) => {
      if (String(payload).includes('full-reload')) reloadMessages++;
    }),
  );
  const origin = process.env.LOCAL_PREVIEW_URL || 'http://127.0.0.1:5174';
  await page.goto(`${origin}/about`);
  await page.waitForSelector('.hero-globe canvas');
  await page.waitForSelector('.about-pixel-section canvas');
  await page.evaluate(() => {
    window.previewCanvases = Array.from(document.querySelectorAll('canvas'));
  });
  page.on('framenavigated', (frame) => {
    if (frame === page.mainFrame()) navigations++;
  });
  const run = promisify(execFile);
  for (const command of ['check', 'build'])
    await run('npm', ['run', command], { cwd: root });
  await page.waitForTimeout(1000);
  const unchanged = await page.evaluate(() =>
    window.previewCanvases?.every((canvas) => canvas.isConnected),
  );
  assert.equal(
    navigations,
    0,
    'Frontend checks/builds reloaded the presentation page',
  );
  assert.equal(
    unchanged,
    true,
    'Frontend checks/builds removed animation canvases',
  );
  assert.doesNotMatch(
    await page.locator('.site-footer').innerText(),
    /oxhack/i,
  );
  const health = await page.request.get(`${origin}/api/health`);
  assert.equal(health.status(), 200, 'Backend proxy is unavailable');
  console.log(
    `Passed: ${reloadMessages} reload messages, ${navigations} document reloads; both canvases retained; backend HTTP 200.`,
  );
} finally {
  await browser.close();
}
