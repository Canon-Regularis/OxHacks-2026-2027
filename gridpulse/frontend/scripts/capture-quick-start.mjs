import { chromium } from '@playwright/test';
import { mkdirSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const output = `${root}static/quick-start`;
mkdirSync(output, { recursive: true });
const origin = process.env.LOCAL_PREVIEW_URL || 'http://127.0.0.1:5174';
const browser = await chromium.launch();
try {
  const page = await browser.newPage({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 1,
    reducedMotion: 'reduce',
  });
  const catalogue = await (
    await page.request.get(`${origin}/api/catalog`)
  ).json();
  const algorithm = catalogue.algorithms.find(
    (a) => a.id === 'price_naive_backtest' && a.latest_run,
  );
  if (!algorithm)
    throw new Error(
      'A completed price_naive_backtest run is needed for real-data screenshots.',
    );
  const snapshot = await (
    await page.request.get(`${origin}/api/snapshot`)
  ).json();
  const to = snapshot.settlement_date;
  const from = new Date(Date.parse(`${to}T12:00:00Z`) - 3 * 86400000)
    .toISOString()
    .slice(0, 10);
  await page.goto(`${origin}/terminal`);
  await page.getByLabel('Terminal start date').fill(from);
  await page.getByLabel('Terminal end date').fill(to);
  await page.waitForSelector('path[data-terminal-series="price"]');
  await page
    .getByLabel('Investigation notes')
    .fill(
      'Question: how do wind forecasts and imbalance prices move together?\n\nHover to compare values. Click a period to inspect the source records.',
    );
  await page.locator('.terminal-status').click();
  await page.waitForFunction(() =>
    document
      .querySelector('.terminal-status')
      ?.textContent?.includes('Saved locally'),
  );
  await page
    .getByRole('slider', { name: 'Chart selected period' })
    .click({ position: { x: 500, y: 150 } });
  await page.waitForTimeout(1200);
  await page.screenshot({ path: `${output}/market-workspace.png` });
  await page.getByRole('button', { name: /Search \/ Add/ }).click();
  await page.getByRole('button', { name: 'Algorithms', exact: true }).click();
  await page
    .getByLabel('Search data algorithms and widgets')
    .fill('price_naive');
  await page.screenshot({ path: `${output}/add-analysis.png` });
  await page
    .getByRole('button', { name: 'Open analysis layout', exact: true })
    .click();
  await page.waitForSelector(
    'path[data-terminal-series="persistence_forecast"]',
    { timeout: 30000 },
  );
  await page.waitForSelector('.terminal-table-scroll tbody tr', {
    timeout: 30000,
  });
  await page
    .getByRole('button', { name: 'Settings for Time series', exact: true })
    .click();
  await page
    .getByRole('group', { name: 'Overlay source data' })
    .getByRole('checkbox')
    .nth(catalogue.series.findIndex((series) => series.id === 'price'))
    .check();
  await page.getByRole('button', { name: 'Close drawer' }).click();
  await page.waitForSelector('path[data-terminal-series="source:price"]');

  await page
    .getByLabel('Investigation notes')
    .fill(
      'Compare the forecast traces with the source prices.\n\nUse the results table to inspect each model’s errors. All companion widgets keep the same stored run.',
    );
  await page.locator('.terminal-status').click();
  await page.waitForFunction(() =>
    document
      .querySelector('.terminal-status')
      ?.textContent?.includes('Saved locally'),
  );
  await page.screenshot({ path: `${output}/analysis-workspace.png` });
  await page.getByRole('button', { name: 'Workspaces', exact: true }).click();
  await page
    .getByLabel('Workspace name', { exact: true })
    .fill('My first price investigation');
  await page.getByLabel('Workspace name', { exact: true }).press('Tab');
  await page.screenshot({ path: `${output}/save-workspace.png` });
  writeFileSync(
    `${root}src/features/quick-start/capture.json`,
    JSON.stringify(
      {
        captured_at_utc: new Date().toISOString(),
        from,
        to,
        algorithm: algorithm.id,
        run_id: algorithm.latest_run.run_id,
      },
      null,
      2,
    ) + '\n',
  );
  console.log(
    `Captured 4 real Terminal screenshots for ${from} to ${to}, stored run #${algorithm.latest_run.run_id}.`,
  );
} finally {
  await browser.close();
}
