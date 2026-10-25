import { test, expect } from '@playwright/test';
import catalogue from '../../contracts/examples/catalog.json' with { type: 'json' };
import tables from '../../contracts/examples/analysis_tables.json' with { type: 'json' };
const date = '2026-10-25';
const times = Array.from({ length: 50 }, (_, i) =>
  new Date(Date.parse('2026-10-24T23:00Z') + i * 1800000).toISOString(),
);
const run = {
  ...tables.run,
  range_start_utc: times[0],
  range_end_utc: '2026-10-26T00:00:00Z',
};
const catalog = {
  ...catalogue,
  algorithms: catalogue.algorithms.map((a) => ({
    ...a,
    latest_run: a.id === 'price_naive_backtest' ? run : a.latest_run,
  })),
};
const requests: string[] = [];
async function addChart(page: import('@playwright/test').Page) {
  await page.getByRole('button', { name: /Search \/ Add/ }).click();
  await page.getByRole('button', { name: 'Widgets', exact: true }).click();
  await page
    .getByRole('button', { name: 'Time-series chart Add +', exact: true })
    .click();
}
test.beforeEach(async ({ page }) => {
  requests.length = 0;
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url());
    requests.push(url.pathname + url.search);
    let body: unknown;
    if (url.pathname === '/api/catalog') body = catalog;
    else if (url.pathname === '/api/health')
      body = { status: 'ok', mode: 'live', database: 'ok', version: 'test' };
    else if (url.pathname === '/api/snapshot') body = { settlement_date: date };
    else if (url.pathname === '/api/series') {
      const ids = (url.searchParams.get('ids') ?? 'price').split(',');
      body = {
        from: date,
        to: date,
        resolution: '30m',
        cutoff_utc: null,
        t: times,
        settlement_date: times.map(() => date),
        settlement_period: times.map((_, i) => i + 1),
        series: Object.fromEntries(
          ids.map((id, j) => [
            id,
            {
              label: catalogue.series.find((s) => s.id === id)?.label ?? id,
              unit: catalogue.series.find((s) => s.id === id)?.unit ?? 'MW',
              values: times.map((_, i) =>
                i === 3
                  ? null
                  : id === 'price'
                    ? Math.sin(i / 5) * 80 + 30
                    : (j + 1) * 5000 + i * 20,
              ),
            },
          ]),
        ),
      };
    } else if (url.pathname === '/api/analysis/runs')
      body = {
        algorithm: catalog.algorithms.find(
          (a) => a.id === url.searchParams.get('algorithm'),
        ),
        runs: [run],
      };
    else if (url.pathname.endsWith('/tables')) body = { ...tables, run };
    else if (url.pathname === '/api/analysis') {
      const algo = catalog.algorithms.find(
        (a) => a.id === url.searchParams.get('algorithm'),
      )!;
      const r = {
        ...run,
        algorithm_id: algo.id,
        run_id: algo.id === 'price_naive_backtest' ? 12 : 7,
      };
      body = {
        algorithm: algo,
        run: r,
        from: date,
        to: date,
        resolution: '30m',
        t: times,
        indicators: Object.fromEntries(
          Object.entries(algo.outputs).map(([id, unit]) => [
            id,
            {
              label: id,
              unit,
              values: times.map((_, i) =>
                i === 3 ? null : 50 + Math.sin(i / 6) * 30,
              ),
            },
          ]),
        ),
        warnings: [],
        tables: Object.keys(algo.tables),
      };
    } else if (url.pathname.startsWith('/api/periods/'))
      body = {
        settlement_date: date,
        settlement_period: Number(url.pathname.split('/').at(-1)),
        delivery_start_utc: times[10],
        delivery_end_utc: times[11],
        values: { price: 40, wind: 12000 },
        vintages: { wind: [{ published_at_utc: times[0], value: 12000 }] },
        analysis: [],
      };
    else
      return route.fulfill({
        status: 404,
        json: {
          detail: { code: 'no_data', message: 'Test route unavailable' },
        },
      });
    await route.fulfill({ json: body });
  });
});
test('Terminal renders linked data, keeps hover local and saves the workspace', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/terminal');
  await expect(
    page.locator('path[data-terminal-series="price"]'),
  ).toBeVisible();
  await expect(
    page.getByRole('button', { name: 'Workspaces', exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole('tab', { name: 'Market investigation', exact: true }),
  ).toBeVisible();
  await expect(
    page
      .locator('.terminal-widget-content')
      .getByRole('button', { name: 'Export chart CSV', exact: true }),
  ).toHaveCount(0);
  await page
    .getByRole('button', { name: 'Settings for Time series', exact: true })
    .click();
  const exportRequests = requests.length;
  const csvDownload = page.waitForEvent('download');
  await page
    .getByRole('button', { name: 'Export chart CSV', exact: true })
    .click();
  const downloaded = await csvDownload;
  expect(downloaded.suggestedFilename()).toBe('eisengrau-series.csv');
  const stream = await downloaded.createReadStream();
  const chunks = [];
  for await (const chunk of stream!) chunks.push(chunk);
  const contents = Buffer.concat(chunks).toString('utf8');
  expect(contents).toContain('"timestamp_utc"');
  expect(contents).toContain('"price (GBP/MWh)"');
  expect(requests.length).toBe(exportRequests);
  await page.getByRole('button', { name: 'Close drawer' }).click();
  const before = requests.length;
  const plot = page.getByRole('slider', { name: 'Chart selected period' });
  await plot.hover({ position: { x: 300, y: 100 } });
  await plot.hover({ position: { x: 400, y: 100 } });
  await expect(page.locator('.terminal-inspector-preview')).toContainText(
    'selected',
  );
  expect(requests.length).toBe(before);
  await page
    .getByRole('button', { name: 'Settings for Time series', exact: true })
    .click();
  await page.getByLabel('Widget title', { exact: true }).fill('Market signals');
  await page.getByLabel('Widget title', { exact: true }).press('Tab');
  await page.getByRole('button', { name: 'Close drawer' }).click();
  await expect(page.locator('.terminal-status')).toContainText('Saved locally');
  await page.reload();
  await expect(
    page.getByRole('tab', { name: 'Market signals', exact: true }),
  ).toBeVisible();
  expect(errors).toEqual([]);
  await page.screenshot({
    path: test.info().outputPath('terminal-default.png'),
  });
});
test('catalogue opens an algorithm workspace with a consistent pinned chart/table run', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/terminal');
  await expect(
    page.locator('path[data-terminal-series="price"]'),
  ).toBeVisible();
  await page.getByRole('button', { name: /Search \/ Add/ }).click();
  await page
    .getByLabel('Search data algorithms and widgets')
    .fill('price_naive');
  await page.getByRole('button', { name: 'Algorithms', exact: true }).click();
  await page
    .getByRole('button', { name: 'Open analysis layout', exact: true })
    .click();
  await expect(
    page.locator('path[data-terminal-series="persistence_forecast"]'),
  ).toBeVisible();
  await expect(page.locator('.terminal-table-scroll')).toContainText(
    'seasonal_naive_forecast',
  );
  await expect(page.locator('.terminal-provenance').first()).toContainText(
    'Run #12',
  );
  expect(
    requests.some((r) => r.startsWith('/api/analysis/runs/12/tables')),
  ).toBe(true);
  await expect(
    page.getByRole('button', { name: /Run computation/ }),
  ).toHaveCount(0);
  await expect(
    page
      .locator('.terminal-widget-content')
      .getByRole('button', { name: 'Export table CSV', exact: true }),
  ).toHaveCount(0);
  await page
    .getByRole('button', { name: 'Settings for Results table', exact: true })
    .click();
  const tableDownload = page.waitForEvent('download');
  await page
    .getByRole('dialog', { name: 'Widget settings', exact: true })
    .getByRole('button', { name: 'Export table CSV', exact: true })
    .click();
  const exportedTable = await tableDownload;
  expect(exportedTable.suggestedFilename()).toBe(
    'eisengrau-metrics-run-12.csv',
  );
  const tableStream = await exportedTable.createReadStream();
  const tableChunks = [];
  for await (const chunk of tableStream!) tableChunks.push(chunk);
  expect(Buffer.concat(tableChunks).toString('utf8')).toContain(
    'seasonal_naive_forecast',
  );
  await page.getByRole('button', { name: 'Close drawer' }).click();
  await page.screenshot({
    path: test.info().outputPath('terminal-algorithm.png'),
  });
});
test('placement rejects unreadable splits, supports tabbing and undo', async ({
  page,
}) => {
  await page.setViewportSize({ width: 900, height: 768 });
  await page.goto('/terminal');
  await expect(
    page.locator('path[data-terminal-series="price"]'),
  ).toBeVisible();
  await page
    .getByRole('button', { name: 'Settings for Time series', exact: true })
    .click();
  await page
    .getByRole('button', { name: 'Duplicate widget', exact: true })
    .click();
  await page
    .getByLabel('Placement destination')
    .selectOption({ label: 'Period inspector' });
  await page.getByLabel('Placement position').selectOption('left');
  await expect(
    page.getByRole('button', { name: 'Place widget', exact: true }),
  ).toBeDisabled();
  await expect(page.getByRole('dialog')).toContainText('Workspace has no room');
  await page.getByLabel('Placement position').selectOption('tab');
  await page.getByRole('button', { name: 'Place widget', exact: true }).click();
  await expect(page.locator('.terminal-widget-bar > button')).toHaveCount(4);
  await expect(page.locator('.terminal-status')).toContainText('Saved locally');
  const desktopLayout = await page.evaluate(
    () =>
      JSON.parse(localStorage.getItem('eisengrau:terminal:v1')!).workspaces[0]
        .layout,
  );
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator('.terminal-widget-bar > button').first().click();
  await page.setViewportSize({ width: 900, height: 768 });
  expect(
    await page.evaluate(
      () =>
        JSON.parse(localStorage.getItem('eisengrau:terminal:v1')!).workspaces[0]
          .layout,
    ),
  ).toEqual(desktopLayout);
  await page.getByRole('button', { name: 'Undo workspace change' }).click();
  await expect(page.locator('.terminal-widget-bar > button')).toHaveCount(3);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
});
test('compact navigation leaves the desktop layout intact and old Explore redirects', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/explore');
  await expect(page).toHaveURL(/\/terminal$/);
  await expect(
    page.locator('path[data-terminal-series="price"]'),
  ).toBeVisible();
  await expect(page.locator('.terminal-status')).toContainText('Saved locally');
  const layout = await page.evaluate(
    () =>
      JSON.parse(localStorage.getItem('eisengrau:terminal:v1')!).workspaces[0]
        .layout,
  );
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator('.terminal-tile-header')).toContainText(
    'Compact view',
  );
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page
    .locator('.terminal-widget-bar > button')
    .filter({ hasText: 'Context & notes' })
    .click();
  await expect(page.getByLabel('Investigation notes')).toBeVisible();
  await page.setViewportSize({ width: 1440, height: 900 });
  await expect(page.locator('.terminal-split').first()).toBeVisible();
  expect(
    await page.evaluate(
      () =>
        JSON.parse(localStorage.getItem('eisengrau:terminal:v1')!).workspaces[0]
          .layout,
    ),
  ).toEqual(layout);
});

test('native drag previews reject small splits and keyboard dividers snap discretely', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 650 });
  await page.goto('/terminal');
  await expect(
    page.locator('path[data-terminal-series="price"]'),
  ).toBeVisible();
  const divider = page.getByRole('slider', { name: 'Resize tiles' }).first();
  const step = Number(await divider.getAttribute('aria-valuenow'));
  await divider.focus();
  await page.keyboard.press('ArrowLeft');
  await expect(divider).toHaveAttribute('aria-valuenow', String(step - 1));
  await page.getByRole('button', { name: 'Undo workspace change' }).click();
  await expect(divider).toHaveAttribute('aria-valuenow', String(step));
  const source = page.getByRole('tab', {
    name: 'Time series',
    exact: true,
  });
  const dest = page.getByRole('group', {
    name: 'Period inspector',
    exact: true,
  });
  const box = (await dest.boundingBox())!;
  const dt = await page.evaluateHandle(() => new DataTransfer());
  await source.dispatchEvent('dragstart', { dataTransfer: dt });
  await dest.dispatchEvent('dragover', {
    dataTransfer: dt,
    clientX: box.x + box.width / 2,
    clientY: box.y + box.height - 3,
  });
  await expect(page.locator('.terminal-drop-preview.invalid')).toContainText(
    'No room',
  );
  await page.keyboard.press('Escape');
  await expect(page.locator('.terminal-drop-preview')).toHaveCount(0);
  const targetTab = page.getByRole('tab', { name: 'Time series', exact: true });
  const contextTab = page.getByRole('tab', {
    name: 'Context & notes',
    exact: true,
  });
  await contextTab.dispatchEvent('dragstart', { dataTransfer: dt });
  await targetTab.dispatchEvent('dragover', { dataTransfer: dt });
  await targetTab.dispatchEvent('drop', { dataTransfer: dt });
  await expect(
    page.locator('.terminal-widget-tabs').first().getByRole('tab'),
  ).toHaveText(['Context & notes', 'Time series']);
  await page.getByRole('button', { name: 'Undo workspace change' }).click();
  await expect(page.locator('.terminal-tile')).toHaveCount(3);
});

test('algorithm overlays retain source data and shared units', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/terminal');
  await expect(
    page.locator('path[data-terminal-series="price"]'),
  ).toBeVisible();
  await page.getByRole('button', { name: /Search \/ Add/ }).click();
  await page
    .getByLabel('Search data algorithms and widgets')
    .fill('price_naive');
  await page.getByRole('button', { name: 'Algorithms', exact: true }).click();
  await page
    .getByRole('button', { name: 'Add outputs to chart', exact: true })
    .click();
  await page
    .getByRole('button', { name: 'Apply algorithm', exact: true })
    .click();
  await expect(
    page.locator('path[data-terminal-series="source:price"]'),
  ).toBeVisible();
  await expect(
    page.locator('path[data-terminal-series="source:wind"]'),
  ).toBeVisible();
  await expect(
    page.locator('path[data-terminal-series="persistence_forecast"]'),
  ).toBeVisible();
});

test('saved presets duplicate into independent workspaces and imports can be undone', async ({
  page,
}) => {
  await page.goto('/terminal');
  await expect(
    page.locator('path[data-terminal-series="price"]'),
  ).toBeVisible();
  await page.getByRole('button', { name: 'Workspaces', exact: true }).click();
  await page.getByLabel('Workspace name', { exact: true }).fill('My analysis');
  await page.getByLabel('Workspace name', { exact: true }).press('Tab');
  await page
    .getByRole('button', { name: 'Save current view as preset', exact: true })
    .click();
  await page.getByLabel('Search saved presets').fill('My analysis');
  await page
    .getByRole('button', { name: 'Load preset My analysis', exact: true })
    .click();
  await expect(
    page.getByRole('tab', { name: 'My analysis', exact: true }),
  ).toHaveCount(2);
  await expect(page.locator('.terminal-status')).toContainText('Saved locally');
  const saved = await page.evaluate(() =>
    localStorage.getItem('eisengrau:terminal:v1')!,
  );
  const parsed = JSON.parse(saved);
  expect(parsed.presets).toHaveLength(1);
  const ids = parsed.workspaces.flatMap(
    (w: { widgets: Record<string, unknown> }) => Object.keys(w.widgets),
  );
  expect(new Set(ids).size).toBe(ids.length);
  await page.getByRole('button', { name: 'Workspaces', exact: true }).click();
  await page.getByLabel('Import configuration', { exact: true }).setInputFiles({
    name: 'workspace.json',
    mimeType: 'application/json',
    buffer: Buffer.from(saved),
  });
  await expect(page.getByText('Import preview', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Cancel import' }).click();
  await page.getByLabel('Import configuration', { exact: true }).setInputFiles({
    name: 'bad.json',
    mimeType: 'application/json',
    buffer: Buffer.from('{"version":999}'),
  });
  await expect(
    page
      .getByRole('status')
      .filter({ hasText: 'Unsupported workspace version' }),
  ).toBeVisible();
});

test('linked groups isolate cursors and compact mode does not issue hover requests', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/terminal');
  await expect(
    page.locator('path[data-terminal-series="price"]'),
  ).toBeVisible();
  await page
    .getByRole('button', { name: 'Settings for Period inspector', exact: true })
    .click();
  await page.getByLabel('Linked group', { exact: true }).selectOption('__new');
  await expect(page.getByLabel('Linked group', { exact: true })).toHaveValue(
    'B',
  );
  await page.getByRole('button', { name: 'Close drawer' }).click();
  const before = requests.length;
  await page
    .getByRole('slider', { name: 'Chart selected period' })
    .hover({ position: { x: 250, y: 100 } });
  await expect(page.locator('.terminal-inspector-preview')).toHaveCount(0);
  expect(requests.length).toBe(before);
  await page
    .getByRole('button', { name: 'Settings for Period inspector', exact: true })
    .click();
  await page.getByLabel('Linked group', { exact: true }).selectOption('A');
  await page.getByRole('button', { name: 'Close drawer' }).click();
  await page
    .getByRole('slider', { name: 'Chart selected period' })
    .hover({ position: { x: 300, y: 100 } });
  await expect(page.locator('.terminal-inspector-preview')).toBeVisible();
});

test('uncomputed algorithms stay local and never submit a computation', async ({
  page,
}) => {
  await page.route('**/api/analysis?**', (r) =>
    r.fulfill({
      status: 404,
      json: {
        detail: {
          code: 'no_run',
          message: 'No stored result for these settings.',
        },
      },
    }),
  );
  await page.goto('/terminal');
  await expect(
    page.locator('path[data-terminal-series="price"]'),
  ).toBeVisible();
  await page.getByRole('button', { name: /Search \/ Add/ }).click();
  await page
    .getByLabel('Search data algorithms and widgets')
    .fill('price_naive');
  await page.getByRole('button', { name: 'Algorithms', exact: true }).click();
  await page
    .getByRole('button', { name: 'Open analysis layout', exact: true })
    .click();
  await expect(page.getByText('Not computed yet', { exact: true })).toHaveCount(
    3,
  );
  await expect(
    page.getByRole('button', { name: 'Workspaces', exact: true }),
  ).toBeEnabled();
});

test('damaged saves are preserved until recovery and storage failure stays usable', async ({
  page,
}) => {
  await page.addInitScript(() =>
    localStorage.setItem('eisengrau:terminal:v1', '{"version":999}'),
  );
  await page.goto('/terminal');
  await expect(page.getByRole('alert')).toContainText('needs recovery');
  await expect(
    page.locator('path[data-terminal-series="price"]'),
  ).toBeVisible();
  expect(
    await page.evaluate(() => localStorage.getItem('eisengrau:terminal:v1')),
  ).toBe('{"version":999}');
  await expect(
    page.getByRole('button', { name: 'Download saved file' }),
  ).toBeVisible();
  await page.getByRole('button', { name: 'Use fresh workspace' }).click();
  await page.getByRole('button', { name: 'Workspaces', exact: true }).click();
  await page.getByLabel('Workspace name', { exact: true }).fill('Recovered');
  await page.getByLabel('Workspace name', { exact: true }).press('Tab');
  await page.getByRole('button', { name: 'Close drawer' }).click();
  await expect(page.locator('.terminal-status')).toContainText('Saved locally');
});

test('a future catalogue algorithm renders without a new frontend adapter', async ({
  page,
}) => {
  const future = {
    ...catalog.algorithms[0],
    id: 'future_signal',
    description: 'A new arbitrary numeric signal',
    outputs: { unknown_output: 'widgets/hour' },
    tables: {},
    latest_run: null,
  };
  await page.route('**/api/catalog', (r) =>
    r.fulfill({
      json: { ...catalog, algorithms: [...catalog.algorithms, future] },
    }),
  );
  await page.route('**/api/analysis?**', (r) =>
    r.fulfill({
      json: {
        algorithm: future,
        run: { ...run, algorithm_id: future.id, run_id: 90 },
        from: date,
        to: date,
        resolution: '30m',
        t: times,
        indicators: {
          unknown_output: {
            label: 'Unfamiliar signal',
            unit: 'widgets/hour',
            values: times.map((_, i) => (i % 4 === 0 ? null : i)),
          },
        },
        warnings: ['Partial coverage in fixture'],
        tables: [],
      },
    }),
  );
  await page.goto('/terminal');
  await expect(
    page.locator('path[data-terminal-series="price"]'),
  ).toBeVisible();
  await page.getByRole('button', { name: /Search \/ Add/ }).click();
  await page
    .getByLabel('Search data algorithms and widgets')
    .fill('future_signal');
  await page.getByRole('button', { name: 'Algorithms', exact: true }).click();
  await page
    .getByRole('button', { name: 'Open analysis layout', exact: true })
    .click();
  await expect(
    page.locator('path[data-terminal-series="unknown_output"]'),
  ).toBeVisible();
  await expect(page.locator('.terminal-chart-legend')).toContainText(
    'widgets/hour',
  );
  await expect(page.locator('.terminal-warnings').first()).toContainText(
    'Partial coverage',
  );
});

test('a valid import replaces only after confirmation and undo restores the document', async ({
  page,
}) => {
  await page.goto('/terminal');
  await expect(
    page.locator('path[data-terminal-series="price"]'),
  ).toBeVisible();
  await expect(page.locator('.terminal-status')).toContainText('Saved locally');
  const initial = await page.evaluate(() =>
    localStorage.getItem('eisengrau:terminal:v1')!,
  );
  const imported = JSON.parse(initial);
  imported.workspaces[0].name = 'Imported research';
  await page.getByRole('button', { name: 'Workspaces', exact: true }).click();
  await page.getByLabel('Import configuration', { exact: true }).setInputFiles({
    name: 'view.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(imported)),
  });
  await expect(
    page.getByRole('tab', { name: 'Market investigation', exact: true }),
  ).toBeVisible();
  await page
    .getByRole('button', { name: 'Replace with imported configuration' })
    .click();
  await page.getByRole('button', { name: 'Close drawer' }).click();
  await expect(
    page.getByRole('tab', { name: 'Imported research', exact: true }),
  ).toBeVisible();
  await page.getByRole('button', { name: 'Undo workspace change' }).click();
  await expect(
    page.getByRole('tab', { name: 'Market investigation', exact: true }),
  ).toBeVisible();
});

test('large chart and table responses remain usable without rendering every table row', async ({
  page,
}) => {
  const manyTimes = Array.from({ length: 5000 }, (_, i) =>
    new Date(Date.parse('2026-07-01T00:00Z') + i * 1800000).toISOString(),
  );
  await page.route('**/api/series?**', async (r) => {
    const ids = (
      new URL(r.request().url()).searchParams.get('ids') ?? 'price'
    ).split(',');
    await r.fulfill({
      json: {
        from: date,
        to: date,
        resolution: '30m',
        cutoff_utc: null,
        t: manyTimes,
        settlement_date: manyTimes.map(() => date),
        settlement_period: manyTimes.map((_, i) => (i % 48) + 1),
        series: Object.fromEntries(
          ids.map((id) => [
            id,
            {
              label: id,
              unit: id === 'price' ? 'GBP/MWh' : 'MW',
              values: manyTimes.map((_, i) =>
                i % 20 === 0 ? null : Math.sin(i / 30) * 100,
              ),
            },
          ]),
        ),
      },
    });
  });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/terminal');
  await expect(
    page.locator('path[data-terminal-series="price"]'),
  ).toBeVisible();
  const plot = page.getByRole('slider', { name: 'Chart selected period' });
  await plot.hover({ position: { x: 300, y: 100 } });
  await expect(page.locator('.terminal-inspector-preview')).toBeVisible();
  await page.route('**/api/analysis/runs/12/tables', (r) =>
    r.fulfill({
      json: {
        run,
        tables: {
          metrics: {
            description: 'Synthetic large result',
            columns: ['model', 'score'],
            rows: Array.from({ length: 20000 }, (_, i) => [
              'Model ' + String(i).padStart(5, '0'),
              i,
            ]),
          },
        },
      },
    }),
  );
  await page.getByRole('button', { name: /Search \/ Add/ }).click();
  await page
    .getByLabel('Search data algorithms and widgets')
    .fill('price_naive');
  await page.getByRole('button', { name: 'Algorithms', exact: true }).click();
  await page
    .getByRole('button', { name: 'Open analysis layout', exact: true })
    .click();
  await expect(page.locator('.terminal-table-tools')).toContainText(
    '20000 / 20000',
  );
  expect(
    await page.locator('.terminal-table-scroll tbody tr').count(),
  ).toBeLessThan(100);
  await page.getByLabel('Search result rows').fill('Model 19999');
  await expect(page.locator('.terminal-table-scroll')).toContainText(
    'Model 19999',
  );
  await expect(page.locator('.terminal-table-tools')).toContainText(
    '1 / 20000',
  );
});

test('disabled browser storage and external changes do not destroy a working session', async ({
  page,
}) => {
  await page.addInitScript(() => {
    Storage.prototype.setItem = () => {
      throw new DOMException('Storage denied', 'QuotaExceededError');
    };
  });
  await page.goto('/terminal');
  await expect(
    page.locator('path[data-terminal-series="price"]'),
  ).toBeVisible();
  await expect(page.locator('.terminal-status')).toContainText('Not saved');
  await page.getByRole('button', { name: 'Workspaces', exact: true }).click();
  await expect(
    page.getByRole('button', { name: 'Export configuration' }),
  ).toBeEnabled();
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export configuration' }).click();
  expect((await download).suggestedFilename()).toBe('eisengrau-terminal.json');
  await page.getByRole('button', { name: 'Close drawer' }).click();
  await page.evaluate(() =>
    window.dispatchEvent(
      new StorageEvent('storage', {
        key: 'eisengrau:terminal:v1',
        newValue: '{"changed":true}',
      }),
    ),
  );
  await expect(
    page.getByRole('status').filter({ hasText: 'Another browser tab' }),
  ).toBeVisible();
  await page.getByRole('button', { name: 'Keep this tab' }).click();
  await expect(
    page.locator('path[data-terminal-series="price"]'),
  ).toBeVisible();
});

test('compact Terminal chrome keeps one content menu and tiles multiple charts', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/terminal');
  await expect(
    page.locator('path[data-terminal-series="price"]'),
  ).toBeVisible();
  await expect(page.getByRole('contentinfo')).toHaveCount(0);
  await expect(page.locator('.terminal-topline')).toHaveCount(0);
  const dateBox = (await page.getByLabel('Terminal start date').boundingBox())!,
    searchBox = (await page
      .getByRole('button', { name: /Search \/ Add/ })
      .boundingBox())!;
  expect(
    Math.abs(
      dateBox.y + dateBox.height / 2 - searchBox.y - searchBox.height / 2,
    ),
  ).toBeLessThan(2);
  expect(dateBox.x).toBeLessThan(searchBox.x);
  const height = await page
    .locator('.terminal-canvas')
    .evaluate((el) => el.clientHeight);
  expect(height).toBeGreaterThan(700);
  await addChart(page);
  await expect(
    page.getByRole('slider', { name: 'Chart selected period' }),
  ).toHaveCount(2);
  const plots = page.getByRole('slider', { name: 'Chart selected period' });
  const a = (await plots.nth(0).boundingBox())!,
    b = (await plots.nth(1).boundingBox())!;
  expect(Math.abs(a.y - b.y)).toBeLessThan(2);
  expect(b.x).toBeGreaterThan(a.x + a.width);
  await addChart(page);
  await addChart(page);
  await expect(
    page.getByRole('slider', { name: 'Chart selected period' }),
  ).toHaveCount(4);
  await expect(page.locator('.terminal-tile')).toHaveCount(6);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollHeight <= innerHeight + 1,
    ),
  ).toBe(true);
  await page.screenshot({ path: test.info().outputPath('terminal-dense.png') });
  await page.getByRole('link', { name: 'About', exact: true }).click();
  await expect(page.getByRole('contentinfo')).toHaveCount(1);
});

test('Search Add has only content categories and Workspaces owns preset loading', async ({
  page,
}) => {
  await page.goto('/terminal');
  await expect(
    page.getByRole('navigation', { name: 'Primary navigation' }),
  ).not.toContainText('Methods');

  await expect(
    page.locator('path[data-terminal-series="price"]'),
  ).toBeVisible();
  await expect(
    page.getByRole('button', { name: '＋ Chart', exact: true }),
  ).toHaveCount(0);
  await page.getByRole('button', { name: /Search \/ Add/ }).click();
  await expect(page.locator('.terminal-filter-tabs button')).toHaveText([
    'Data',
    'Algorithms',
    'Widgets',
  ]);
  await expect(page.locator('.terminal-filter-tabs button.active')).toHaveText(
    'Data',
  );
  await expect(
    page.getByRole('button', { name: 'All', exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByRole('button', { name: 'Presets', exact: true }),
  ).toHaveCount(0);
  await page.getByRole('button', { name: 'Close drawer' }).click();
  await page.getByRole('button', { name: 'Workspaces', exact: true }).click();
  const manager = page.getByRole('dialog', { name: 'Workspaces', exact: true });
  await expect(
    manager.getByRole('button', {
      name: 'Reset to market preset',
      exact: true,
    }),
  ).toBeVisible();
  await expect(
    manager.getByRole('button', {
      name: 'Close current workspace',
      exact: true,
    }),
  ).toBeVisible();
  await expect(manager.locator('summary')).toHaveCount(0);

  await expect(
    page.getByRole('button', { name: 'Market preset', exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole('button', {
      name: 'Save current view as preset',
      exact: true,
    }),
  ).toBeVisible();
});

test('widget group badges switch groups without opening settings', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/terminal');
  await expect(
    page.locator('path[data-terminal-series="price"]'),
  ).toBeVisible();
  const badge = page.getByRole('button', {
    name: 'Change group for Period inspector',
    exact: true,
  });
  await badge.click();
  const menu = page.getByRole('menu', {
    name: 'Link group for Period inspector',
  });
  await expect(menu).toBeVisible();
  await expect(
    menu.getByRole('menuitemradio', { name: 'Group A', exact: true }),
  ).toHaveAttribute('aria-checked', 'true');
  await menu.getByRole('menuitem', { name: 'Create linked group…' }).click();
  await expect(badge).toContainText('B');
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await badge.click();
  await page
    .getByRole('menuitemradio', { name: 'Independent', exact: true })
    .click();
  await expect(badge).toContainText('—');
  await badge.click();
  await page
    .getByRole('menuitemradio', { name: 'Group A', exact: true })
    .click();
  await expect(badge).toContainText('A');
  await badge.focus();
  await page.keyboard.press('ArrowDown');
  await expect(menu).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(menu).toHaveCount(0);
  await expect(badge).toBeFocused();
  await badge.click();
  await page.getByRole('button', { name: 'Workspaces', exact: true }).click();
  await expect(menu).toHaveCount(0);
  await page.getByRole('button', { name: 'Close drawer' }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  await page
    .locator('.terminal-widget-bar > button')
    .filter({ hasText: 'Period inspector' })
    .click();
  const compactBadge = page.getByRole('button', {
    name: 'Change group for Period inspector',
    exact: true,
  });
  await compactBadge.click();
  await page
    .getByRole('menuitemradio', { name: 'Group B', exact: true })
    .click();
  await expect(compactBadge).toContainText('B');
});

test('keyboard panel has its own navbar button, traps focus and blocks workspace shortcuts behind it', async ({
  page,
}) => {
  await page.goto('/terminal');
  await expect(
    page.locator('path[data-terminal-series="price"]'),
  ).toBeVisible();
  const keyboard = page.getByRole('button', {
    name: 'Open keyboard shortcuts',
    exact: true,
  });
  const gear = page.getByRole('button', { name: 'Open settings', exact: true });
  const keyboardBox = await keyboard.boundingBox();
  const gearBox = await gear.boundingBox();
  expect(keyboardBox!.x + keyboardBox!.width).toBeLessThanOrEqual(gearBox!.x);
  await keyboard.click();
  const shortcuts = page.getByRole('dialog', {
    name: 'Keyboard shortcuts',
    exact: true,
  });
  await expect(shortcuts).toBeVisible();
  await expect(
    shortcuts.getByRole('table', { name: 'Charts', exact: true }),
  ).toContainText('Home / End');
  await expect(shortcuts.getByRole('radio')).toHaveCount(0);
  const close = shortcuts.getByRole('button', {
    name: 'Close keyboard shortcuts',
    exact: true,
  });
  await close.focus();
  await page.keyboard.press('Tab');
  await expect(close).toBeFocused();
  await page.keyboard.press('Shift+Tab');
  await expect(close).toBeFocused();
  await page.keyboard.press('Control+k');
  await expect(shortcuts).toBeVisible();
  await expect(
    page.getByRole('dialog', { name: 'Search / Add', exact: true }),
  ).toHaveCount(0);
  await page.keyboard.press('Escape');
  await expect(shortcuts).not.toBeVisible();
  await expect(keyboard).toBeFocused();
  await gear.click();
  const settings = page.getByRole('dialog', { name: 'Settings', exact: true });
  await expect(settings.getByRole('table')).toHaveCount(0);
  await page.keyboard.press('Escape');
  await page.keyboard.press('Control+k');
  await expect(
    page.getByRole('dialog', { name: 'Search / Add', exact: true }),
  ).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(
    page.getByRole('dialog', { name: 'Search / Add', exact: true }),
  ).toHaveCount(0);
  const tabs = page
    .getByRole('tablist', { name: 'Workspaces', exact: true })
    .getByRole('tab');
  await page
    .getByRole('button', { name: 'New blank workspace', exact: true })
    .click();
  await expect(tabs).toHaveCount(2);
  await page.keyboard.press('Control+z');
  await expect(tabs).toHaveCount(1);
  await page.keyboard.press('Control+Shift+z');
  await expect(tabs).toHaveCount(2);
});
