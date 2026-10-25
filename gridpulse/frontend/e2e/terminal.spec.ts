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

test('AI opens a dockable conversation, applies a view and can undo it', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.route('**/api/agent/turns/stream', async (route) => {
    const turn = route.request().postDataJSON();
    const messages = [
      { type: 'started' },
      { type: 'answer', text: 'Here is your price view.', references: [] },
      {
        type: 'view',
        workspace_id: turn.workspace_id,
        revision: turn.revision,
        view: {
          version: 1,
          from_date: date,
          to_date: date,
          widgets: [
            {
              key: 'price',
              existing: null,
              type: 'chart',
              title: 'AI price evidence',
              series: ['price'],
              algorithm: null,
              run_id: null,
              outputs: [],
              table: '',
              group: 'A',
              relative_to: null,
              position: 'tab',
              notes: '',
            },
          ],
          remove: [],
          background_chat: true,
          new_workspace: false,
          name: 'AI view',
        },
      },
      { type: 'completed' },
    ];
    await route.fulfill({
      contentType: 'text/event-stream',
      body: messages
        .map(
          (event, i) =>
            `data: ${JSON.stringify({ version: 1, turn_id: turn.turn_id, sequence: i + 1, ...event })}\n\n`,
        )
        .join(''),
    });
  });
  await page.goto('/terminal');
  await page.getByRole('button', { name: '✦ AI', exact: true }).click();
  await page
    .getByRole('textbox', { name: 'Message AI' })
    .fill('Set up a price workspace');
  await page.getByRole('button', { name: 'Send ↗', exact: true }).click();
  await expect(
    page.getByRole('tab', { name: 'AI price evidence', exact: true }),
  ).toBeVisible();
  await page.getByRole('button', { name: '✦ AI', exact: true }).click();
  await expect(
    page.getByText('View updated. Use Go back to undo.', { exact: false }),
  ).toBeVisible();
  await page.getByRole('button', { name: '↶ Go back', exact: true }).click();
  await expect(
    page.getByRole('tab', { name: 'AI price evidence', exact: true }),
  ).toHaveCount(0);
  await page.reload();
  await page.getByRole('button', { name: '✦ AI', exact: true }).click();
  await expect(
    page.getByText('Set up a price workspace', { exact: true }),
  ).toBeVisible();
  const persistent = await page.evaluate(() =>
    localStorage.getItem('eisengrau:terminal:v1'),
  );
  expect(persistent).not.toContain('"type":"chat"');
  expect(errors).toEqual([]);
});

test('empty workspace starts with chat and promotes it after submission', async ({
  page,
}) => {
  await page.route('**/api/agent/turns/stream', async (route) => {
    const turn = route.request().postDataJSON();
    await route.fulfill({
      contentType: 'text/event-stream',
      body: [
        {
          type: 'answer',
          text: 'Which wind series would you like?',
          references: [],
        },
        { type: 'completed' },
      ]
        .map(
          (event, i) =>
            `data: ${JSON.stringify({ version: 1, turn_id: turn.turn_id, sequence: i + 1, ...event })}\n\n`,
        )
        .join(''),
    });
  });
  await page.goto('/terminal');
  await page
    .getByRole('button', { name: 'New blank workspace', exact: true })
    .click();
  await expect(
    page.getByRole('heading', { name: 'What would you like to investigate?' }),
  ).toBeVisible();
  await page.getByRole('textbox', { name: 'Message AI' }).fill('Show wind');
  await page.getByRole('button', { name: 'Send ↗', exact: true }).click();
  await expect(
    page.getByRole('tab', { name: 'AI chat', exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText('Which wind series would you like?', { exact: true }),
  ).toBeVisible();
});

test('multiple AI tabs keep conversations separate and cancel a pending turn', async ({
  page,
}) => {
  await page.route('**/api/agent/turns/stream', async (route) => {
    const turn = route.request().postDataJSON();
    await new Promise((resolve) => setTimeout(resolve, 250));
    await route
      .fulfill({
        contentType: 'text/event-stream',
        body: [
          { type: 'answer', text: `Answer to ${turn.message}`, references: [] },
          { type: 'completed' },
        ]
          .map(
            (event, i) =>
              `data: ${JSON.stringify({ version: 1, turn_id: turn.turn_id, sequence: i + 1, ...event })}\n\n`,
          )
          .join(''),
      })
      .catch(() => {});
  });
  await page.goto('/terminal');
  await page.getByRole('button', { name: '✦ AI', exact: true }).click();
  await page
    .getByRole('textbox', { name: 'Message AI' })
    .fill('First conversation');
  await page.getByRole('button', { name: 'Send ↗', exact: true }).click();
  await expect(
    page.getByText('Answer to First conversation', { exact: true }),
  ).toBeVisible();
  await page.getByRole('button', { name: 'New chat', exact: true }).click();
  const inputs = page.getByRole('textbox', { name: 'Message AI' });
  await expect(inputs).toHaveCount(2);
  await inputs.last().fill('Second conversation');
  await page
    .getByRole('button', { name: 'Send ↗', exact: true })
    .last()
    .click();
  await page
    .getByRole('button', { name: 'Cancel', exact: true })
    .last()
    .click();
  await expect(
    page.getByText('Cancelled. Completed results remain available.', {
      exact: true,
    }),
  ).toBeVisible();
  await expect(
    page.getByText('Answer to Second conversation', { exact: true }),
  ).toHaveCount(0);
});

test('AI computation confirmation is explicit and carries its approval token', async ({
  page,
}) => {
  const submissions: Record<string, unknown>[] = [];
  await page.route('**/api/agent/turns/stream', async (route) => {
    const turn = route.request().postDataJSON();
    submissions.push(turn);
    const events = turn.approval
      ? [
          {
            type: 'answer',
            text: 'Approved analysis completed.',
            references: [],
          },
          { type: 'completed' },
        ]
      : [
          {
            type: 'confirmation',
            approval: 'test-approval',
            request: {
              algorithm: 'price_naive_backtest',
              from_date: date,
              to_date: date,
              params: {},
            },
            message: 'Confirm this backtest before running.',
          },
          { type: 'completed' },
        ];
    await route.fulfill({
      contentType: 'text/event-stream',
      body: events
        .map(
          (event, i) =>
            `data: ${JSON.stringify({ version: 1, turn_id: turn.turn_id, sequence: i + 1, ...event })}\n\n`,
        )
        .join(''),
    });
  });
  await page.goto('/terminal');
  await page.getByRole('button', { name: '✦ AI', exact: true }).click();
  await page
    .getByRole('textbox', { name: 'Message AI' })
    .fill('Run a backtest');
  await page.getByRole('button', { name: 'Send ↗', exact: true }).click();
  await expect(
    page.getByRole('button', { name: 'Run analysis', exact: true }),
  ).toBeVisible();
  expect(submissions).toHaveLength(1);
  await page.getByRole('button', { name: 'Run analysis', exact: true }).click();
  await expect(
    page.getByText('Approved analysis completed.', { exact: true }),
  ).toBeVisible();
  expect(submissions[1].approval).toBe('test-approval');
});

test('manual edits during an AI turn cause one refresh instead of being overwritten', async ({
  page,
}) => {
  let count = 0;
  await page.route('**/api/agent/turns/stream', async (route) => {
    const turn = route.request().postDataJSON();
    count++;
    await new Promise((resolve) => setTimeout(resolve, 300));
    await route.fulfill({
      contentType: 'text/event-stream',
      body: [
        {
          type: 'view',
          workspace_id: turn.workspace_id,
          revision: turn.revision,
          view: {
            version: 1,
            from_date: date,
            to_date: date,
            widgets: [],
            remove: [],
            background_chat: false,
            new_workspace: false,
            name: 'View',
          },
        },
        { type: 'completed' },
      ]
        .map(
          (event, i) =>
            `data: ${JSON.stringify({ version: 1, turn_id: turn.turn_id, sequence: i + 1, ...event })}\n\n`,
        )
        .join(''),
    });
  });
  await page.goto('/terminal');
  await page.getByRole('button', { name: '✦ AI', exact: true }).click();
  await page
    .getByRole('textbox', { name: 'Message AI' })
    .fill('Inspect the workspace');
  await page.getByRole('button', { name: 'Send ↗', exact: true }).click();
  await expect.poll(() => count).toBe(1);
  await page
    .getByRole('button', { name: 'Layout unlocked', exact: true })
    .click();
  await expect.poll(() => count).toBe(2);
  await page.getByRole('button', { name: '🔒 Locked', exact: true }).click();
  await expect(
    page.getByText('Workspace changed again.', { exact: false }),
  ).toBeVisible();
  expect(count).toBe(2);
  await expect(
    page.getByRole('button', { name: 'Layout unlocked', exact: true }),
  ).toBeVisible();
});

test('AI chat opens in compact mode without discarding the desktop tree', async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/terminal');
  await expect(
    page.getByRole('button', { name: '✦ AI', exact: true }),
  ).toBeVisible();
  await page.getByRole('button', { name: '✦ AI', exact: true }).click();
  await expect(page.getByRole('textbox', { name: 'Message AI' })).toBeVisible();
  await page.setViewportSize({ width: 1440, height: 900 });
  await expect(page.getByRole('textbox', { name: 'Message AI' })).toBeVisible();
  await expect(
    page.locator('path[data-terminal-series="price"]'),
  ).toBeVisible();
});

test('AI renders a February–March investigation with charts placed relative to newly created charts', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.route('**/api/agent/turns/stream', async (route) => {
    const turn = route.request().postDataJSON();
    const base = {
      existing: null,
      type: 'chart',
      series: ['price'],
      algorithm: null,
      run_id: null,
      outputs: [],
      table: '',
      group: 'A',
      relative_to: null,
      position: 'right',
      notes: '',
    };
    const events = [
      {
        type: 'answer',
        text: 'Here are the February–March views.',
        references: [],
      },
      {
        type: 'view',
        workspace_id: turn.workspace_id,
        revision: turn.revision,
        view: {
          version: 1,
          from_date: '2026-02-01',
          to_date: '2026-03-31',
          widgets: [
            { ...base, key: 'price', title: 'February–March prices' },
            {
              ...base,
              key: 'wind',
              title: 'February–March wind',
              series: ['wind'],
              relative_to: 'price',
              position: 'bottom',
            },
            {
              ...base,
              key: 'demand',
              title: 'February–March demand',
              series: ['demand'],
              relative_to: 'wind',
              position: 'tab',
            },
          ],
          remove: [],
          background_chat: false,
          new_workspace: false,
          name: 'Market relationships',
        },
      },
      { type: 'completed' },
    ];
    await route.fulfill({
      contentType: 'text/event-stream',
      body: events
        .map(
          (event, i) =>
            `data: ${JSON.stringify({ version: 1, turn_id: turn.turn_id, sequence: i + 1, ...event })}\n\n`,
        )
        .join(''),
    });
  });
  await page.goto('/terminal');
  await page
    .getByRole('button', { name: 'New blank workspace', exact: true })
    .click();
  await page
    .getByRole('textbox', { name: 'Message AI' })
    .fill('Visualise prices, wind and demand from February to March 2026');
  await page.getByRole('button', { name: 'Send ↗', exact: true }).click();
  for (const title of [
    'February–March prices',
    'February–March wind',
    'February–March demand',
  ])
    await expect(
      page.getByRole('tab', { name: title, exact: true }),
    ).toBeVisible();
  await expect(
    page.getByText('View updated. Use Go back to undo.', { exact: false }),
  ).toBeVisible();
  await expect(page.locator('.terminal-chat [role="alert"]')).toHaveCount(0);
  await expect
    .poll(() =>
      requests.some(
        (url) =>
          url.includes('/api/series?') &&
          url.includes('from=2026-02-01') &&
          url.includes('to=2026-03-31'),
      ),
    )
    .toBe(true);
  await page.getByRole('button', { name: '↶ Go back', exact: true }).click();
  for (const title of [
    'February–March prices',
    'February–March wind',
    'February–March demand',
  ])
    await expect(
      page.getByRole('tab', { name: title, exact: true }),
    ).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('AI source data tables render wind and demand rows instead of an algorithm placeholder', async ({
  page,
}) => {
  await page.route('**/api/agent/turns/stream', async (route) => {
    const turn = route.request().postDataJSON();
    const events = [
      {
        type: 'view',
        workspace_id: turn.workspace_id,
        revision: turn.revision,
        view: {
          version: 1,
          from_date: date,
          to_date: date,
          widgets: [
            {
              key: 'data',
              existing: null,
              type: 'table',
              title: 'Wind and demand data',
              series: ['wind', 'demand'],
              algorithm: null,
              run_id: null,
              outputs: [],
              table: '',
              group: 'A',
              relative_to: null,
              position: 'tab',
              notes: '',
            },
          ],
          remove: [],
          background_chat: false,
          new_workspace: false,
          name: 'Data',
        },
      },
      { type: 'completed' },
    ];
    await route.fulfill({
      contentType: 'text/event-stream',
      body: events
        .map(
          (event, i) =>
            `data: ${JSON.stringify({ version: 1, turn_id: turn.turn_id, sequence: i + 1, ...event })}\n\n`,
        )
        .join(''),
    });
  });
  await page.goto('/terminal');
  await page.getByRole('button', { name: '✦ AI', exact: true }).click();
  await page
    .getByRole('textbox', { name: 'Message AI' })
    .fill('Show a table of wind and demand');
  await page.getByRole('button', { name: 'Send ↗', exact: true }).click();
  await page
    .getByRole('tab', { name: 'Wind and demand data', exact: true })
    .click();
  await expect(
    page.getByRole('columnheader', { name: /Wind forecast/ }),
  ).toBeVisible();
  await expect(
    page.getByRole('columnheader', { name: /National demand/ }),
  ).toBeVisible();
  await expect(
    page.getByText('Choose an algorithm binding in widget settings.', {
      exact: true,
    }),
  ).toHaveCount(0);
});

function failedTableProposal() {
  return {
    version: 1,
    layout_mode: 'preserve',
    from_date: date,
    to_date: date,
    widgets: [
      {
        key: 'numbers',
        existing: null,
        type: 'table',
        title: 'Repaired numerical data',
        series: ['wind', 'demand'],
        algorithm: null,
        run_id: null,
        outputs: [],
        table: '',
        group: 'A',
        relative_to: null,
        position: 'tab',
        size_step: 1,
        notes: '',
      },
    ],
    remove: [],
    background_chat: false,
    new_workspace: false,
    name: 'Data',
  };
}
test('AI receives a minimum-size error and repairs the layout without publishing false success', async ({
  page,
}) => {
  const submissions: {
    layout_feedback: { error: string; attempt: number } | null;
    approval: string | null;
    history: { content: string }[];
  }[] = [];
  await page.route('**/api/agent/turns/stream', async (route) => {
    const turn = route.request().postDataJSON();
    submissions.push(turn);
    const view = failedTableProposal();
    if (turn.layout_feedback) {
      view.layout_mode = 'research';
      view.widgets[0].size_step = 8;
    }
    const events = [
      {
        type: 'answer',
        text: turn.layout_feedback
          ? 'Prepared a corrected numerical table.'
          : 'I already added the table successfully.',
        references: [],
      },
      {
        type: 'view',
        workspace_id: turn.workspace_id,
        revision: turn.revision,
        view,
      },
      { type: 'completed' },
    ];
    await route.fulfill({
      contentType: 'text/event-stream',
      body: events
        .map(
          (event, i) =>
            `data: ${JSON.stringify({ version: 1, turn_id: turn.turn_id, sequence: i + 1, ...event })}\n\n`,
        )
        .join(''),
    });
  });
  await page.goto('/terminal');
  await page.getByRole('button', { name: '✦ AI', exact: true }).click();
  await page
    .getByRole('textbox', { name: 'Message AI' })
    .fill('Add a numerical wind and demand table');
  await page.getByRole('button', { name: 'Send ↗', exact: true }).click();
  await expect(
    page.getByRole('tab', { name: 'Repaired numerical data', exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText('Prepared a corrected numerical table.', { exact: false }),
  ).toBeVisible();
  await expect(
    page.getByText('I already added the table successfully.', { exact: false }),
  ).toHaveCount(0);
  expect(submissions).toHaveLength(2);
  expect(submissions[1].layout_feedback!.error).toBe(
    'Minimum widget size reached.',
  );
  expect(submissions[1].layout_feedback!.attempt).toBe(1);
  expect(submissions[1].approval).toBeNull();
  expect(
    submissions[1].history.some((message) =>
      message.content.includes('already added'),
    ),
  ).toBe(false);
  await page.getByRole('button', { name: '↶ Go back', exact: true }).click();
  await expect(
    page.getByRole('tab', { name: 'Repaired numerical data', exact: true }),
  ).toHaveCount(0);
});
test('AI stops after two failed layout repairs and keeps the previous workspace', async ({
  page,
}) => {
  let requests = 0;
  await page.route('**/api/agent/turns/stream', async (route) => {
    const turn = route.request().postDataJSON();
    requests++;
    const events = [
      {
        type: 'answer',
        text: 'I already added the table successfully.',
        references: [],
      },
      {
        type: 'view',
        workspace_id: turn.workspace_id,
        revision: turn.revision,
        view: failedTableProposal(),
      },
      { type: 'completed' },
    ];
    await route.fulfill({
      contentType: 'text/event-stream',
      body: events
        .map(
          (event, i) =>
            `data: ${JSON.stringify({ version: 1, turn_id: turn.turn_id, sequence: i + 1, ...event })}\n\n`,
        )
        .join(''),
    });
  });
  await page.goto('/terminal');
  await page.getByRole('button', { name: '✦ AI', exact: true }).click();
  await page
    .getByRole('textbox', { name: 'Message AI' })
    .fill('Add a numerical table');
  await page.getByRole('button', { name: 'Send ↗', exact: true }).click();
  await expect(
    page.getByText('Could not apply the view after 2 repair attempts.', {
      exact: false,
    }),
  ).toBeVisible();
  expect(requests).toBe(3);
  await expect(
    page.getByRole('tab', { name: 'Repaired numerical data', exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByText('I already added the table successfully.', { exact: false }),
  ).toHaveCount(0);
  await expect(
    page.locator('path[data-terminal-series="price"]'),
  ).toBeVisible();
});
