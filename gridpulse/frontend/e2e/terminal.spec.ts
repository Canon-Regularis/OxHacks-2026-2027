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
  await page.setViewportSize({ width: 1366, height: 768 });
  await page.goto('/terminal');
  await expect(
    page.locator('path[data-terminal-series="price"]'),
  ).toBeVisible();
  await page.getByRole('button', { name: /Search \/ Add/ }).click();
  await page.getByLabel('Search data algorithms and widgets').fill('chart');
  await page
    .getByRole('button', { name: 'Time-series chart Add +', exact: true })
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
  await page.setViewportSize({ width: 1440, height: 900 });
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
    name: 'Context & notes',
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
    clientX: box.x + 4,
    clientY: box.y + box.height / 2,
  });
  await expect(page.locator('.terminal-drop-preview.invalid')).toContainText(
    'No room',
  );
  await page.keyboard.press('Escape');
  await expect(page.locator('.terminal-drop-preview')).toHaveCount(0);
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
  await page.getByRole('button', { name: 'Close drawer' }).click();
  await page.getByRole('button', { name: /Search \/ Add/ }).click();
  await page
    .getByLabel('Search data algorithms and widgets')
    .fill('My analysis');
  await page
    .getByRole('button', { name: 'My analysis Saved preset ↗', exact: true })
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
