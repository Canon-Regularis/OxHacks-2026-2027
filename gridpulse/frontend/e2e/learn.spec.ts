import { test, expect } from '@playwright/test';
import fixture from '../../contracts/examples/catalog.json' with { type: 'json' };
const catalog = {
  ...fixture,
  datasets: [
    ...fixture.datasets,
    {
      dataset: 'clock_fixture',
      source: 'Synthetic test source',
      notes: 'Test-only metadata',
      first_utc: '2026-10-25T00:30:00Z',
      last_utc: '2026-10-25T01:30:00Z',
      loaded_at_utc: null,
      approx_rows: 0,
    },
    {
      dataset: 'openmeteo_era5',
      source: 'Synthetic weather source',
      notes: null,
      first_utc: null,
      last_utc: null,
      loaded_at_utc: null,
      approx_rows: null,
    },
  ],
};
async function catalogueRoutes(page: import('@playwright/test').Page) {
  await page.route('**/api/catalog', (route) =>
    route.fulfill({ json: catalog }),
  );
  await page.route('**/api/health', (route) =>
    route.fulfill({
      json: {
        status: 'ok',
        mode: 'replay',
        database: 'not_used',
        version: 'test',
      },
    }),
  );
}

test('sources show real catalogue fields, distinguish stored-only data and filter without refetching', async ({
  page,
}) => {
  let requests = 0;
  page.on('request', (request) => {
    if (request.url().endsWith('/api/catalog')) requests++;
  });
  await catalogueRoutes(page);
  await page.goto('/sources');
  await expect(page.locator('.sources-table tbody tr')).toHaveCount(
    catalog.datasets.length,
  );
  await expect(page.getByText('Cached replay', { exact: true })).toBeVisible();
  const missing = page
    .locator('tbody tr')
    .filter({ hasText: 'openmeteo_era5' });
  await expect(missing).toContainText('Reanalysis');
  await expect(missing).toContainText('No catalogue chart series');
  await expect(missing).toContainText('Not supplied');
  const clock = page
    .locator('.sources-table tbody tr')
    .filter({ hasText: 'clock_fixture' });
  await expect(clock.getByRole('cell').last()).toHaveText('0');
  await page.getByLabel('Search datasets').fill('elexon_system_prices');
  await expect(page.locator('.sources-table tbody tr')).toHaveCount(1);
  await expect(page.locator('.sources-table tbody')).toContainText('GBP/MWh');
  await page.getByLabel('Search datasets').fill('nothing_matches');
  await expect(
    page.getByText('No matching datasets.', { exact: true }),
  ).toBeVisible();
  await page.getByRole('button', { name: 'Clear filters' }).click();
  await page.getByLabel('Filter data type').selectOption('Reanalysis');
  await expect(page.locator('.sources-table tbody tr')).toHaveCount(1);
  expect(requests).toBe(1);
});

test('source timestamps honour UK clock changes without conflating the repeated hour', async ({
  page,
}) => {
  await page.addInitScript(() =>
    localStorage.setItem(
      'gridpulse:settings:v1',
      JSON.stringify({
        version: 1,
        timeDisplay: 'local',
        motion: 'reduce',
        gridlines: true,
        lineThickness: 'standard',
        rememberView: true,
      }),
    ),
  );
  await catalogueRoutes(page);
  await page.goto('/sources');
  const clock = page
    .locator('.sources-table tbody tr')
    .filter({ hasText: 'clock_fixture' });
  await expect(clock).toContainText('01:30 BST');
  await expect(clock).toContainText('01:30 GMT');
});

test('source catalogue handles failure, retry and empty results', async ({
  page,
}) => {
  let attempts = 0;
  await page.route('**/api/catalog', (route) =>
    ++attempts === 1
      ? route.fulfill({
          status: 503,
          json: { detail: { message: 'Catalogue temporarily unavailable' } },
        })
      : route.fulfill({ json: { ...catalog, datasets: [], series: [] } }),
  );
  await page.route('**/api/health', (route) =>
    route.fulfill({
      json: {
        status: 'degraded',
        mode: 'live',
        database: 'unavailable',
        version: 'test',
      },
    }),
  );
  await page.goto('/sources');
  await expect(page.getByRole('alert')).toContainText(
    'Catalogue temporarily unavailable',
  );
  await page.getByRole('button', { name: 'Refresh catalogue' }).click();
  await expect(
    page.getByText('No datasets reported.', { exact: true }),
  ).toBeVisible();
  await expect(page.getByRole('alert')).toHaveCount(0);
});

test('quick start has four usable screenshots, working section links and no market API requests', async ({
  page,
}) => {
  const apiRequests: string[] = [];
  page.on('request', (request) => {
    if (request.url().includes('/api/')) apiRequests.push(request.url());
  });
  await page.goto('/quick-start');
  await expect(page).toHaveTitle('Quick start - Eisengrau');
  await expect(page.locator('.quick-step')).toHaveCount(4);
  for (const image of await page.locator('.quick-figure img').all()) {
    await image.scrollIntoViewIfNeeded();
    await expect
      .poll(() =>
        image.evaluate((node) => (node as HTMLImageElement).naturalWidth),
      )
      .toBe(1440);
  }
  await page
    .getByRole('navigation', { name: 'Tutorial steps' })
    .getByRole('link', { name: /Save & export/ })
    .click();
  await expect(
    page.getByRole('heading', { name: 'Leave yourself a way back.' }),
  ).toBeInViewport();
  await expect(
    page.getByText(/Screenshots are examples captured from the app/),
  ).toBeVisible();
  expect(apiRequests).toEqual([]);
});

test('new pages and navigation fit narrow screens while the dataset table scrolls locally', async ({
  page,
}) => {
  await catalogueRoutes(page);
  for (const path of ['/sources', '/quick-start']) {
    await page.goto(path);
    await expect(page.locator('.site-footer')).toHaveCount(
      path === '/sources' ? 0 : 1,
    );
    for (const width of [1440, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 900 });
      const nav = page.getByRole('navigation', { name: 'Primary navigation' });
      for (const name of [
        'Overview',
        'Terminal',
        'Data sources',
        'Quick start',
        'Status',
        'About',
      ])
        await expect(
          nav.getByRole('link', { name, exact: true }),
        ).toBeInViewport();
      await expect(
        page.getByRole('button', { name: 'Open settings' }),
      ).toBeInViewport();
      await expect(
        page.getByRole('button', {
          name: 'Open keyboard shortcuts',
          exact: true,
        }),
      ).toBeInViewport();
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      ).toBe(true);
      if (width === 1440 || width === 390)
        await page.screenshot({
          path: test.info().outputPath(`${path.slice(1)}-${width}.png`),
          fullPage: true,
        });
    }
  }
});

test('quick start exposes the same Terminal keyboard reference', async ({
  page,
}) => {
  await page.goto('/quick-start');
  await page
    .getByRole('navigation', { name: 'Tutorial steps' })
    .getByRole('link', { name: /Keyboard shortcuts/ })
    .click();
  const reference = page.getByRole('region', {
    name: 'Keep the keys close.',
    exact: true,
  });
  await expect(reference).toBeInViewport();
  await expect(
    reference.getByRole('table', { name: 'Workspace', exact: true }),
  ).toContainText('Open Search / Add');
  await expect(
    reference.getByRole('table', { name: 'Layout', exact: true }),
  ).toContainText('snap step');
});
