import { test, expect } from '@playwright/test';

const iso = (hoursAgo: number) =>
  new Date(Date.now() - hoursAgo * 3_600_000).toISOString();

test('status page shows health, stale data and out-of-date runs', async ({
  page,
}) => {
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path.endsWith('/health'))
      return route.fulfill({
        json: { status: 'ok', mode: 'live', database: 'ok', version: '0.1.0' },
      });
    if (path.endsWith('/catalog'))
      return route.fulfill({
        json: {
          series: [],
          limits: { max_points: 1, max_range_days: 1 },
          data_updated_at_utc: iso(1),
          datasets: [
            {
              dataset: 'elexon_ndf',
              source: 'Test fixture',
              notes: null,
              first_utc: iso(500),
              last_utc: iso(-20),
              loaded_at_utc: iso(1),
              approx_rows: 1200,
            },
            {
              dataset: 'elexon_windfor',
              source: 'Test fixture',
              notes: null,
              first_utc: iso(500),
              last_utc: iso(-20),
              loaded_at_utc: iso(40),
              approx_rows: 900,
            },
          ],
          algorithms: [
            {
              id: 'price_baseline',
              version: '1',
              latest_run: {
                datasets: { elexon_ndf: iso(5) },
                finished_at_utc: iso(5),
              },
            },
          ],
        },
      });
    return route.fulfill({ status: 404, json: {} });
  });
  await page.goto('/status');
  await expect(
    page.getByRole('heading', { name: /Health & status/ }),
  ).toBeVisible();
  await expect(page.getByRole('status')).toContainText(
    'Operational, with warnings',
  );
  await expect(page.getByRole('cell', { name: /Problem/ })).toContainText(
    'Problem',
  );
  await expect(page.getByText('Inputs reloaded since this run')).toBeVisible();
  await expect(
    page
      .getByRole('navigation', { name: 'Primary navigation' })
      .getByRole('link', { name: 'Terminal', exact: true }),
  ).toBeVisible();
});

test('status page reports an unreachable API', async ({ page }) => {
  await page.route('**/api/**', (route) =>
    route.fulfill({
      status: 503,
      json: { detail: { code: 'backend_unavailable', message: 'Down.' } },
    }),
  );
  await page.goto('/status');
  await expect(page.getByRole('status')).toContainText('API unreachable');
});
