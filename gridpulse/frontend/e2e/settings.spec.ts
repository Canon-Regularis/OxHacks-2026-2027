import { test, expect, type Page } from '@playwright/test';
const settingsKey = 'gridpulse:settings:v1';
const viewKey = 'gridpulse:view:v1';
async function fixtures(page: Page) {
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.endsWith('/health'))
      return route.fulfill({
        json: { status: 'ok', mode: 'live', database: 'ok' },
      });
    if (url.pathname.endsWith('/snapshot'))
      return route.fulfill({
        json: {
          settlement_date: '2026-10-25',
          data_updated_at_utc: '2026-10-25T00:30:00Z',
        },
      });
    if (url.pathname.endsWith('/weather/locations'))
      return route.fulfill({ json: { locations: [], warnings: [] } });
    if (url.pathname.endsWith('/catalog'))
      return route.fulfill({
        json: {
          series: [],
          datasets: [],
          algorithms: [],
          limits: { max_points: 5000, max_range_days: 3660 },
          data_updated_at_utc: '2026-10-25T00:30:00Z',
        },
      });
    const day = url.searchParams.get('to') ?? '2026-10-25';
    const count = day === '2026-10-25' ? 50 : 48;
    const start = Date.parse(day + 'T00:00Z') - 3600000;
    await route.fulfill({
      json: {
        from: day,
        to: day,
        resolution: '30m',
        t: Array.from({ length: count }, (_, i) =>
          new Date(start + i * 1800000).toISOString(),
        ),
        settlement_date: Array(count).fill(day),
        settlement_period: Array.from({ length: count }, (_, i) => i + 1),
        series: Object.fromEntries(
          [
            'price',
            'wind',
            'demand',
            'wind_outturn',
            'demand_outturn',
            'residual_demand',
            'niv',
          ].map((id, j) => [
            id,
            {
              label: id,
              unit: j ? 'MW' : 'GBP/MWh',
              values: Array.from({ length: count }, (_, i) => i + j * 100),
            },
          ]),
        ),
      },
    });
  });
}
async function open(page: Page) {
  await page
    .getByRole('button', { name: 'Open settings', exact: true })
    .click();
  await expect(
    page.getByRole('dialog', { name: 'Settings', exact: true }),
  ).toBeVisible();
}
async function close(page: Page) {
  await page
    .getByRole('button', { name: 'Close settings', exact: true })
    .click();
}
test.beforeEach(async ({ page }) => {
  await fixtures(page);
});

test('drawer traps keyboard focus, dismisses with Escape and backdrop, and restores focus', async ({
  page,
}) => {
  await page.goto('/');
  await open(page);
  const dialog = page.getByRole('dialog');
  await page
    .getByRole('button', { name: 'Close settings', exact: true })
    .focus();
  await page.keyboard.press('Shift+Tab');
  await expect(
    page.getByRole('button', { name: 'Reset settings', exact: true }),
  ).toBeFocused();
  await page.keyboard.press('Tab');
  await expect(
    page.getByRole('button', { name: 'Close settings', exact: true }),
  ).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(dialog).not.toBeVisible();
  await expect(
    page.getByRole('button', { name: 'Open settings', exact: true }),
  ).toBeFocused();
  await open(page);
  await page.mouse.click(10, 100);
  await expect(dialog).not.toBeVisible();
});

test('chart preferences apply immediately without refetching or changing selection', async ({
  page,
}) => {
  let requests = 0;
  page.on('request', (r) => {
    if (r.url().includes('/api/series?')) requests++;
  });
  await page.goto('/');
  await expect(page.locator('path[data-series]')).toHaveCount(3);
  await page.locator('.scan-period').nth(10).click();
  const selected = await page.getByRole('slider').getAttribute('aria-valuenow');
  const before = requests;
  await open(page);
  await page.getByLabel('Chart gridlines', { exact: true }).uncheck();
  await page
    .getByLabel('Chart line thickness', { exact: true })
    .selectOption('bold');
  await expect(page.locator('[data-gridline]')).toHaveCount(0);
  await expect(page.locator('path[data-series="price"]')).toHaveAttribute(
    'stroke-width',
    '4',
  );
  await close(page);
  await expect(page.getByRole('slider')).toHaveAttribute(
    'aria-valuenow',
    selected!,
  );
  expect(requests).toBe(before);
  await page.reload();
  await expect(page.locator('path[data-series="price"]')).toHaveAttribute(
    'stroke-width',
    '4',
  );
  await expect(page.locator('[data-gridline]')).toHaveCount(0);
});

test('time display distinguishes repeated UK hours without changing settlement identities', async ({
  page,
}) => {
  await page.goto('/');
  await expect(page.locator('.scan-period')).toHaveCount(50);
  await expect(page.locator('.scan-period').nth(2)).toHaveAttribute(
    'aria-label',
    /^00:00 UTC,/,
  );
  await open(page);
  await page.getByLabel('Time display', { exact: true }).selectOption('local');
  await close(page);
  await expect(page.locator('.scan-period').nth(2)).toHaveAttribute(
    'aria-label',
    /^01:00 BST,/,
  );
  await expect(page.locator('.scan-period').nth(4)).toHaveAttribute(
    'aria-label',
    /^01:00 GMT,/,
  );
  await expect(page.getByLabel('Settlement date')).toHaveValue('2026-10-25');
  await page.locator('.scan-period').nth(2).click();
  await expect(page.getByRole('slider')).toHaveAttribute(
    'aria-valuetext',
    '01:00 BST',
  );
  await expect(page.locator('.time-badge')).toHaveText('01:00 BST');
  await expect(
    page.getByRole('region', { name: 'Day at a glance' }),
  ).toContainText('BST');
  await page.getByRole('link', { name: 'Status', exact: true }).click();
  await expect(
    page.getByRole('columnheader', { name: 'Coverage (UK local)' }),
  ).toBeVisible();
});

test('appearance and motion survive navigation and reload; system appearance follows the device', async ({
  page,
}) => {
  await page.emulateMedia({
    colorScheme: 'dark',
    reducedMotion: 'no-preference',
  });
  await page.goto('/');
  await open(page);
  await page.getByLabel('Appearance', { exact: true }).selectOption('light');
  await page.getByLabel('Motion', { exact: true }).selectOption('reduce');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await expect(page.locator('html')).toHaveAttribute('data-motion', 'reduce');
  await close(page);
  await page.getByRole('link', { name: 'Status', exact: true }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.reload();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await open(page);
  await page.getByLabel('Appearance', { exact: true }).selectOption('system');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await page.emulateMedia({ colorScheme: 'light' });
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.getByLabel('Motion', { exact: true }).selectOption('system');
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await expect(page.locator('html')).toHaveAttribute('data-motion', 'full');
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await expect(page.locator('html')).toHaveAttribute('data-motion', 'reduce');
});

test('turning off remembered views clears storage but keeps current view until reload', async ({
  page,
}) => {
  await page.goto('/');
  await expect(page.locator('.scan-period')).toHaveCount(50);
  await page.getByLabel('Settlement date').fill('2026-10-24');
  await page
    .getByRole('checkbox', { name: 'Actual wind GW', exact: true })
    .check();
  await open(page);
  await page.getByRole('checkbox', { name: /Remember my view/ }).uncheck();
  await expect
    .poll(() => page.evaluate((key) => localStorage.getItem(key), viewKey))
    .toBeNull();
  await close(page);
  await expect(page.getByLabel('Settlement date')).toHaveValue('2026-10-24');
  await page.getByRole('link', { name: 'About', exact: true }).click();
  await page.getByRole('link', { name: 'Overview', exact: true }).click();
  await expect(page.getByLabel('Settlement date')).toHaveValue('2026-10-24');
  await page.reload();
  await expect(page.getByLabel('Settlement date')).toHaveValue('2026-10-25');
  await expect(page.locator('.metric-card')).toHaveCount(3);
  await open(page);
  await expect(
    page.getByRole('checkbox', { name: /Remember my view/ }),
  ).not.toBeChecked();
  await page.getByRole('checkbox', { name: /Remember my view/ }).check();
  await close(page);
  await page.getByLabel('Settlement date').fill('2026-10-24');
  await page.reload();
  await expect(page.getByLabel('Settlement date')).toHaveValue('2026-10-24');
});

test('reset restores all defaults and resets the visible day and property choices', async ({
  page,
}) => {
  await page.goto('/');
  await expect(page.locator('.scan-period')).toHaveCount(50);
  await page.getByLabel('Settlement date').fill('2026-10-24');
  await page
    .getByRole('checkbox', { name: 'Actual wind GW', exact: true })
    .check();
  await page.getByLabel('Scan property').selectOption('wind_outturn');
  await open(page);
  await page.getByLabel('Appearance', { exact: true }).selectOption('light');
  await page.getByLabel('Time display', { exact: true }).selectOption('local');
  await page.getByLabel('Motion', { exact: true }).selectOption('reduce');
  await page.getByLabel('Chart gridlines', { exact: true }).uncheck();
  await page
    .getByLabel('Chart line thickness', { exact: true })
    .selectOption('bold');
  await page
    .getByRole('button', { name: 'Reset settings', exact: true })
    .click();
  await expect(page.getByLabel('Appearance', { exact: true })).toHaveValue(
    'dark',
  );
  await expect(page.getByLabel('Time display', { exact: true })).toHaveValue(
    'utc',
  );
  await expect(page.getByLabel('Motion', { exact: true })).toHaveValue(
    'system',
  );
  await expect(
    page.getByLabel('Chart gridlines', { exact: true }),
  ).toBeChecked();
  await expect(
    page.getByLabel('Chart line thickness', { exact: true }),
  ).toHaveValue('standard');
  await close(page);
  await expect(page.getByLabel('Settlement date')).toHaveValue('2026-10-25');
  await expect(page.getByLabel('Scan property')).toHaveValue('price');
  await expect(page.locator('.metric-card')).toHaveCount(3);
  await expect(page.locator('[data-gridline]')).toHaveCount(3);
  await page.reload();
  await expect(page.getByLabel('Settlement date')).toHaveValue('2026-10-25');
});

test('settings and saved views synchronize across tabs', async ({
  page,
  context,
}) => {
  await page.goto('/');
  await expect(page.locator('.scan-period')).toHaveCount(50);
  const other = await context.newPage();
  await fixtures(other);
  await other.goto('/');
  await expect(other.locator('.scan-period')).toHaveCount(50);
  await open(page);
  await page.getByLabel('Appearance', { exact: true }).selectOption('light');
  await close(page);
  await expect(other.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.getByLabel('Settlement date').fill('2026-10-24');
  await expect(other.getByLabel('Settlement date')).toHaveValue('2026-10-24');
  await other.close();
});

test('invalid preferences and blocked storage do not break the app', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.addInitScript(() => {
    Object.defineProperty(window, 'localStorage', {
      get() {
        throw new DOMException('denied', 'SecurityError');
      },
    });
  });
  await page.goto('/');
  await expect(page.locator('.scan-period')).toHaveCount(50);
  await open(page);
  await page.getByLabel('Appearance', { exact: true }).selectOption('light');
  await close(page);
  await page.getByLabel('Settlement date').fill('2026-10-24');
  await expect(page.locator('.scan-period')).toHaveCount(48);
  await page.getByRole('link', { name: 'About', exact: true }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  expect(errors).toEqual([]);
});

test('corrupt storage recovers to defaults', async ({ page }) => {
  await page.addInitScript(
    ({ settingsKey, viewKey }) => {
      localStorage.setItem(settingsKey, '{broken');
      localStorage.setItem(viewKey, '{broken');
    },
    { settingsKey, viewKey },
  );
  await page.goto('/');
  await expect(page.locator('.scan-period')).toHaveCount(50);
  await open(page);
  await expect(page.getByLabel('Appearance', { exact: true })).toHaveValue(
    'dark',
  );
  await close(page);
  await expect(page.getByLabel('Settlement date')).toHaveValue('2026-10-25');
});

test('navbar and drawer remain usable on narrow and short screens', async ({
  page,
}) => {
  await page.goto('/about');
  for (const [width, height] of [
    [320, 568],
    [390, 844],
    [768, 600],
    [1366, 768],
  ]) {
    await page.setViewportSize({ width, height });
    await open(page);
    const dialog = page.getByRole('dialog');
    const box = await dialog.boundingBox();
    expect(box!.x).toBeGreaterThanOrEqual(0);
    expect(box!.x + box!.width).toBeLessThanOrEqual(width + 1);
    await page
      .getByRole('button', { name: 'Reset settings', exact: true })
      .scrollIntoViewIfNeeded();
    await expect(
      page.getByRole('button', { name: 'Reset settings', exact: true }),
    ).toBeInViewport();
    await close(page);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
  }
});
