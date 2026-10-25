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
  await page.goto('/status');
  await expect(
    page.getByRole('columnheader', { name: 'Coverage (UK local)' }),
  ).toBeVisible();
});

test('dark theme stays fixed and motion preferences survive navigation and reload', async ({
  page,
}) => {
  await page.emulateMedia({
    colorScheme: 'light',
    reducedMotion: 'no-preference',
  });
  await page.goto('/');
  await open(page);
  await expect(page.getByLabel('Appearance', { exact: true })).toHaveCount(0);
  await page.getByLabel('Motion', { exact: true }).selectOption('reduce');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await expect(page.locator('html')).toHaveAttribute('data-motion', 'reduce');
  await close(page);
  await page.goto('/status');
  await page.reload();
  await expect(page.locator('html')).toHaveAttribute('data-motion', 'reduce');
  await open(page);
  await page.getByLabel('Motion', { exact: true }).selectOption('system');
  await expect(page.locator('html')).toHaveAttribute('data-motion', 'full');
  await page.emulateMedia({ colorScheme: 'dark', reducedMotion: 'reduce' });
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await expect(page.locator('html')).toHaveAttribute('data-motion', 'reduce');
});
test('legacy light or system preferences cannot re-enable light mode', async ({
  page,
}) => {
  await page.emulateMedia({ colorScheme: 'light' });
  await page.addInitScript(
    (key) =>
      localStorage.setItem(
        key,
        JSON.stringify({
          version: 1,
          appearance: 'light',
          timeDisplay: 'local',
          gridlines: false,
        }),
      ),
    settingsKey,
  );
  await page.goto('/');
  await open(page);
  await expect(page.locator('html')).toHaveClass('dark');
  await expect(page.getByLabel('Appearance', { exact: true })).toHaveCount(0);
  await expect(page.getByLabel('Time display', { exact: true })).toHaveValue(
    'local',
  );
  await expect(
    page.getByLabel('Chart gridlines', { exact: true }),
  ).not.toBeChecked();
  await expect
    .poll(() =>
      page.evaluate(
        (key) => JSON.parse(localStorage.getItem(key) ?? '{}').appearance,
        settingsKey,
      ),
    )
    .toBeUndefined();
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
  await page.getByLabel('Time display', { exact: true }).selectOption('local');
  await page.getByLabel('Motion', { exact: true }).selectOption('reduce');
  await page.getByLabel('Chart gridlines', { exact: true }).uncheck();
  await page
    .getByLabel('Chart line thickness', { exact: true })
    .selectOption('bold');
  await page
    .getByRole('button', { name: 'Reset settings', exact: true })
    .click();
  await expect(page.getByLabel('Appearance', { exact: true })).toHaveCount(0);
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
  await page.getByLabel('Time display', { exact: true }).selectOption('local');
  await close(page);
  await open(other);
  await expect(other.getByLabel('Time display', { exact: true })).toHaveValue(
    'local',
  );
  await close(other);
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
  await page.getByLabel('Time display', { exact: true }).selectOption('local');
  await close(page);
  await page.getByLabel('Settlement date').fill('2026-10-24');
  await expect(page.locator('.scan-period')).toHaveCount(48);
  await page.getByRole('link', { name: 'About', exact: true }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
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
  await expect(page.getByLabel('Appearance', { exact: true })).toHaveCount(0);
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
    await page
      .getByRole('button', { name: 'Open keyboard shortcuts', exact: true })
      .click();
    const shortcuts = page.getByRole('dialog', {
      name: 'Keyboard shortcuts',
      exact: true,
    });
    const shortcutBox = await shortcuts.boundingBox();
    expect(shortcutBox!.x).toBeGreaterThanOrEqual(0);
    expect(shortcutBox!.x + shortcutBox!.width).toBeLessThanOrEqual(width + 1);
    expect(
      await shortcuts.evaluate(
        (element) => element.scrollWidth <= element.clientWidth,
      ),
    ).toBe(true);
    await page.keyboard.press('Escape');
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
  }
});

test('pastel accents apply immediately, persist, sync between tabs and preserve chart colours', async ({
  page,
  context,
}) => {
  await page.goto('/');
  const price = page.locator('path[data-series="price"]');
  await expect(price).toBeVisible();
  const original = await price.getAttribute('stroke');
  await open(page);
  await page.getByRole('radio', { name: 'Pastel red', exact: true }).check();
  await expect(page.locator('html')).toHaveAttribute('data-accent', 'red');
  await expect(page.locator('.site-header .brand svg')).toHaveCSS(
    'color',
    'rgb(230, 160, 160)',
  );
  await expect(price).toHaveAttribute('stroke', original!);
  await close(page);
  await page.getByRole('link', { name: 'Data sources', exact: true }).click();
  await expect(page.locator('.learn-heading h1 span')).toHaveCSS(
    'color',
    'rgb(230, 160, 160)',
  );
  await page.reload();
  await open(page);
  await expect(
    page.getByRole('radio', { name: 'Pastel red', exact: true }),
  ).toBeChecked();
  const other = await context.newPage();
  await other.goto('/quick-start');
  await expect(other.locator('html')).toHaveAttribute('data-accent', 'red');
  await page.getByRole('radio', { name: 'Pastel green', exact: true }).check();
  await expect(other.locator('html')).toHaveAttribute('data-accent', 'green');
  await expect(other.locator('.learn-heading h1 span')).toHaveCSS(
    'color',
    'rgb(168, 201, 173)',
  );
  await page
    .getByRole('button', { name: 'Reset settings', exact: true })
    .click();
  await expect(
    page.getByRole('radio', { name: 'Pastel orange', exact: true }),
  ).toBeChecked();
  await expect(other.locator('html')).toHaveAttribute('data-accent', 'orange');
  await other.close();
});
