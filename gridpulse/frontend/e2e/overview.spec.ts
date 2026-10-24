import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url());
    let body: unknown;
    if (url.pathname.endsWith('/snapshot'))
      body = {
        settlement_date: '2026-10-25',
        as_of_utc: '2026-10-25T23:30:00Z',
        settlement_period: 50,
        data_updated_at_utc: '2026-10-25T23:59:00Z',
      };
    else if (url.pathname.endsWith('/health'))
      body = { status: 'ok', mode: 'live', database: 'ok' };
    else if (url.pathname.endsWith('/weather/locations'))
      body = {
        locations: [
          {
            id: 'dogger_bank',
            label: 'Dogger Bank',
            latitude: 54.9,
            longitude: 2,
          },
        ],
        warnings: [],
      };
    else if (url.pathname.endsWith('/weather'))
      body = {
        location: {
          id: 'dogger_bank',
          label: 'Dogger Bank',
          latitude: 54.9,
          longitude: 2,
        },
        points: Array.from({ length: 26 }, (_, i) => ({
          t: new Date(
            Date.parse(
              (url.searchParams.get('date') ?? '2026-10-25') + 'T00:00Z',
            ) +
              (i - 1) * 3600000,
          ).toISOString(),
          temperature: 10 + i * 0.1,
          pressure: 1000 + i * 0.2,
          temperature_change: 0.1,
          pressure_change: 0.2,
          price_change: i,
          temperature_correlation: 0.1,
          pressure_correlation: -0.2,
          temperature_samples: 168,
          pressure_samples: 168,
        })),
        warnings: ['Fixture weather unavailable for this period.'],
        source: 'Test fixture',
        window_hours: 168,
      };
    else {
      const day = url.searchParams.get('to') ?? '2026-10-25';
      const count = day === '2026-10-25' ? 50 : 48;
      body = {
        from: day,
        to: day,
        resolution: '30m',
        t: Array.from({ length: count }, (_, i) =>
          new Date(
            Date.parse(day + 'T00:00Z') + (i - 2) * 1800000,
          ).toISOString(),
        ),
        settlement_date: Array(count).fill(day),
        settlement_period: Array.from({ length: count }, (_, i) => i + 1),
        series: Object.fromEntries(
          [
            'price',
            'wind',
            'demand',
            'residual_demand',
            'wind_outturn',
            'demand_outturn',
            'niv',
          ].map((id, j) => [
            id,
            {
              label: id,
              unit: j ? 'MW' : 'GBP/MWh',
              values: Array.from({ length: count }, (_, i) =>
                id === 'demand_outturn'
                  ? null
                  : j
                    ? (j + 1) * 1000 + i * 100
                    : i === 1
                      ? null
                      : i * 10 - 100,
              ),
            },
          ]),
        ),
      };
    }
    await route.fulfill({ json: body });
  });
});

test('50 periods, hover synchronisation and keyboard selection do not refetch', async ({
  page,
}) => {
  let requests = 0;
  page.on('request', (r) => {
    if (r.url().includes('/api/series?')) requests++;
  });
  await page.goto('/');
  await expect(page.locator('.scan-period')).toHaveCount(50);
  await page.locator('.scan-period').nth(12).hover();
  await expect(page.locator('.metric-value').first()).toHaveText('20.00');
  await expect(page.locator('.scan-period').nth(12)).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  const slider = page.getByRole('slider').first();
  await slider.focus();
  await page.keyboard.press('Home');
  await expect(page.locator('.metric-value').first()).toHaveText('-100.00');
  await page.keyboard.press('ArrowRight');
  await expect(page.locator('.metric-value').first()).toHaveText('No data');
  const box = await slider.boundingBox();
  if (box)
    await page.mouse.move(box.x + box.width * 0.7, box.y + box.height * 0.5);
  await expect(page.locator('.metric-value').first()).not.toHaveText('No data');
  expect(requests).toBe(1);
  await expect(
    page.getByRole('button', { name: 'Explore Soon' }),
  ).toBeDisabled();
});

test('date selection survives reload and empty storage is optional', async ({
  page,
}) => {
  await page.goto('/');
  await expect(page.locator('.scan-period')).toHaveCount(50);
  await page.getByLabel('Settlement date').fill('2026-10-24');
  await expect(page.locator('.scan-period')).toHaveCount(48);
  await page.reload();
  await expect(page.getByLabel('Settlement date')).toHaveValue('2026-10-24');
  await expect(page.locator('.scan-period')).toHaveCount(48);
  await page.getByRole('button', { name: 'Latest available' }).click();
  await expect(page.getByLabel('Settlement date')).toHaveValue('2026-10-25');
});

test('mobile layout, touch selection and unavailable data are usable', async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await expect(page.locator('.scan-period')).toHaveCount(50);
  await expect(page.locator('.hero')).toHaveCount(0);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.locator('.scan-period').nth(20).click();
  await expect(page.locator('.metric-value').first()).toHaveText('100.00');
  await page.route('**/api/snapshot', (r) =>
    r.fulfill({
      status: 503,
      json: {
        detail: {
          code: 'database_unavailable',
          message: 'Database unavailable',
        },
      },
    }),
  );
  await page.reload();
  await expect(page.getByRole('alert')).toContainText('Database unavailable');
  await expect(page.getByRole('button', { name: 'Try again' })).toBeVisible();
});

test('compact desktop fits one screen and combines independent axes', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/');
  await expect(page.locator('.scan-period')).toHaveCount(50);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollHeight <= innerHeight + 1,
    ),
  ).toBe(true);
  await expect(page.getByRole('slider')).toHaveCount(1);
  await expect(page.locator('path[data-series]')).toHaveCount(3);
  await expect(page.locator('path[data-series="residual_demand"]')).toHaveCount(
    0,
  );
  await page
    .getByRole('checkbox', { name: 'Residual demand GW', exact: true })
    .check();
  await expect(page.locator('path[data-series="residual_demand"]')).toHaveCount(
    1,
  );
  await page
    .getByRole('button', { name: 'About imbalance price', exact: true })
    .hover();
  await expect(page.getByRole('tooltip')).toContainText('contracted');
  await page
    .getByRole('button', { name: 'About imbalance price', exact: true })
    .focus();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('tooltip')).toHaveCount(0);
});

test('About contains the animations without fetching market data and preserves the date', async ({
  page,
}) => {
  await page.goto('/');
  await expect(page.locator('.scan-period')).toHaveCount(50);
  await page.getByLabel('Settlement date').fill('2026-10-24');
  await expect(page.locator('.scan-period')).toHaveCount(48);
  let requests = 0;
  page.on('request', (r) => {
    if (r.url().includes('/api/')) requests++;
  });
  await page.getByRole('link', { name: 'About', exact: true }).click();
  await expect(
    page.getByRole('heading', { name: 'What is imbalance?' }),
  ).toBeVisible();
  await expect(page.locator('.hero-globe canvas')).toBeVisible();
  expect(requests).toBe(0);
  await page.getByRole('link', { name: 'Overview', exact: true }).click();
  await expect(page.getByLabel('Settlement date')).toHaveValue('2026-10-24');
});

test('selector keeps cards, two axes, scan and storage synchronised', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/');
  await expect(page.locator('.metric-card')).toHaveCount(3);
  await expect(page.locator('.scan-period')).toHaveCount(50);
  await expect(
    page.getByRole('checkbox', { name: 'Temperature °C', exact: true }),
  ).toHaveCount(0);
  const option = (name: string) =>
    page.getByRole('checkbox', { name, exact: true });
  await expect(option('Net imbalance volume MWh')).toBeDisabled();
  await option('Actual wind GW').check();
  await option('Actual demand GW').check();
  await expect(page.locator('.metric-card')).toHaveCount(5);
  await expect(option('Residual demand GW')).toBeDisabled();
  await page.screenshot({
    path: test.info().outputPath('five-properties.png'),
  });
  await expect(
    page
      .locator('.metric-card')
      .filter({ hasText: 'Actual demand' })
      .locator('.metric-value'),
  ).toHaveText('No data');
  expect(
    await page.evaluate(
      () => document.documentElement.scrollHeight <= innerHeight + 1,
    ),
  ).toBe(true);
  await option('Actual demand GW').uncheck();
  await option('Actual wind GW').uncheck();
  await option('Imbalance price £/MWh').uncheck();
  await expect(page.locator('.metric-card')).toHaveCount(2);
  await expect(page.locator('path[data-series="price"]')).toHaveCount(0);
  await option('Net imbalance volume MWh').check();
  await expect(option('Imbalance price £/MWh')).toBeDisabled();
  await expect(page.locator('svg[data-axis-right]')).toHaveAttribute(
    'data-axis-right',
    'MWh',
  );
  await page.getByLabel('Scan property').selectOption('niv');
  await expect(page.locator('.scan-controls')).toContainText('MWh');
  await page.screenshot({ path: test.info().outputPath('volume-scan.png') });
  await page.reload();
  await expect(page.locator('.metric-card')).toHaveCount(3);
  await expect(page.getByLabel('Scan property')).toHaveValue('niv');
  await option('Forecast wind GW').uncheck();
  await option('Forecast demand GW').uncheck();
  await expect(option('Net imbalance volume MWh')).toBeDisabled();
  await expect(page.locator('.metric-card')).toHaveCount(1);
  await expect(page.getByRole('slider')).toHaveCount(1);
  await expect(page.locator('.daily-average')).toContainText('Daily AVG');
  await page.getByRole('slider').focus();
  await page.keyboard.press('ArrowLeft');
  await expect(page.locator('.metric-value')).not.toHaveText('No data');
});

test('daily context uses the selected day and stays fixed during hover', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1366, height: 768 });
  await page.goto('/');
  const summary = page.getByRole('region', { name: 'Day at a glance' });
  await expect(summary).toContainText('149.80');
  await expect(summary).toContainText('-100.00 £/MWh');
  await expect(summary).toContainText('390.00 £/MWh');
  await expect(summary).toContainText('9 periods · 4.5 h');
  await expect(summary).toContainText('7.90 GW');
  await expect(summary).toContainText('49/50 price periods available');
  const before = await summary.textContent();
  await page.locator('.scan-period').nth(20).hover();
  expect(await summary.textContent()).toBe(before);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollHeight <= innerHeight + 1,
    ),
  ).toBe(true);
  await page.screenshot({ path: test.info().outputPath('daily-context.png') });
});

test('daily summary fits all text at five selections on desktop', async ({
  page,
}) => {
  await page.goto('/');
  await expect(page.locator('.scan-period')).toHaveCount(50);
  await page
    .getByRole('checkbox', { name: 'Actual wind GW', exact: true })
    .check();
  await page
    .getByRole('checkbox', { name: 'Actual demand GW', exact: true })
    .check();
  for (const width of [1440, 1366, 1024]) {
    await page.setViewportSize({ width, height: 768 });
    await expect(page.locator('.day-average')).toHaveCount(5);
    expect(
      await page
        .locator('.day-summary-body')
        .evaluate((el) => el.scrollHeight <= el.clientHeight + 1),
    ).toBe(true);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollHeight <= innerHeight + 1,
      ),
    ).toBe(true);
    const coverage = await page.locator('.day-coverage').boundingBox();
    const panel = await page.locator('.day-summary').boundingBox();
    expect(coverage!.y + coverage!.height).toBeLessThanOrEqual(
      panel!.y + panel!.height,
    );
  }
});
