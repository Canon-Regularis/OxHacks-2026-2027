import { test, expect } from '@playwright/test';

test('About, its decorative bands, header and footer fill the viewport', async ({
  page,
}) => {
  await page.goto('/about');
  await expect(
    page.getByRole('heading', {
      name: 'What is imbalance?',
    }),
  ).toBeVisible();
  await expect(
    page.locator('header').getByText('PUBLIC MARKET DATA'),
  ).toHaveCount(0);
  for (const width of [1920, 1440, 1024, 768, 390]) {
    await page.setViewportSize({ width, height: 900 });
    for (const selector of [
      '.site-header',
      '.site-footer',
      '.about-page',
      '.about-spacer',
      '.about-pixel-section',
    ]) {
      const box = await page.locator(selector).boundingBox();
      expect(box, `${selector} at ${width}px`).not.toBeNull();
      expect(
        Math.abs(box!.x),
        `${selector} left edge at ${width}px`,
      ).toBeLessThan(1);
      expect(
        Math.abs(box!.x + box!.width - width),
        `${selector} right edge at ${width}px`,
      ).toBeLessThan(1);
    }
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
  }
});

test('favicon serves the navbar logo as SVG on every page', async ({
  page,
}) => {
  for (const path of ['/about', '/', '/status']) {
    await page.goto(path);
    const icon = page.locator('link[rel="icon"]');
    await expect(icon).toHaveAttribute('type', 'image/svg+xml');
    const href = await icon.getAttribute('href');
    const response = await page.request.get(new URL(href!, page.url()).href);
    expect(response.ok()).toBe(true);
    expect(response.headers()['content-type']).toContain('image/svg+xml');
    const svg = await response.text();
    const matches = await page.evaluate((source) => {
      const favicon = new DOMParser().parseFromString(source, 'image/svg+xml');
      return (
        favicon.querySelector('path')?.getAttribute('d') ===
          document.querySelector('.brand svg path')?.getAttribute('d') &&
        favicon.querySelector('circle')?.getAttribute('r') === '13'
      );
    }, svg);
    expect(matches).toBe(true);
  }
});
