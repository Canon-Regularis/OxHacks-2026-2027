import { test, expect } from '@playwright/test';

test('About and its decorative bands fill the viewport at every breakpoint', async ({
  page,
}) => {
  await page.goto('/about');
  await expect(
    page.getByRole('heading', {
      name: 'Published sources. Clear distinctions.',
    }),
  ).toBeVisible();
  await expect(
    page.locator('header').getByText('PUBLIC MARKET DATA'),
  ).toHaveCount(0);
  for (const width of [1920, 1440, 1024, 768, 390]) {
    await page.setViewportSize({ width, height: 900 });
    for (const selector of [
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
