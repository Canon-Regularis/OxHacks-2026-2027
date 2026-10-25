import { test, expect } from '@playwright/test';

test.use({ launchOptions: { args: ['--enable-unsafe-swiftshader'] } });

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

test('globe keeps its canvas and rendered image across responsive layouts and motion changes', async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/about');
  const globe = page.locator('.hero-globe canvas');
  await expect(globe).toBeVisible();
  const original = await globe.elementHandle();
  for (const width of [840, 860, 390, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    await expect
      .poll(() =>
        globe.evaluate((canvas, previous) => canvas === previous, original),
      )
      .toBe(true);
    await expect
      .poll(() =>
        globe.evaluate((node) => {
          const canvas = node as HTMLCanvasElement;
          const pixels = canvas
            .getContext('2d')!
            .getImageData(0, 0, canvas.width, canvas.height).data;
          return pixels.some((value, i) => i % 4 === 3 && value > 0);
        }),
      )
      .toBe(true);
  }
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await expect
    .poll(() =>
      globe.evaluate((canvas, previous) => canvas === previous, original),
    )
    .toBe(true);
  await expect(globe).toBeVisible();
});

test.describe('dot animation visibility', () => {
  test('dot field stays visible through an accelerated, seeded animation cycle', async ({
    page,
  }) => {
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    await page.setViewportSize({ width: 1000, height: 800 });
    await page.addInitScript(() => {
      // Fix the initial noise phase and accelerate its clock without waiting minutes.
      crypto.getRandomValues = <T extends ArrayBufferView | null>(
        array: T,
      ): T => {
        if (array)
          new Uint8Array(array.buffer, array.byteOffset, array.byteLength).fill(
            0,
          );
        return array;
      };
      const now = performance.now.bind(performance);
      performance.now = () => now() * 30;
      const samples: number[] = [];
      Object.assign(window, { pixelCoverage: samples });
      const original = HTMLCanvasElement.prototype.getContext;
      HTMLCanvasElement.prototype.getContext = function (...args: any[]): any {
        const gl = original.apply(
          this,
          args as any,
        ) as WebGL2RenderingContext | null;
        if (args[0] === 'webgl2' && gl) {
          const draw = gl.drawElements.bind(gl);
          let frames = 0;
          gl.drawElements = (...params) => {
            draw(...params);
            if (++frames % 3) return;
            const pixels = new Uint8Array(
              gl.drawingBufferWidth * gl.drawingBufferHeight * 4,
            );
            gl.readPixels(
              0,
              0,
              gl.drawingBufferWidth,
              gl.drawingBufferHeight,
              gl.RGBA,
              gl.UNSIGNED_BYTE,
              pixels,
            );
            let visible = 0;
            for (let i = 3; i < pixels.length; i += 4)
              if (pixels[i] > 5) visible++;
            samples.push(visible / (pixels.length / 4));
          };
        }
        return gl;
      } as typeof original;
    });
    await page.goto('/about');
    await page.locator('.about-pixel-section').scrollIntoViewIfNeeded();
    await expect
      .poll(() => page.evaluate(() => (window as any).pixelCoverage.length), {
        timeout: 15000,
      })
      .toBeGreaterThanOrEqual(30);
    const coverage = await page.evaluate(
      () => (window as any).pixelCoverage as number[],
    );
    expect(Math.min(...coverage)).toBeGreaterThan(0.05);
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await expect(page.locator('.about-pixel-section canvas')).toHaveCount(0);
    await expect(page.locator('.about-pixel-copy')).toBeVisible();
  });
});
