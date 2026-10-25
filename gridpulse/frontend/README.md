# GridPulse frontend

The Overview is implemented in SvelteKit 2 / Svelte 5 with Tailwind 4 and selected KCPC source components. Explore and Methods are visibly disabled; About is a read-only explanation page. Gemini and weather mapping remain extensions.

## Local development

Use Node 22+ and npm. From `gridpulse/frontend`:

```bash
npm ci
npm run dev
```

Start with the [API reference](../../docs/gridpulse/FRONTEND_API.md) and the [guide to displaying algorithm results](../../docs/gridpulse/ALGORITHM_DISPLAY.md); response types are in [`src/lib/types.ts`](src/lib/types.ts). Use [contract examples](../contracts/examples/) during parallel development, visibly labelled as fixtures. Read the [request flow](../../docs/gridpulse/REPOSITORY_STRUCTURE.md#request-flow) before connecting the API. Database credentials and Gemini keys belong only on the backend.

Open <http://localhost:5173>. Run the FastAPI backend on port 8000 in another terminal (see [backend setup](../backend/README.md)). The SvelteKit `/api/*` handler proxies GET requests to `http://127.0.0.1:8000` in development and production. To change it, set server-side `API_PROXY_TARGET`; never put database credentials or service keys in public frontend variables.

## Verification and production

```bash
npm run check
npm run test
npx playwright install chromium --only-shell
npm run test:e2e
npm run build
node build
```

The production Node server defaults to port 3000; configure `HOST`, `PORT`, and `ORIGIN` for deployment. The backend must remain separately running. Browser tests use intercepted, clearly synthetic API responses and check 50-period days, hover without refetching, keyboard selection, date persistence, mobile layout and connection errors.

## Working areas

- `src/routes/`: SvelteKit entry points, per-request view context and server-only API proxy.
- `src/features/overview/`: Overview orchestration, property selector, day scan, metric registry and formatting helpers.
- `src/components/charts/`: unit-aware SVG chart with up to two axes with keyboard/pointer selection.
- `src/components/ui/InfoTooltip.svelte`: shared progressive-disclosure behaviour.
- `src/components/layout/`: shared header/footer.
- `src/features/about/`: globe, PixelBlast/TextShimmer explainer and read-only context.
- `src/lib/api.ts`: typed API boundary; no direct SQL or credentials.
- `src/lib/state/view.svelte.ts`: versioned browser preferences, shared through root layout context; never shared between server requests.
- `src/lib/vendor/kcpc/`: selected upstream components and provenance; no dependency on sibling checkouts at runtime.
- `src/styles/`: neutral KCPC theme tokens and responsive application layout.

Cards, legends and series requests use the `METRICS` registry in `model.ts`. The registry also defines unit families, scan colours and selector groups; backend `WEATHER_METRICS` defines available weather analysis. Changes to response fields require updating runtime schemas, TypeScript types and tests together.

## Behaviour

On first use, fetch the latest priced settlement day. Date changes persist in `localStorage` under `gridpulse:view:v1`, along with weather location, selected properties and scan property; the latest-day mode is stored as a null date. Reload restores that choice and fetches fresh data. Storage failures do not prevent use. Storage events synchronise open tabs; later pages can consume the same root context. No historical data is silently replaced by synthetic values.

Fetch the selected day plus its previous day once to calculate period-to-period card differences at midnight. Hovering a chart or day-scan period updates the shared selected index, cards and chart markers locally. The scan has a distance-based magnification centred on the current selection, retained after pointer exit. Keyboard and touch selection are supported. API responses are aborted when superseded.

The scan dropdown offers selected properties, using each property’s units and a matching sequential colour scale over its daily min/max; colours describe magnitude, not risk. The scan uses actual 46/48/50-period identities. Weather context is fetched per day/location, not on hover. It uses completed UTC hours and retrospective archived forecasts from the existing database; hourly temperature/pressure-change correlations with price changes use a trailing seven-day window, at least 24 finite pairs and nonconstant data. Null values remain missing. Source, units, sample count and limitations are visible.

See the [confirmed frontend brief](../../docs/gridpulse/FRONTEND_DESIGN.md) and [component provenance](src/lib/vendor/kcpc/README.md).

## Compact layout and colour

Overview fits one desktop viewport without clipping; phones stack and scroll. `src/styles/overview.css` defines compact layout, pastel orange accents and the gradual peach price scale. Users select 1–5 properties with at most two unit families; incompatible additions are disabled. Price is optional. Defaults are price, forecast wind and forecast demand. Each selected property has a card showing the hovered value and muted Daily AVG; missing values show No data and do not enter averages. The chart derives its labelled axes from the selection. Definitions, methods and source details use shared tooltips rather than repeated paragraphs. Keep essential units, timestamps and source/missing-data badges visible. About carries the globe and KCPC-style PixelBlast/TextShimmer section; it makes no market API calls. See the [frontend brief](../../docs/gridpulse/FRONTEND_DESIGN.md#compact-dashboard-and-progressive-disclosure) before adding future dashboard text.

## Property catalogue and alignment

The catalogue offers seven electricity series: price, forecast/actual wind, forecast/actual demand, residual demand and net imbalance volume. Stepped weather, hourly-change and correlation options are removed from Overview; saved selections automatically discard those properties. At the fixed half-hour resolution price minima/maxima duplicate price, so they are omitted; sample counts remain coverage metadata. All market series are preloaded for the date, so property toggles and hover need no new market requests. Weather is loaded per date/location and aligned to completed UTC hours; hourly lines are stepped rather than interpolated. Missing actuals are never replaced by forecasts. Extend `METRICS` and the typed API boundary together when adding properties.

## Daily context

`DaySummary.svelte` sits below the content-sized selector. It uses the selected day’s preloaded records for averages of selected properties, price extremes and their first UTC timestamps, negative-price periods/duration, and peak forecast demand. It remains fixed during hover. Missing records are excluded, and price coverage is shown so incomplete days remain identifiable. Each settlement period contributes 30 minutes, including 46/50-period clock-change days.

Deployment configuration is documented in [deploy/README.md](../deploy/README.md). Its current frontend image expects static output; the local frontend uses a Node server and server-side API proxy, so deployment compatibility needs reconciliation before deploying this work.
