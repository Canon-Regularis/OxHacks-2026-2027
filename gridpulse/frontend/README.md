# GridPulse frontend

The Overview is implemented in SvelteKit 2 / Svelte 5 with Tailwind 4 and selected KCPC source components. Terminal is implemented at `/terminal`; Methods is omitted from navigation; About is a read-only explanation page. Gemini and weather mapping remain extensions.

## Local development

Use Node 22+ and npm. From `gridpulse/frontend`:

```bash
npm ci
npm run dev
```

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

## Stable local presentation

For a local presentation, run `npm run build` followed by `npm run serve:local`, then open <http://localhost:5174>. This serves a temporary snapshot of the production build with the existing backend on port 8000. It has no Vite hot reload, and subsequent checks/builds do not alter the files being served. Stop and restart it after rebuilding to show new changes. `HOST`, `PORT`, `ORIGIN` and `API_PROXY_TARGET` can be set when starting it.

Running `npm run check` or `npm run build` alongside `npm run dev` regenerates SvelteKit files watched by Vite. This can reload the entire page, briefly removing and recreating both About animation canvases. Use the stable server when inspecting continuous animation or presenting; development hot reload remains available through `npm run dev`. With the stable server running, `npm run test:preview` opens About in a test browser, runs checks and a fresh build, and verifies that neither the document nor its animation canvases are replaced.

## Working areas

- `src/routes/`: SvelteKit entry points, per-request view context and server-only API proxy.
- `src/features/overview/`: Overview orchestration, property selector, day scan, metric registry and formatting helpers.
- `src/components/charts/`: unit-aware SVG chart with up to two axes with keyboard/pointer selection.
- `src/components/ui/InfoTooltip.svelte`: shared progressive-disclosure behaviour.
- `src/components/layout/`: shared header/footer.
- `src/features/status/`: Status page (`/status`); `model.ts` holds the pure health/freshness rules (thresholds are heuristics: 6 h ageing, 24 h stale since the loader last wrote).
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

Start with the [API reference](../../docs/gridpulse/FRONTEND_API.md) and [algorithm display guide](../../docs/gridpulse/ALGORITHM_DISPLAY.md) for Terminal and future page integration; additional response types are in `src/lib/types.ts`. Deployment uses the SvelteKit Node server; see [deployment setup](../deploy/README.md).

## Terminal workspace

See [Terminal usage and extension seams](../../docs/gridpulse/TERMINAL_WORKSPACE.md) and the [implementation plan](../../docs/gridpulse/TERMINAL_IMPLEMENTATION_PLAN.md). `/explore` redirects to `/terminal`. The page discovers stored algorithms through the shared catalogue and uses the canonical types in `src/lib/types.ts`. It supports docked snapping tiles, linked groups, source/algorithm overlays, tables/comparisons, notes, local workspaces/presets, import/export, recovery and compact navigation. Computation submission and agent execution remain deferred.

## Workspace settings

The app uses a dark theme. The navbar gear opens a keyboard-accessible modal drawer on every page. Preferences apply immediately and persist in `gridpulse:settings:v1`: UTC or UK local time, device/reduced motion, Overview chart gridlines, standard/bold Overview chart lines, and whether to remember the current view. UK time labels include BST/GMT, including the repeated clock-change hour; settlement dates and API requests retain their original meaning.

The root layout owns settings and saved-view persistence for all routes, with storage events synchronising tabs. Turning off **Remember my view** removes the saved view while keeping the current session's selection; reloading starts with the latest day and default properties. **Reset settings** restores all settings and view defaults. Browser storage failures leave the current session usable. Market values are never saved as preferences.

`src/lib/state/settings.ts` holds validated defaults and timestamp formatting; `settings.svelte.ts` provides per-layout reactive state. `SettingsDrawer.svelte` uses a native modal dialog with focus wrapping, Escape/backdrop dismissal and focus restoration. Presentation styles live in `src/styles/settings.css`. Navigation shows Overview, Terminal, Data sources, Quick start and About alongside the settings gear. On phones the two learning-page labels shorten to Data and Guide. Automated checks cover storage corruption/denial, reloads, cross-tab updates, clock changes, chart behavior, keyboard focus, and mobile layouts.

Terminal retains its own workspace and per-widget chart preferences. The navbar time display applies to Overview and Status; Terminal time labels follow its existing workspace settings.

## About page narrative

About introduces Eisengrau as a workspace for algorithmic analysis of GB electricity data. Terminal is the primary destination for forecasts, backtests, rolling signals, correlations and diagnostics; Overview is the daily starting point. About is presentation copy for the finished project, including the AI agent in the present tense at the user’s request. It describes algorithm comparison, linked views, saved investigations and CSV/JSON exports. This copy does not establish the implementation status of the agent; refer to the agent code and technical docs for that.

## Data Sources and Quick Start

`/sources` reads the current `/api/catalog` and `/api/health` metadata. `features/sources/model.ts` joins datasets to catalogue series and supplies names/source-frequency descriptions for known datasets. Coverage dates, load timestamps, counts, source notes and chart-series units come from the API. Unrecognised datasets remain visible with honest unknown fields. The page supports search, type filtering, refresh, cached-mode labels, unavailable/empty states and UTC/UK time settings. Ranges describe earliest/latest records, not completeness or historical forecast availability.

`/quick-start` is a four-step static Terminal walkthrough. Its four screenshots under `static/quick-start/` were captured from the real local app; capture dates, date range and stored run are recorded in `features/quick-start/capture.json`. Images can be opened at full size. To recapture them, run `node scripts/capture-quick-start.mjs` against a local frontend/backend with an available `price_naive_backtest` run (`LOCAL_PREVIEW_URL` overrides the default `http://127.0.0.1:5174`). The script uses an isolated browser context and does not change the user's workspace. Screenshots illustrate a saved investigation rather than current data.

Both pages use `styles/learn.css` for KCPC-style editorial headings, neutral borders and the shared pastel-orange accent. Browser tests cover catalogue fields, unknown/empty data, errors/retry, filtering, clock-change labels, mobile navigation and tutorial images/anchors. Unit tests cover dataset/series joins and chronological coverage bounds.
