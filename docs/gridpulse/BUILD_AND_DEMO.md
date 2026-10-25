# 25-hour implementation and demo plan

## Ownership

| Workstream | Responsibilities | First integration deliverable |
|---|---|---|
| Data/database | Existing API exporter, CSV mapping, Tiger Data schema/import, quality and coverage | Real sample CSVs and repeatable import |
| Backend/algorithms | Integrate existing algorithms, persist results, expose API, coordinate hosting | Real algorithm output behind an agreed response schema |
| Frontend/product | Dashboard, charts, states, polish, demo and README | Main screen rendering a clearly labelled fixture matching the API |

Assign named owners at kickoff; adapt roles to preserve the existing exporter and algorithm work. Agree CSV schemas, API examples, units, date conventions and one demo path in the first hour. Frameworks are a team choice, not an additional research task.

## Checkpoints

| Elapsed time | Deliverable |
|---|---|
| 0–1h | Confirm real CSV samples and algorithm interface; choose stack; agree responses and dashboard layout |
| 1–6h | Import into Tiger Data; wrap algorithms; render frontend fixtures; prepare Vultr deployment |
| 6–12h | Complete CSV → database → algorithm → API → frontend flow with real data; verify repeated import |
| 12–18h | Historical exploration, period details, data-quality states and visual polish; get hosted app reachable |
| 18–22h | Fix integration/hosting issues; freeze cached demo; attempt Gemini extension only if core is stable |
| 22–25h | Freeze features, smoke-test, rehearse, capture backup video and submit |

If blocked on the database, develop against the agreed CSV fixture while fixing ingestion. If an algorithm lacks required inputs, expose only supported results and explain missing coverage. Preserve an honest cached demonstration if live connectivity fails.

## Dashboard priorities

**Confirmed frontend scope:** Overview, Terminal (formerly Explore) and Methods are MVP pages. Overview and Terminal are implemented; About holds the globe and animated explanations. Methods remains planned. Investigate/Gemini and the synoptic weather map remain extensions.


Follow the [confirmed frontend brief](FRONTEND_DESIGN.md) for KCPC responsiveness, theme, globe, fade animation, section/card patterns and repository exploration. Reuse selected components rather than reproducing the entire reference website. Gemini-driven animated view changes and the synoptic/electricity map are extension requirements, not core delivery gates.


**Core screen:** strong visual hierarchy, coherent colours/typography, responsive layout, date/range controls, freshness and source labels, summary cards, aligned wind/demand/price charts and available algorithm indicators. Charts need clear units, readable tooltips and support for negative prices.

**Next:** historical replay, selected-period details, day comparisons, missing-data explanations and CSV export. A heatmap is useful when an algorithm provides a meaningful period-level indicator; use its documented scale rather than inventing probability percentages.

**Optional:** Gemini question panel and scenario controls supported by the actual algorithms. Do not add a wind slider backed by an invented formula. A retrospective reveal of actual prices must not imply that all displayed inputs were known beforehand.

Loading, empty, error and cached-mode states are part of the core UI. Synthetic fixtures used during development must be labelled and replaced with real data for the real-data demo.

## Vultr hosting

Use Vultr as the application hosting target and Tiger Data as the managed database. Choose the simplest deployment the team can operate, such as one VM serving frontend and backend. Containerise only if it helps the team ship faster.

Keep `DATABASE_URL`, `GEMINI_API_KEY` and any optional Snowflake credentials on the server. Configure frontend/backend routing, permitted frontend origins where needed, database connectivity and HTTPS for the public app. Document deployed URL, deployment/update steps and local startup commands in the root README. Verify reboot/restart behaviour and test the URL from another device before relying on it for judging.

## Repository layout

Follow the [repository structure and request flow](REPOSITORY_STRUCTURE.md). The skeleton now exists, with frontend, backend, database migrations, shared contracts, data, scripts and deployment areas. The backend, the P0 algorithm layer and the Vultr deployment are implemented; the SvelteKit frontend (Overview, Terminal, About, Status) is implemented; Methods and Gemini are not.

The algorithm integration location is `gridpulse/backend/src/gridpulse/analysis/algorithms/`. The shared frontend HTTP boundary is `gridpulse/frontend/src/lib/api.ts`; backend routes live in `gridpulse/backend/src/gridpulse/api/`. Keep one repository and avoid extra services beyond the frontend, backend and database.

## Demo sequence (2–3 minutes)

1. Explain the electricity-market problem and show the real-data dashboard.
2. Explore wind, demand and price changes over a selected historical range.
3. Open one period and explain the existing algorithm's output and inputs.
4. Demonstrate Tiger Data's role and the Vultr-hosted application.
5. If complete, ask Gemini a question and show the evidence used in its answer.
6. State limitations and show reproducible data/run metadata.

## Final checks

- [ ] Reimporting the same CSV does not duplicate logical data.
- [ ] Charts and analysis reflect real source records and documented algorithms.
- [ ] Negative prices, missing inputs and clock-change days display correctly.
- [ ] Hosted app works from another device; keys are absent from client bundles and Git.
- [ ] Cached fallback launches locally with network unavailable and is clearly labelled.
- [ ] Root README has exact commands and configuration placeholders.
- [ ] Demo video, screenshots, attribution and submission are ready before the deadline.
