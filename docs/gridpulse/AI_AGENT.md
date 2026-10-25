# Terminal AI agent

Implemented on `feat/terminal-ai-agent`; live CanopyWave/Kimi evidence retrieval and a stored algorithm execution have been verified. The [approved plan](AI_AGENT_IMPLEMENTATION_PLAN.md) records product decisions; this page describes the current code.

## User flow

AI immediately before Search / Add reopens the last chat. New chat creates an independent dockable conversation; empty workspaces start with a central prompt. Chat counts toward the 12-widget workspace limit. Workspace setup requests can put chat second in an existing analytical tab stack, with analysis active. Questions keep chat visible when the layout permits.

Validated views apply immediately at full colour, in one undo transaction. Go back inside chat uses chronological undo, including intervening manual changes. Current-workspace edits are the default; another workspace requires an explicit request. Invalid layouts leave the workspace unchanged. If manual edits race an investigation, the client regenerates once against the new state and then stops on another conflict.

Chats and their placements survive refresh in sessionStorage, independently per browser tab; transcripts are not in localStorage exports or saved presets. Closing a chat stops its active turn. Navigating away cancels active work. One investigation per workspace, at most two per API process, is admitted. Conversations do not share transcripts. Each turn includes bounded recent messages, an extractive older summary, current workspace metadata and up to three recent states. There are at most twelve retained session conversations; inactive closed chats may be evicted at this limit.

## Tools and algorithms

The capability catalogue is generated from registered backend algorithms and source series. Typed tools reuse the existing API implementation/services: coverage, series, runs, analysis, tables, period and execute. No separate calculations or algorithm-specific prompt branches are needed for future algorithms using the shared contract.

Execution first reuses an available matching run. The conservative automatic tier is `calendar_fourier_terms`, `price_rolling_signals` and `wind_price_correlation`, only at default parameters and at most seven days. Other missing runs require Run analysis confirmation with algorithm, dates and parameters. Approval receipts expire after five minutes, belong to their session/conversation and are consumed once.

Requests allow at most 366 days, bounded schema-declared parameter values, six model rounds, at most one additional schema-repair call, and twelve tool calls. Jobs run in spawned processes with their own DB connection and one fit worker; they have a 90-second default deadline. Cancellation first terminates gracefully and then kills an unresponsive process. Completed runs remain reusable. If a process requires a hard kill, its interrupted run can remain marked running; it is not served as a successful result.

The view compiler validates source/output/table identifiers, pins algorithm run IDs, protects chat, checks two-unit chart limits, resolves relative placement and optional snapped divider steps, and uses existing atomic workspace actions. Clickable evidence opens the corresponding widget/run; uninspected run lists are not treated as a specific result. Data widgets continue fetching through the normal request pool, and hover remains local.

## Configuration and deployment

Backend reads `CWAVE_API_KEY`, `AGENT_BASE_URL` and `AGENT_MODEL`; defaults select CanopyWave Kimi K2.6. The [provider note](AI_AGENT_PROVIDER_RESEARCH.md) records endpoint details and live-check evidence. Earlier Gemini fields remain optional compatibility settings, not fallback behaviour.

Set the same random `AGENT_ACCESS_TOKEN` in the backend environment and the SvelteKit server environment. Local development uses root/application `.env` for the backend and `gridpulse/frontend/.env` for SvelteKit. Never prefix it PUBLIC_. Docker Compose passes only that proxy token to the web service; inference credentials stay backend-only. A matching local token was generated in ignored env files during implementation.

Run **one backend worker** (`WEB_CONCURRENCY=1`, now the image default). Admission, deduplication and approvals are process-local. Multiple workers/replicas require a shared coordinator before enabling agent traffic. Public end-user authentication is not implemented; same-origin checks, a server-to-server token and conservative global/session hourly caps protect the initial endpoint, but do not constitute user accounts. The default global limit is 60 turns/hour and session limit 20; conversation tokens cannot bypass the global cap.

`POST /api/agent/turns/stream` is proxied by a dedicated SvelteKit handler. It verifies Origin, creates an HttpOnly same-site session cookie, injects the backend token and forwards the abort signal. It never accepts arbitrary upstream URLs. API replay mode disables AI; chat streams are excluded from demo recording. SSE responses use no-store and disable proxy buffering. Provider calls currently return complete decisions, so progress is live but answer tokens are not streamed incrementally.

## Tests and remaining verification

Backend tests cover schema rejection, bounded tools, fabricated runs, evidence loops, reuse, confirmation ownership, endpoint authentication, body limits, deduplication and lock release. Frontend tests cover atomic layout rejection, protected chat, persistence boundaries and docking. Browser tests exercise chat-to-view, Undo, refresh, blank prompts, separate conversations and expensive-job confirmation.

Live verification: a real Kimi turn fetched price/wind evidence for 23 October 2026 and produced a chart view, which passes the frontend compiler contract test. A separate real worker ran `calendar_fourier_terms` for that date, stored run **49** and returned six readable output series. After the initial schema failure, a provider regression test and one bounded schema repair were added; the original investigation then succeeded.

Deployment-specific cancellation/proxy streaming remains to be checked after deployment. Mocked tests establish application behaviour, not broad model quality. One live example is not evidence that all investigations will choose the best algorithms.

A real headless-browser smoke test also traversed the local SvelteKit POST proxy, authenticated backend, Kimi and Tiger Data, then rendered a price/wind chart with no browser or chat errors. Single-day evidence retains 30-minute settlement values to match the plotted data; wider daily aggregates and partial samples must be labelled in answers.

Final local checks: backend agent/API suite **61 passed**; frontend unit suite **39 passed**; browser suite **34 passed**; Svelte check reported zero errors/warnings; production build succeeded. The browser suite includes compact-screen chat docking and manual-edit conflict handling. Deployed-stack checks remain separate from these local results.
