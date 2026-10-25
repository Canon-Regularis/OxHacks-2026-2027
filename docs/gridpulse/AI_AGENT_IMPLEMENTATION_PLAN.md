# Terminal AI agent implementation plan

Status: user-approved design, 25 October 2026. Implementation begins on `feat/terminal-ai-agent`; completion is tracked below. Product name: Eisengrau; existing GridPulse paths remain unchanged.

## Confirmed scope

An AI button on Terminal opens a chat tab. The user asks a question; the agent discovers suitable available algorithms and views, retrieves supporting results, chooses widgets and their relative orientation, and loads the view automatically. Follow-up questions can refine the view. Keep the system modular so new catalogue algorithms become discoverable without adding a prompt branch for each one.

Use the configured OpenRouter agent model; the current default is GPT-5.4 mini with low reasoning. This supersedes the earlier Kimi default. Earlier Gemini-only descriptions are superseded for this feature. Do not train ML models or invent missing results. Reuse matching stored results first; run missing registered analyses with supported parameters, requiring explicit confirmation for expensive jobs. Continue using the shared [frontend API](FRONTEND_API.md), [algorithm display contract](ALGORITHM_DISPLAY.md) and [Terminal workspace rules](TERMINAL_WORKSPACE.md).

UX decisions below were confirmed through the grilling interview. Provider endpoint compatibility remains a technical prerequisite; `OPENROUTER_API_KEY` is configured server-side with an explicit provider, base URL and model.

## Confirmed interaction and state rules

1. Place an AI button immediately before Search / Add. It reopens the last active chat; New chat creates another independent conversation as a dockable Terminal tab.
2. Empty workspaces show a centred chat prompt. After submission it becomes a docked chat tab alongside the resulting content.
3. Modify the current workspace by default. Create a workspace only when explicitly requested; otherwise ask before branching an investigation.
4. Apply complete validated layouts immediately at full colour. There is no grey preview or Accept step. Keep Go back / Undo inside chat, using chronological history: manual edits after an AI change undo before that AI change.
5. The agent can add, remove analytical widgets, resize and regroup. Never delete a conversation. For workspace setup, put chat second in an analytical tab stack and activate the analysis. For conversational explanations, keep chat visible alongside evidence when space permits. The AI button or tab restores it.
6. Answer questions as well as arranging views. Know the supported tools; select relevant evidence, give concise explanations, clickable references and expandable details. Distinguish facts, correlations and hypotheses.
7. Use existing stored results first. Execute missing analyses through a bounded API reusing the algorithm contract/service. Automatically run short bounded jobs; ask for confirmation with algorithm, dates and parameters before expensive work. Parameters must be schema-supported and server-limited.
8. Persist independent chats in browser sessionStorage: survive refresh, expire with the browser-tab session. Separate browser tabs are independent. Keep transcript and protocol context bounded; send recent history, a compact older summary, relevant evidence IDs and at most three recent workspace snapshots.
9. One layout-changing turn per workspace. Other conversations cannot read each other's history. Each turn receives fresh current context. If manual edits make the proposal stale, regenerate once against the new snapshot; on another conflict offer Retry with current layout.
10. Resolve dates/location from the current workspace and state assumptions briefly. Ask when ambiguity materially changes the investigation. Never substitute unavailable dates silently.
11. Keep existing content until usable replacements are ready. Display partial successes and specific missing/failed evidence. Cancel pending work on chat close/navigation; stop running computations where supported, keep completed reusable runs and never apply late unseen layouts.
12. Follow existing motion, minimum-size, discrete tiling, linked-group, compact-view and accessibility conventions. Respect reduced motion.

## Existing foundations and reference inspiration

Terminal already provides pure `applyActions(document, actions, size)`, validated documents, widget/run bindings, linking groups, undo, minimum sizes and a discrete split/tab layout. Reuse these rather than introducing a second layout engine. Current renderers are the capability boundary: specialised probability bands, maps and other proposed widgets must not be advertised before they exist.

Reviewed ieka v2 at commit `72836de452770eced76d1c24737347758d330e9e`:

| Reference | Useful pattern | Adaptation here |
|---|---|---|
| [agent_client.py](https://github.com/daviddudun/ieka-v2/blob/72836de452770eced76d1c24737347758d330e9e/src/agent/agent_client.py) | Separate agent orchestration from tools | Async provider calls, bounded multi-step loop and turn-local state; its synchronous HTTP inside async code and single tool iteration are unsuitable here |
| [agent_tools.py](https://github.com/daviddudun/ieka-v2/blob/72836de452770eced76d1c24737347758d330e9e/src/agent/agent_tools.py) | Named allowlisted tool executor | Typed schemas and explicit failures; malformed JSON must fail validation rather than silently become “no tool” |
| [conversation_history.py](https://github.com/daviddudun/ieka-v2/blob/72836de452770eced76d1c24737347758d330e9e/src/conversation_history.py) | Bounded per-conversation history | Isolate by session/conversation, preserve complete tool-call/result pairs and keep shared mutable memory out of concurrent turns |

This is architectural inspiration, not a wholesale port of Discord, web scraping or finance tools. Check upstream licensing before copying implementation.

## Request and layout flow

```text
Terminal chat + bounded workspace snapshot
  → dedicated same-origin POST proxy
  → backend agent orchestrator → configured model adapter
                            ↔ allowlisted data/catalogue tools
  ← answer/progress/evidence + complete typed view proposal
  → frontend view compiler → dry-run validated actions
  → atomic commit + undo → apply acknowledgement
```

The model chooses content and relative layout intent; the browser decides whether it fits. It never receives arbitrary DOM, JavaScript, CSS, SQL or shell execution tools.

### Algorithm and view discovery

Build one capability registry from existing catalogue and renderer definitions: algorithm IDs, descriptions, required inputs, output keys/units, tables, limitations, supported view types and stored-run availability. Include catalogue/schema versions in turn context. Supplement terse metadata with reviewed interpretation notes, not the entire source repository.

Tools should reuse backend services behind the existing API rather than call this server over HTTP or duplicate calculations:

| Tool | Bounded result |
|---|---|
| `list_capabilities` | Searchable algorithms, metrics and supported widgets |
| `get_coverage` | Available date ranges, gaps, latest usable date and source distinctions |
| `find_runs` | Stored runs matching an algorithm and date; explicit run IDs and parameters |
| `read_run` | Run metadata, metrics and selected table rows |
| `read_series` | Requested series over a bounded range, explicit resolution/units and missingness |
| `inspect_period` | Source values for a specific UTC/settlement identity |
| `propose_view` | A typed content/layout proposal; no server-side UI mutation |

The agent can inspect all registered algorithms, but only select compatible available outputs. Pin a single run ID across companion charts/tables. Whole-run metrics remain labelled as such when a chart is zoomed. Retrospective correlation does not establish causality or a real-time predictive signal. Never describe an unavailable run as queued or completed.

### View proposal contract

Define versioned Pydantic schemas plus matching frontend types and fixtures in `gridpulse/contracts/agent/`. Reuse current metric, binding and algorithm identifiers. A proposal includes:

- Turn/proposal IDs, schema and catalogue versions, destination workspace and base workspace revision/hash.
- Bounded widget specifications with stable proposal-local keys, supported renderer, registered series/output/table references, run IDs, title and limited settings.
- Relative placement list: left/right/top/bottom/tab against a proposal key or existing widget, with optional snapped parent divider steps; linked-group membership and explicit date/range context.
- Evidence references and a short user-facing reason for each analytical widget.

The frontend compiler maps local keys to application IDs and existing action batches. Validate nesting depth, references, units/axes, widget count, bindings and layout legality before `applyActions`. A deterministic compiler can adjust proportions to minimum sizes; it must not silently drop requested content or replace evidence. If fitting fails, return a typed reason and offer a new workspace or fewer widgets. Keep partial JSON invisible to the action layer.

A successful proposal applies immediately and becomes one undo transaction. Failed compilation leaves the document unchanged. An application acknowledgement records what actually appeared; the assistant must not claim “loaded” before this acknowledgement. Keep proposals separate from rendered data: widgets fetch authoritative data through the existing request pool.

## Provider and transport

See [provider/model research](AI_AGENT_PROVIDER_RESEARCH.md). The selected provider is OpenRouter (`https://openrouter.ai/api/v1`), model `openai/gpt-5.4-mini`, key `OPENROUTER_API_KEY`, reasoning effort `low`. CanopyWave remains explicit legacy configuration only; do not silently fall back to it or another model.

The adapter requests JSON-object output and excludes private reasoning, then validates locally. One schema repair per investigation and one presentation correction within the existing six-round budget handle malformed decisions and incompatible/empty widgets. The dedicated SSE endpoint, same-origin proxy, cancellation, execution approval and evidence rules remain unchanged. Model identity is supplied in the started event rather than hardcoded in chat.

Presentation uses `layout_mode=research` by default: the frontend assigns wide chart geometry and compact chat/supporting tabs. Use `preserve` for explicit relative placement/resize requests. The model still selects content, outputs, groups, table metrics/categories and primary ordering; it cannot create an unsupported renderer. Source tables are real time-series data tables; comparisons require inspected numeric algorithm tables. See [the implementation guide](AI_AGENT.md) for current layout and compatibility rules.

Exclude agent streams from response caching/recording and buffering middleware; review GZip, deployment proxy buffering and timeouts. Never log full prompts, secrets or provider bodies by default. Provider reasoning fields needed for continuation remain server-side; do not expose chain-of-thought as chat content.

## State and edge cases

| Case | Required behaviour |
|---|---|
| User edits layout while the agent thinks | Compare destination revision before applying; request a fresh proposal against the new snapshot, never overwrite silently |
| Duplicate submit, reconnect or retry | Idempotent turn/proposal IDs; at most one application per proposal; no automatic retry after uncertain paid request completion |
| Cancel, navigate away or disconnect | Stop provider/tool work where possible; do not apply an unseen late proposal; preserve completed chat text and current workspace |
| Widget data request fails after layout commit | Existing widget error/retry UI; chat reports partial availability rather than claiming the whole investigation succeeded |
| Missing dates, forecast revisions or run mismatch | Return structured absence; ask for another period or show available coverage; no synthetic values |
| Full workspace, locked tree or narrow screen | Respect current limits/minimums and lock semantics; explicit alternative workspace; preserve desktop tree in compact mode |
| User resizes browser before apply | Revalidate against current viewport; do not use model-supplied pixel dimensions |
| Linked contexts or mixed units | Use existing group rules and supported axes; add another chart when necessary; never coerce incompatible units |
| Undo or manual changes after an AI turn | History records actual application state; do not replay undone actions when chat is restored |
| New/removed algorithm or renderer | Version/capability check; reject stale references and rediscover |
| Long conversation or large data | Bounded history, metadata-first retrieval, limited rows and explicit downsampling; preserve tool message pairs during trimming |
| Malicious source text or user prompt | Treat retrieved strings as data; allowlisted tools and strict validation enforce capability limits independently of instructions |
| Provider unavailable, rate-limited or out of credit | Clear recoverable chat error; manual Terminal remains functional; no fabricated fallback answer |

Use one active turn per workspace and no unattended background jobs. Initial budgets: 6 model rounds plus at most one schema-repair request, 12 tool calls, 180-second turn deadline, 30-second read timeout, 90-second killable job deadline, 5,000 output tokens per model response and at most 100 compiled actions (existing action limit). Tune using measured fixtures and provider latency. Reuse existing query point/range/timeout caps. No model calls on hover or drag.

Store a bounded chat transcript in sessionStorage separately from the workspace document, with a schema version and clear/reset control. Do not duplicate credentials or raw fetched datasets in browser storage. Backend protocol messages are turn-local; bounded sessionStorage history supplies subsequent context. Expiring computation approvals are isolated by session/conversation. If the deployment has no authentication, introduce session isolation and rate/cost controls before publicly exposing a paid inference endpoint; an unguessable conversation ID alone is not ownership. Check Origin/CSRF for cookie-backed mutation routes and validate the same limits on the server.

## Module boundaries

Proposed additions; final filenames can follow established conventions:

```text
gridpulse/backend/src/gridpulse/agent/
  schemas.py       # messages, evidence, tools, proposals, events
  provider.py      # async model adapter and protocol capability handling
  service.py       # bounded orchestration and turn lifecycle
  tools.py         # catalogue/data services, no duplicated maths
  # Protocol history is turn-local; policy and expiring approvals currently live in service.py.
  routes.py        # streaming transport, cancellation, acknowledgement

gridpulse/frontend/src/features/terminal/agent/
  client.ts        # streaming parser, cancellation, reconnect policy
  investigation.ts # bounded conversation history and evidence stream lifecycle
  view.ts          # proposal validation/compiler and apply acknowledgement
  Chat.svelte      # messages, progress, input, retry, evidence and undo

gridpulse/contracts/agent/  # versioned fixtures and contract examples
```

Backend tools own data access; the orchestrator owns decisions/budgets; the frontend owns layout feasibility and application. Avoid a second catalogue, bespoke algorithm renderers in chat, or a model-specific branch in each widget.

## Implementation sequence and acceptance gates

1. **Contracts and capability inventory.** Use the confirmed UX rules and verify provider capabilities; inspect latest main for newly registered algorithms; capture schemas and fixtures. Pass when a fixture can describe real existing views and invalid references are rejected. Commit.
2. **Chat shell and deterministic view application.** Implement AI button, proposed tab, transcript and a mocked proposal through compiler/undo. Pass when one prompt fixture opens linked widgets with legal minimum sizes and one Undo restores the view. Commit.
3. **Backend read tools and session lifecycle.** Add typed read tools, isolated histories, budgets, cancellation and streaming route/proxy using a fake provider. Pass when concurrent conversations cannot share state and disconnection stops pending work. Commit.
4. **Provider adapter.** Wire confirmed settings, run opt-in compatibility checks, then integrate the bounded tool loop. Pass when tool evidence and a proposal reach the UI without keys or private reasoning reaching the browser. Commit.
5. **Grounded investigations and recovery.** Test follow-ups, stale proposals, missing runs, full layouts, provider errors and apply acknowledgements. Add motion/accessibility and deployment streaming checks. Commit tested increments.
6. **Documentation and PR.** Update API reference, capability contract, environment placeholders and deployment notes to describe implemented behaviour. Include fixtures and validation evidence; keep unbuilt capabilities explicitly deferred.

Use fake provider transcripts for routine tests: malformed/fragmented tool arguments, unknown tools/IDs, truncation, budget exhaustion, tool failures and out-of-order events. Add frontend unit tests for proposal compilation, atomic rejection, revision conflicts, idempotency and undo. Add Playwright coverage for chat-to-linked-view, follow-up modification, narrow layouts, cancellation, keyboard access and reduced motion. Run backend tests appropriate to tools/contracts and frontend `npm run check`, `npm run test`, `npm run build`, `npm run test:e2e` for relevant implementation increments. Live model calls are opt-in integration checks, not CI requirements.

First usable milestone: “show price and wind with an available relevant algorithm” opens a grounded, linked view from one prompt, followed by “put the table below it” producing an undoable rearrangement. Autonomous research experiments, arbitrary code execution, new map renderers and multi-agent orchestration remain outside this first build.

## Implementation checklist

- [x] User-approved interaction, context and execution rules recorded.
- [x] Versioned agent contracts, capability discovery and proposal compiler.
- [x] Dockable session chat, empty prompt, context snapshots and undo.
- [x] Backend model adapter, bounded tools and same-origin streaming transport.
- [x] Missing-analysis execution, expensive-job confirmation and cancellation.
- [x] Integration tests, provider compatibility evidence and documentation.

Implementation details, limits, tests and remaining live verification are tracked in [AI_AGENT.md](AI_AGENT.md). Live price/wind retrieval and a real Fourier computation (stored run 49) succeeded. Deployed-stack streaming/cancellation verification remains a deployment check.
