# Eisengrau Terminal implementation plan

Status: architecture and delivery plan for review; implementation has not started.
Branch: `feat/terminal-workspace`, based on main `7d68fc3`.

This is the implementation reference for the page previously called Explore. The website remains Eisengrau; existing GridPulse folders and backend names remain unchanged. User-confirmed decisions take precedence over the earlier Explore mockup.

## 1. Confirmed product decisions

- Rename the visible Explore page to **Terminal**. Use `/terminal`; redirect `/explore` to it.
- Provide dockable tiles, tab stacks and discrete, snapping resize steps. Never shrink widgets below a readable minimum size.
- Reject placements that cannot fit. Drag previews turn red and explain why. Menu additions offer another workspace tab when there is no room; they never silently rearrange everything or create a tab.
- Include a widget bar listing widgets, a brief description of their contents, focus and linked-group membership.
- Linked groups share date range and selected settlement period; compatible time charts share zoom. Series, algorithm settings and pinned runs remain independent.
- Discover algorithms through a searchable catalogue drawer. Offer: add outputs to an existing chart, open a new chart, open tables, or open an analysis layout.
- Build reusable widgets: time-series chart, generic results table, metric comparison, period inspector and context/notes.
- Save named workspaces locally, autosave configuration, support JSON import/export, duplicate/reset and undo. Built-in presets remain immutable.
- Desktop docking is primary. Compact screens show one widget at a time without overwriting the desktop arrangement.
- Users consume existing stored algorithm runs. **No computation submission or arbitrary new parameter execution in this release.**
- Define the workspace so a future agent can use the same validated actions as manual controls. Gemini itself is outside this build.
- Commit working increments regularly and create/run meaningful tests throughout implementation.

## 2. Existing contracts and limits

Read these before implementing related modules:

- [Frontend API](FRONTEND_API.md): requests, runs, source coverage, errors and limits.
- [Algorithm display](ALGORITHM_DISPLAY.md): semantics, formatting and limitations.
- [Frontend brief](FRONTEND_DESIGN.md): theme, responsive behaviour and progressive disclosure.
- [Algorithm inventory](ALGORITHMS_AND_ANALYSIS.md): implemented methods versus registered adapters.
- [Frontend README](../../gridpulse/frontend/README.md): local commands and existing modules.

Verified from current code:

- Registered algorithms: `wind_price_correlation` and `price_naive_backtest`. Additional Orchid functions exist, but the eight proposed adapters are not registered. Fixtures are test data, not available user tools.
- All analysis endpoints read completed runs. Parameter overrides select an existing variant; `404 no_run` means not computed, not a request to compute.
- `/api/catalog` supplies series, algorithm outputs/units, tables, defaults, descriptions, limits and dataset lineage.
- Parameter defaults/descriptions are not a full validation schema. This release selects stored runs and displays their parameters; it does not invent generic parameter editors.
- `/api/analysis` returns numeric time series and warnings. `/api/analysis/runs/{id}/tables` returns whole-run tables, without a range filter.
- The run list returns up to 100 recent results and has no pagination. Label the list as recent runs; do not imply it is exhaustive.
- `/api/series` supports historical cutoff at 30-minute resolution. Analysis selects cutoff runs by run ID, not a cutoff query. Period-detail analysis lists default runs only.
- UTC instants identify periods. Settlement days follow Europe/London and can contain 46/48/50 periods. Ranges and responses must respect catalogue limits (currently 5,000 points per series and 3,660 days).
- Existing Overview chart/state depend on its metric registry. Terminal needs arbitrary catalogue units and its own versioned workspace state. Preserve Overview behaviour.

## 3. Screen and control hierarchy

```text
Shared Eisengrau navigation: Overview | Terminal | Methods | About
Terminal toolbar: workspace tabs | Search/Add | date range | undo/redo | layout menu
Widget bar: [Price + wind · A] [Backtest metrics · A] [Period inspector · A]
Dock canvas: split tiles and tab stacks
Status strip: mode, freshness, active group and request state
Transient drawer: catalogue OR focused-widget settings OR workspace manager
```

- Retain the neutral dark surfaces and pastel orange accent. Use compact readable sans-serif labels and monospaced numeric values; avoid tiny text and decorative dashboard heroes.
- One drawer occupies a consistent overlay position; opening another replaces it. Escape closes transient UI before cancelling a workspace action.
- Settings remain outside chart plotting areas. Headers expose title, focus/group identity and a small menu; duplicate/maximise/close are secondary actions.
- One widget has keyboard focus. Hover changes the linked cursor, not widget focus or the settings target.
- Widget-bar entries include widget type, short content description, group letter/colour and warning/loading state. Clicking reveals its tab and focuses it. Long lists scroll horizontally rather than expanding the toolbar.
- Distinguish active workspace tabs from tabs stacked inside one tile.
- Tooltips support hover/focus/tap. Essential units, coverage, current run and warning indicators remain discoverable without hovering.
- Starter workspace: source-data chart, results table and period inspector. Resolve all bindings from the catalogue; when runs are absent show a useful placeholder rather than fake results.

## 4. Docking and snapping rules

Use a split-tree layout: split nodes have horizontal/vertical orientation and a discrete ratio; leaf nodes hold one or more widget tabs and the active tab ID. Keep pixel measurements out of persisted documents.

- Quantise divider targets to a logical 24-column by 16-row canvas. Compute physical minimum-size feasibility from current canvas dimensions; quantisation does not override minimums.
- Minimums are declared by widget type and include headers, labels and warnings. Initial targets: chart 360×220 px, table 300×200, comparison 300×200, inspector 280×220, context 240×160. Validate these visually and adjust in one registry before release.
- Resize previews snap to legal steps. Clamp to the nearest feasible step; if none exists, disable that divider and explain why. Reflow chart labels and plot size after committed resize.
- Edge drops create splits; centre drops create tab stacks. Clear previews distinguish these destinations. Neither changes persistent state until drop succeeds.
- A split is legal only if both child subtrees fit their minimums, including dividers. Reject illegal drops without moving the source widget. Red feedback includes text/icon, not colour alone.
- Dropping onto a tab header reorders tabs; dropping onto the same location is a no-op. Drag thresholds prevent clicks becoming accidental drags.
- Menu-based placement uses the same feasibility calculation. If no space exists, show **Workspace has no room** with choices to cancel, choose an existing tab stack, or create a new workspace tab. Existing content remains intact until an explicit choice.
- A full analysis preset is placed atomically. If it cannot fit, offer a new workspace; never leave a half-created bundle.
- Closing the last tab removes that leaf and collapses its redundant parent split. Closing the final widget leaves an Add widget empty state.
- Maximise is temporary and reversible; it does not rewrite docking ratios. Hidden tabs remain in the widget bar.
- Workspace lock disables drag/resize only; charts and settings remain usable. Provide keyboard/menu alternatives for move, resize and placement.
- When the browser becomes too small for the current arrangement, switch to compact mode rather than violating minimums or clipping tiles. Restore the unchanged desktop layout when space returns.
- Dragging beyond the canvas, cancelling, losing pointer capture or navigating away rolls back the preview. Resizing is one undo action per gesture, not one per pointer move.

## 5. Linking, time and selection

Each widget stores a group ID or is independent. Groups have a readable name, letter and colour. The first default group is A.

- Group state owns the settlement-date range, selected UTC instant and committed time zoom. Joining a group adopts its context; leaving snapshots that context into the widget's independent state.
- Capability flags define participation in range, cursor and zoom. Time-series charts can share all three. Tables, comparisons and notes need not support every capability.
- Hover uses in-memory data and a transient group cursor. Clicking/touching pins the selected period; keyboard actions provide the same selection. Avoid announcing every mouse move to screen readers.
- A group range change clears a selected instant outside the range. A missing point inside the range stays selected and displays No data; it does not jump to another period.
- Selection is an instant, not an array index. Align different responses by timestamp and resolution. Never assume independently fetched arrays line up.
- Zoom previews are local during a gesture; committed zoom can refetch an appropriate resolution. Deduplicate the resulting group requests and keep the full settlement-date query range distinct from the visible intraday window.
- Non-time axes (lag, category, forecast lead time) do not inherit time zoom. Tables remain labelled **Whole run** unless an explicit local row filter is applied; filtering rows does not recompute scorecards.
- Output from a stored cutoff run and retrospective source data must be labelled separately. Do not invent a universal cutoff toggle that appears to recompute algorithms.
- Inspector period details come from `/periods`. Its default-run outputs are labelled as such; a pinned chart's value/run is shown separately, not substituted with a default result.
- Import Overview's saved date only when creating the first Terminal workspace. Subsequent Terminal ranges persist independently. Offer an explicit Open this day in Terminal action later; opening Terminal does not rewrite Overview preferences.

## 6. Catalogue, run binding and chart builder

The Add drawer searches Data, Algorithms, Widgets and Presets. Support keyboard navigation and meaningful empty results. Only catalogue-registered algorithms appear as usable tools.

Algorithm actions:

1. **Add outputs to chart:** choose an existing compatible chart, outputs and binding. Disable incompatible choices with a reason.
2. **Open chart:** create a new chart with selected outputs.
3. **Open tables:** choose declared tables; disable when none exist.
4. **Open analysis layout:** assemble charts/tables/context using the registry and available outputs. Table-only algorithms remain valid.

Stored-run rules:

- Opening an algorithm resolves one run binding. Related charts/tables share that binding and actual run ID; do not mix a chart response with `catalog.latest_run` tables.
- Pin the resolved run for a reproducible investigation. Offer explicit **Use latest** and **Refresh to latest**, then atomically update related widgets after resolution. A range change never silently substitutes a different run.
- Run picker shows date/coverage, version, settings, cutoff and finished time. Show parameters read-only; variants are selected from completed runs.
- No matching run is a Not computed yet state. Do not expose a misleading Run button or execute CLI commands from browser errors.
- Preserve run lineage/warnings, including source loads newer than computation, partial coverage, cached mode and older versions.
- Older runs currently use some current catalogue metadata. If version differs, require compatible output names/units and show a schema warning. Do not silently interpret a changed output; use an unsupported-version placeholder when compatibility cannot be established. Versioned per-run output metadata is a follow-up backend improvement.

Chart builder:

- Source and algorithm series use the same data-binding interface. Names/units come from responses/catalogue, not the Overview registry.
- Allow overlays with at most two explicit unit axes, aligned separate panes, or small multiples. Adding a third unit offers a separate pane or another widget.
- Offer raw values by default. Any indexed/normalised display is explicit, labelled and never overwrites source values; constant/missing series must not divide by zero.
- Choose output visibility, colour, line style and pane placement through the settings drawer. Duplicate a chart for independent comparison.
- Retain missing-value gaps, negative prices, zero, correct timestamps and source forecast/outturn labels. Do not automatically smooth or impute.
- Future arbitrary outputs get generic lines/tables. Special roles such as forecast bands use reviewed presentation adapters following `ALGORITHM_DISPLAY.md`, not guessed names alone.

## 7. Initial widget registry

| Type | Data and behaviour | Important limits |
|---|---|---|
| Time series | Any catalogue source or analysis numeric output; zoom, selection, multiple panes | Units and timestamp alignment; generic fallback for new output names |
| Results table | Named run table; sort/filter/search; accessible headers | Mixed/null cells, 20,000-row cap; virtualise large tables; whole-run coverage |
| Metric comparison | Select a numeric column and category/model from a table | No invented better/worse direction; show coverage, horizon and sample differences |
| Period inspector | Source values, forecast vintages, declared analysis readings | Missing/invalid period, default versus pinned run, clock-change labels |
| Context/notes | Editable plain-text notes plus structured source/run details | User text is text, not executable HTML; limitations cannot be erased by editing notes |

The registry defines renderer, settings renderer, defaults, minimum size, supported links and configuration validation. Generic widgets work before specialised forecast/correlogram widgets. Metadata enhancements are additive; old API responses continue to render.

## 8. Workspace model and module seams

```text
features/terminal/
  model/          versioned documents, validation, actions, history, migrations
  layout/         split tree, snapped placement, feasibility and compact adaptation
  data/           catalogue, request cache, run resolution and timestamp alignment
  widgets/        registry, common shell, chart/table/comparison/inspector/context
  controls/       Add drawer, widget bar, settings, run picker and workspace manager
  presets/        immutable built-in workspace factories
  Terminal.svelte orchestration
routes/terminal/   page entry
routes/explore/    redirect
```

Modules expose small interfaces: apply a workspace action, resolve a data binding, and render a registered widget. Layout operations contain placement complexity; widgets do not manipulate tree internals. Requests stay out of widget presentation logic.

Document shape:

```text
TerminalDocument(version, activeWorkspaceId, workspaces[])
Workspace(id, name, layoutTree, widgetsById, bindingsById, groupsById, independentContexts)
Widget(id, type, title, bindingId?, groupId?, settings)
Binding(id, kind, sourceIds? | algorithmId/runId, requestedOutputs, coverage)
Layout = Split(axis, ratioStep, children[2]) | Stack(widgetIds[], activeWidgetId)
```

- The layout is the authority for widget placement. Every widget appears exactly once; references must resolve, IDs are unique, trees are acyclic and split ratios are legal.
- Runtime state (loading, requests, errors, hover, focus, drag previews) is separate from the saved document. Do not persist fetched market values, DOM nodes or credentials.
- Actions include add/move/resize/tab/rebind/link/remove, configure, rename, duplicate and preset/import. Validate the complete result before committing it.
- Manual controls and future agent tools call the same action interface. A multi-widget operation is atomic and one undo entry. The agent extension needs no second layout engine.
- A widget error boundary isolates renderer failures. Removed algorithms and unknown widget types retain recoverable placeholders with remove/rebind/export actions.

## 9. Requests and performance

- Use the canonical response types from `src/lib/types.ts`; consolidate overlapping API helpers without changing Overview behaviour. Validate externally supplied responses before treating them as renderable data.
- Cache keys include endpoint, canonical sorted parameters, range, resolution, cutoff and run ID. Parameter serialisation remains deterministic for offline replay.
- Shared subscribers reuse in-flight requests. Abort only when no subscriber still needs the request. Use request generations so stale results never repaint a changed binding.
- Bound parallel requests; batch compatible source-series queries. Hover/drag/resize do not refetch. Hidden tabs do not mount expensive charts or poll independently.
- Preserve existing data on refresh with an Updating badge; changing a binding cannot present old data as the new binding. Show errors locally with retry.
- Obey catalogue limits before requests; cap DOM work through chart point limits and table virtualisation. Don't allocate a chart for every search result or algorithm.
- Initial configurable resource guards: 12 widget instances per workspace, 16 workspaces, 1 MiB configuration import. These are engineering defaults to verify during implementation, not backend limits. Reaching a guard produces a clear message and never discards existing work.
- Global status distinguishes live/record/replay and data freshness. Stored-run widgets need explicit refresh rather than a polling loop that changes their pinned result.

## 10. Save, import, undo and compact mode

- Use a separate storage key, `eisengrau:terminal:v1`; preserve `gridpulse:view:v1` for Overview.
- Autosave committed document changes with a short debounce; flush on workspace switch and lifecycle events when possible. Storage failure shows a persistent save-status indicator and keeps the current session usable.
- Keep a last-known-valid backup. Parse/schema failures offer recovery, export and reset without silently overwriting the broken document. Future schema versions open read-only recovery rather than being interpreted as v1.
- Import validates file size, depth, IDs, widget settings, references and known types. Imported URLs/code are not executed. Missing catalogue items remain placeholders. Import into a new workspace/document preview; confirm before replacing current work.
- Export includes configuration/schema version only. Duplicate remaps IDs and shared bindings correctly. Built-in presets instantiate new IDs; editing a workspace never mutates the preset.
- Undo/redo covers layout, configuration, group and workspace actions, with bounded history. It excludes hover, fetch results and automatic refresh. New actions clear redo; bulk changes undo atomically.
- Other-browser-tab changes trigger an update notification; do not overwrite an active drag or newer unsaved changes. Choose reload or keep this tab's workspace; cloud sync is not implied.
- Compact mode uses widget navigation and one active widget, preserving groups/settings. Desktop layout is unchanged, even after reload in compact mode. Drawers fit the viewport, touch targets remain usable and chart controls have keyboard equivalents.
- Honour reduced motion. Announce completed changes and errors accessibly; group membership has names/letters as well as colours. Resize handles and placement menus are keyboard accessible.

## 11. Delivery stages and commit discipline

Each stage ends in a reviewable commit after its checks pass. Split large stages further rather than keeping the whole Terminal uncommitted. Commit messages describe working behaviour; never commit secrets/generated builds/test reports. Do not merge/publish a PR until requested or authorised by the implementation workflow.

| Stage | Deliverable | Required evidence before commit |
|---|---|---|
| 0 | Review this plan; confirm shared understanding before implementation | Check confirmed scope and recorded engineering defaults |
| 1 | Workspace document/actions, registry contracts, import validation and migrations | Unit tests of invariants, atomic rejection, history and corrupt/old saves |
| 2 | Terminal route, widget bar, docking/tab shell, snapping and minimum sizes | Pure placement tests plus real-browser drag/resize/keyboard/compact checks |
| 3 | Typed data/cache layer, catalogue drawer, run picker, generic chart/table | Contract fixtures plus deduplication, abort/race, unavailable-run and unknown-output tests |
| 4 | Linking, chart settings, comparison/inspector/context widgets | Group isolation, UTC selection, whole-run tables and consistent run-binding tests |
| 5 | Presets, autosave/import/export, workspace manager and recovery | Round-trip, duplication, storage failure, multi-tab and invalid-import tests |
| 6 | Accessibility, responsive/performance pass, deployment regression and docs | Complete acceptance matrix; production build and existing Overview regression suite |

Use fixtures only in tests/development contexts labelled synthetic. Smoke-test both currently registered algorithms against real available runs; absence of a run must exercise the real empty state, not a fixture substitution.

## 12. Verification matrix

| Area | Cases and expected result |
|---|---|
| Placement | Valid edge/tab drop; full canvas; invalid minimum; same-position drop; drop outside; Escape/pointer-capture loss; atomic multi-widget placement |
| Sizing | Snapped discrete steps; nested splits; five labels/long titles; 1024×768, 1366×768, 1440×900 and wider; browser zoom; compact transition and restoration |
| Lifecycle | Close last tab/widget; duplicate/remap IDs; maximise/restore; lock layout; removed active widget/group; undo/redo and new-action redo invalidation |
| Linking | Independent groups; joined/left group; cursor across gaps; selected instant outside range; non-time widget; compatible zoom; focus unaffected by hover |
| Time | 46/48/50 periods; repeated London hour; UTC identity; multi-day zoom and resolution changes; boundaries and missing points |
| Runs | Default resolution to a pinned run; chart/table same ID; variants; table-only outputs; missing/partial run; old version/schema mismatch; reloaded datasets; cutoff versus retrospective |
| Tables/comparisons | Whole-run scores unchanged by chart zoom; missing/mixed cells; zero/constant metrics; long text; stable sorting; 20,000 rows; incomparable coverage labelled |
| Network | Loading/offline/503/504; malformed responses; unknown algorithm; replay-cache miss; shared request abort; rapid range/binding changes; stale response ignored |
| Save/import | Storage denied/quota; invalid JSON; version migration; future schema; duplicate IDs/cycles/dangling references; oversized input; missing catalogue entry; backup restore |
| Accessibility | Keyboard add/move/resize/tab/select; focus after closing; tooltip dismissal; screen-reader warning state; group colour alternatives; reduced motion; touch |
| Regression | Overview local hover/date persistence and About effects still work; visible navigation says Terminal; no user-facing compute controls; production proxy still works |

Run targeted unit/browser tests while building each stage. At each integrated milestone run `npm run check`, `npm run test`, `npm run test:e2e`, and `npm run build`. Run backend tests when changing response contracts or backend behaviour. Re-run failed/newly affected checks after fixes; avoid repeatedly running unrelated heavy suites for text-only changes.

## 13. Acceptance and deferred work

The release is complete when users can discover current data/registered algorithms, open their stored results, arrange readable snapping tiles, link compatible views, inspect a selected period, compare run results, save/recover a workspace and use a compact screen without corrupting the desktop layout. Unknown future numeric outputs and tables must work with generic widgets. All visible values retain units, missingness, run identity and source/mode context.

Deferred: submitting computations, job progress/cancellation, arbitrary parameter editors, registering the eight planned algorithm adapters, specialised views beyond the initial set, agent execution, cloud accounts/collaboration, unrestricted floating windows and guaranteed semantic rendering for every future output shape. Additional capabilities require explicit scope decisions rather than being silently folded into this build.
