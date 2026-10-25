# Using and extending Terminal

Terminal is Eisengrau's docked research workspace at `/terminal`. The old `/explore` route redirects there. It reads existing source data and stored algorithm results through [the shared API](FRONTEND_API.md); manual data controls remain read-only. The [AI chat](AI_AGENT.md) can request bounded missing analyses, with explicit confirmation for expensive jobs.

## User workflow

The Terminal uses one compact toolbar: workspace tabs, date range, Search / Add and Workspaces. Its title block and site footer are omitted to leave more space for tiles; other pages retain their footer.

1. Open **Search / Add** (Ctrl/Cmd+K). Choose **Data**, **Algorithms** or **Widgets**, then search that category. Data opens by default; there is no All tab.
2. For an algorithm, choose outputs in an existing chart, a new chart, its tables or a complete analysis workspace. Only `/api/catalog` entries are offered. Not computed yet is a normal empty state.
3. Choosing content in Search / Add automatically chooses a legal visible split, prioritising the focused tile and then available space. When no visible split fits, choose a tab or a new workspace. Choose a destination and dock position, stack as a tab, or create another workspace tab. Invalid splits explain that there is no room. Drag tile tabs to edge/centre targets for the equivalent actions; invalid previews are red and carry text.
4. Drag dividers to snap-resize; focused dividers support arrow keys. Horizontal splits use 24 steps, vertical splits 16, constrained by both subtrees' minimum readable sizes. Use settings to move without dragging.
5. Click the letter/arrow badge at the top right of a widget to choose a linked group, create a group or make it Independent. This works in docked, compact and maximised views; Escape or an outside click dismisses the menu. Focus widgets through the widget bar. Settings also expose source selection, stored runs, output visibility/styles, overlay versus separate unit panes, explicit indexing, group membership, duplication and placement.
6. Linked time charts share range/zoom and selected UTC instant. Hover reads cached values locally; click or arrow-key selection loads full period details. The inspector preloads source values for hover. Clicking a whole-run table's period outside the current range opens that day in its linked group.
7. Use **Workspaces** to rename, duplicate, reset, export/import or save the current view as a reusable preset. Save and search/load presets in Workspaces; they are not part of the content drawer. Built-ins remain unchanged.

A chart accepts at most two overlay unit axes; extra units use separate panes. More than 12 selected outputs require choosing visible outputs in settings. Separate panes scroll when the available tile cannot display them readably. Indexing is explicit (first non-zero value = 100); all-zero series cannot be indexed. Raw values remain in the cursor readout.

Market preset: source chart, period inspector and notes/context. Algorithm preset: algorithm chart, same-run table and notes/context. Generic comparisons choose a recognised error metric when present (MASE, MAE, RMSE), otherwise a numeric column; users can change it. Bar length describes magnitude and does not imply a ranking. Tables retain whole-run coverage when charts zoom.

## Persistent and runtime state

`eisengrau:terminal:v1` stores a validated version-1 document containing named workspaces, reusable presets, split/tab layouts, widgets, bindings and groups. This is separate from Overview's `gridpulse:view:v1`; its selected date is only inherited on first creation. Save preferences and notes, never market responses or credentials.

The registry defines readable minimum sizes: chart 320×200 px, table/comparison 280×180, inspector 240×180 and context 220×140. Placement adjusts existing divider ratios only when required, in snapped steps nearest their previous ratios; tile order and minimums are preserved. This avoids rejecting a viable split solely because a divider started at 50/50. When the physical viewport cannot fit the desktop layout, compact mode displays one widget at a time and keeps the desktop tree unchanged. Workspace lock stops placement/drag/resize while charts and settings remain usable.

Undo/redo records committed changes (Ctrl/Cmd+Z; Shift for redo), not hover. Bulk imports/presets are one transaction. Imports are previewed and confirmed before replacement. Validation checks version, byte limit, tree depth, duplicate IDs, references, settings and resource limits. Missing algorithms and unknown widget types retain recoverable placeholders. Old algorithm versions require a compatible versioned result schema; currently unverified versions are rejected rather than interpreted using current units.

A valid backup is kept. Invalid or future-version saves are preserved until the user chooses recovery; download the saved file or restore a backup. Storage denial/quota failures leave the session usable and exportable. Changes from another browser tab show a choice instead of silently replacing work.

Initial limits: 12 widgets/workspace, 16 workspaces, 16 custom presets, 1 MiB document/import, 40 undo entries. These are frontend guards, not backend API limits.

## Implementation seams

| Module | Responsibility |
|---|---|
| `features/terminal/model/types.ts` | Workspace, layout, widget, binding and group shapes; central limits/minimums |
| `model/document.ts` | Parsing/validation, presets, duplication, closure and bounded undo history |
| `model/actions.ts` | Atomic validated commands; shared manual-control/future-agent interface |
| `layout/engine.ts` | Pure placement, feasibility, split collapse and snapped resize |
| `layout/DockNode.svelte` | Recursive tile/tab rendering, drag previews and pointer/keyboard divider gestures |
| `data/client.ts` | Typed API reads, runtime validation, request coalescing, four-request concurrency, cancellation and run binding |
| `widgets/WidgetBody.svelte` | Independent loading/error/warning/provenance state and renderer selection |
| `widgets/Chart.svelte` | Arbitrary units/outputs, precomputed scales/paths, local cursor, committed zoom and source overlays |
| `widgets/ResultTable.svelte` | Browser filter/sort and virtualised row rendering |
| `controls/` | Single drawer, searchable catalogue, run/settings/placement controls |
| `Terminal.svelte` | Workspace orchestration, view preferences, controls and runtime focus/cursor state |

To extend an algorithm, register it on the backend and preserve the shared schema. New numeric outputs and tables work generically. Do not add algorithm IDs to the frontend to make discovery work. Special forecast bands, diagnostics or other semantic visualisations need a reviewed presentation adapter and meaningful contract metadata; arbitrary table columns do not imply their units or axis roles.

To extend a widget, add its type, minimum size, validated configuration, supported link behaviour, renderer/settings and tests. Keep non-time axes independent of shared time zoom. Retain visible warnings/limitations and meaningful error states.

Future agent tools should call `applyActions(document, actions, size)` rather than manipulating DOM or split trees. The function clones and validates a transaction, applies minimum-size/placement rules and returns the next document only when the entire transaction succeeds. The AI view compiler now uses this interface; see [AI_AGENT.md](AI_AGENT.md) for chat, streaming and execution boundaries.

## Data correctness

- Resolve an algorithm binding once and retain its run ID across associated widgets. Changing range never silently changes that run. Use the run picker explicitly for latest/default/variant selection.
- Adding an algorithm to an existing source chart retains its source selections as overlays. Related widgets sharing a binding continue to identify the same run.
- Align by UTC timestamps and compatible API resolution; preserve null gaps. London labels include timezone to distinguish repeated clock-change hours.
- Whole-run tables are not recomputed for chart zoom. Source overlays are retrospective; historical-cutoff runs carry an explicit mismatch warning.
- Period detail's default-run outputs are labelled separately from pinned chart variants. Query full history only on committed selection, not hover.
- Hidden widget tabs do not render charts or request data independently. Returning to them uses the shared response cache. Stale/aborted loads cannot repaint a newer binding.

## Verification

Run frontend `check`, `test`, `test:e2e` and `build`. The browser suite uses synthetic intercepted API responses; it tests catalogue-only future algorithms, stored-run consistency, overlays, docking rejection, snapping, group isolation, compact restoration, persistence, presets, imports, recovery, offline/storage failures and large responses. Existing Overview tests remain in the suite.

Live smoke testing additionally verified source charts and the real naive-backtest output against Tiger Data, with zero hover requests and no browser/renderer errors after restarting the backend to load main's current API schema.

Deferred features and the full acceptance matrix are in [the implementation plan](TERMINAL_IMPLEMENTATION_PLAN.md). Dropping onto a tab header inserts before that tab; settings offer the equivalent tab placement. Source/algorithm data and generic renderers work; specialised forecast bands and correlograms remain deferred as planned.
