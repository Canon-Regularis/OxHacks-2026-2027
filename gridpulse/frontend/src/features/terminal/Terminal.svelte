<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import { dev } from '$app/environment';
  import type { Catalog, Health } from '$lib/types';
  import {
    createDocument,
    preset,
    emptyWorkspace,
    duplicateWorkspace,
    parseDocument,
    History,
    closeWidget,
    reveal,
    STORAGE_KEY,
  } from './model/document';
  import {
    uid,
    createWidget,
    context,
    LIMITS,
    type TerminalDocument,
    type Workspace,
    type Widget,
    type WidgetType,
    type Binding,
    type NewBinding,
    type LinkContext,
    type Position,
  } from './model/types';
  import {
    fits,
    place,
    resize,
    findStack,
    stack,
    suggestPlacement,
  } from './layout/engine';
  import { bootstrap, pool, dateOf } from './data/client';
  import DockNode from './layout/DockNode.svelte';
  import WidgetBody from './widgets/WidgetBody.svelte';
  import Drawer from './controls/Drawer.svelte';
  import Catalogue from './controls/Catalogue.svelte';
  import Settings from './controls/Settings.svelte';
  import GroupPicker from './controls/GroupPicker.svelte';
  import { applyActions, type WorkspaceAction } from './model/actions';
  import Chat from './agent/Chat.svelte';
  import {
    conversation,
    type Conversation,
    type Evidence,
    type View,
  } from './agent/types';
  import { compileView, persistentDocument, chatCompanion } from './agent/view';
  import { investigate } from './agent/investigation';
  import { applyWithRepairs } from './agent/repair';
  import './terminal.css';
  let chats = $state<Record<string, Conversation>>({});
  let lastChat = $state<Record<string, string>>({});
  let running = $state<Record<string, string>>({});
  const controllers = new Map<string, AbortController>();
  const blankChat = conversation();
  const browserInstance = uid();
  const SESSION_KEY = 'eisengrau:agent:v1';
  const SESSION_LAYOUT = 'eisengrau:terminal:session:v1';
  function chatState(id: string): Conversation {
    return chats[id] ?? (chats[id] = conversation());
  }
  function cancelChat(id: string) {
    controllers.get(id)?.abort();
  }
  function openAI(fresh = false): string | null {
    if (!workspace || !doc) return null;
    const previous = lastChat[workspace.id];
    const found =
      !fresh &&
      (workspace.widgets[previous]?.type === 'chat'
        ? previous
        : Object.values(workspace.widgets).find((w) => w.type === 'chat')?.id);
    if (found) {
      activate(found);
      return found;
    }
    if (Object.keys(chats).length >= 12) {
      const used = new Set(
        doc.workspaces.flatMap((ws) => Object.keys(ws.widgets)),
      );
      for (const id of Object.keys(chats))
        if (!used.has(id)) {
          delete chats[id];
          break;
        }
      if (Object.keys(chats).length >= 12) {
        notice =
          'Maximum 12 open session conversations. Close an unused chat first.';
        return null;
      }
    }
    if (workspace.locked) {
      notice = 'Unlock the layout before opening a new chat.';
      return null;
    }
    const widget = createWidget('chat', range);
    if (
      !fresh &&
      previous &&
      chats[previous] &&
      !doc.workspaces.some(
        (ws) => ws.id !== workspace!.id && Object.hasOwn(ws.widgets, previous),
      )
    )
      widget.id = previous;
    const draft = `${workspace.id}-chat`;
    if (!widgets.length) widget.id = draft;
    widget.group = null;
    const candidate = plain(workspace);
    candidate.widgets[widget.id] = widget;
    const size = {
      width: canvasWidth < 850 ? 1200 : canvasWidth,
      height: canvasWidth < 850 ? 700 : canvasHeight,
    };
    const suggestion = suggestPlacement(candidate, widget.id, size, focused);
    const destination = findStack(
      workspace.layout,
      suggestion?.widget ?? focused ?? widgets[0]?.id ?? '',
    );
    dispatch(
      [
        {
          type: 'widget.add',
          widget,
          target: destination ?? null,
          position: suggestion?.position ?? 'tab',
        },
      ],
      true,
      size,
    );
    if (!workspace?.widgets[widget.id]) return null;
    chatState(widget.id);
    lastChat[workspace.id] = widget.id;
    activate(widget.id);
    return widget.id;
  }
  function openEvidence(item: Evidence) {
    if (!workspace || !catalog) return;
    const request = item.request;
    const matching = Object.values(workspace.widgets).find((w) => {
      const binding = w.binding ? workspace!.bindings[w.binding] : undefined;
      const ctx = workspace!.groups[w.group ?? ''] ?? w.context;
      if (ctx.from > request.from_date || ctx.to < request.to_date)
        return false;
      return request.algorithm
        ? binding?.kind === 'algorithm' &&
            binding.algorithm === request.algorithm &&
            (!request.run_id || binding.runId === request.run_id)
        : binding?.kind === 'source' &&
            request.ids?.every((id) => binding.ids.includes(id));
    });
    if (matching) {
      activate(matching.id);
      return;
    }
    if (request.algorithm && !request.run_id) {
      notice =
        'This reference describes available runs. Ask AI to inspect a specific run before opening its results.';
      return;
    }
    if (!request.algorithm && !request.ids?.length) {
      notice = item.label;
      return;
    }
    const chat =
      lastChat[workspace.id] ??
      Object.values(workspace.widgets).find((w) => w.type === 'chat')?.id ??
      '';
    const algo = catalog.algorithms.find((a) => a.id === request.algorithm);
    const view: View = {
      version: 1,
      from_date: request.from_date,
      to_date: request.to_date,
      remove: [],
      background_chat: false,
      new_workspace: false,
      name: 'Evidence',
      widgets: [
        {
          key: 'evidence',
          existing: null,
          type: request.tool === 'tables' ? 'table' : 'chart',
          title: item.label.slice(0, 100),
          series: request.algorithm ? [] : (request.ids ?? []),
          algorithm: request.algorithm ?? null,
          run_id: request.run_id ?? null,
          outputs: Object.keys(algo?.outputs ?? {}).slice(0, 2),
          table: request.table ?? '',
          group: null,
          relative_to: chat || null,
          position: 'tab',
          notes: '',
        },
      ],
    };
    try {
      const priorIds = new Set(Object.keys(workspace.widgets));
      const next = compileView(plain(doc!), view, catalog, canvasSize(), chat);
      history.record(plain(doc!), next);
      historyTick++;
      doc = next;
      const added = Object.values(
        next.workspaces.find((w) => w.id === next.active)!.widgets,
      ).find((w) => !priorIds.has(w.id));
      if (added) activate(added.id);
    } catch (error) {
      notice =
        error instanceof Error ? error.message : 'Could not open evidence.';
    }
  }

  async function askAI(id: string, message: string, approval?: string) {
    if (!doc || !workspace || !catalog) return;
    if (!workspace.widgets[id]) {
      const promoted = openAI();
      if (!promoted) return;
      id = promoted;
    }
    const wsId = workspace.id;
    if (running[wsId]) {
      chatState(id).error =
        'Another investigation is running in this workspace. Wait or cancel it first.';
      return;
    }
    const chat = chatState(id);
    const controller = new AbortController();
    controllers.set(id, controller);
    running[wsId] = id;
    try {
      await applyWithRepairs({
        workspaceId: wsId,
        signal: controller.signal,
        current: () =>
          doc
            ? {
                document: plain(doc),
                size:
                  canvasWidth < 850
                    ? { width: 1200, height: 700 }
                    : canvasSize(),
              }
            : null,
        request: ({ workspace, size, retry, feedback }) =>
          investigate({
            chat,
            id,
            workspace,
            browserInstance,
            message,
            approval: retry ? undefined : approval,
            retry,
            feedback,
            size,
            signal: controller.signal,
          }),
        compile: (document, view, size) =>
          compileView(document, view, catalog!, size, id),
        status: (message) => (chat.status = message),
        complete: ({ document: next, before, view, answer }) => {
          if (next && view) {
            history.record(plain(doc!), next);
            historyTick++;
            doc = next;
            chat.states = [...chat.states, before].slice(-3);
            chat.messages.push(
              answer
                ? {
                    ...answer,
                    content:
                      answer.content + '\n\nView updated. Use Go back to undo.',
                  }
                : {
                    role: 'assistant',
                    content: 'View updated. Use Go back to undo.',
                  },
            );
            if (!view.background_chat && view.layout_mode !== 'separate')
              activate(id);
            else {
              compactWidget = chatCompanion(
                doc.workspaces.find((w) => w.id === doc!.active)!,
                id,
              );
              focused = compactWidget;
            }
            lastChat[doc.active] = id;
          } else if (answer) chat.messages.push(answer);
        },
      });
    } catch (err) {
      if (!controller.signal.aborted)
        chat.error =
          err instanceof Error ? err.message : 'Investigation failed.';
    } finally {
      chat.status = controller.signal.aborted
        ? 'Cancelled. Completed results remain available.'
        : '';
      controllers.delete(id);
      delete running[wsId];
    }
  }

  let doc = $state<TerminalDocument | null>(null),
    catalog = $state<Catalog | null>(null),
    health = $state<Health | null>(null);
  let error = $state(''),
    notice = $state(''),
    saving = $state('Loading'),
    recovery = $state(false),
    external = $state(false),
    storedError = $state(''),
    storedRaw = $state('');
  let focused = $state<string | null>(null),
    maximised = $state<string | null>(null),
    dragging = $state<string | null>(null),
    drawer = $state<
      'add' | 'settings' | 'manage' | 'place' | 'existing' | null
    >(null);
  let canvasWidth = $state(1000),
    canvasHeight = $state(600),
    refresh = $state(0),
    historyTick = $state(0),
    cursor = $state<Record<string, string>>({});
  let widgetStatus = $state<Record<string, string>>({});
  let widgetExports = $state<Record<string, () => void>>({});
  let placementNote = $state('');
  let presetSearch = $state('');
  let pending = $state<{ widget: Widget; binding?: Binding } | null>(null),
    target = $state(''),
    placement = $state<Position>('tab'),
    existingAlgorithm = $state(''),
    imported = $state<TerminalDocument | null>(null);
  const history = new History();
  let mounted = $state(false);
  let saveTimer: ReturnType<typeof setTimeout>;
  let compactWidget = $state<string | null>(null);
  let workspace = $derived(
    doc?.workspaces.find((w) => w.id === doc?.active) ?? null,
  );
  let widgets = $derived(Object.values(workspace?.widgets ?? {}));
  let compact = $derived(
    canvasWidth < 850 ||
      (workspace?.layout
        ? !fits(workspace.layout, workspace, {
            width: canvasWidth,
            height: canvasHeight,
          })
        : false),
  );
  let activeId = $derived(
    compact || maximised
      ? maximised && workspace?.widgets[maximised]
        ? maximised
        : compactWidget && workspace?.widgets[compactWidget]
          ? compactWidget
          : focused && workspace?.widgets[focused]
            ? focused
            : widgets[0]?.id
      : null,
  );
  let selectedWidget = $derived(workspace?.widgets[focused ?? ''] ?? null);
  let activeGroup = $derived(selectedWidget?.group ?? 'A');
  let range = $derived(
    workspace?.groups[activeGroup] ??
      selectedWidget?.context ??
      workspace?.groups.A ?? { from: '', to: '', selected: null, zoom: null },
  );
  const canvasSize = () => ({ width: canvasWidth, height: canvasHeight });
  const plain = <T,>(value: T): T => JSON.parse(JSON.stringify(value));
  function mutate(
    fn: (ws: Workspace, document: TerminalDocument) => void,
    record = true,
  ) {
    if (!doc || !workspace) return;
    try {
      const before = plain(doc),
        next = plain(doc),
        ws = next.workspaces.find((w) => w.id === next.active)!;
      fn(ws, next);
      parseDocument(JSON.stringify(next));
      if (record) {
        history.record(before, next);
        historyTick++;
      }
      doc = next;
      notice = '';
    } catch (e) {
      notice = e instanceof Error ? e.message : 'Workspace action failed';
    }
  }
  function dispatch(
    actions: WorkspaceAction[],
    record = true,
    size = canvasSize(),
  ) {
    if (!doc) return;
    try {
      const before = plain(doc),
        next = applyActions(before, actions, size);
      if (record) {
        history.record(before, next);
        historyTick++;
      }
      doc = next;
      notice = '';
    } catch (e) {
      notice = (e as Error).message;
    }
  }
  function changeWorkspace(next: Workspace) {
    mutate((_ws, d) => {
      d.workspaces[d.workspaces.findIndex((w) => w.id === d.active)] = next;
    });
  }
  function focus(id: string) {
    focused = id;
    compactWidget = id;
  }
  function activate(id: string) {
    if (!workspace) return;
    if (workspace.widgets[id]?.type === 'chat') lastChat[workspace.id] = id;
    if (maximised) maximised = id;
    if (compact) {
      focus(id);
      return;
    }
    const next = reveal(plain(workspace), id);
    mutate((ws) => {
      ws.layout = next.layout;
    }, false);
    focus(id);
  }
  function close(id: string) {
    cancelChat(id);
    dispatch([{ type: 'widget.close', id }]);
    if (focused === id) focused = null;
    if (maximised === id) maximised = null;
  }
  function settings(id: string) {
    focus(id);
    drawer = 'settings';
  }
  function widgetContext(w: Widget): LinkContext {
    const base = workspace?.groups[w.group ?? ''] ?? w.context;
    return { ...base, selected: cursor[w.group ?? w.id] ?? base.selected };
  }
  function select(w: Widget, t: string, pin = false) {
    const key = w.group ?? w.id;
    cursor[key] = t;
    if (pin)
      mutate((ws) => {
        const base = w.group ? ws.groups[w.group] : ws.widgets[w.id].context;
        if (base) {
          const day = dateOf(t);
          if (day < base.from || day > base.to) {
            base.from = day;
            base.to = day;
            base.zoom = null;
          }
          base.selected = t;
        }
      }, false);
  }
  function pin(w: Widget, runId: number) {
    if (!w.binding) return;
    mutate((ws) => {
      const b = ws.bindings[w.binding!];
      if (b?.kind === 'algorithm' && b.runId === null) b.runId = runId;
    }, false);
  }
  function zoom(w: Widget, z: [number, number] | null) {
    mutate((ws) => {
      const c = w.group ? ws.groups[w.group] : ws.widgets[w.id].context;
      c.zoom = z;
    });
  }
  function updateWidget(w: Widget) {
    dispatch([
      {
        type: 'widget.configure',
        id: w.id,
        title: w.title,
        settings: plain(w.settings),
      },
    ]);
  }
  function rebind(w: Widget, b: NewBinding | null) {
    mutate((ws) => {
      const previous = w.binding ? ws.bindings[w.binding] : undefined;
      if (
        b &&
        previous &&
        b.kind === 'algorithm' &&
        previous.kind === 'algorithm' &&
        b.algorithm === previous.algorithm
      ) {
        ws.bindings[previous.id] = { ...b, id: previous.id };
        ws.widgets[w.id].settings.outputs = [];
        return;
      }
      const id = uid();
      if (b) ws.bindings[id] = { ...b, id } as Binding;
      ws.widgets[w.id].binding = b ? id : null;
      ws.widgets[w.id].settings.outputs = [];
      ws.widgets[w.id].settings.table = '';
    });
  }
  function link(w: Widget, group: string | null) {
    mutate((ws) => {
      if (group === '__new') {
        const ids = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
        const id = ids.find((id) => !ws.groups[id]);
        if (!id || Object.keys(ws.groups).length >= 12)
          throw new Error('Maximum 12 linked groups.');
        const colors = ['#e6b080', '#a2c7b8', '#b8afd0', '#9cb8cf', '#d2a1a1'];
        ws.groups[id] = {
          ...context(ws.groups[w.group ?? ''] ?? w.context),
          id,
          name: `Group ${id}`,
          color: colors[Object.keys(ws.groups).length % colors.length],
        };
        group = id;
      }
      if (!group)
        ws.widgets[w.id].context = plain(ws.groups[w.group ?? ''] ?? w.context);
      ws.widgets[w.id].group = group;
    });
    cursor = {};
  }
  function setRange(key: 'from' | 'to', value: string) {
    if (!value) return;
    mutate((ws) => {
      const c =
        ws.groups[activeGroup] ??
        (selectedWidget ? ws.widgets[selectedWidget.id].context : ws.groups.A);
      const next = { ...c, [key]: value };
      if (next.from > next.to)
        throw new Error('The start date must not follow the end date.');
      if (
        (Date.parse(next.to) - Date.parse(next.from)) / 86400000 + 1 >
        (catalog?.limits.max_range_days ?? 3660)
      )
        throw new Error('Range exceeds the catalogue limit.');
      c[key] = value;
      c.zoom = null;
      c.selected = null;
    });
    cursor = {};
  }
  function newWorkspace(
    kind: 'market' | 'blank' | 'analysis',
    algorithm?: string,
  ) {
    mutate((_ws, d) => {
      if (d.workspaces.length >= LIMITS.workspaces)
        throw new Error('Maximum 16 workspaces. Export or close one first.');
      const next = preset({ from: range.from, to: range.to }, kind, algorithm);
      d.workspaces.push(next);
      d.active = next.id;
    });
    drawer = null;
    maximised = null;
    focused = null;
  }
  function loadPreset(id: string) {
    mutate((_ws, d) => {
      if (d.workspaces.length >= LIMITS.workspaces)
        throw new Error('Maximum 16 workspaces.');
      const saved = d.presets?.find((p) => p.id === id);
      if (!saved) return;
      const next = duplicateWorkspace(saved);
      next.name = saved.name;
      d.workspaces.push(next);
      d.active = next.id;
    });
    drawer = null;
    focused = null;
    maximised = null;
  }
  function add(type: WidgetType, source?: string, algorithm?: string) {
    if (!workspace) return;
    if (widgets.length >= LIMITS.widgets) {
      notice =
        'Maximum 12 widgets in a workspace. Create another workspace tab.';
      return;
    }
    const id = uid();
    const b: Binding | undefined = algorithm
      ? { id, kind: 'algorithm', algorithm, runId: null }
      : source
        ? { id, kind: 'source', ids: [source] }
        : type === 'chart'
          ? { id, kind: 'source', ids: [catalog?.series[0]?.id ?? 'price'] }
          : undefined;
    const widget = createWidget(type, range, b?.id ?? null);
    if (source)
      widget.title =
        catalog?.series.find((s) => s.id === source)?.label ?? source;
    if (algorithm)
      widget.title =
        algorithm.replaceAll('_', ' ') + (type === 'table' ? ' · table' : '');
    pending = { widget, binding: b };
    target = focused ?? widgets[0]?.id ?? '';
    placement = 'tab';
    const candidate = plain(workspace);
    candidate.widgets[widget.id] = widget;
    const suggestion = suggestPlacement(
      candidate,
      widget.id,
      canvasSize(),
      focused,
    );
    placementNote = workspace.locked
      ? 'Unlock the layout or create a new workspace.'
      : suggestion
        ? 'Suggested placement uses available space. Dividers adjust in snapped steps while keeping every tile readable.'
        : 'No room for another visible tile. Stack it as a tab or create a new workspace.';
    if (suggestion) {
      target = suggestion.widget;
      placement = suggestion.position;
    }
    if (suggestion) {
      finishAdd();
      return;
    }
    if (!workspace.layout) {
      finishAdd();
      return;
    }
    drawer = 'place';
  }
  function canPlace(id: string, destination: string, pos: Position) {
    if (!workspace) return false;
    try {
      place(plain(workspace), id, destination, pos, canvasSize());
      return true;
    } catch {
      return false;
    }
  }
  function canResize(id: string, step: number) {
    if (!workspace || step < 1 || step > 23) return false;
    try {
      resize(plain(workspace), id, step, canvasSize());
      return true;
    } catch {
      return false;
    }
  }
  function move(
    id: string,
    destination: string,
    pos: Position,
    before?: string,
  ) {
    dispatch([
      { type: 'widget.move', id, target: destination, position: pos, before },
    ]);
  }
  function finishAdd(newTab = false) {
    if (!pending || !workspace) return;
    const addition = plain(pending);
    let success = false;
    mutate((ws, d) => {
      if (ws.locked && !newTab)
        throw new Error('Unlock the layout before adding widgets.');
      if (newTab) {
        if (d.workspaces.length >= LIMITS.workspaces)
          throw new Error('Maximum 16 workspaces.');
        const next = emptyWorkspace(range, addition.widget.title);
        addition.widget.group = 'A';
        next.widgets[addition.widget.id] = addition.widget;
        if (addition.binding)
          next.bindings[addition.binding.id] = addition.binding;
        next.layout = stack(addition.widget.id);
        d.workspaces.push(next);
        d.active = next.id;
      } else {
        if (Object.keys(ws.widgets).length >= LIMITS.widgets)
          throw new Error('Maximum 12 widgets.');
        ws.widgets[addition.widget.id] = addition.widget;
        if (addition.binding)
          ws.bindings[addition.binding.id] = addition.binding;
        const destination = target ? findStack(ws.layout, target) : null;
        const placed = place(
          ws,
          addition.widget.id,
          destination,
          placement,
          canvasSize(),
        );
        ws.layout = placed.layout;
      }
      success = true;
    });
    if (success) {
      focus(addition.widget.id);
      pending = null;
      drawer = null;
    }
  }
  let placementPossible = $derived.by(() => {
    if (!pending || !workspace) return false;
    try {
      const ws = plain(workspace);
      ws.widgets[pending.widget.id] = plain(pending.widget);
      if (pending.binding)
        ws.bindings[pending.binding.id] = plain(pending.binding);
      place(
        ws,
        pending.widget.id,
        findStack(ws.layout, target),
        placement,
        canvasSize(),
      );
      return true;
    } catch {
      return false;
    }
  });
  function duplicateWidget(w: Widget) {
    if (!workspace) return;
    const copy = plain(w);
    copy.id = uid();
    copy.title += ' copy';
    const old = w.binding ? workspace.bindings[w.binding] : undefined;
    const binding = old ? { ...plain(old), id: uid() } : undefined;
    copy.binding = binding?.id ?? null;
    pending = { widget: copy, binding };
    placementNote =
      'Choose a tab, a legal split or a new workspace for the copy.';
    target = w.id;
    placement = 'tab';
    drawer = 'place';
  }
  function algorithm(
    id: string,
    action: 'chart' | 'table' | 'existing' | 'layout',
  ) {
    if (action === 'layout') {
      newWorkspace('analysis', id);
      return;
    }
    if (action === 'existing') {
      existingAlgorithm = id;
      target = widgets.find((w) => w.type === 'chart')?.id ?? '';
      drawer = 'existing';
      return;
    }
    add(action === 'table' ? 'table' : 'chart', undefined, id);
  }
  function addExisting() {
    const w = workspace?.widgets[target];
    if (!w || w.type !== 'chart') return;
    rebind(w, {
      kind: 'algorithm',
      algorithm: existingAlgorithm,
      runId: null,
      sources:
        w.binding && workspace?.bindings[w.binding]?.kind === 'source'
          ? (
              workspace.bindings[w.binding] as Extract<
                Binding,
                { kind: 'source' }
              >
            ).ids
          : [],
    } as NewBinding);
    drawer = null;
  }
  function undo(redo = false) {
    if (!doc) return;
    doc = redo ? history.redo(plain(doc)) : history.undo(plain(doc));
    historyTick++;
    focused = null;
    maximised = null;
    cursor = {};
  }
  function exportDoc(value: TerminalDocument | null = doc) {
    if (!value) return;
    const blob = new Blob(
      [JSON.stringify(persistentDocument(plain(value)), null, 2)],
      {
        type: 'application/json',
      },
    );
    const href = URL.createObjectURL(blob),
      a = document.createElement('a');
    a.href = href;
    a.download = 'eisengrau-terminal.json';
    a.click();
    setTimeout(() => URL.revokeObjectURL(href), 1000);
  }
  async function importFile(event: Event) {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (!file) return;
    try {
      if (file.size > LIMITS.bytes)
        throw new Error('Workspace import exceeds 1 MiB.');
      imported = parseDocument(await file.text());
      notice =
        'Import validated. Review and replace below; current data is unchanged.';
    } catch (e) {
      notice = (e as Error).message;
    }
    (event.target as HTMLInputElement).value = '';
  }
  function applyImport() {
    if (!doc || !imported) return;
    history.record(plain(doc), plain(imported));
    doc = imported;
    historyTick++;
    imported = null;
    recovery = false;
    focused = null;
    maximised = null;
    notice = 'Imported workspace configuration.';
  }
  function save() {
    if (!doc || recovery || external) return;
    try {
      const raw = JSON.stringify(persistentDocument(plain(doc))),
        previous = localStorage.getItem(STORAGE_KEY);
      if (previous) {
        try {
          parseDocument(previous);
          localStorage.setItem(STORAGE_KEY + ':backup', previous);
        } catch {
          /* Keep the valid backup. */
        }
      }
      localStorage.setItem(STORAGE_KEY, raw);
      sessionStorage.setItem(
        SESSION_LAYOUT,
        JSON.stringify({ baseline: raw, document: doc }),
      );
      saving = 'Saved locally';
    } catch {
      saving = 'Not saved · browser storage unavailable';
    }
  }
  $effect(() => {
    if (!doc) return;
    JSON.stringify(doc);
    if (mounted) {
      saving = recovery
        ? 'Recovery required'
        : external
          ? 'Changed in another tab'
          : 'Saving…';
      clearTimeout(saveTimer);
      saveTimer = setTimeout(() => untrack(save), 250);
    }
  });
  async function initialize() {
    error = '';
    try {
      const result = await bootstrap();
      catalog = result.catalog;
      health = result.health;
      if (!doc) {
        let date = result.date;
        try {
          const old = JSON.parse(
            localStorage.getItem('gridpulse:view:v1') ?? 'null',
          );
          if (/^\d{4}-\d{2}-\d{2}$/.test(old?.date)) date = old.date;
        } catch {
          /* Optional Overview preference. */
        }
        doc = createDocument({ from: date, to: date });
      }
    } catch (e) {
      error = (e as Error).message;
    }
  }
  onMount(() => {
    mounted = true;
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      try {
        const saved = sessionStorage.getItem(SESSION_KEY);
        if (saved && saved.length <= 600000) {
          const value = JSON.parse(saved);
          if (
            value.version === 1 &&
            value.chats &&
            typeof value.chats === 'object'
          ) {
            for (const [id, candidate] of Object.entries(value.chats).slice(
              0,
              12,
            )) {
              const c = candidate as Conversation;
              if (
                Array.isArray(c.messages) &&
                typeof c.summary === 'string' &&
                Array.isArray(c.states) &&
                c.states.length <= 3 &&
                c.messages.length <= 42 &&
                c.messages.every(
                  (m) =>
                    m &&
                    ['user', 'assistant'].includes(m.role) &&
                    typeof m.content === 'string' &&
                    m.content.length <= 12000,
                )
              ) {
                c.status = '';
                c.error = '';
                c.confirmation = null;
                chats[id] = c;
              }
            }
          }
        }
        const savedMeta = JSON.parse(
          sessionStorage.getItem(SESSION_KEY) ?? 'null',
        );
        if (savedMeta?.lastChat && typeof savedMeta.lastChat === 'object') {
          for (const [key, id] of Object.entries(savedMeta.lastChat).slice(
            0,
            16,
          ))
            if (typeof id === 'string' && chats[id]) lastChat[key] = id;
        }
        const session = JSON.parse(
          sessionStorage.getItem(SESSION_LAYOUT) ?? 'null',
        );
        if (session?.baseline === raw)
          doc = parseDocument(JSON.stringify(session.document));
      } catch {
        /* Session recovery must never prevent ordinary workspace loading. */
      }
      if (raw) {
        try {
          if (!doc) doc = parseDocument(raw);
        } catch (e) {
          storedError = (e as Error).message;
          storedRaw = raw;
          recovery = true;
        }
      }
    } catch {
      saving = 'Browser storage unavailable';
    }
    void initialize();
    const storage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY && e.newValue !== JSON.stringify(doc))
        external = true;
    };
    const key = (e: KeyboardEvent) => {
      if (
        e.defaultPrevented ||
        e.isComposing ||
        document.querySelector('dialog[open]')
      )
        return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        drawer = 'add';
      } else if (e.key === 'Escape') {
        if (drawer) drawer = null;
        else if (dragging) dragging = null;
        else maximised = null;
      } else if (
        (e.ctrlKey || e.metaKey) &&
        e.key.toLowerCase() === 'z' &&
        !(
          e.target instanceof HTMLInputElement ||
          e.target instanceof HTMLTextAreaElement ||
          e.target instanceof HTMLSelectElement ||
          (e.target instanceof HTMLElement && e.target.isContentEditable)
        )
      ) {
        e.preventDefault();
        undo(e.shiftKey);
      }
    };
    const flush = () => save();
    window.addEventListener('storage', storage);
    window.addEventListener('keydown', key);
    window.addEventListener('pagehide', flush);
    return () => {
      clearTimeout(saveTimer);
      save();
      window.removeEventListener('storage', storage);
      window.removeEventListener('keydown', key);
      window.removeEventListener('pagehide', flush);
      for (const controller of controllers.values()) controller.abort();
      mounted = false;
    };
  });

  $effect(() => {
    if (!mounted) return;
    const raw = JSON.stringify({ version: 1, chats, lastChat });
    try {
      if (raw.length <= 600000) sessionStorage.setItem(SESSION_KEY, raw);
    } catch {
      /* Chat remains usable without storage. */
    }
  });
  $effect(() => {
    const ws = workspace;
    if (ws)
      untrack(() => {
        const ids = Object.values(ws.widgets)
          .filter((w) => w.type === 'chat')
          .map((w) => w.id);
        if (!Object.keys(ws.widgets).length) ids.push(`${ws.id}-chat`);
        for (const id of ids) chatState(id);
      });
  });
  $effect(() => {
    const active = doc?.active;
    for (const [workspaceId, chatId] of Object.entries(running))
      if (workspaceId !== active) cancelChat(chatId);
  });
</script>

<main class="terminal-page" aria-label="Eisengrau Terminal">
  {#if error}<div class="terminal-alert" role="alert">
      {error}<button onclick={initialize}>Retry connection</button>
    </div>{/if}
  {#if recovery}<div class="terminal-alert" role="alert">
      Saved workspace needs recovery: {storedError}<button
        onclick={() => {
          try {
            const raw = localStorage.getItem(STORAGE_KEY + ':backup');
            if (!raw) throw new Error('No valid backup found');
            if (!doc) doc = parseDocument(raw);
            recovery = false;
            storedError = '';
          } catch (e) {
            notice = (e as Error).message;
          }
        }}>Restore backup</button
      ><button
        onclick={() => {
          recovery = false;
          storedError = '';
        }}>Use fresh workspace</button
      ><button
        onclick={() => {
          const href = URL.createObjectURL(
            new Blob([storedRaw], { type: 'application/json' }),
          );
          const a = document.createElement('a');
          a.href = href;
          a.download = 'eisengrau-terminal-recovery.json';
          a.click();
          setTimeout(() => URL.revokeObjectURL(href), 1000);
        }}>Download saved file</button
      >
    </div>{/if}
  {#if external}<div class="terminal-alert" role="status">
      Another browser tab changed this workspace.<button
        onclick={() => {
          try {
            doc = parseDocument(localStorage.getItem(STORAGE_KEY) ?? '');
            external = false;
          } catch (e) {
            notice = (e as Error).message;
          }
        }}>Load other tab</button
      ><button
        onclick={() => {
          external = false;
          save();
        }}>Keep this tab</button
      >
    </div>{/if}
  {#if notice}<div class="terminal-notice" role="status">
      {notice}<button aria-label="Dismiss notice" onclick={() => (notice = '')}
        >×</button
      >
    </div>{/if}
  {#if workspace && doc && catalog}
    <div class="terminal-toolbar">
      <div
        class="terminal-workspace-tabs"
        role="tablist"
        aria-label="Workspaces"
      >
        {#each doc.workspaces as ws}<button
            role="tab"
            aria-selected={ws.id === doc.active}
            onclick={() => {
              save();
              mutate((_ws, d) => (d.active = ws.id), false);
              focused = null;
              maximised = null;
              cursor = {};
            }}>{ws.name}</button
          >{/each}<button
          aria-label="New blank workspace"
          onclick={() => newWorkspace('blank')}>＋</button
        >
      </div>
      <div class="terminal-toolbar-controls">
        <div class="terminal-range">
          <span>{workspace.groups[activeGroup]?.name ?? 'Independent'}</span
          ><input
            type="date"
            aria-label="Terminal start date"
            value={range.from}
            onchange={(e) => setRange('from', e.currentTarget.value)}
          /><span>→</span><input
            type="date"
            aria-label="Terminal end date"
            value={range.to}
            onchange={(e) => setRange('to', e.currentTarget.value)}
          />
        </div>
        <button onclick={() => openAI()}>✦ AI</button>
        <button class="terminal-primary" onclick={() => (drawer = 'add')}
          >＋ Search / Add <kbd>⌘ K</kbd></button
        >
        <button onclick={() => (drawer = 'manage')}>Workspaces</button>
      </div>
    </div>
    <div class="terminal-widget-bar" aria-label="Workspace widgets">
      {#each widgets as w}<button
          class:active={focused === w.id}
          onclick={() => activate(w.id)}
          ondblclick={() => settings(w.id)}
          ><i
            class="terminal-group-dot"
            style={`background:${workspace.groups[w.group ?? '']?.color ?? '#888'}`}
          ></i><strong>{w.title}</strong><small
            >{w.binding
              ? workspace.bindings[w.binding]?.kind === 'source'
                ? (
                    workspace.bindings[w.binding] as Extract<
                      Binding,
                      { kind: 'source' }
                    >
                  ).ids.join(' + ')
                : `run ${(workspace.bindings[w.binding] as Extract<Binding, { kind: 'algorithm' }>)?.runId ?? 'latest'}`
              : w.type} · {w.group ?? 'independent'}{widgetStatus[w.id]
              ? ' · ' + widgetStatus[w.id]
              : ''}</small
          ></button
        >{/each}
      <div class="terminal-history" data-history={historyTick}>
        <button
          aria-label="Undo workspace change"
          disabled={!history.canUndo}
          onclick={() => undo()}>↶</button
        ><button
          aria-label="Redo workspace change"
          disabled={!history.canRedo}
          onclick={() => undo(true)}>↷</button
        ><button
          aria-pressed={workspace.locked}
          onclick={() => mutate((ws) => (ws.locked = !ws.locked))}
          >{workspace.locked ? '🔒 Locked' : 'Layout unlocked'}</button
        >{#if selectedWidget}<button onclick={() => (drawer = 'settings')}
            >Widget settings</button
          >{/if}
      </div>
    </div>
    <div
      class="terminal-canvas"
      bind:clientWidth={canvasWidth}
      bind:clientHeight={canvasHeight}
    >
      {#if !widgets.length}<div
          class="terminal-empty-workspace terminal-empty-ai"
        >
          {@render renderChat(`${workspace.id}-chat`)}
          <button onclick={() => (drawer = 'add')}
            >Or add widgets manually</button
          >
        </div>
      {:else if activeId}
        {@const w = workspace.widgets[activeId]}
        <section class="terminal-tile focused">
          <header class="terminal-tile-header">
            <strong>{w.title}</strong><span class="terminal-hint"
              >{compact
                ? 'Compact view · desktop layout preserved'
                : 'Maximised'}</span
            ><GroupPicker
              groups={workspace.groups}
              group={w.group}
              title={w.title}
              onchange={(group) => link(w, group)}
              onfocus={() => focus(w.id)}
            /><button onclick={() => settings(w.id)}>Settings</button
            >{#if maximised}<button onclick={() => (maximised = null)}
                >Restore layout</button
              >{/if}
          </header>
          <div class="terminal-widget-content">
            {@render renderWidget(w.id)}
          </div>
        </section>
      {:else if workspace.layout}<DockNode
          node={workspace.layout}
          {workspace}
          {focused}
          {dragging}
          {renderWidget}
          onfocus={focus}
          onactivate={activate}
          ondrag={(id) => (dragging = id)}
          onmove={move}
          onresize={(id, step) => {
            try {
              changeWorkspace(
                resize(plain(workspace!), id, step, canvasSize()),
              );
            } catch (e) {
              notice = (e as Error).message;
            }
          }}
          onsettings={settings}
          onlink={(id, group) => {
            const widget = workspace?.widgets[id];
            if (widget) link(widget, group);
          }}
          onmaximise={(id) => (maximised = id)}
          onclose={close}
          {canPlace}
          {canResize}
        />{/if}
    </div>
    <div class="terminal-status">
      <span class:warning={health?.mode !== 'live'}
        >{health?.mode === 'live'
          ? '● LIVE SOURCE DATA'
          : `● ${health?.mode?.toUpperCase() ?? 'UNKNOWN'} · CACHED DATA`} · {saving}</span
      ><span
        >Data loaded {catalog.data_updated_at_utc ?? 'unknown'} · {compact
          ? 'Compact navigation'
          : 'Snapping tiles'} · Hover to inspect, drag chart to zoom</span
      >
    </div>
  {:else if !error}<div class="terminal-empty" role="status">
      Loading catalogue and workspace…
    </div>{/if}
</main>

{#snippet renderWidget(id: string)}
  {#if workspace && catalog && workspace.widgets[id]}
    {@const w = workspace.widgets[id]}{@const ctx = widgetContext(w)}
    <svelte:boundary
      onerror={(error) => {
        if (dev) console.error('Terminal renderer failed', error);
      }}
    >
      {#if w.type === 'chat'}{@render renderChat(w.id)}{:else}<WidgetBody
          widget={w}
          binding={w.binding ? workspace.bindings[w.binding] : undefined}
          {ctx}
          {catalog}
          onselect={(t) => select(w, t)}
          onpin={(t) => select(w, t, true)}
          inspection={(workspace.groups[w.group ?? ''] ?? w.context).selected}
          onzoom={(z) => zoom(w, z)}
          onrun={(runId) => pin(w, runId)}
          onnotes={(notes) =>
            updateWidget({
              ...plain(w),
              settings: { ...plain(w.settings), notes },
            })}
          {refresh}
          onexport={(exportCsv) => {
            if (exportCsv) widgetExports[id] = exportCsv;
            else delete widgetExports[id];
          }}
          onstatus={(status) => (widgetStatus[id] = status)}
        />{/if}{#snippet failed(_error, reset)}<div
          class="terminal-widget-error"
        >
          <p>
            This widget could not render. Open settings to rebind or remove it.
          </p>
          <button onclick={reset}>Retry rendering</button>
        </div>{/snippet}
    </svelte:boundary>
  {/if}
{/snippet}

{#snippet renderChat(id: string)}
  {@const chat = chats[id] ?? blankChat}
  <Chat
    {chat}
    busy={!!running[workspace?.id ?? '']}
    onsubmit={(message, approval) => askAI(id, message, approval)}
    oncancel={() => cancelChat(running[workspace?.id ?? ''] ?? id)}
    onundo={() => undo()}
    canUndo={historyTick >= 0 && history.canUndo}
    onnew={() => openAI(true)}
    onreference={openEvidence}
    onclear={() => {
      delete chats[id];
      chats[id] = conversation();
    }}
  />
{/snippet}

{#if drawer === 'add' && catalog}<Drawer
    title="Search / Add"
    onclose={() => (drawer = null)}
    ><Catalogue {catalog} onadd={add} onalgorithm={algorithm} /></Drawer
  >
{:else if drawer === 'settings' && selectedWidget && workspace && catalog}<Drawer
    title="Widget settings"
    onclose={() => (drawer = null)}
    ><Settings
      widget={selectedWidget}
      exportCsv={widgetExports[selectedWidget.id]}
      {workspace}
      {catalog}
      onchange={updateWidget}
      onbinding={(binding) => rebind(selectedWidget!, binding)}
      onrun={(id) => {
        if (selectedWidget?.binding)
          mutate((ws) => {
            const b = ws.bindings[selectedWidget!.binding!];
            if (b.kind === 'algorithm') b.runId = id;
          });
      }}
      onlink={(group) => link(selectedWidget!, group)}
      onmove={(id, pos) => {
        const destination = findStack(workspace!.layout, id);
        if (destination)
          move(
            selectedWidget!.id,
            destination,
            pos,
            pos === 'tab' ? id : undefined,
          );
      }}
      onduplicate={() => duplicateWidget(selectedWidget!)}
      onclose={() => {
        close(selectedWidget!.id);
        drawer = null;
      }}
    /></Drawer
  >
{:else if drawer === 'place' && pending && workspace}<Drawer
    title="Place widget"
    onclose={() => {
      drawer = null;
      pending = null;
    }}
    ><p class="terminal-hint">
      {pending.widget.title}. Tile sizes snap to discrete steps; existing
      widgets keep readable minimums.
    </p>
    <p class="terminal-hint">{placementNote}</p>
    <label
      >Destination<select aria-label="Placement destination" bind:value={target}
        >{#each widgets as w}<option value={w.id}>{w.title}</option
          >{/each}</select
      ></label
    ><label
      >Placement<select aria-label="Placement position" bind:value={placement}
        >{#each ['tab', 'left', 'right', 'top', 'bottom'] as pos}<option
            value={pos}>{pos === 'tab' ? 'Stack as tab' : `Dock ${pos}`}</option
          >{/each}</select
      ></label
    >{#if !placementPossible}<p class="terminal-alert">
        Workspace has no room for this placement at minimum readable sizes.
        Choose a tab or create a new workspace.
      </p>{/if}
    <div class="terminal-action-row">
      <button
        class="terminal-primary"
        disabled={!placementPossible}
        onclick={() => finishAdd()}>Place widget</button
      ><button onclick={() => finishAdd(true)}>Create new workspace tab</button>
    </div></Drawer
  >
{:else if drawer === 'existing' && workspace}<Drawer
    title="Add algorithm to chart"
    onclose={() => (drawer = null)}
    ><p class="terminal-hint">
      Use {existingAlgorithm} in a chart. This replaces its previous algorithm binding,
      if any.
    </p>
    <label
      >Destination chart<select aria-label="Existing chart" bind:value={target}
        ><option value="">Choose chart…</option
        >{#each widgets.filter((w) => w.type === 'chart') as w}<option
            value={w.id}>{w.title}</option
          >{/each}</select
      ></label
    ><button disabled={!target} onclick={addExisting}>Apply algorithm</button
    ><button onclick={() => add('chart', undefined, existingAlgorithm)}
      >Open new chart instead</button
    ></Drawer
  >
{:else if drawer === 'manage' && doc && workspace}<Drawer
    title="Workspaces"
    onclose={() => (drawer = null)}
  >
    <label
      >Workspace name<input
        aria-label="Workspace name"
        value={workspace.name}
        maxlength="80"
        onchange={(e) =>
          mutate(
            (ws) => (ws.name = e.currentTarget.value || 'Untitled workspace'),
          )}
      /></label
    >
    <div class="terminal-action-row">
      <button onclick={() => newWorkspace('market')}>Market preset</button
      ><button onclick={() => newWorkspace('blank')}>Blank workspace</button
      ><button
        onclick={() => {
          mutate((ws, d) => {
            if (d.workspaces.length >= LIMITS.workspaces)
              throw new Error('Maximum 16 workspaces.');
            const copy = duplicateWorkspace(ws);
            d.workspaces.push(copy);
            d.active = copy.id;
          });
        }}>Duplicate workspace</button
      >
    </div>
    <button
      onclick={() => {
        pool.invalidate();
        refresh++;
        void initialize();
        notice = 'Refreshing data. Pinned runs remain pinned.';
      }}>Refresh data</button
    >
    <button
      onclick={() => {
        mutate((ws, d) => {
          if ((d.presets?.length ?? 0) >= LIMITS.workspaces)
            throw new Error('Maximum 16 saved presets.');
          const saved = duplicateWorkspace(
            persistentDocument(plain(d)).workspaces.find(
              (w) => w.id === ws.id,
            )!,
          );
          saved.name = ws.name;
          let suffix = 2;
          while (d.presets?.some((p) => p.name === saved.name))
            saved.name = `${ws.name} (${suffix++})`;
          d.presets = [...(d.presets ?? []), saved];
        });
        notice = 'Saved as a reusable preset in Workspaces.';
      }}>Save current view as preset</button
    >
    <h3>Saved presets</h3>
    <input
      aria-label="Search saved presets"
      placeholder="Search saved presets…"
      bind:value={presetSearch}
    />
    {#each (doc.presets ?? []).filter((p) => p.name
        .toLowerCase()
        .includes(presetSearch.toLowerCase())) as saved}<div
        class="terminal-action-row"
      >
        <button
          class="terminal-load-preset"
          onclick={() => loadPreset(saved.id)}>Load preset {saved.name}</button
        ><button
          aria-label={`Delete preset ${saved.name}`}
          onclick={() =>
            mutate(
              (_ws, d) =>
                (d.presets = d.presets?.filter((p) => p.id !== saved.id)),
            )}>Delete</button
        >
      </div>{/each}
    {#if !(doc.presets ?? []).some((p) => p.name
        .toLowerCase()
        .includes(presetSearch.toLowerCase()))}<p class="terminal-hint">
        {doc.presets?.length
          ? 'No matching saved presets.'
          : 'Save the current view as a preset to reuse it later.'}
      </p>{/if}
    <button onclick={() => exportDoc()}>Export configuration</button>
    <label
      >Import configuration<input
        type="file"
        accept="application/json,.json"
        aria-label="Import configuration"
        onchange={importFile}
      /></label
    >
    {#if imported}<div class="terminal-import-preview">
        <h3>Import preview</h3>
        <p>
          {imported.workspaces.length} workspaces: {imported.workspaces
            .map((w) => w.name)
            .join(', ')}
        </p>
        <p>Replace the current configuration? This can be undone.</p>
        <button onclick={applyImport}
          >Replace with imported configuration</button
        ><button onclick={() => (imported = null)}>Cancel import</button>
      </div>{/if}
    <div class="terminal-workspace-actions">
      <p class="terminal-hint">These actions can be undone.</p>
      <button
        onclick={() => {
          mutate((ws, d) => {
            const next = preset(range);
            d.workspaces[d.workspaces.indexOf(ws)] = next;
            d.active = next.id;
          });
          drawer = null;
        }}>Reset to market preset</button
      ><button
        class="danger"
        onclick={() => {
          mutate((ws, d) => {
            d.workspaces = d.workspaces.filter((w) => w.id !== ws.id);
            if (!d.workspaces.length) d.workspaces = [emptyWorkspace(range)];
            d.active = d.workspaces[0].id;
          });
          drawer = null;
        }}>Close current workspace</button
      >
    </div>
  </Drawer>{/if}
