<script lang="ts">
  const groups = [
    {
      name: 'Workspace',
      note: 'Use while Terminal is open. Undo and redo leave text editing to the focused field.',
      rows: [
        ['Ctrl / ⌘ + K', 'Open Search / Add'],
        ['Ctrl / ⌘ + Z', 'Undo a workspace change'],
        ['Ctrl / ⌘ + Shift + Z', 'Redo a workspace change'],
        [
          'Escape',
          'Close the active drawer, cancel a drag or leave a maximised widget',
        ],
      ],
    },
    {
      name: 'Charts',
      note: 'Tab to a chart’s time-selection control first.',
      rows: [
        ['← / →', 'Inspect and pin the previous or next visible point'],
        ['Home / End', 'Inspect and pin the first or last visible point'],
      ],
    },
    {
      name: 'Layout',
      note: 'Focus a divider. Resizing respects minimum tile sizes and workspace lock.',
      rows: [
        ['← / ↑', 'Move the divider back one snap step'],
        ['→ / ↓', 'Move the divider forward one snap step'],
      ],
    },
    {
      name: 'Linked groups',
      note: 'Focus the letter badge at the top right of a widget.',
      rows: [
        ['↓', 'Open the group picker'],
        ['↑ / ↓', 'Move between options in an open picker'],
        ['Home / End', 'Jump to the first or last group option'],
        ['Enter / Space', 'Choose the focused option'],
        ['Escape', 'Close the group picker'],
      ],
    },
  ];
</script>

<div class="shortcut-reference">
  {#each groups as group}
    <div class="shortcut-group">
      <table>
        <caption>{group.name}</caption>
        <thead class="sr-only"
          ><tr><th scope="col">Keys</th><th scope="col">Action</th></tr></thead
        >
        <tbody
          >{#each group.rows as [keys, action]}<tr
              ><th scope="row"><kbd>{keys}</kbd></th><td>{action}</td></tr
            >{/each}</tbody
        >
      </table>
      <p>{group.note}</p>
    </div>
  {/each}
</div>

<style>
  .shortcut-reference {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 280px), 1fr));
    gap: 24px;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
    font-size: 12px;
    line-height: 1.7;
  }
  caption {
    text-align: left;
    color: var(--orange);
    font:
      10px/1.7 'DM Mono',
      monospace;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding-bottom: 10px;
  }
  th,
  td {
    text-align: left;
    vertical-align: top;
    padding: 10px 0;
    border-top: 1px solid var(--border);
  }
  tbody th {
    width: 44%;
    padding-right: 16px;
    font-weight: 400;
  }
  td {
    color: var(--muted-foreground);
  }
  kbd {
    color: var(--foreground);
    font:
      10px/1.9 'DM Mono',
      monospace;
    background: var(--secondary);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 3px 5px;
    box-decoration-break: clone;
    -webkit-box-decoration-break: clone;
  }
  .shortcut-group > p {
    color: var(--muted-foreground);
    font-size: 11px;
    line-height: 1.8;
    margin-top: 10px;
  }
</style>
