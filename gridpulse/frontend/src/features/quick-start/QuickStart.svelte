<script lang="ts">
  import capture from './capture.json';
  import TerminalShortcuts from '../../components/help/TerminalShortcuts.svelte';
  const steps = [
    { id: 'start', number: '01', label: 'Start with the data' },
    { id: 'analysis', number: '02', label: 'Add an algorithm' },
    { id: 'workspace', number: '03', label: 'Make the view yours' },
    { id: 'save', number: '04', label: 'Save & export' },
    { id: 'shortcuts', number: '⌨', label: 'Keyboard shortcuts' },
  ];
  const day = (value: string) =>
    new Date(value).toLocaleDateString('en-GB', {
      day: 'numeric',
      month: 'short',
      year: 'numeric',
      timeZone: 'UTC',
    });
</script>

<a class="skip-link" href="#quick-start-main">Skip to quick start</a>
<main class="page-shell learn-page quick-start-page" id="quick-start-main">
  <header class="learn-heading">
    <div>
      <p class="learn-eyebrow">QUICK START / YOUR FIRST INVESTIGATION</p>
      <h1>A question. A workspace.<br /><span>A clearer picture.</span></h1>
      <p class="learn-intro">
        Get comfortable with Terminal in four steps. Start with source data,
        bring in an algorithm and build a view you can come back to.
      </p>
    </div>
    <a class="learn-button" href="/terminal"
      >Open Terminal <span aria-hidden="true">↗</span></a
    >
  </header>
  <div class="quick-layout">
    <aside class="quick-sidebar">
      <nav aria-label="Tutorial steps">
        <p class="learn-eyebrow">IN THIS GUIDE</p>
        {#each steps as step}<a href={`#${step.id}`}
            ><span>{step.number}</span>{step.label}</a
          >{/each}
      </nav>
      <p>
        Each chart, table or notes tile is a <strong>widget</strong>. You can
        arrange and link them to investigate the same data from different
        angles.
      </p>
      <a class="learn-link" href="/sources">Explore the data sources ↗</a>
    </aside>
    <div class="quick-steps">
      <section class="quick-step" id="start" aria-labelledby="start-heading">
        <p class="learn-eyebrow">01 / START WITH THE DATA</p>
        <h2 id="start-heading">Choose a window into the market.</h2>
        <p>
          Open Terminal to see your saved workspace, or the starter market view
          on your first visit. To create a fresh starting point, choose <strong
            >Workspaces → Market preset</strong
          >.
        </p>
        <ol>
          <li>
            Set the start and end dates in the toolbar. A few days is a useful
            first window.
          </li>
          <li>
            Hover over the chart to compare values across linked views. Click a
            period, or use the chart’s arrow keys, to pin it and inspect its
            source records.
          </li>
          <li>
            Write a question in the notes tile. Click outside the field to save
            it.
          </li>
        </ol>
        <figure class="quick-figure">
          <a
            href="/quick-start/market-workspace.png"
            target="_blank"
            rel="noreferrer"
            aria-label="Open full-size screenshot of the market workspace"
            ><img
              src="/quick-start/market-workspace.png"
              width="1440"
              height="900"
              alt="Terminal showing a wind and imbalance-price chart, a selected-period inspector and investigation notes."
              fetchpriority="high"
            /></a
          >
          <figcaption>
            <span
              >A source chart, period inspector and notes share one workspace.</span
            ><a
              href="/quick-start/market-workspace.png"
              target="_blank"
              rel="noreferrer">View full size ↗</a
            >
          </figcaption>
        </figure>
      </section>
      <section
        class="quick-step"
        id="analysis"
        aria-labelledby="analysis-heading"
      >
        <p class="learn-eyebrow">02 / ADD AN ALGORITHM</p>
        <h2 id="analysis-heading">Bring a method to your question.</h2>
        <p>
          Choose <strong>Search / Add</strong>, or press <kbd>Ctrl</kbd>/<kbd
            >⌘</kbd
          >
          + <kbd>K</kbd>. Open <strong>Algorithms</strong> and search for a method.
          Each entry explains its inputs, outputs and limitations.
        </p>
        <ol>
          <li>
            For a simple first comparison, look for <strong
              >price_naive_backtest</strong
            >. It compares baseline forecasts with settled prices.
          </li>
          <li>
            Choose <strong>Open analysis layout</strong> to create a workspace with
            an algorithm chart, its results table and notes. You can also add outputs
            to an existing chart.
          </li>
          <li>
            The app opens a completed, stored run. If a method has no results
            yet, it shows “Not computed yet”.
          </li>
        </ol>
        <figure class="quick-figure">
          <a
            href="/quick-start/add-analysis.png"
            target="_blank"
            rel="noreferrer"
            aria-label="Open full-size screenshot of the algorithm catalogue"
            ><img
              src="/quick-start/add-analysis.png"
              width="1440"
              height="900"
              alt="Search / Add drawer with the Algorithms tab selected and the price naive backtest’s chart, table and analysis-layout actions."
              loading="lazy"
            /></a
          >
          <figcaption>
            <span
              >Find an algorithm, check its limitations and choose how to open
              it.</span
            ><a
              href="/quick-start/add-analysis.png"
              target="_blank"
              rel="noreferrer">View full size ↗</a
            >
          </figcaption>
        </figure>
      </section>
      <section
        class="quick-step"
        id="workspace"
        aria-labelledby="workspace-heading"
      >
        <p class="learn-eyebrow">03 / MAKE THE VIEW YOURS</p>
        <h2 id="workspace-heading">Keep the evidence side by side.</h2>
        <p>
          Use the chart to explore the forecast traces and the table to inspect
          their errors. Companion widgets retain the same run ID, so you can
          tell which calculation produced the results.
        </p>
        <ol>
          <li>
            To compare with actual prices, open the chart’s gear and select <strong
              >Imbalance price</strong
            >
            under <strong>Overlay source data</strong>.
          </li>
          <li>
            Drag a tile tab to an edge to split the view, or to the centre to
            stack it as a tab. Drag a divider to resize. Widget settings also
            offer <strong>Move without dragging</strong>.
          </li>
          <li>
            Use each widget’s gear to change its series, outputs, display or
            selected result table. Use the letter badge to link widgets into the
            same group, or make one independent.
          </li>
          <li>
            On narrow screens, use the widget bar to switch views. Your desktop
            arrangement stays saved.
          </li>
        </ol>
        <figure class="quick-figure">
          <a
            href="/quick-start/analysis-workspace.png"
            target="_blank"
            rel="noreferrer"
            aria-label="Open full-size screenshot of the analysis workspace"
            ><img
              src="/quick-start/analysis-workspace.png"
              width="1440"
              height="900"
              alt="A docked analysis workspace with baseline forecast traces, a same-run results table and notes about comparing model errors."
              loading="lazy"
            /></a
          >
          <figcaption>
            <span
              >Charts, tables and notes give the investigation a shape you can
              customise.</span
            ><a
              href="/quick-start/analysis-workspace.png"
              target="_blank"
              rel="noreferrer">View full size ↗</a
            >
          </figcaption>
        </figure>
      </section>
      <section class="quick-step" id="save" aria-labelledby="save-heading">
        <p class="learn-eyebrow">04 / SAVE &amp; EXPORT</p>
        <h2 id="save-heading">Leave yourself a way back.</h2>
        <p>
          Workspace changes save in this browser automatically. Open <strong
            >Workspaces</strong
          >
          to name or duplicate the view, or choose
          <strong>Save current view as preset</strong> to reuse its layout.
        </p>
        <ul>
          <li>
            <strong>Export data:</strong> open a chart or table’s gear, then use
            <strong>Data export → Export chart CSV</strong>
            or <strong>Export table CSV</strong>. Charts export their loaded
            data; tables export the selected whole-run result.
          </li>
          <li>
            <strong>Back up your setup:</strong> choose
            <strong>Workspaces → Export configuration</strong> for a JSON file containing
            layouts, settings and notes.
          </li>
          <li>
            <strong>Restore a setup:</strong> choose
            <strong>Import configuration</strong>, select the JSON file, review
            the preview and confirm replacement. You can undo the replacement.
          </li>
        </ul>
        <p class="quick-tip">
          CSV contains data. JSON configuration contains your workspace setup,
          not a copy of market data.
        </p>
        <figure class="quick-figure">
          <a
            href="/quick-start/save-workspace.png"
            target="_blank"
            rel="noreferrer"
            aria-label="Open full-size screenshot of the Workspaces panel"
            ><img
              src="/quick-start/save-workspace.png"
              width="1440"
              height="900"
              alt="Workspaces panel showing workspace naming, presets, JSON export and configuration import."
              loading="lazy"
            /></a
          >
          <figcaption>
            <span
              >Save a reusable view or take a configuration backup with you.</span
            ><a
              href="/quick-start/save-workspace.png"
              target="_blank"
              rel="noreferrer">View full size ↗</a
            >
          </figcaption>
        </figure>
      </section>
      <section
        class="quick-shortcuts"
        id="shortcuts"
        aria-labelledby="quick-shortcuts-heading"
      >
        <p class="learn-eyebrow">KEYBOARD REFERENCE</p>
        <h2 id="quick-shortcuts-heading">Keep the keys close.</h2>
        <p>
          Open this reference any time with the keyboard icon beside Settings.
          These controls apply while Terminal is open.
        </p>
        <TerminalShortcuts />
      </section>
      <p class="quick-capture-note">
        Screenshots are examples captured from the app using real stored data
        for {day(capture.from)}–{day(capture.to)} and {capture.algorithm} run #{capture.run_id}.
        Your available dates, runs and layout may differ.
      </p>
      <div class="quick-finish">
        <div>
          <p class="learn-eyebrow">YOUR NEXT QUESTION</p>
          <h2>Build your first investigation.</h2>
          <p>Start with a small date range and a question you can test.</p>
        </div>
        <a class="learn-button" href="/terminal"
          >Open Terminal <span aria-hidden="true">↗</span></a
        >
      </div>
    </div>
  </div>
</main>
