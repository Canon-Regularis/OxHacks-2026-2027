# Confirmed frontend direction

This records requirements supplied by the team in chat and Discord. It is a design brief, not a claim that the components or features are implemented. The core dashboard remains the MVP; weather mapping and the Gemini agent remain extensions.

For the approved evolution of Explore into Terminal, read [the Terminal implementation plan](TERMINAL_IMPLEMENTATION_PLAN.md). It records docking, linked widgets, saved workspaces, stored-run constraints and staged verification; earlier Explore references below describe the original scope.

## Confirmed pages and immediate focus

The frontend MVP includes **Overview, Terminal (formerly Explore) and Methods**. Overview and Terminal are implemented; Methods remains planned. **Investigate** and the synoptic/weather-map view remain extensions, not MVP delivery requirements.

- **Overview:** summarise electricity prices, wind/demand forecasts and available analysis for a selected timeframe.
- **Terminal:** inspect datasets and stored algorithm results through docked, linked widgets and saved workspaces.
- **Methods:** explain source data, algorithm inputs/outputs, assumptions and limitations.

The mockups are visual references. Overview and Terminal (the renamed Explore page) are implemented; Methods remains unimplemented. The Gemini panel shown in the overview concept is an extension and is not required for the first Overview build.

## Overview: “Scan the day” interaction

The team's 16:50 Discord instruction confirms that the scan should be centred on the last mouse-hovered position and move as the mouse moves over it. Keep the scan focus anchored to the most recently hovered settlement period when the pointer leaves, until the user chooses another period. Preserve this behaviour across resizing and support the actual 46/48/50 periods for the selected day. Provide an equivalent keyboard/touch selection so the information is accessible without a mouse.

This interaction is confirmed even though the rest of the mockup's exact layout remains exploratory. It is implemented in the interactive frontend; static PNG/SVG previews cannot demonstrate it.

## Compact dashboard and progressive disclosure

Overview should fit a normal desktop viewport at 100% zoom (verified at 1440×900, 1366×768 and 1024×768), including the footer. Small screens stack and scroll naturally; never clip content or shrink text to force a mobile screen fit. Decorative and long-form content belongs on the read-only **About** page, now an approved addition to the navigation.

- Overview has no hero/globe and no “01 / context” strip. About owns the globe, KCPC PixelBlast/TextShimmer explainer and committee-style decorative reveal.
- Selected properties share one chart/time cursor and up to two labelled unit axes. Defaults are imbalance price, forecast wind and forecast demand; users can deselect price. See the selection rules below.
- Use a pastel orange accent for navigation, selection and price, with muted mint/lavender companion series. Use a gradual sequential scan scale matching the chosen property’s colour. Negative prices retain their numerical sign in the legend; missing values remain neutral grey.
- Remove secondary descriptive text from cards and the property selector. Put definitions, source details and methods behind consistent **ⓘ tooltips** beside the relevant label.

**Project-wide design convention:** keep monitoring screens concise. Use progressive disclosure for secondary explanations, with tooltips available on hover, keyboard focus and tap, dismissible with Escape. Keep essential units, timestamps, missing-data states and short source/interpretation labels visible. Do not hide information required to interpret a number safely. Put longer explanations in read-only About/Methods content instead of repeatedly adding paragraphs to dashboards.

`InfoTooltip.svelte` is the shared implementation. Add future tooltip behaviour there rather than creating different variants per panel. Header/footer are shared across routes; moving to About preserves the saved view selection. About does not fetch market data.

## Reuse KCPC selectively

WWW and Components are sources of reusable code and visual patterns, not a requirement to reproduce the KCPC website or include every animation. The later Discord brief specifically requests these patterns where applicable:

| Requirement | Reference |
|---|---|
| Preserve existing resizing/responsive support | WWW and Components layouts |
| Reuse the application's colour scheme | Existing KCPC/Components theme and styling |
| Reuse the spinning globe | [KCPC home](https://kcpc.orthant.co.uk/) |
| Reuse the fade-in/fade-out “bit” animation near “work with us” | [KCPC home](https://kcpc.orthant.co.uk/) |
| Use the Internal/External committee animation treatment for decorative content where useful | [About](https://kcpc.orthant.co.uk/about) |
| Split long-form information into sections like UKIEPC Legacy | [About](https://kcpc.orthant.co.uk/about) |
| Use Software-style cards for those sections, rather than directly copying Legacy cards with reported formatting issues | [Software](https://kcpc.orthant.co.uk/software) |
| Make repository exploration fine-grained | [Software](https://kcpc.orthant.co.uk/software) |

Repository exploration refers to detailed browsing of software/repository information, not a confirmed requirement for an arbitrary file browser. Exact GridPulse content and navigation remain to be agreed. Preserve the selected components' responsive behaviour during adaptation, and keep transitions compatible with readable charts and reduced-motion preferences.

The cloned sources are siblings of this repository: `../www/` and `../components/`. The KCPC app is in `../www/apps/kcpc/`. Inspect actual components and their usage before adapting them; preserve required upstream attribution. Cloning is complete, but reuse and integration are not.

## Agent-driven visual investigation

The user wants Gemini to change the user's view while investigating a question, with animations that make the retrieved evidence visibly appear. The frontend should support this capability as part of the Gemini extension, while retaining manual navigation and usable data views.

This is a confirmed interaction goal, not an approved UI-action API or page-builder design. Define and validate supported view changes jointly with the backend during implementation. Show actual retrieved data and analysis results; animation must not imply that an unperformed calculation has occurred. Respect reduced-motion preferences and allow the user to interrupt agent-driven navigation.

## Synoptic and electricity map extension

Reuse/adapt the global map from [KCPC Network](https://kcpc.orthant.co.uk/network) for weather mapping. The Discord brief asks for:

- Synoptic visualisation of wind, pressure and meteorological fronts, with correctly represented isobars.
- A secondary, independently visible UK electricity-use and forecast heatmap overlay.
- The ability to view the weather/front layer and the electricity layer together or separately.
- Display of these charts within the agentic analysis/modelling section when supported by the available data and tools.

Time-align the weather fields, temperature changes, electricity demand/forecasts and imbalance prices where source coverage permits. If a dataset cannot support the selected time or region, show that limitation rather than implying an alignment.

The Discord claim of a directly proportional relationship between temperature differences and energy-price upticks is a hypothesis to investigate, not a confirmed formula or established causal effect. The overlay should expose evidence and comparisons without assuming that relationship.

This is an extension, not an expansion of the 25-hour core acceptance criteria. Geographic data and chart semantics must be verified before implementation. See [data requirements](DATA_PIPELINE.md#weather-and-regional-data-for-the-map-extension).

## Confirmed implementation decisions

- Use the compact monitoring layout with KCPC neutral surfaces, pastel orange accents and selected components; globe and PixelBlast live on About.
- Below the content-sized Chart data selector, **Day at a glance** shows daily averages for selected properties, minimum/maximum price with UTC times, negative-price period count/duration and peak forecast demand with its time. Statistics use loaded records for the selected day, remain fixed during hover and exclude missing readings; show price coverage for incomplete days.
- A right-hand property selector replaces the Weather × Market panel. Select **1–5 properties** from at most **two unit groups**; show units beside every option and grey out incompatible additions. Price counts towards the limit and can be deselected.
- Show one responsive summary card per selected property. Cards display the hovered period’s value and a muted daily average labelled **Daily AVG**; averages exclude missing values. Missing readings show **No data**, with gaps in chart lines.
- Available properties: imbalance price, forecast/actual wind, forecast/actual demand, residual demand and net imbalance volume. Stepped hourly weather, hourly-change and correlation options have been removed from Overview at the user’s request; the backend weather API remains available for future analysis. The shared metric catalogue defines units, formatting, colours and chart axes.
- The combined chart and scan update cards, markers and selection on hover, using loaded data without per-hover API calls.
- The scan has a property dropdown immediately before the period count and unit label. Offer only selected properties; use the chosen property’s own units, sequential colour scale and explicit daily min/max. Missing values are neutral grey. Preserve last-hover-centred magnification. If its property is deselected, fall back to the first remaining property.
- Use existing stored Open-Meteo archived forecasts, labelled retrospective; no new weather ingestion.
- Seven-day rolling correlation of hourly price changes versus temperature/pressure changes, minimum 24 valid pairs; show source, location, units and sample counts.
- Default to latest priced settlement day, but preserve user-selected date, location, properties and scan property in browser storage for reload and future cross-page synchronisation.
- Terminal is an active navigation link; Methods appears disabled; About is a working read-only page. Investigate is omitted from MVP navigation.

## Scope boundary

The existing overview mockup is a design exploration, not an approved page specification. Its illustrative values are not production data; confirmed behaviour is listed above. Overview, Terminal (formerly Explore) and Methods are confirmed MVP pages; Investigate remains an extension. Extension navigation and exact Gemini frontend tools remain undecided; the synoptic-map source is still unselected.

### Navigation and informational pages

The header groups Overview and Terminal as workspace pages, separated by a divider from Data sources, Quick start, Status and About. Preserve all links, active-page indicators and narrow-screen support. Quick start omits the screenshot capture disclaimer; image provenance remains in its capture metadata.

About's final three story rows pair their text with KCPC Shape Blur (square), Cubes and Liquid Ether illustrations in the alternating spare columns. Load decorative effects near the viewport, respect the shared motion preference and retain static fallbacks when WebGL is unavailable. Source attribution is recorded in the vendored KCPC README.

Status uses specific older-data/input-update labels and an explicit old-data notice instead of the ambiguous “Attention” badge. Preserve connection failures, missing-data states, coverage and timestamps so historical-data warnings do not hide real operational problems.
