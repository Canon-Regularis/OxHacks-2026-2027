# GridPulse frontend mockups

Confirmed MVP pages: **Overview, Explore and Methods**; implementation currently focuses on **Overview only**. Investigate and the weather map are extension concepts. The Overview mockup includes an optional Gemini panel; it is not required for the initial build.

Open `gridpulse-overview.png` for the preview. Import `gridpulse-overview.svg` into Figma or Penpot for an editable vector mockup; this was created locally, not inside either service. No account connection was available.

Recommendation: compact editorial hero, KCPC-style rotating globe placement, four summary cards, a large historical chart and period heatmap, and a contextual investigation panel. All values and plotted lines are illustrative. Gemini and real indicators are not implemented by this mockup.

The mockup references the neutral light/dark design tokens and DM Sans / DM Serif typography found in Components. SVG text specifies those fonts with fallbacks; the PNG uses installed DejaVu substitutes. Chart accents are proposed semantic colours, not verified existing brand colours. Globe is a drawn placement illustration, not copied KCPC animation code. For implementation reuse the actual globe and motion components after attribution checks.

For narrow screens: collapse navigation, use two columns then one for summary cards, stack the chart and investigation panel, preserve the time-series inspection area and move investigation below the selected-period detail. Motion must respect reduced-motion preferences. A weather map belongs in Explore as an extension, not in the initial Overview.


## Additional page mockups

Open `index.html` in a browser to browse all five concepts, including the existing overview. Each has a PNG preview and editable SVG suitable for import into Figma or Penpot:

- `gridpulse-explore`: dataset controls, aligned series and selected-period details.
- `gridpulse-weather-map`: UK map view with a synthetic GB relative-demand overlay for **24 October 2026, 17:00–17:30 UTC**, separate wind/pressure controls and a time scrubber. Northern Ireland has no electricity data in this example. Heat bands are not administrative regions. Pressure contours and arrows are schematic, not operational meteorological data; fronts are shown as an inactive layer option.
- `gridpulse-investigate`: illustrative Gemini conversation, view activity and evidence panels. No actual investigation or UI automation has run.
- `gridpulse-methods`: source catalogue, pipeline and algorithm-contract exploration.

`gridpulse-pages.png` is a contact sheet of the five concepts. Page scope is recorded above; detailed layouts and controls remain design explorations and do not expand the MVP. Static previews cannot demonstrate animation.

Map land outlines were drawn from the existing reference asset `components/static/world.json` in the sibling `orthant-kcpc/components` clone at revision `9626af5`. This is low-resolution land geometry, not verified regional boundaries. Confirm upstream dataset attribution/licensing and obtain suitable boundaries before shipping a production map. No application code was copied from the reference repositories.
