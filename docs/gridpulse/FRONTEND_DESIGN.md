# Confirmed frontend direction

This records requirements supplied by the team in chat and Discord. It is a design brief, not a claim that the components or features are implemented. The core dashboard remains the MVP; weather mapping and the Gemini agent remain extensions.

## Confirmed pages and immediate focus

The frontend MVP includes **Overview, Explore and Methods**. Work currently focuses only on **Overview**; Explore and Methods remain in MVP scope for subsequent implementation. **Investigate** and the synoptic/weather-map view remain extensions, not MVP delivery requirements.

- **Overview:** summarise electricity prices, wind/demand forecasts and available analysis for a selected timeframe.
- **Explore:** inspect datasets, time ranges and supporting records in more detail.
- **Methods:** explain source data, algorithm inputs/outputs, assumptions and limitations.

The mockups are visual references, not implemented pages. The Gemini panel shown in the overview concept is an extension and is not required for the first Overview build.

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

This is an extension, not an expansion of the 25-hour core acceptance criteria. Geographic data and chart semantics must be verified before implementation. See [data requirements](DATA_PIPELINE.md#weather-and-regional-data-for-the-map-extension).

## Scope boundary

The existing overview mockup is a design exploration, not an approved page specification. Its exact layout, labels, illustrative values and chart choices do not establish product requirements. Overview, Explore and Methods are confirmed MVP pages; Investigate remains an extension. Detailed layouts, particular control defaults and extension navigation are not confirmed. The weather provider and exact Gemini frontend tools also remain undecided.
