> **Current agent direction (25 October 2026):** OpenRouter (GPT-5.4 mini by default) supersedes the earlier Gemini and Kimi choices for Terminal chat and automatic workspace layouts. See [the approved implementation plan](AI_AGENT_IMPLEMENTATION_PLAN.md) and [provider research](AI_AGENT_PROVIDER_RESEARCH.md). The Gemini descriptions below record the earlier proposal; they are not the selected implementation.

# Extensions, sponsor integrations and limitations

## Main extension: Gemini-powered analysis agent

After the core dashboard works, add a backend agent using the **Gemini API**. It should answer higher-level questions by querying the same stored data and calling the same analysis functions used by the dashboard.

Example questions:

- “What changed around this price spike?”
- “Which periods had the largest downward wind forecast revisions?”
- “Compare demand and algorithm indicators across these two days.”

Suggested flow:

```text
Question + selected date range
    → backend Gemini integration
    → bounded data/analysis tools
    → structured evidence
    → answer with periods, units, source and algorithm context
```

Start with a small read-only tool set: available coverage, fetch bounded series, fetch existing analysis and inspect a period. Add new analysis execution only if runtime is predictable. Validate tool arguments, cap rows/tool calls, set timeouts and keep credentials server-side. Feed data as evidence, not instructions.

Answers should identify supporting periods and calculations, distinguish observations from interpretations and state when data is missing. The agent must not invent measurements or claim a causal explanation from correlation. Keep the dashboard operational when Gemini is unavailable; label any recorded response used as a demo fallback.

Select an available Gemini model during implementation; no particular model ID or account entitlement is promised. See [Gemini documentation](https://ai.google.dev/gemini-api/docs) and [function calling](https://ai.google.dev/gemini-api/docs/function-calling).

## Agent-driven dashboard and synoptic extension

The user has requested that Gemini be able to change the displayed view during an investigation, using animations to bring up supporting information. This belongs to the agent extension. The exact tool names, transport and workspace layout remain implementation decisions; no arbitrary UI execution interface is agreed.

The Discord brief also confirms a map extension: adapt KCPC Network's global map to display wind/pressure/fronts with correct isobars, plus an independently selectable UK electricity-use and forecast heatmap. Permit supported charts to be displayed within agent-led investigations. Correct regional inputs and a verified synoptic source are prerequisites, not existing capabilities.

See [FRONTEND_DESIGN.md](FRONTEND_DESIGN.md) for the confirmed visual requirements and [DATA_PIPELINE.md](DATA_PIPELINE.md#weather-and-regional-data-for-the-map-extension) for the outstanding data requirements. The Royal Meteorological Society mention is unverified; no weather provider is selected.

## Hosting: Vultr

Vultr is the chosen application hosting target, described in the [build plan](BUILD_AND_DEMO.md). Demonstrate the actual deployed frontend/backend and describe what is running there. GPU infrastructure is unnecessary for this plan. Keep a local cached demo as backup.

## Optional: Snowflake API

Snowflake is a possible additional track, not an MVP dependency. Its useful role is still undecided. One candidate is an optional comparison between Gemini and a Snowflake-hosted model answering the same evidence-backed question. Another is a specific analytical capability that solves a demonstrated gap.

Before implementing, establish available account access, a concrete user benefit, a small integration contract and enough remaining time. Keep Tiger Data as the core store; avoid duplicating the entire pipeline merely to add a sponsor name. If no useful integration is completed, describe Snowflake as unimplemented exploration and omit any claim of using it.

Consult [Snowflake Cortex REST API documentation](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-rest-api) when implementing; model availability and credentials depend on the account. Keep credentials server-side. Sponsor eligibility must be checked against the event's current rules at submission time.

## Other enhancements

Only after the MVP is stable: day comparisons, export, richer historical exploration and scenario controls that the existing algorithms actually support. Preserve explanations of input assumptions and units. No model-training or learned-model explanation work is scheduled.

## Limitations to communicate

- Algorithm capabilities and interpretation come from their owner; complexity alone does not establish accuracy.
- Indicators are not probabilities unless the algorithm supports that interpretation with evidence.
- Retrospective observations differ from forecasts available at a historical cutoff.
- Historical price records may reflect later settlement revisions.
- Missing historical forecast versions limit what can be claimed about earlier-known signals.
- Selected historical events illustrate behaviour; they do not establish representative predictive performance.
- Wind and demand alone cannot explain all market movements, and this app does not measure physical grid reliability.
- AI answers must be grounded in retrieved evidence and cannot establish causality by narration.

## Background references

These references motivate the domain; they are not validation of the team's algorithms.

- [Elexon Insights API](https://bmrs.elexon.co.uk/api-documentation/endpoint): source datasets and endpoint catalogue.
- [Elexon settlement guidance](https://www.elexon.co.uk/bsc/settlement/): settlement-period context.
- Goodarzi, Perera & Bunn (2019), [renewable forecast errors and electricity prices](https://doi.org/10.1016/j.enpol.2019.06.035): German-market research context.
- Browell & Gilbert (2022), [GB imbalance-price forecasting context](https://doi.org/10.3390/en15103645): background rather than a deliverable to reproduce.
