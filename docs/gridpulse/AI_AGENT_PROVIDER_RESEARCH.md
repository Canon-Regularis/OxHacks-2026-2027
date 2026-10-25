# Terminal agent provider and model

Updated 25 October 2026. The user requested **OpenRouter**, replacing the previous CanopyWave/Kimi default. Credentials remain backend-only.

| Setting | Current default |
|---|---|
| Provider | `openrouter` |
| Base URL | `https://openrouter.ai/api/v1` |
| Model | `openai/gpt-5.4-mini` |
| Key | `OPENROUTER_API_KEY` |
| Reasoning | `low`, private reasoning excluded from responses |
| Output | JSON object, followed by local schema and widget validation |

GPT-5.4 mini is selected as a compact agent model with tool/structured-output capabilities. OpenRouter currently lists $0.75/M input tokens and $4.50/M output tokens. Sonnet 4.6 was considered but lists $3/M input and $15/M output; the smaller model is a deliberate efficiency choice, with application-side layout planning providing predictable geometry. These published prices are not a per-investigation cost guarantee. [GPT-5.4 mini](https://openrouter.ai/openai/gpt-5.4-mini), [Sonnet 4.6 pricing](https://openrouter.ai/anthropic/claude-sonnet-4.6/pricing)

OpenRouter documents response-format controls and normalized reasoning parameters. The integration requests JSON-object output and low reasoning with `exclude: true`, while retaining strict Pydantic and renderer checks. It does not rely on provider JSON syntax alone for safe widget creation. Only providers supporting requested parameters are eligible. [Structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs), [reasoning parameters](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)

## Verification

The configured key passed a live JSON decision check. A real browser investigation over February–March 2026 then created a combined wind/demand/price chart and a wind/demand source table using Tiger Data. The chart measured **1053 × 406 px**, aspect ratio **2.6**, with a compact chat sidebar. Actual timestamped rows and MW headers rendered without chat/browser errors.

Live testing exposed two pre-existing application issues beyond model choice: source output names were incorrectly treated as algorithm keys, and unbound context widgets failed an optional-binding identifier check. Regression tests cover both. A captured OpenRouter proposal is replayed in frontend tests as [a fixture](../../gridpulse/contracts/agent/view.openrouter.example.json). No credentials or private reasoning appear in it.

This is a functional UX check, not a broad benchmark establishing model superiority. Routine tests use mocked provider responses; live checks remain opt-in. CanopyWave is retained only as an explicitly configured legacy provider (`AGENT_PROVIDER=canopywave`, its base URL/model and `CWAVE_API_KEY`); there is no silent provider/model fallback.
