# AI agent provider research

Research date: 25 October 2026. This is provider research for the Terminal agent plan, not an implemented integration. No credential values were read and no authenticated inference requests were made.

## Confirmed provider facts

CoreWeave Serverless Inference is a viable candidate for the requested Kimi K2.6 model. Its current catalogue lists `moonshotai/Kimi-K2.6` as generally available. This is the CoreWeave model identifier; it differs from the first-party Kimi identifier. [CoreWeave model catalogue](https://docs.coreweave.com/products/inference/serverless/models)

CoreWeave documents the following connection settings:

| Setting | Documented value |
|---|---|
| Base URL | `https://api.inference.wandb.ai/v1` |
| Authentication | CoreWeave Forge API key; HTTP bearer authentication |
| Account requirement | Forge account with Serverless Inference credits |
| Optional attribution | `OpenAI-Project: team/project` |

These settings belong to hosted Serverless Inference, rather than CoreWeave's separate infrastructure/deployment management API. [Serverless API reference](https://docs.coreweave.com/products/inference/serverless/api-reference)

The service documents OpenAI-compatible `POST /chat/completions`. Its published example covers ordinary messages and assistant text responses; compatibility does not establish that every OpenAI request parameter is supported. [Chat completions](https://docs.coreweave.com/products/inference/serverless/api-reference/chat-completions)

`GET /models` lists available model IDs and can confirm account access. The published catalogue alone does not confirm this project's account entitlement. No authenticated discovery was performed during this research. [List models](https://docs.coreweave.com/products/inference/serverless/api-reference/list-models)

## Features requiring endpoint validation

The reviewed CoreWeave Serverless reference and usage examples do not explicitly specify Kimi K2.6 support for function tools, streamed tool-call deltas, JSON Schema structured outputs, or thinking-mode controls. Treat these as **unverified for the chosen CoreWeave endpoint**, rather than unsupported. Before implementation depends on them, verify:

- `tools`, `tool_choice`, tool-call IDs and continuation after tool results.
- `stream: true`, SSE termination, usage chunks, cancellation and fragmented function arguments.
- `response_format` with JSON Schema, schema restrictions and refusal/truncation handling.
- Thinking defaults and any required preservation of provider-specific assistant fields.

This distinction matters because model capabilities and a particular hosting API's exposed capabilities can differ. [CoreWeave API reference](https://docs.coreweave.com/products/inference/serverless/api-reference), [CoreWeave examples](https://docs.coreweave.com/products/inference/serverless/examples)

## First-party Kimi alternative and reference

Kimi's own service documents Kimi K2.6 as `kimi-k2.6`, with thinking and non-thinking modes. This does not establish the identity or validity of the project's existing key. [Kimi model selection](https://www.kimi.ai/help/kimi-api/api-model-selection)

The first-party Chat Completions reference uses `https://api.moonshot.ai/v1/chat/completions`. It documents function tools and matching `tool_call_id` results, SSE streaming, and `response_format` modes including JSON Schema. Tool continuation should retain the complete assistant message, including provider-specific reasoning fields. K2.6 supports `thinking.type`; K3 examples and reasoning controls must not be copied indiscriminately. These are **first-party Kimi API facts, not evidence of CoreWeave feature parity**. [Kimi Chat Completions reference](https://platform.kimi.ai/docs/api/chat)

## Integration recommendations

These are proposed application decisions:

- Keep provider, base URL and model ID explicit server-side configuration. `CWAVE_API_KEY` is only a variable name; confirm the intended issuer/endpoint without exposing its value. Do not silently switch providers or fall back to Gemini.
- Hide provider details behind one adapter that returns application events and typed tool calls. Never pass provider credentials to the browser.
- Validate tool names, arguments, algorithm/run identifiers and workspace actions locally even when provider structured output is available. Apply a layout only after its complete action batch passes validation; partial streamed JSON must never mutate the workspace.
- Keep protocol continuation fields server-side; stream user-facing answers and tool progress separately. Assemble streamed tool arguments completely before validation or execution.
- Bound request duration, tool rounds, output tokens and concurrency. Handle rate limits, missing model access, disconnections and cancellation without duplicating layout changes.
- Use mocked provider fixtures for routine tests. Make any live compatibility check a small, explicitly enabled integration check; record actual endpoint/model capabilities before selecting a fallback strategy.

Provider/endpoint confirmation and a compatibility check remain implementation prerequisites. The research establishes a documented CoreWeave candidate, not a tested integration with the supplied credential.
