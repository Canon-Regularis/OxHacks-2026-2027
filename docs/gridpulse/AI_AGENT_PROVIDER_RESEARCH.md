# AI agent provider research

Updated 25 October 2026. The user confirmed **CanopyWave** as the issuer of `CWAVE_API_KEY` and supplied the endpoint/model example below. The earlier CoreWeave interpretation of that variable name was incorrect and is superseded.

| Setting | Selected value |
|---|---|
| Base URL | `https://inference.canopywave.io/v1` |
| Endpoint | `POST /chat/completions` |
| Model | `moonshotai/kimi-k2.6` |
| Authentication | Bearer key, backend-only `CWAVE_API_KEY` |

CanopyWave's own [Kimi K2.6 model page](https://canopywave.com/models/kimi-k2-6) documents that endpoint, identifier and ordinary Chat Completions request shape. The user's example agrees. Minimax was also offered by the user but is not selected and there is no automatic model fallback.

## Implementation choice

The adapter currently uses ordinary asynchronous Chat Completions with a bounded, schema-validated JSON decision envelope. Application SSE streams progress, evidence and complete answers/views; provider-native token streaming, function calling and JSON Schema response-format support are not assumed. This keeps execution behind the same local validation even if the model returns malformed instructions. Provider credentials, raw provider errors and private reasoning are never sent to the browser.

## Verification evidence

After the user corrected the local credential, CanopyWave returned a valid Kimi decision. A real read-only investigation retrieved price and wind evidence from Tiger Data for 23 October 2026 and returned a chart view. That view is preserved as [a contract fixture](../../gridpulse/contracts/agent/view.example.json) and replayed through the frontend compiler test.

The first full prompt revealed a schema error: Kimi included a protected chat widget in an analytical view. A minimised regression test reproduces this; the adapter now allows one schema-only repair per investigation, without relaxing validation. The original real investigation succeeded after the fix. This validates ordinary Chat Completions/JSON-envelope behaviour, not provider-native function tools or token streaming.

The previous 401 responses were resolved by updating the credential; they are not a remaining blocker. No credential values or private reasoning were captured in fixtures.

Routine tests use fake provider transcripts. Do not label mocked investigations as live-provider validation. No credentials are recorded in these documents or test fixtures.
