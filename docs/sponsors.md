# OxHack 2026 — MLH Sponsor Prizes & Technology Implementation Reference

> **Purpose:** A quick, implementation-focused reference for choosing and integrating sponsor technologies during OxHack (24–25 October 2026). **Start with one reliable core feature.** Only add an integration if it materially improves the demo.
>
> **Status:** Prize names and rewards verified on the [official MLH OxHack ’26 prize page](https://www.mlh.com/events/oxhack-26/prizes) and [event page](https://www.mlh.com/events/oxhack-26) on **24 October 2026**. Vendor implementation notes are based on official documentation; model IDs, trial terms, credits and rules may change.

## GridPulse project choices

The current [GridPulse plan](gridpulse/README.md) is a **25-hour electricity-market analysis dashboard** using existing algorithms, with no model training.

- **Tiger Data — MVP:** central store for CSV-imported electricity data and analysis results, queried by the backend.
- **Vultr — hosting:** target for the frontend/backend deployment, with a local cached demo fallback.
- **Gemini API — extension:** backend agent using data and analysis tools to answer higher-level questions after the main dashboard works.
- **Snowflake API — optional exploration:** a useful role must be established before implementation; it is not a core dependency or a completed integration.

The broader ideas and technology rankings below are generic sponsor reference material, not GridPulse implementation priorities. Follow the [25-hour build plan](gridpulse/BUILD_AND_DEMO.md) for project scope. Check track requirements when submitting; planned usage alone is not a completed entry.

## 0. Read this first: prizes and technology choice

### Official MLH prize categories (8 total)

| MLH category | Advertised reward | Main qualifying technology/action |
|---|---|---|
| **Best Use of ElevenLabs** | Wireless Earbuds | Make expressive generated speech or audio an integral part of the hack |
| **Best Use of Gemini API** | MLH Swag Kits | Use Google Gemini API for a meaningful AI-powered feature |
| **Best Use of Solana** | Unruggable Hardware Wallets | Demonstrate a real on-chain use case such as transactions or ownership |
| **Best Use of Tiger Data** | Stream Deck Mini | Use Tiger Data meaningfully for SQL, time-series, real-time data or analytics |
| **Best Use of Presage** | Fitbit Inspire & Presage Perks | Use a Presage SDK for live human sensing from a camera |
| **Best Use of Vultr** | M5Stack Official Tab5 | Demonstrate Vultr's compute or cloud infrastructure powering the project |
| **DEV Writing Challenge** | $100 | Publish a substantive project write-up on DEV; selected monthly |
| **Best Use of Snowflake API** | Arduino Student Kit | Build with Snowflake APIs, e.g. hosted models, custom AI or retrieval |

**Source:** https://www.mlh.com/events/oxhack-26/prizes (cross-checked with https://www.mlh.com/events/oxhack-26). The website advertises these prize types, **not** a guarantee that every member receives a particular item. Check any individual category rules, eligibility, shipping, judging and submission conditions. These are **MLH sponsor prizes**, not necessarily OxHack's main judging tracks.

**Additional published details:** Presage advertises development-credit refills over the next three months and a 30% discount on usage charges for the first year after go-live. Snowflake advertises a special **120-day student trial**. The DEV editorial team selects at least one favourite project **per month**, with additional selections depending on volume; this is not necessarily an OxHack-only prize.

### Implementation shortlist

| Technology | Does | Best use | Integration risk | Priority |
|---|---|---|---|---|
| **Google Gemini** | Multimodal LLM and tool/function calling | AI agents, reasoning, interpreting scenes, generating actions | Low–medium | **1 — core AI** |
| **ElevenLabs** | Text-to-speech / voice AI | Narration, voiced agents/NPCs, accessible interfaces | Low | **2 — when voice adds value** |
| **Presage SmartSpectra** | Estimates pulse, breathing and other camera-derived signals | Physiology-responsive interactive demo | Medium–high (camera and platform setup) | Distinctive if feasible |
| **Snowflake Cortex REST API** | Hosted LLMs via an API; associated AI/data capabilities | Multi-model assistants, RAG and data-driven AI | Medium | Alternative AI backend |
| **Tiger Data** | PostgreSQL + time-series storage and analytics | Live agent telemetry, history, simulation dashboards | Low–medium | Optional |
| **Vultr** | Cloud compute/hosting, including GPU options | Deployed backend, hosted simulations/LLMs | Medium | Only if infrastructure needed |
| **Solana** | Blockchain transactions and on-chain programs | Verifiable asset transfers, agent marketplaces | High | Stretch only |
| **DEV Writing Challenge** | Publish a project post | A thoughtful technical write-up after the hackathon | Low | Post-event opportunity |

**Fast choice:**

- Building an AI agent? **Gemini**; optionally add **ElevenLabs** if voice is central.
- Building an AI information assistant or model-comparison tool? Explore **Snowflake**; don't integrate Snowflake and Gemini unless you need both.
- Building an adaptive camera-driven game? **Presage** is distinctive, but validate webcam metrics before committing.
- Building agent analytics? **Tiger Data** if timestamped observations genuinely matter; SQLite/Postgres may be sufficient for an MVP.
- Need public hosting or substantial compute? **Vultr**; check available trial credits and cost.
- For the **DEV Writing Challenge**, capture useful screenshots, trade-offs and lessons during the event and write afterward.

**Strategy:** Start with one reliable vertical slice and one defensible primary sponsor integration; adding every sponsor API is not a winning strategy.

## 1. Google Gemini — first choice for AI functionality

**MLH category / reward:** Best Use of Gemini API — **MLH Swag Kits**. The event page highlights personal assistants, summarising complex material and creative generation; a visible, functioning Gemini integration matters more than a generic “AI” label.

**What it is:** Google's hosted multimodal models. Useful for text/image interpretation, planning suggestions, classification, structured output and *function calling*. Model output is not itself an action: **your application executes approved actions and checks the result**.

**Setup:** Obtain an API key via [Google AI Studio](https://aistudio.google.com/app/apikey), check model availability and quotas, and use the current `google-genai` package (not the older `google-generativeai` SDK).

```bash
python -m pip install -U google-genai python-dotenv
```

Create a local `.env` (do **not** commit it):

```dotenv
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.8-flash
```

Minimal Python smoke test (`gemini_test.py`):

```python
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_content(
    model=os.environ["GEMINI_MODEL"],
    contents="Suggest one action for a robot blocked by a wall. One sentence.",
)
print(response.text)
```

Run with `python gemini_test.py`. **The model ID is an example from current documentation, not a guarantee for your account.** Change `GEMINI_MODEL` to one available for your key if necessary.

**Practical agent architecture:**

```text
Game/simulation state -> compact state description -> Gemini proposes action
         ^                                               |
         |                                               v
State + result <- validator + bounded executor <- structured action
```

Keep the environment *authoritative*. Define a small set of legal actions (`move`, `inspect`, `interact`, `wait`). Reject unknown actions; impose per-step timeouts, bounded retries and a max actions/turn budget. Use mock/recorded responses for your offline demo fallback. Capture a plan-vs-actual trace so the judge can see how the agent recovered.

**Good projects:** agent re-planning when a path becomes blocked; multi-agent coordination with a live trace; visual assistant with explicit tool use.

**Pitfalls:** latency, token costs/rate limits, hallucinated actions, changing model availability, accidentally exposing API keys from frontend code. Make calls from a backend (or another securely controlled environment).

**Docs:** [Gemini getting started](https://ai.google.dev/gemini-api/docs/get-started) · [Models](https://ai.google.dev/gemini-api/docs/models) · [Function calling](https://ai.google.dev/gemini-api/docs/function-calling)

---

## 2. ElevenLabs — easiest way to add a polished audio interface

**MLH category / reward:** Best Use of ElevenLabs — **Wireless Earbuds**. The advertised use cases include interactive companions, narration and voice-enabled apps. Prefer audio that meaningfully changes the experience, rather than decorative text-to-speech.

**What it is:** Speech synthesis and related voice AI capabilities. A particularly easy win when spoken feedback is essential to how the app is used, not merely a sound effect.

**Setup:** Create an API key, pick a voice from the library, and install the official SDK:

```bash
python -m pip install elevenlabs python-dotenv
```

`.env`:

```dotenv
ELEVENLABS_API_KEY=your_key_here
ELEVENLABS_VOICE_ID=JBFqnCBsd6RMkjVDRZzb
```

Minimal Python example (writes an MP3, so it does not require a system audio player):

```python
import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()
client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

chunks = client.text_to_speech.convert(
    voice_id=os.environ["ELEVENLABS_VOICE_ID"],
    model_id="eleven_flash_v2_5",  # example low-latency TTS model; verify access
    text="Route blocked. Recalculating a safer path.",
    output_format="mp3_22050_32",
)
with open("sample.mp3", "wb") as f:
    for chunk in chunks:
        if chunk:
            f.write(chunk)
print("Created sample.mp3")
```

**Typical integration:** `Gemini response / app event -> short spoken message -> ElevenLabs -> audio player`. Do **not** send a 2,000-word reasoning trace to TTS: generate concise human-facing narration. Cache recurring lines, avoid generating identical speech repeatedly, and fall back to on-screen text if synthesis fails.

**Good projects:** AI guide that narrates actions; multiple characters speaking with distinct voices; conversational assistive application.

**Pitfalls:** time to first audio, noisy venue (use headphones/captions), credit limits and API cost, browser playback restrictions. Do not clone a real person's voice without permission.

**Docs:** [ElevenLabs TTS quickstart](https://elevenlabs.io/docs/eleven-api/guides/cookbooks/text-to-speech) · [Streaming TTS](https://elevenlabs.io/docs/eleven-api/guides/how-to/text-to-speech/streaming)

---

## 3. Presage SmartSpectra — impressive, but prove feasibility immediately

**MLH category / reward:** Best Use of Presage — **Fitbit Inspire & Presage Perks**. The official listing mentions breathing/heart-rate estimates, movement, facial expressions and engagement-related capabilities. It also advertises development-credit refills for three months and 30% off usage charges in the first year after go-live; check exact eligibility. **Do not interpret estimated measurements as a medical diagnosis or as proof of an emotional state.**

**What it is:** Camera-based physiological measurements. Depending on platform/SDK, it can supply **pulse**, **breathing**, facial landmarks and other metrics, with confidence/validation feedback. It is intended for **wellness/informational** use, not medical diagnosis.

**Important platform decision:** The Node/Electron JavaScript option is **`@smartspectra/node-sdk`**, using native components; Presage offers other SDK options as well. **It is not a drop-in browser-only React API.** A pure website will require a different architecture or another supported platform. Windows x64, macOS Apple Silicon and recent Linux distributions are supported by the documented Node SDK; **Node 20+** is required. Electron can be used if the app needs a graphical camera feed.

**First test (before designing a product around it):**

1. Get a key from the [Presage developer portal](https://physiology.presagetech.com/).
2. Check supported OS/architecture and whether you have a working camera.
3. `npm install @smartspectra/node-sdk` — the SDK downloads substantial native runtime dependencies.
4. Run the [official Node quickstart](https://smartspectra.presagetech.com/docs/nodejs) or [Electron sample](https://github.com/Presage-Security/SmartSpectra) unchanged first.
5. Confirm you receive *real* metrics with useful confidence/validation status before integrating a game/UI.

Minimal Node camera smoke test based on the vendor's SDK shape (save as `presage_test.mjs`; use a local environment variable for the key):

```javascript
import {
  SmartSpectraSDK, CameraSelection,
  cardioMetrics, decodeMetrics
} from '@smartspectra/node-sdk';

const sdk = new SmartSpectraSDK({
  apiKey: process.env.PRESAGE_API_KEY,
  requestedMetrics: [...cardioMetrics],
});

sdk.on('metrics', (buf) => console.log(decodeMetrics(buf)));
sdk.on('validationStatus', (code, ts, hint) =>
  console.log('Validation:', code, hint));
sdk.on('error', (code, msg) => console.error(code, msg));
sdk.useCamera(CameraSelection.default);
sdk.start();

process.on('SIGINT', async () => {
  await sdk.stopAsync();
  await sdk.destroy();
  process.exit(0);
});
```

Set `PRESAGE_API_KEY` in your shell, then run `node presage_test.mjs`. **Verify the installed SDK's methods against its versioned docs.** If the key or hardware fails, switch to another project quickly rather than sinking the day into setup.

**Measurement constraints:** pulse typically requires about **12 seconds** of usable video; breathing rate around **30 seconds**. Good light, stable camera, visible face/upper chest and limited movement matter. A model cannot reliably infer a person's emotional state or diagnose stress solely from pulse rate. Only collect measurements with consent; avoid saving raw video if not necessary.

**Good projects:** a breathing-feedback visualiser; an interactive experience affected by measured physiological signals; live biometrics dashboard. For an *action game* with lots of movement, measurements may be unreliable — that demo needs careful design.

**Fallback:** show an explicitly labeled simulated reading for UI development/demonstration; never present mock values as live biomedical data.

**Docs:** [Presage overview](https://smartspectra.presagetech.com/docs) · [Node/Electron SDK](https://smartspectra.presagetech.com/docs/nodejs) · [Getting a good measurement](https://smartspectra.presagetech.com/docs/measurement-quality)

---

## 4. Tiger Data — useful for agent/simulation telemetry

**MLH category / reward:** Best Use of Tiger Data — **Stream Deck Mini**. The event highlights standard SQL, unified relational/time-series workloads, continuous aggregates and compression. It rewards meaningful, performant data usage; illustrate the data benefit in the demo.

**What it is:** PostgreSQL with time-series features (TimescaleDB). Ideal if you want to log many timestamped decisions and show metrics/aggregations over time.

**Integration:** Create a Tiger Cloud database and connect to it using a normal PostgreSQL driver (`psycopg` for Python or `pg` for Node). A sample time-series table on a TimescaleDB version supporting the modern table option:

```sql
CREATE TABLE agent_events (
    ts TIMESTAMPTZ NOT NULL,
    run_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    latency_ms DOUBLE PRECISION
) WITH (
    tsdb.hypertable,
    tsdb.partition_column = 'ts'
);

SELECT time_bucket('1 minute', ts) AS minute,
       count(*) AS events,
       avg(latency_ms) AS avg_latency_ms
FROM agent_events
GROUP BY 1 ORDER BY 1;
```

Use continuous aggregates if you genuinely need them. For an MVP with a few hundred events, ordinary SQL might be sufficient. **Make the use of time-series functionality demonstrably important** if entering its sponsor challenge.

**Docs:** [Tiger Data documentation](https://docs.tigerdata.com/) · [TimescaleDB hypertables](https://docs.tigerdata.com/use-timescale/latest/hypertables/)

## 5. Vultr — only if hosting/compute is key

**MLH category / reward:** Best Use of Vultr — **M5Stack Official Tab5**. The event highlights fast cloud deployment, scalable compute and cloud GPUs. It also advertises promotional cloud credits; verify account eligibility, actual balances and usage costs.

**What it is:** Virtual machines and cloud infrastructure, including GPU options. Useful if judges need a public demo or a server is doing substantial computation.

**Suggested route:** Build/test locally -> containerise backend -> provision VM -> SSH into it -> deploy -> HTTPS URL. Keep secrets in server environment variables; limit open ports; monitor cost/credit consumption. **Confirm credits and GPU availability before promising a hosted LLM.** A locally running app with a recorded video is a better fallback than a half-configured cloud deployment.

**Docs:** [Vultr provisioning](https://docs.vultr.com/products/compute/instances/cloud-compute/provisioning) · [SSH connection](https://docs.vultr.com/products/compute/instances/cloud-compute/connection/openssh)

## 6. Solana — stretch goal, only with a genuine blockchain use case

**MLH category / reward:** Best Use of Solana — **Unruggable Hardware Wallets**. The event highlights fast, low-cost transactions, with example domains including games, social apps, trading, lending, identity, supply chains and payments.

**What it is:** A blockchain for token transfers, ownership and executable on-chain programs. Useful for an agent marketplace or verifiable shared state, *not* as a replacement for a normal database.

**Suggested route:** Use **devnet** and test tokens; build a minimal wallet + transaction flow first. Avoid real funds. Writing custom on-chain programs in Rust/Anchor can consume most of a 2-day event; don't start there unless someone on the team already knows Solana.

**Docs:** [Solana quick start](https://solana.com/docs/intro/quick-start)

---

## 7. Snowflake API — NEW: AI models and retrieval behind one platform

**MLH category / reward:** Best Use of Snowflake API — **Arduino Student Kit**. MLH encourages use of industry-leading hosted LLMs, custom AI apps and retrieval-augmented generation (RAG). It advertises a **special 120-day student trial**, subject to current sign-up and account conditions.

**What it is:** Snowflake is a data-and-AI platform. Its **Cortex REST API** provides an OpenAI-compatible chat-completions endpoint for multiple hosted models, so a Python backend can access a model through Snowflake without running local inference. Snowflake also has data search and AI functions that can be appropriate for a RAG application.

**Start here:**

1. Review [Snowflake's Cortex REST API documentation](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-rest-api).
2. Sign up through the hackathon's [Snowflake offer](https://www.mlh.com/events/oxhack-26) if eligible, and check the trial's exact limits.
3. Identify the Snowflake **account URL**, generate a **programmatic access token (PAT)**, and choose a model available in your region and account.
4. Install the OpenAI-compatible client: `python -m pip install openai python-dotenv`.
5. Keep the PAT server-side, in your local `.env` file (never commit it).

Example `.env`:

```dotenv
SNOWFLAKE_ACCOUNT_URL=https://your-account-id.snowflakecomputing.com
SNOWFLAKE_PAT=your_pat_here
SNOWFLAKE_MODEL=openai-gpt-5
```

The model identifier above is an **example shown in Snowflake's official quickstart**, not a guaranteed entitlement. Check [model availability](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-rest-api) for your account before running it.

Minimal Python smoke test (`snowflake_test.py`):

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.environ["SNOWFLAKE_PAT"],
    base_url=os.environ["SNOWFLAKE_ACCOUNT_URL"].rstrip("/") + "/api/v2/cortex/v1",
)
response = client.chat.completions.create(
    model=os.environ["SNOWFLAKE_MODEL"],
    messages=[{"role": "user", "content": "Suggest one safe action for a robot blocked by a wall."}],
)
print(response.choices[0].message.content)
```

**Good projects:** multi-model AI comparison with provenance; document assistant grounded in retrieved data; analytics assistant that reasons over stored observations; enterprise-style knowledge explorer.

**Pitfalls:** account permissions, PAT setup, model/region availability, credit use, and attempting to build an entire data platform for a trivial feature. **A compelling Snowflake entry needs a real reason to use Snowflake**, not a proxy call disguised as novel functionality.

**Official sources:** [Snowflake Cortex REST API](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-rest-api) · [MLH OxHack prize descriptions](https://www.mlh.com/events/oxhack-26)

---

## 8. DEV Writing Challenge — NEW: submit a technical project write-up

**MLH category / reward:** DEV Writing Challenge — **$100**. This is different from the technology-based sponsor challenges: the aim is to explain what you built and help other developers learn from it.

MLH says the DEV editorial team chooses **at least one favourite project each month**, with additional selections possible depending on volume. They consider **technical substance, originality, storytelling and community popularity**. This means the advertised reward is **not an automatic payment**, nor necessarily an award judged only among OxHack attendees.

**Suggested approach:** Collect evidence during the hackathon, but **write and submit after the core demo is complete**. Check the exact entry instructions on the [event's DEV link](https://www.mlh.com/events/oxhack-26) and [DEV](https://dev.to/).

A good article should cover:

1. The user problem and original pitch.
2. The technical architecture and principal sponsor API.
3. Key implementation decisions and what you built during OxHack.
4. A hard bug, unexpected result or failed approach and how you resolved it.
5. Screenshots, short code excerpts and working demo.
6. Limitations, next steps and GitHub repository link.

**Tip:** Clear photos of the demo and a truthful explanation of trade-offs help more than a generic marketing-style announcement.

---

## 9. Recommended implementation workflow

### First 30–60 minutes after team formation

- [ ] Define the problem, intended user and **one-sentence demo pitch**.
- [ ] Choose **one** central technical achievement, plus at most one sponsor integration to start.
- [ ] Check published hackathon rules and submission deadline; choose relevant categories. Note that the MLH categories do not necessarily cover OxHack’s main prizes.
- [ ] Agree on stack, responsibilities, Git branching and who owns integration.
- [ ] Make a minimal repo with `README.md`, `.gitignore`, `.env.example` and startup instructions.
- [ ] Each API owner runs a standalone smoke test and shares its actual output.
- [ ] Decide on a fallback if the API, camera or internet fails.

### Build order

1. **Vertical slice:** input -> core logic -> UI/result; hardcoded or simulated data is okay at first, **but label mock data**.
2. Integrate the main sponsor API and confirm genuine results.
3. Add visible instrumentation (agent plan, confidence, latency, state changes, failure recovery).
4. Make the demo repeatable and robust (timeouts, retries with limits, graceful fallbacks).
5. Add polish and any second API only after the main flow works.
6. Reserve time for README, screenshots, demo video and submission; finish these before the deadline.

### Never forget

- **Secrets:** `.env` in `.gitignore`; commit only `.env.example` with placeholder values. Rotate any accidentally committed key.
- **Privacy:** camera/microphone consent, do not retain unnecessary data, don't treat physiology estimates as medical data.
- **Reliability:** prepare a recorded demo if venue Wi-Fi/API quota fails; do not pretend a fallback is live.
- **Attribution:** credit SDKs/services, make clear what was built during the event; follow OxHack's rules for pre-existing code and AI tooling.
- **Story:** show a real problem, the technical challenge, and a 60–90 second successful demo.

## 10. Ideas matching the team's likely AI/full-stack strengths

| Idea | Core sponsor tech | Optional tech | 60-second demonstration |
|---|---|---|---|
| **Replanning Agent** | Gemini | ElevenLabs, Tiger Data | Block the agent's route; watch the updated plan and successful recovery |
| **Living Simulation Dashboard** | Gemini | Tiger Data | Change a scenario; compare agents' choices, results and live history |
| **Breathing-guided Interactive Experience** | Presage | ElevenLabs | Camera measurement drives visual/audio breathing guidance after confidence stabilises |
| **Voiced AI World** | Gemini + ElevenLabs | Vultr | Speak to characters, show coordinated actions in a scene, hear responses |
| **Retrieval Assistant** | Snowflake | Tiger Data | Ask a question, show retrieved context and a grounded answer |
| **Multi-model Comparison** | Snowflake | Gemini | Compare answers from supported models, with cost/latency notes |

**Best default:** Build the **Replanning Agent** with Gemini first. It fits an AI-planning emphasis, is easy to explain to judges and does not depend on fragile physiological capture. Add a short ElevenLabs narrator or Tiger Data analytics only if time permits.

---

## Official documentation index

- Gemini: https://ai.google.dev/gemini-api/docs/get-started
- ElevenLabs: https://elevenlabs.io/docs/eleven-api/guides/cookbooks/text-to-speech
- Presage: https://smartspectra.presagetech.com/docs/nodejs
- Tiger Data: https://docs.tigerdata.com/
- Vultr: https://docs.vultr.com/
- Solana: https://solana.com/docs/intro/quick-start
- Snowflake Cortex REST API: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-rest-api
- DEV Writing Challenge: https://dev.to/
- Official MLH OxHack 2026 prizes: https://www.mlh.com/events/oxhack-26/prizes
- Official MLH OxHack 2026 event details: https://www.mlh.com/events/oxhack-26
- OxHack Discord: https://26.oxhack.net/discord

*Prize names and advertised rewards cross-checked against MLH on 24 October 2026. Implementation guidance only; verify category rules, vendor entitlements and current API parameters when coding.*
