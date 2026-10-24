# Shared API contracts

Frontend and backend owners agree changes here before integration. This directory is a shared specification, not another runtime service. Backend runtime request/response validation belongs in `backend/src/gridpulse/api/schemas.py`.

Proposed endpoints:

| Method/path | Purpose |
|---|---|
| `GET /api/health` | Readiness and database/cache mode |
| `GET /api/coverage` | Available dates, source freshness and missing coverage |
| `GET /api/series?from=YYYY-MM-DD&to=YYYY-MM-DD` | Source series over an inclusive settlement-date range |
| `GET /api/analysis?from=YYYY-MM-DD&to=YYYY-MM-DD` | Completed analysis over that range |
| `GET /api/periods/{date}/{period}` | Period-level source and analysis details |

Endpoints are not implemented. Agree range limits, error responses and forecast-version selection before coding handlers. Use UTC instants with an explicit offset, settlement date/period identities, explicit units and algorithm/run provenance. Never assume 48 periods per day.

`examples/analysis.json` is a draft synthetic empty response for UI integration. It provides no real algorithm values. A real response must set `is_fixture` to false, supply real freshness/run metadata and include records using the agreed indicator schema. Do not turn missing values into zero.

Update example responses and backend schemas together. Add examples for source series, period details and errors once their actual data shapes are known.
