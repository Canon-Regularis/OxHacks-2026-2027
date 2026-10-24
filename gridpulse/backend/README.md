# Backend work area

Data/database owner: `ingestion/`, `db/` and `gridpulse/db/migrations/` (relative to the repository root).
Algorithm/backend owner: `analysis/`, `api/`, application wiring and later `agent/`.

FastAPI + psycopg 3 over Tiger Data. Read the [backend design](../../docs/gridpulse/BACKEND_DESIGN.md) first. Algorithm authors only need [`analysis/contract.py`](src/gridpulse/analysis/contract.py) and the example in [`analysis/algorithms/`](src/gridpulse/analysis/algorithms/). The Orchid P0 adapters in the same package are documented in [algorithms and analysis](../../docs/gridpulse/ALGORITHMS_AND_ANALYSIS.md#p0-algorithms-issue-6).

## Setup

From this directory (`gridpulse/backend`):

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Database: save the credentials file downloaded from Tiger Cloud (`tiger-cloud-<service>-credentials.env`) in the repository root, where it is picked up automatically and gitignored, or set `DATABASE_URL` in `gridpulse/.env` (see `../.env.example`).

## Commands

```bash
pytest                                              # unit tests, no database needed
pytest -n auto                                      # the same, one worker per core (what CI runs)
RUNSLOW=1 pytest                                    # also the 31 slow Orchid smoothing tests
GRIDPULSE_INTEGRATION=1 pytest                      # also run read-only checks against Tiger Data
ruff check . && ruff format --check . && mypy       # strict lint and types for the Orchid adapter layer
gridpulse migrate                                   # create/update the gridpulse schema and views (safe to repeat)
gridpulse analyse wind_price_correlation --from 2026-07-01 --to 2026-10-23
gridpulse algorithms                                # list registered algorithms
uvicorn gridpulse.main:app --reload                 # API on :8000, docs at /docs
```

Source data is loaded into the `raw` schema by the team's loader; the backend only reads it, through the `gridpulse.*` views.

Offline demo: run once with `GRIDPULSE_MODE=record`, click through the demo, then start with `GRIDPULSE_MODE=replay` (no database needed).

Keep routes thin, SQL in `db/queries.py`, orchestration in `analysis/service.py` and calculations in `analysis/algorithms/`. See the [repository structure](../../docs/gridpulse/REPOSITORY_STRUCTURE.md) and [API contract](../contracts/README.md).
