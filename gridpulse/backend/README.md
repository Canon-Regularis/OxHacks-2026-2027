# Backend work area

Data/database owner: `ingestion/`, `db/` and `gridpulse/db/migrations/` (relative to the repository root).
Algorithm/backend owner: `analysis/`, `api/`, application wiring and later `agent/`.

Python code lives in `src/gridpulse/`. `analysis/algorithms/` holds the Orchid P0 adapters (pure calculations, no I/O). All other modules are documentation-only stubs: no server or database connection is implemented yet. Choose the HTTP framework and database driver when integrating real data; add dependencies to `pyproject.toml` and commit the chosen lockfile.

Create a local environment with `python3 -m venv .venv` from this directory. After activation, `python -m pip install -e ".[dev]"` installs the package with the development tools. Installation does not start a server. Add the actual launch command here once application wiring exists.

Checks, run from this directory:

- `pytest` runs the adapter tests and the ported Orchid suites. It takes about 1.5 minutes, and 31 slow smoothing tests are skipped unless `RUNSLOW=1` is set. NumPy and SciPy are used only by the suites' differential checks.
- `ruff check .` and `ruff format --check .`
- `mypy`
FastAPI + psycopg 3 over Tiger Data. Read the [backend design](../../docs/gridpulse/BACKEND_DESIGN.md) first. Algorithm authors only need [`analysis/contract.py`](src/gridpulse/analysis/contract.py) and the example in [`analysis/algorithms/`](src/gridpulse/analysis/algorithms/).

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
GRIDPULSE_INTEGRATION=1 pytest                      # also run read-only checks against Tiger Data
gridpulse migrate                                   # create/update the gridpulse schema and views (safe to repeat)
gridpulse analyse wind_price_correlation --from 2026-07-01 --to 2026-10-23
gridpulse algorithms                                # list registered algorithms
uvicorn gridpulse.main:app --reload                 # API on :8000, docs at /docs
```

Source data is loaded into the `raw` schema by the team's loader; the backend only reads it, through the `gridpulse.*` views.

Offline demo: run once with `GRIDPULSE_MODE=record`, click through the demo, then start with `GRIDPULSE_MODE=replay` (no database needed).

Keep routes thin, SQL in `db/queries.py`, orchestration in `analysis/service.py` and calculations in `analysis/algorithms/`. See the [repository structure](../../docs/gridpulse/REPOSITORY_STRUCTURE.md) and [API contract](../contracts/README.md).
