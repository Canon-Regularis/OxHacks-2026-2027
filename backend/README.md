# Backend work area

Data/database owner: `ingestion/`, `db/` and root `db/migrations/`.
Algorithm/backend owner: `analysis/`, `api/`, application wiring and later `agent/`.

Python code lives in `src/gridpulse/`. All modules are documentation-only stubs: no server, database connection or algorithm is implemented yet. Choose the HTTP framework and database driver when integrating real data; add dependencies to `pyproject.toml` and commit the chosen lockfile.

Create a local environment with `python3 -m venv .venv` from this directory. After activation, `python -m pip install -e .` installs the package for development. Installation does not start a server. Add the actual launch command here once application wiring exists.

Keep routes thin, SQL in `db/`, orchestration in `analysis/service.py` and calculation code in `analysis/algorithms/`. Refer to the [repository structure](../docs/gridpulse/REPOSITORY_STRUCTURE.md) and [API contract](../contracts/README.md).
