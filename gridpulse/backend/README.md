# Backend work area

Data/database owner: `ingestion/`, `db/` and `gridpulse/db/migrations/` (relative to the repository root).
Algorithm/backend owner: `analysis/`, `api/`, application wiring and later `agent/`.

Python code lives in `src/gridpulse/`. `analysis/algorithms/` holds the Orchid P0 adapters (pure calculations, no I/O). All other modules are documentation-only stubs: no server or database connection is implemented yet. Choose the HTTP framework and database driver when integrating real data; add dependencies to `pyproject.toml` and commit the chosen lockfile.

Create a local environment with `python3 -m venv .venv` from this directory. After activation, `python -m pip install -e ".[dev]"` installs the package with the development tools. Installation does not start a server. Add the actual launch command here once application wiring exists.

Checks, run from this directory:

- `pytest` runs the adapter tests and the ported Orchid suites. It takes about 1.5 minutes, and 31 slow smoothing tests are skipped unless `RUNSLOW=1` is set. NumPy and SciPy are used only by the suites' differential checks.
- `ruff check .` and `ruff format --check .`
- `mypy`

Keep routes thin, SQL in `db/`, orchestration in `analysis/service.py` and calculation code in `analysis/algorithms/`. Refer to the [repository structure](../../docs/gridpulse/REPOSITORY_STRUCTURE.md) and [API contract](../contracts/README.md).
