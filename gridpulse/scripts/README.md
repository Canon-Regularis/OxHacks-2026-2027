# Operational helpers

Batch commands are the `gridpulse` CLI installed by the backend package (`gridpulse/backend/src/gridpulse/cli.py`): `migrate`, `analyse`, `refresh` and `algorithms`. See the [backend README](../backend/README.md) for usage. Source data is loaded into Tiger Data's `raw` schema by the team's loader, not by these commands.

- [`record_demo.py`](record_demo.py): records API responses for the offline demo without a UI, for a date range and optionally an exact list of the frontend's requests. Run it with the backend's virtual environment; see its docstring.

Scheduled analysis runs are the *Refresh analysis* GitHub workflow calling `gridpulse refresh`, not a script here. Add thin wrappers here only if they help. They should call the CLI rather than duplicate SQL or algorithms. Deployment-specific configuration belongs in `deploy/`.
