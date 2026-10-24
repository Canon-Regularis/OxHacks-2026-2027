# Operational helpers

Batch commands are the `gridpulse` CLI installed by the backend package (`gridpulse/backend/src/gridpulse/cli.py`): `migrate`, `analyse` and `algorithms`. See the [backend README](../backend/README.md) for usage. Source data is loaded into Tiger Data's `raw` schema by the team's loader, not by these commands.

Add thin shell wrappers here only if they help, for example a nightly analysis run or a demo-recording script. They should call the CLI rather than duplicate SQL or algorithms. Deployment-specific configuration belongs in `deploy/`.
