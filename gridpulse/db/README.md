# Database schema

Data/database owner: add ordered SQL migrations under `migrations/` after validating real CSV samples. Use names such as `001_initial_schema.sql`; choose and document a migration runner before applying them. No schema is implemented yet.

This directory holds schema changes; Python connections and SQL queries belong in `gridpulse/backend/src/gridpulse/db/`. Keep credentials in server environment configuration. See the [data pipeline](../../docs/gridpulse/DATA_PIPELINE.md) for proposed tables and identity rules.
