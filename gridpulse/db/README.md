# Database schema

Source data lives in the `raw` schema of Tiger Data, written by the team's data loader and described in `raw.dataset_catalog`. The backend never modifies it.

The backend's own objects live in the `gridpulse` schema. They are created by the ordered SQL migrations in `migrations/`, applied with `gridpulse migrate` from `gridpulse/backend`. The runner applies every file each time, so every statement must be re-runnable (`IF NOT EXISTS`, `CREATE OR REPLACE VIEW`, `if_not_exists => TRUE`). Statements run one at a time in autocommit mode, because TimescaleDB cannot create continuous aggregates inside a transaction. End each statement with `;` at the end of a line, and do not use `$$` bodies.

| File | Contents |
|---|---|
| `001_app_schema.sql` | `gridpulse` schema, `analysis_runs`, `analysis_results` hypertable |
| `002_series_views.sql` | `*_30m` series views and forecast-vintage views over `raw`: **the data-modelling seam** |
| `003_analysis_tables.sql` | `analysis_tables`: non-time-series results per run |

To model the data later, replace a view in `002` with a continuous aggregate or table that has the same name and columns; the backend needs no change. `CREATE OR REPLACE VIEW` cannot turn a view into a materialized view, so drop the view first in a new migration. Python connections and SQL queries belong in `gridpulse/backend/src/gridpulse/db/`. See the [backend design](../../docs/gridpulse/BACKEND_DESIGN.md#data-in-tiger-data).
