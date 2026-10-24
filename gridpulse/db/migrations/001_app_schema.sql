-- The backend's own schema. Source data stays in `raw` (written by the data loader) and is never
-- modified here. Every migration is re-runnable and is applied with `gridpulse migrate`.
-- The runner splits statements on a `;` at the end of a line, so keep one statement per `;`.

CREATE SCHEMA IF NOT EXISTS gridpulse;

-- Stored algorithm runs and their period-level outputs.
-- The API only serves results from runs with status 'succeeded'.
CREATE TABLE IF NOT EXISTS gridpulse.analysis_runs (
    run_id            bigserial   PRIMARY KEY,
    algorithm_id      text        NOT NULL,
    algorithm_version text        NOT NULL,
    params            jsonb       NOT NULL DEFAULT '{}',
    range_start_utc   timestamptz NOT NULL,
    range_end_utc     timestamptz NOT NULL,
    cutoff_utc        timestamptz,              -- NULL means retrospective (all data)
    status            text        NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    error             text,
    result_rows       bigint,
    started_at_utc    timestamptz NOT NULL DEFAULT now(),
    finished_at_utc   timestamptz
);

CREATE INDEX IF NOT EXISTS analysis_runs_lookup
    ON gridpulse.analysis_runs (algorithm_id, status, finished_at_utc DESC);

-- Long format: one row per (run, period, indicator). NULL value means missing, not zero.
CREATE TABLE IF NOT EXISTS gridpulse.analysis_results (
    run_id             bigint           NOT NULL,
    delivery_start_utc timestamptz      NOT NULL,
    indicator          text             NOT NULL,
    value              double precision,
    PRIMARY KEY (run_id, delivery_start_utc, indicator)
);

SELECT create_hypertable('gridpulse.analysis_results', by_range('delivery_start_utc', INTERVAL '30 days'),
                         if_not_exists => TRUE);
