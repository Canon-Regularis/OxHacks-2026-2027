-- Results of a run that are not one value per settlement period: ACF by lag, metric scorecards,
-- fitted parameters, residual diagnostics, backtest fold detail.
-- One row per (run, table). `data` is {"columns": [...], "rows": [[...], ...]}, as served by the API.

CREATE TABLE IF NOT EXISTS gridpulse.analysis_tables (
    run_id    bigint NOT NULL,
    name      text   NOT NULL,
    row_count bigint NOT NULL,
    data      jsonb  NOT NULL,
    PRIMARY KEY (run_id, name)
);
