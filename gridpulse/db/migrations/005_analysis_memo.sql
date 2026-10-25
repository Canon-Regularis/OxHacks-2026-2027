-- Intermediate results an algorithm may reuse in its next run (see analysis/contract.py, memo):
-- today a backtest's folds, as JSON. `namespace` hashes the algorithm id, its version and the
-- adapter code, so a code change starts afresh; `key` hashes everything one entry depends on.
-- A cache only: deleting rows costs a slower next run, never a different result. Rows unused
-- for a week are deleted when a run saves.

CREATE TABLE IF NOT EXISTS gridpulse.analysis_memo (
    namespace    text        NOT NULL,
    key          text        NOT NULL,
    algorithm_id text        NOT NULL,
    value        text        NOT NULL,
    used_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (namespace, key)
);

CREATE INDEX IF NOT EXISTS analysis_memo_used_at ON gridpulse.analysis_memo (used_at);
