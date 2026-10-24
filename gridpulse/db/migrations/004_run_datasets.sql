-- The raw datasets each run read, with each one's raw.dataset_catalog load time when the run
-- started: {"elexon_system_prices": "2026-10-24T14:17:03+00:00", ...}. A later load of the same
-- dataset means the run may predate new or revised data. NULL for runs made before this column.

ALTER TABLE gridpulse.analysis_runs ADD COLUMN IF NOT EXISTS datasets jsonb;
