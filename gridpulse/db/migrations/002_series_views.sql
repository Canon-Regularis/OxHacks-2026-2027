-- The seam between raw data and the API.
--
-- Each *_30m view has one row per settlement period: `bucket` (period start, UTC) plus named value
-- columns. The API reads only these views (see backend/src/gridpulse/catalog.py) and buckets them to
-- 1h/1d on the fly. Prices and outturns are plain views over `raw`, one row per period already.
-- The latest-forecast views read continuous aggregates, because picking the latest of ~60 NDF
-- versions per period on every request made forecast series 5-25x slower than prices.
--
-- "Latest forecast" means the last version published at or before the period starts. Later versions
-- are excluded: each WINDFOR publish re-issues the previous ~27 hours, and a value published after
-- delivery is not a forecast. The *_vintages views keep every version for inspection.

-- Prices and imbalance volume (DISEBSP). GB has used a single imbalance price since 2015, so the
-- system sell price is "the price".
CREATE OR REPLACE VIEW gridpulse.price_30m AS
SELECT start_time           AS bucket,
       system_sell_price    AS price_gbp_mwh,
       net_imbalance_volume AS niv_mwh
FROM raw.elexon_system_prices;

-- Every published wind forecast version (WINDFOR) on the settlement-period grid. WINDFOR is hourly,
-- so each hourly value is applied to both settlement periods in that hour.
-- source_start_utc is the raw row's start_time. A filter on delivery_start_utc cannot skip chunks
-- for the shifted half-hour, so range queries should also bound source_start_utc.
CREATE OR REPLACE VIEW gridpulse.wind_forecast_vintages AS
SELECT start_time AS delivery_start_utc, publish_time AS published_at_utc, generation AS wind_mw,
       start_time AS source_start_utc
FROM raw.elexon_windfor
UNION ALL
SELECT start_time + INTERVAL '30 minutes', publish_time, generation, start_time
FROM raw.elexon_windfor;

-- Every published national demand forecast version (NDF, national boundary only).
CREATE OR REPLACE VIEW gridpulse.demand_forecast_vintages AS
SELECT start_time AS delivery_start_utc, publish_time AS published_at_utc, demand AS demand_mw,
       start_time AS source_start_utc
FROM raw.elexon_ndf
WHERE boundary = 'N';

-- Latest pre-delivery forecasts, kept by TimescaleDB. The refresh policies only recompute ranges
-- the loader has written to (including backfills and revisions), and real-time aggregation adds
-- periods newer than the last refresh, so readers never see a stale value.
-- WINDFOR is hourly: each hour carries the latest value for both of its settlement periods, since
-- a version published during the first half-hour counts for the second.
CREATE MATERIALIZED VIEW IF NOT EXISTS gridpulse.wind_forecast_latest_1h
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT time_bucket(INTERVAL '1 hour', start_time) AS hour_start_utc,
       last(generation, publish_time) FILTER (WHERE publish_time <= start_time) AS first_half_mw,
       last(generation, publish_time) AS second_half_mw
FROM raw.elexon_windfor
WHERE publish_time <= start_time + INTERVAL '30 minutes'
GROUP BY 1
WITH DATA;

CREATE MATERIALIZED VIEW IF NOT EXISTS gridpulse.demand_forecast_latest_30m
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT time_bucket(INTERVAL '30 minutes', start_time) AS bucket, last(demand, publish_time) AS demand_mw
FROM raw.elexon_ndf
WHERE boundary = 'N' AND publish_time <= start_time
GROUP BY 1
WITH DATA;

SELECT add_continuous_aggregate_policy('gridpulse.wind_forecast_latest_1h', start_offset => NULL,
    end_offset => INTERVAL '1 hour', schedule_interval => INTERVAL '30 minutes', if_not_exists => TRUE);

SELECT add_continuous_aggregate_policy('gridpulse.demand_forecast_latest_30m', start_offset => NULL,
    end_offset => INTERVAL '1 hour', schedule_interval => INTERVAL '30 minutes', if_not_exists => TRUE);

CREATE OR REPLACE VIEW gridpulse.wind_30m AS
SELECT hour_start_utc AS bucket, first_half_mw AS wind_mw
FROM gridpulse.wind_forecast_latest_1h
UNION ALL
SELECT hour_start_utc + INTERVAL '30 minutes', second_half_mw
FROM gridpulse.wind_forecast_latest_1h;

CREATE OR REPLACE VIEW gridpulse.demand_30m AS
SELECT bucket, demand_mw FROM gridpulse.demand_forecast_latest_30m;

-- Outturns. FUELHH WIND excludes embedded wind. NESO demand arrives about 21 days in arrears.
CREATE OR REPLACE VIEW gridpulse.wind_outturn_30m AS
SELECT start_time AS bucket, generation AS wind_outturn_mw
FROM raw.elexon_fuelhh
WHERE fuel_type = 'WIND';

CREATE OR REPLACE VIEW gridpulse.demand_outturn_30m AS
SELECT start_time AS bucket, nd AS demand_outturn_mw
FROM raw.neso_historic_demand
WHERE forecast_actual_indicator IS DISTINCT FROM 'F';

-- Residual demand: latest demand forecast minus latest wind forecast.
-- Postgres does not carry a time-range filter across a join, so the wind side of this join reads
-- its whole aggregate (one row per hour) on every request. That is cheap now it is materialised.
CREATE OR REPLACE VIEW gridpulse.residual_30m AS
SELECT d.bucket, d.demand_mw - w.wind_mw AS residual_demand_mw
FROM gridpulse.demand_30m d
JOIN gridpulse.wind_30m w ON w.bucket = d.bucket;
