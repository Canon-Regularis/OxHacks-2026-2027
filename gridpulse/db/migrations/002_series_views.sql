-- The seam between raw data and the API.
--
-- Each *_30m view has one row per settlement period: `bucket` (period start, UTC) plus named value
-- columns. The API reads only these views (see backend/src/gridpulse/catalog.py) and buckets them to
-- 1h/1d on the fly. They are plain views over `raw` for now. Once the team decides how to model the
-- data, a view can be replaced by a continuous aggregate or table of the same name and columns,
-- with no backend change.
--
-- "Latest forecast" means the last version published for that period, as in the data loaded today.

-- Prices and imbalance volume (DISEBSP). GB has used a single imbalance price since 2015, so the
-- system sell price is "the price".
CREATE OR REPLACE VIEW gridpulse.price_30m AS
SELECT start_time           AS bucket,
       system_sell_price    AS price_gbp_mwh,
       net_imbalance_volume AS niv_mwh
FROM raw.elexon_system_prices;

-- Every published wind forecast version (WINDFOR) on the settlement-period grid. WINDFOR is hourly,
-- so each hourly value is applied to both settlement periods in that hour.
CREATE OR REPLACE VIEW gridpulse.wind_forecast_vintages AS
SELECT start_time AS delivery_start_utc, publish_time AS published_at_utc, generation AS wind_mw
FROM raw.elexon_windfor
UNION ALL
SELECT start_time + INTERVAL '30 minutes', publish_time, generation
FROM raw.elexon_windfor;

-- Every published national demand forecast version (NDF, national boundary only).
CREATE OR REPLACE VIEW gridpulse.demand_forecast_vintages AS
SELECT start_time AS delivery_start_utc, publish_time AS published_at_utc, demand AS demand_mw
FROM raw.elexon_ndf
WHERE boundary = 'N';

CREATE OR REPLACE VIEW gridpulse.wind_30m AS
SELECT delivery_start_utc AS bucket, last(wind_mw, published_at_utc) AS wind_mw
FROM gridpulse.wind_forecast_vintages
GROUP BY delivery_start_utc;

CREATE OR REPLACE VIEW gridpulse.demand_30m AS
SELECT delivery_start_utc AS bucket, last(demand_mw, published_at_utc) AS demand_mw
FROM gridpulse.demand_forecast_vintages
GROUP BY delivery_start_utc;

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
-- Written as one aggregate over both sources rather than a join of demand_30m and wind_30m:
-- Postgres does not carry a time-range filter across a join, so the join version aggregated
-- the whole of one table on every request.
CREATE OR REPLACE VIEW gridpulse.residual_30m AS
SELECT delivery_start_utc AS bucket,
       last(value, published_at_utc) FILTER (WHERE source = 'demand')
     - last(value, published_at_utc) FILTER (WHERE source = 'wind') AS residual_demand_mw
FROM (
    SELECT 'demand' AS source, delivery_start_utc, published_at_utc, demand_mw AS value
    FROM gridpulse.demand_forecast_vintages
    UNION ALL
    SELECT 'wind', delivery_start_utc, published_at_utc, wind_mw
    FROM gridpulse.wind_forecast_vintages
) AS forecasts
GROUP BY delivery_start_utc;
