-- Existing archived Open-Meteo model forecasts. Publication time is not stored here;
-- this view is retrospective context, not proof of forecast availability at a past cutoff.
-- Values use the source's standard units: temperature_2m in Celsius and pressure_msl in hPa.
CREATE OR REPLACE VIEW gridpulse.weather_hourly AS
SELECT valid_time AS bucket, location, lat AS latitude, lon AS longitude,
       temperature_2m AS temperature_c, pressure_msl AS pressure_hpa
FROM raw.openmeteo_previous_runs;
