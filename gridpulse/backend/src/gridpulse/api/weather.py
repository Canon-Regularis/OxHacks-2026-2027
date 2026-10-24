"""Bounded, read-only weather context from the existing Tiger Data archived forecasts."""
from datetime import date, timedelta
from fastapi import APIRouter, Depends, Query
from psycopg import Connection
from psycopg.errors import UndefinedTable
from gridpulse.analysis.weather import WINDOW_HOURS, weather_statistics
from gridpulse.api.deps import api_error, get_conn
from gridpulse.api.schemas import WeatherLocation, WeatherLocations, WeatherResponse
from gridpulse.db import queries
from gridpulse.settlement import range_utc

router = APIRouter(tags=["weather"])
SOURCE = "Open-Meteo archived model forecasts · retrospective · temperature °C / sea-level pressure hPa"


def _locations(conn: Connection) -> list[WeatherLocation]:
    try:
        return [WeatherLocation(**row) for row in queries.weather_locations(conn)]
    except UndefinedTable as exc:
        raise api_error(503, "weather_not_configured", "Weather view is unavailable; apply database migration 004.") from exc


@router.get("/weather/locations", response_model=WeatherLocations)
def locations(conn: Connection = Depends(get_conn)) -> WeatherLocations:
    rows = _locations(conn)
    return WeatherLocations(locations=rows, warnings=[] if rows else ["No weather records are loaded."])


@router.get("/weather", response_model=WeatherResponse)
def weather(date_: date = Query(alias="date"), location: str = Query(min_length=1, max_length=100),
            conn: Connection = Depends(get_conn)) -> WeatherResponse:
    selected = next((item for item in _locations(conn) if item.id == location), None)
    if selected is None:
        raise api_error(404, "unknown_weather_location", "Choose a location from /api/weather/locations.")
    start, end = range_utc(date_, date_)
    read_start = start - timedelta(hours=WINDOW_HOURS + 1)
    rows = queries.weather_values(conn, location, read_start, end)
    prices = queries.hourly_weather_prices(conn, read_start, end)
    return WeatherResponse(location=selected, points=weather_statistics(rows, prices, start, end),
                           source=SOURCE, window_hours=WINDOW_HOURS,
                           warnings=["Completed UTC hours only; minimum 24 valid paired changes for correlation. "
                                     "Archived forecasts have no verified publication timestamp here; this is retrospective context."])
