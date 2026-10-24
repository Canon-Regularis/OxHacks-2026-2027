import pytest
from fastapi.testclient import TestClient
from gridpulse.api import weather
from gridpulse.api.deps import get_conn
from gridpulse.config import get_settings
from gridpulse.main import create_app

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv('DATABASE_URL', '')
    monkeypatch.setenv('TIMESCALE_SERVICE_URL', '')
    get_settings.cache_clear()
    app = create_app()
    app.dependency_overrides[get_conn] = lambda: object()
    with TestClient(app) as instance:
        yield instance
    get_settings.cache_clear()


def test_weather_locations_expose_coordinates_without_secrets(client, monkeypatch):
    monkeypatch.setattr(weather.queries, 'weather_locations', lambda conn: [
        {'id':'dogger_bank','label':'Dogger Bank','latitude':54.9,'longitude':2.0}])
    body = client.get('/api/weather/locations').json()
    assert body['locations'][0]['id'] == 'dogger_bank'
    assert set(body['locations'][0]) == {'id','label','latitude','longitude'}


def test_unknown_location_is_not_interpolated_into_sql(client, monkeypatch):
    monkeypatch.setattr(weather.queries, 'weather_locations', lambda conn: [])
    response = client.get('/api/weather', params={'date':'2026-10-23','location':"unknown'; DROP TABLE raw.x;--"})
    assert response.status_code == 404
    assert response.json()['detail']['code'] == 'unknown_weather_location'


def test_missing_weather_data_preserves_clock_change_day(client, monkeypatch):
    monkeypatch.setattr(weather.queries, 'weather_locations', lambda conn: [
        {'id':'dogger_bank','label':'Dogger Bank','latitude':54.9,'longitude':2.0}])
    monkeypatch.setattr(weather.queries, 'weather_values', lambda *args: [])
    monkeypatch.setattr(weather.queries, 'hourly_weather_prices', lambda *args: [])
    response = client.get('/api/weather', params={'date':'2026-10-25','location':'dogger_bank'})
    assert response.status_code == 200
    body = response.json()
    assert len(body['points']) == 25
    assert all(p['temperature'] is None and p['temperature_samples'] == 0 for p in body['points'])
    assert body['window_hours'] == 168
