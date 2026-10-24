"""API behaviour that needs no database: health, validation, resolution choice and demo replay."""

from datetime import date

import pytest
from fastapi.testclient import TestClient

from gridpulse.api.deps import DateRange, choose_resolution
from gridpulse.config import get_settings
from gridpulse.main import create_app
from gridpulse.settlement import range_utc


def make_client(monkeypatch, **env) -> TestClient:
    # Blank both names so a local credentials file cannot connect these tests to a real database.
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("TIMESCALE_SERVICE_URL", "")
    for name, value in env.items():
        monkeypatch.setenv(name, str(value))
    get_settings.cache_clear()
    return TestClient(create_app())


@pytest.fixture(autouse=True)
def reset_settings():
    yield
    get_settings.cache_clear()


def test_health_without_database(monkeypatch):
    with make_client(monkeypatch) as client:
        body = client.get("/api/health").json()
    assert body["status"] == "degraded"
    assert body["database"] == "not_configured"


def test_data_endpoints_report_missing_database(monkeypatch):
    with make_client(monkeypatch) as client:
        response = client.get("/api/series", params={"from": "2026-10-24", "to": "2026-10-25"})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "database_unavailable"


def test_reversed_range_rejected_before_touching_database(monkeypatch):
    with make_client(monkeypatch) as client:
        response = client.get("/api/series", params={"from": "2026-10-25", "to": "2026-10-24"})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_range"


def _range(days: int) -> DateRange:
    start, end = range_utc(date(2026, 1, 1), date.fromordinal(date(2026, 1, 1).toordinal() + days - 1))
    return DateRange(date(2026, 1, 1), date(2026, 1, 1), start, end)


@pytest.mark.parametrize("days, expected", [(1, "30m"), (90, "30m"), (200, "1h"), (1000, "1d")])
def test_auto_resolution_fits_point_budget(days, expected):
    assert choose_resolution(_range(days), "auto") == expected


def test_explicit_resolution_too_fine_is_rejected():
    with pytest.raises(Exception) as error:
        choose_resolution(_range(1000), "30m")
    assert error.value.detail["code"] == "too_many_points"


def test_record_then_replay_without_database(monkeypatch, tmp_path):
    def add_ping(client):
        client.app.get("/api/ping")(lambda: {"pong": True})
        return client

    with add_ping(make_client(monkeypatch, GRIDPULSE_MODE="record", DEMO_CACHE_DIR=tmp_path)) as client:
        assert client.get("/api/ping?b=2&a=1").json() == {"pong": True}
    assert len(list(tmp_path.glob("*.json"))) == 1

    with make_client(monkeypatch, GRIDPULSE_MODE="replay", DEMO_CACHE_DIR=tmp_path) as client:
        replayed = client.get("/api/ping?a=1&b=2")  # same query, different order
        missing = client.get("/api/ping?a=9")
        health = client.get("/api/health").json()
    assert replayed.json() == {"pong": True}
    assert replayed.headers["X-GridPulse-Mode"] == "replay"
    assert missing.json()["detail"]["code"] == "not_recorded"
    assert health == {**health, "mode": "replay", "database": "not_used", "status": "ok"}


@pytest.mark.parametrize(
    "params, code",
    [
        ({"algorithm": "nope"}, "unknown_algorithm"),
        ({"algorithm": "price_naive_backtest", "params": "[1]"}, "invalid_params"),
        ({"algorithm": "price_naive_backtest", "params": "{bad json"}, "invalid_params"),
        ({"algorithm": "price_naive_backtest", "params": '{"horizon": 1}'}, "unknown_param"),
        ({"algorithm": "price_naive_backtest", "params": "{}", "run_id": 3}, "invalid_params"),
    ],
)
def test_run_selection_rejected_before_touching_database(monkeypatch, params, code):
    with make_client(monkeypatch) as client:
        response = client.get("/api/analysis", params={"from": "2026-10-24", "to": "2026-10-24", **params})
    assert response.json()["detail"]["code"] == code
    assert response.status_code == (404 if code == "unknown_algorithm" else 400)


def test_known_params_reach_the_database(monkeypatch):
    query = {"algorithm": "price_naive_backtest", "from": "2026-10-24", "to": "2026-10-24",
             "params": '{"horizon_periods": 24}'}
    with make_client(monkeypatch) as client:
        response = client.get("/api/analysis", params=query)
    assert response.json()["detail"]["code"] == "database_unavailable"
