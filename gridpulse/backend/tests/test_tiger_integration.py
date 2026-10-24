"""End-to-end checks against the real Tiger Data instance. Read-only.

Skipped unless GRIDPULSE_INTEGRATION=1 and a database URL is configured. Run after `gridpulse migrate`:

    GRIDPULSE_INTEGRATION=1 pytest tests/test_tiger_integration.py
"""

import os

import pytest
from fastapi.testclient import TestClient

from gridpulse.config import get_settings
from gridpulse.main import create_app

pytestmark = pytest.mark.skipif(
    os.environ.get("GRIDPULSE_INTEGRATION") != "1" or not get_settings().database_url,
    reason="set GRIDPULSE_INTEGRATION=1 and DATABASE_URL to run against Tiger Data",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as client:
        yield client


def test_database_is_reachable(client):
    assert client.get("/api/health").json()["database"] == "ok"


def test_catalog_lists_raw_datasets(client):
    datasets = {d["dataset"] for d in client.get("/api/catalog").json()["datasets"]}
    assert {"elexon_system_prices", "elexon_windfor", "elexon_ndf"} <= datasets


def test_every_series_reads_on_a_clock_change_day(client):
    ids = ",".join(s["id"] for s in client.get("/api/catalog").json()["series"])
    body = client.get("/api/series", params={"ids": ids, "from": "2025-10-26", "to": "2025-10-26"}).json()
    assert len(body["t"]) == 50 and body["settlement_period"][-1] == 50
    assert all(v is not None for v in body["series"]["price"]["values"])


def test_tables_of_latest_runs(client):
    for algorithm in client.get("/api/catalog").json()["algorithms"]:
        if algorithm["latest_run"] and algorithm["tables"]:
            body = client.get(f"/api/analysis/runs/{algorithm['latest_run']['run_id']}/tables").json()
            assert set(body["tables"]) == set(algorithm["tables"])
    assert client.get("/api/analysis/runs/999999999/tables").json()["detail"]["code"] == "no_run"


def test_snapshot_and_period_detail(client):
    snapshot = client.get("/api/snapshot").json()
    period = client.get(f"/api/periods/{snapshot['settlement_date']}/{snapshot['settlement_period']}")
    assert period.status_code == 200
    assert period.json()["values"]["price"] == snapshot["items"][0]["value"]
