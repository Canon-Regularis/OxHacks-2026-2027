import json
import pytest
from fastapi.testclient import TestClient
from gridpulse.config import get_settings
from gridpulse.main import create_app
from gridpulse.agent import routes


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv('DATABASE_URL', '')
    monkeypatch.setenv('TIMESCALE_SERVICE_URL', '')
    monkeypatch.setenv('AGENT_ACCESS_TOKEN', 'test-token')
    monkeypatch.setenv('GRIDPULSE_MODE', 'live')
    get_settings.cache_clear()
    routes.ACTIVE.clear(); routes.SEEN.clear(); routes.ADMISSIONS.clear()
    with TestClient(create_app()) as client:
        yield client
    get_settings.cache_clear()


HEADERS = {'Authorization': 'Bearer test-token', 'X-Agent-Session': 'session-12345678901234567890'}
BODY = {'turn_id': 'route-turn', 'conversation': 'chat', 'workspace_id': 'workspace', 'revision': '{}', 'message': 'Hello', 'snapshot': {}}


def test_route_requires_proxy_authentication(client):
    response = client.post('/api/agent/turns/stream', json=BODY)
    assert response.status_code == 503


def test_stream_is_uncached_and_deduplicated(client, monkeypatch):
    async def fake(*args):
        yield {'version': 1, 'turn_id': 'route-turn', 'type': 'completed'}
    monkeypatch.setattr(routes, 'investigate', fake)
    response = client.post('/api/agent/turns/stream', json=BODY, headers=HEADERS)
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('text/event-stream')
    assert response.headers['cache-control'] == 'no-store'
    assert 'content-encoding' not in response.headers
    assert 'completed' in response.text
    assert not routes.ACTIVE
    repeated = client.post('/api/agent/turns/stream', json=BODY, headers=HEADERS)
    assert repeated.status_code == 409


def test_large_context_rejected_before_model(client):
    response = client.post('/api/agent/turns/stream', json={**BODY, 'snapshot': {'notes': 'x'*200001}}, headers=HEADERS)
    assert response.status_code == 413


def test_failure_releases_workspace_lock(client, monkeypatch):
    async def broken(*args):
        raise ValueError('Invalid view')
        yield
    monkeypatch.setattr(routes, 'investigate', broken)
    response = client.post('/api/agent/turns/stream', json=BODY, headers=HEADERS)
    assert 'Invalid view' in response.text
    assert not routes.ACTIVE
