"""Actual child processes, no database/model: verify hard deadlines and cancellation cleanup."""
import asyncio
import multiprocessing
import time
from types import SimpleNamespace
import pytest
from gridpulse.agent import jobs
from gridpulse.agent.schemas import Read


def slow_worker(*args):
    time.sleep(30)


def test_job_timeout_terminates_child(monkeypatch):
    monkeypatch.setattr(jobs, '_worker', slow_worker)
    before = {p.pid for p in multiprocessing.active_children()}
    settings = SimpleNamespace(database_url='', query_timeout_seconds=1, agent_job_timeout_seconds=0.3)
    spec = Read(tool='execute', algorithm='calendar_fourier_terms', from_date='2026-10-25', to_date='2026-10-25')
    with pytest.raises(TimeoutError):
        asyncio.run(jobs.execute(settings, spec))
    assert {p.pid for p in multiprocessing.active_children()} == before


def test_job_cancellation_terminates_child(monkeypatch):
    monkeypatch.setattr(jobs, '_worker', slow_worker)
    before = {p.pid for p in multiprocessing.active_children()}
    settings = SimpleNamespace(database_url='', query_timeout_seconds=1, agent_job_timeout_seconds=30)
    spec = Read(tool='execute', algorithm='calendar_fourier_terms', from_date='2026-10-25', to_date='2026-10-25')
    async def cancel():
        task = asyncio.create_task(jobs.execute(settings, spec))
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(cancel())
    assert {p.pid for p in multiprocessing.active_children()} == before
