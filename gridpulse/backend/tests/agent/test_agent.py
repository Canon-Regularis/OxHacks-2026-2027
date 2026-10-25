import asyncio
from types import SimpleNamespace
import pytest
from pydantic import ValidationError
from gridpulse.agent.schemas import Decision, Read, Turn
from gridpulse.agent import tools, service


def test_contract_rejects_code_and_invalid_ranges():
    with pytest.raises(ValidationError):
        Decision.model_validate({"javascript": "alert(1)"})
    with pytest.raises(ValidationError):
        Read(tool="execute", algorithm="x", from_date="2026-10-26", to_date="2026-10-25")


def test_tools_reject_unknown_algorithms_and_parameters():
    settings = SimpleNamespace(max_range_days=3660)
    with pytest.raises(ValueError, match="Unknown algorithm"):
        tools.validate(Read(tool="execute", algorithm="invented", from_date="2026-10-25", to_date="2026-10-25"), settings)
    with pytest.raises(ValueError):
        tools.validate(Read(tool="execute", algorithm="price_naive_backtest", params={"invented": 3}, from_date="2026-10-25", to_date="2026-10-25"), settings)


def test_model_cannot_fabricate_run():
    decision = Decision.model_validate({"view": {"from_date": "2026-10-25", "to_date": "2026-10-25", "widgets": [{"key": "a", "type": "table", "title": "Results", "algorithm": "price_naive_backtest", "run_id": 99}]}})
    with pytest.raises(ValueError, match="Inspect a stored run"):
        tools.validate_view(decision.view, [])


def test_tool_loop_uses_evidence_and_finishes(monkeypatch):
    class Fake:
        calls = 0
        async def decide(self, messages):
            self.calls += 1
            if self.calls == 1:
                return Decision.model_validate({"reads": [{"tool": "series", "ids": ["price"], "from_date": "2026-10-25", "to_date": "2026-10-25"}]})
            assert 'e1' in messages[-1]['content']
            return Decision(answer="Retrieved price data.", evidence=["e1"])
    monkeypatch.setattr(tools, "read", lambda *_: {"series": {"price": {"values": [20]}}})
    turn = Turn(turn_id="t1", conversation="c1", workspace_id="w1", revision="v1", message="Show prices", snapshot={})
    async def run():
        return [event async for event in service.investigate(turn, None, SimpleNamespace(max_range_days=3660), "session", Fake())]
    events = asyncio.run(run())
    assert [e['type'] for e in events] == ['started', 'progress', 'evidence', 'answer', 'completed']
    assert events[-2]['references'] == ['e1']


def test_foreign_approval_is_rejected():
    service.APPROVALS['token'] = ('another-session', 'c1', float('inf'), None)
    turn = Turn(turn_id="t1", conversation="c1", workspace_id="w1", revision="v1", message="Yes", snapshot={}, approval="token")
    async def run():
        return [event async for event in service.investigate(turn, None, None, "session")]
    with pytest.raises(ValueError, match="another conversation"):
        asyncio.run(run())


def test_expensive_analysis_waits_for_user_confirmation(monkeypatch):
    class Fake:
        async def decide(self, messages):
            return Decision.model_validate({"reads": [{"tool": "execute", "algorithm": "price_naive_backtest", "from_date": "2026-10-25", "to_date": "2026-10-25"}]})
    monkeypatch.setattr(tools, "read", lambda *_: {"error": "no_run"})
    async def forbidden(*args):
        raise AssertionError("Unapproved job executed")
    monkeypatch.setattr(service.jobs, "execute", forbidden)
    turn = Turn(turn_id="confirm", conversation="c1", workspace_id="w1", revision="v1", message="Investigate", snapshot={})
    async def run():
        return [event async for event in service.investigate(turn, None, SimpleNamespace(max_range_days=3660), "session", Fake())]
    events = asyncio.run(run())
    confirmation = next(e for e in events if e['type'] == 'confirmation')
    assert confirmation['request']['algorithm'] == 'price_naive_backtest'
    assert service.APPROVALS[confirmation['approval']][:2] == ('session', 'c1')
    assert events[-1]['type'] == 'completed'


def test_matching_result_does_not_launch_job(monkeypatch):
    class Fake:
        calls = 0
        async def decide(self, messages):
            self.calls += 1
            if self.calls == 1:
                return Decision.model_validate({"reads": [{"tool": "execute", "algorithm": "price_naive_backtest", "from_date": "2026-10-25", "to_date": "2026-10-25"}]})
            return Decision(answer="Existing result reused.", evidence=['e1'])
    monkeypatch.setattr(tools, "read", lambda *_: {"run": {"run_id": 7, "algorithm_id": "price_naive_backtest"}})
    async def forbidden(*args):
        raise AssertionError("Cached result recomputed")
    monkeypatch.setattr(service.jobs, "execute", forbidden)
    turn = Turn(turn_id="cached", conversation="c1", workspace_id="w1", revision="v1", message="Investigate", snapshot={})
    async def run():
        return [event async for event in service.investigate(turn, None, SimpleNamespace(max_range_days=3660), "session", Fake())]
    events = asyncio.run(run())
    assert not any(e['type'] == 'confirmation' for e in events)
    assert events[-1]['type'] == 'completed'


def test_single_day_evidence_retains_settlement_resolution(monkeypatch):
    from contextlib import nullcontext
    from gridpulse.api import market
    observed=[]
    def series(dates, ids, resolution, cutoff, conn):
        observed.append(resolution)
        return SimpleNamespace(model_dump=lambda **_: {'resolution':resolution,'series':{}})
    monkeypatch.setattr(market,'series',series)
    pool=SimpleNamespace(connection=lambda **_: nullcontext(object()))
    spec=Read(tool='series',ids=['price','wind'],from_date='2026-10-25',to_date='2026-10-25')
    result=tools.read(pool,spec,SimpleNamespace(max_range_days=3660))
    assert result['resolution']=='30m'
    assert observed==['30m']
