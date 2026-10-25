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


def test_comparison_cannot_use_source_series_without_an_algorithm_table():
    view=Decision.model_validate({'view':{'from_date':'2026-02-01','to_date':'2026-03-31','widgets':[{'key':'bad','type':'comparison','title':'Wind and demand','series':['wind','demand']}]}}).view
    with pytest.raises(ValueError,match='algorithm'):
        tools.validate_view(view,[])


def test_algorithm_table_requires_a_named_available_table():
    view=Decision.model_validate({'view':{'from_date':'2026-02-01','to_date':'2026-03-31','widgets':[{'key':'bad','type':'table','title':'Results','algorithm':'calendar_fourier_terms','run_id':49}]}}).view
    with pytest.raises(ValueError,match='table'):
        tools.validate_view(view,[{'data':{'run_id':49,'algorithm':'calendar_fourier_terms'}}])


def test_invalid_widget_is_corrected_before_emitting_a_view(monkeypatch):
    class Fake:
        calls=0
        async def decide(self,messages):
            self.calls+=1
            if self.calls==1:
                return Decision.model_validate({'view':{'from_date':'2026-02-01','to_date':'2026-03-31','widgets':[{'key':'bad','type':'comparison','title':'Wind and demand','series':['wind','demand']}]}})
            assert 'Application widget validation rejected' in messages[-1]['content']
            return Decision.model_validate({'answer':'Wind and demand share MW; use a time-series chart.','view':{'from_date':'2026-02-01','to_date':'2026-03-31','widgets':[{'key':'chart','type':'chart','title':'Wind and demand','series':['wind','demand']}]}})
    turn=Turn(turn_id='repair',conversation='chat',workspace_id='ws',revision='v',message='Visualise wind and demand',snapshot={})
    async def run():
        return [e async for e in service.investigate(turn,None,SimpleNamespace(max_range_days=3660),'session',Fake())]
    events=asyncio.run(run())
    views=[e['view'] for e in events if e['type']=='view']
    assert len(views)==1 and views[0]['widgets'][0]['type']=='chart'
    assert any(e['type']=='progress' and 'compatibility' in e['message'] for e in events)


def test_empty_final_answer_gets_one_presentation_repair():
    class Fake:
        calls=0
        async def decide(self,messages):
            self.calls+=1
            if self.calls==1: return Decision()
            assert 'cannot be empty' in messages[-1]['content']
            return Decision(answer='I can prepare a wind and demand chart using the selected dates.')
    turn=Turn(turn_id='explain',conversation='chat',workspace_id='ws',revision='v',message='Show wind and demand',snapshot={})
    async def run():
        return [e async for e in service.investigate(turn,None,SimpleNamespace(max_range_days=3660),'session',Fake())]
    events=asyncio.run(run())
    assert next(e['text'] for e in events if e['type']=='answer').startswith('I can prepare')


def test_chart_unit_limit_is_checked_before_the_view_reaches_the_browser():
    view=Decision.model_validate({'view':{'from_date':'2026-02-01','to_date':'2026-03-31','widgets':[{'key':'mixed','type':'chart','title':'Too many axes','series':['price','wind','niv']}]}}).view
    with pytest.raises(ValueError,match='two distinct units'):
        tools.validate_view(view,[])


def test_layout_repair_is_read_only_and_receives_frontend_error(monkeypatch):
    from gridpulse.agent.schemas import LayoutFeedback, View
    class Fake:
        calls=0
        async def decide(self,messages):
            self.calls+=1
            if self.calls==1:
                assert 'Minimum widget size reached.' in messages[0]['content']
                return Decision.model_validate({'reads':[{'tool':'execute','algorithm':'calendar_fourier_terms','from_date':'2026-02-01','to_date':'2026-02-01'}]})
            assert 'read-only' in messages[-1]['content']
            return Decision(answer='I can rearrange the existing results without running another computation.')
    async def forbidden(*args): raise AssertionError('Layout repair launched a computation')
    monkeypatch.setattr(service.jobs,'execute',forbidden)
    feedback=LayoutFeedback(attempt=1,error='Minimum widget size reached.',failed_view=View(from_date='2026-02-01',to_date='2026-03-31',widgets=[]))
    turn=Turn(turn_id='layout-repair',conversation='chat',workspace_id='ws',revision='v',message='Add a data table',snapshot={},layout_feedback=feedback)
    async def run(): return [e async for e in service.investigate(turn,None,SimpleNamespace(max_range_days=3660),'session',Fake())]
    events=asyncio.run(run())
    assert not any(e['type']=='confirmation' for e in events)
    assert events[-1]['type']=='completed'


def test_layout_feedback_cannot_approve_jobs_or_exceed_two_repairs():
    body={'turn_id':'layout-repair','conversation':'chat','workspace_id':'ws','revision':'v','message':'Add a table','snapshot':{},'layout_feedback':{'attempt':1,'error':'Minimum widget size reached.','failed_view':{'from_date':'2026-02-01','to_date':'2026-03-31','widgets':[]}}}
    with pytest.raises(ValidationError): Turn.model_validate({**body,'approval':'token'})
    with pytest.raises(ValidationError): Turn.model_validate({**body,'layout_feedback':{**body['layout_feedback'],'attempt':3}})


def test_layout_only_request_recovers_stale_evidence_without_reading_data(monkeypatch):
    class Fake:
        calls=0
        async def decide(self,messages):
            self.calls+=1
            if self.calls==1:
                return Decision.model_validate({'answer':'Prepared separate panels.','evidence':['e1'],'view':{'from_date':'2026-02-01','to_date':'2026-03-31','layout_mode':'preserve','widgets':[]}})
            assert 'evidence' in messages[-1]['content'].lower()
            return Decision.model_validate({'answer':'Prepared separate panels without changing their data.','view':{'from_date':'2026-02-01','to_date':'2026-03-31','layout_mode':'preserve','widgets':[]}})
    monkeypatch.setattr(tools,'read',lambda *_: (_ for _ in ()).throw(AssertionError('Layout-only request fetched data')))
    turn=Turn(turn_id='separate',conversation='chat',workspace_id='ws',revision='v',message='Separate the current tabs except AI',snapshot={})
    provider=Fake()
    async def run(): return [event async for event in service.investigate(turn,None,SimpleNamespace(max_range_days=3660),'session',provider)]
    events=asyncio.run(run())
    assert provider.calls==2
    assert next(e for e in events if e['type']=='answer')['references']==[]
    assert events[-1]['type']=='completed'


def test_unretrieved_numerical_evidence_is_still_rejected_after_correction():
    class Fake:
        calls=0
        async def decide(self,messages):
            self.calls+=1
            return Decision(answer='The price was 999.',evidence=['invented'])
    turn=Turn(turn_id='invalid-evidence',conversation='chat',workspace_id='ws',revision='v',message='What was the price?',snapshot={})
    provider=Fake()
    async def run(): return [e async for e in service.investigate(turn,None,SimpleNamespace(max_range_days=3660),'session',provider)]
    with pytest.raises(ValueError,match='evidence'):
        asyncio.run(run())
    assert provider.calls==2
