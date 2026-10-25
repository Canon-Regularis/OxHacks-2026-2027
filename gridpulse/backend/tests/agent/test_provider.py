"""Replay a minimised real Kimi failure: it included protected chat in a view."""
import asyncio
import json
from types import SimpleNamespace
import httpx
import pytest
from gridpulse.agent.provider import Provider, ProviderError


def response(content):
    return httpx.Response(200, request=httpx.Request('POST', 'https://provider.invalid/v1/chat/completions'), json={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(content)}}]})


def test_invalid_chat_widget_gets_one_schema_repair(monkeypatch):
    invalid = {'reads':[{'tool':'series','ids':['price','wind'],'from_date':'2026-10-23','to_date':'2026-10-23'}], 'view':{'from_date':'2026-10-23','to_date':'2026-10-23','widgets':[{'key':'chat','type':'chat','title':'AI chat'}]}}
    payloads=[]
    async def post(self, *args, **kwargs):
        payloads.append(kwargs['json'])
        return response(invalid if len(payloads)==1 else {'reads':invalid['reads']})
    monkeypatch.setattr(httpx.AsyncClient,'post',post)
    provider=Provider(SimpleNamespace(cwave_api_key='test',agent_base_url='https://provider.invalid/v1',agent_model='test',agent_max_tokens=1000))
    result=asyncio.run(provider.decide([{'role':'user','content':'Show price and wind'}]))
    assert result.reads[0].ids == ['price','wind']
    assert len(payloads)==2
    assert 'view.widgets.0.type' in payloads[1]['messages'][-1]['content']


def test_repeated_invalid_decisions_fail_without_executing(monkeypatch):
    calls=[]
    async def post(self,*args,**kwargs):
        calls.append(1)
        return response({'execute_code':'bad'})
    monkeypatch.setattr(httpx.AsyncClient,'post',post)
    provider=Provider(SimpleNamespace(cwave_api_key='test',agent_base_url='https://provider.invalid/v1',agent_model='test',agent_max_tokens=1000))
    with pytest.raises(ProviderError):
        asyncio.run(provider.decide([{'role':'user','content':'Hello'}]))
    assert len(calls)==2
