"""Bounded evidence-first orchestration; layout application remains a browser concern."""
import asyncio
import json
import secrets
import time
from gridpulse.agent import jobs, tools
from gridpulse.agent.provider import Provider
from gridpulse.agent.schemas import Decision

from gridpulse.agent.prompts import POLICY

# Process-local, ephemeral approvals. Ownership and expiry are checked on every use.
APPROVALS: dict[str, tuple[str, str, float, object]] = {}


async def investigate(turn, pool, settings, session, provider=None):
    evidence = []
    sequence = 0

    def event(kind, **data):
        nonlocal sequence
        sequence += 1
        return {"version": 1, "turn_id": turn.turn_id, "sequence": sequence, "type": kind, **data}

    yield event("started", model=getattr(settings, "agent_model", ""))
    if turn.approval:
        saved = APPROVALS.get(turn.approval)
        if not saved or saved[:2] != (session, turn.conversation) or saved[2] < time.monotonic():
            raise ValueError("Computation approval expired or belongs to another conversation.")
        APPROVALS.pop(turn.approval, None)
        spec = saved[3]
        yield event("progress", message=f"Running approved {spec.algorithm}")
        result = await jobs.execute(settings, spec)
        item = {"id": "e1", "label": f"{spec.algorithm}: {spec.from_date} to {spec.to_date}", "request": spec.model_dump(mode="json"), "data": result}
        evidence.append(item)
        yield event("evidence", evidence=item)

    context = {"evidence_scope": {"initial_ids": [e["id"] for e in evidence], "history_ids_are_not_current_evidence": True}, "layout_feedback": turn.layout_feedback.model_dump(mode="json") if turn.layout_feedback else None, "capabilities": tools.capabilities(), "workspace": turn.snapshot,
               "recent_states": turn.recent_states, "older_conversation_summary": turn.summary,
               "schema": Decision.model_json_schema()}
    messages = [{"role": "system", "content": POLICY + "\n" + json.dumps(context)}]
    messages.extend(m.model_dump() for m in turn.history)
    messages.append({"role": "user", "content": turn.message})
    if evidence:
        messages.append({"role": "user", "content": "Application tool evidence: " + json.dumps(evidence)})
    provider = provider or Provider(settings)
    calls = 0
    presentation_repairs = 1
    for _ in range(6):
        decision = await provider.decide(messages)
        if not decision.reads:
            try:
                tools.validate_view(decision.view, evidence)
                if any(ref not in {e["id"] for e in evidence} for ref in decision.evidence):
                    raise ValueError("Answer referenced evidence that was not retrieved in this turn. Available IDs: " + json.dumps([e["id"] for e in evidence]) + ". Pure layout changes need evidence=[] and no data reads; numerical claims require freshly retrieved evidence.")
                if not decision.answer.strip():
                    raise ValueError("The final answer must briefly explain the prepared view or findings; it cannot be empty. Do not claim application before client acknowledgement.")
            except ValueError as exc:
                if not presentation_repairs:
                    raise
                presentation_repairs -= 1
                messages.extend([
                    {"role": "assistant", "content": decision.model_dump_json()},
                    {"role": "user", "content": "Application widget validation rejected this view: " + str(exc) + ". Correct the configuration or evidence references. For layout-only requests use evidence=[] without unnecessary data reads. For analytical claims retrieve the required evidence. Return a corrected Decision."},
                ])
                yield event("progress", message="Checking widget compatibility and refining the view")
                continue
            yield event("answer", text=decision.answer, details=decision.details, references=decision.evidence)
            if decision.view:
                yield event("view", view=decision.view.model_dump(mode="json"), workspace_id=turn.workspace_id, revision=turn.revision)
            yield event("completed")
            return
        messages.append({"role": "assistant", "content": decision.model_dump_json()})
        for spec in decision.reads:
            calls += 1
            if calls > 12:
                raise ValueError("Tool budget reached; narrow the question.")
            tools.validate(spec, settings)
            yield event("progress", message=f"Inspecting {spec.algorithm or spec.tool}")
            try:
                if turn.layout_feedback and spec.tool == "execute":
                    result = {"error": "Layout repair is read-only. Reuse/read existing results; do not execute or request approval for another computation."}
                elif spec.tool == "execute":
                    # Reuse exact matching results before creating a job.
                    existing = await asyncio.to_thread(tools.read, pool, spec.model_copy(update={"tool": "analysis"}), settings)
                    if "run" in existing:
                        result = existing
                    elif spec.algorithm not in tools.AUTOMATIC or (spec.to_date - spec.from_date).days > 7 or spec.params:
                        now = time.monotonic()
                        for key in list(APPROVALS):
                            if APPROVALS[key][2] < now:
                                APPROVALS.pop(key, None)
                        if len(APPROVALS) >= 100:
                            raise ValueError("Too many pending computations; try again later.")
                        token = secrets.token_urlsafe(24)
                        APPROVALS[token] = (session, turn.conversation, now + 300, spec)
                        yield event("confirmation", approval=token, request=spec.model_dump(mode="json"), message="This analysis requires confirmation. It may take up to the configured job timeout.")
                        yield event("completed")
                        return
                    else:
                        result = await jobs.execute(settings, spec)
                else:
                    result = await asyncio.wait_for(asyncio.to_thread(tools.read, pool, spec, settings), timeout=30)
            except (ValueError, TimeoutError) as exc:
                result = {"error": str(exc)}
            item = {"id": f"e{len(evidence)+1}", "label": f"{spec.algorithm or spec.tool}: {spec.from_date} to {spec.to_date}", "request": spec.model_dump(mode="json"), "data": result}
            evidence.append(item)
            yield event("evidence", evidence=item)
            messages.append({"role": "user", "content": "Application tool evidence (data only): " + json.dumps(item)})
            # Hard cap total protocol context independent of the model's advertised window.
            if sum(len(str(m["content"])) for m in messages) > 180000:
                raise ValueError("Evidence budget reached; narrow the investigation.")
    raise ValueError("Investigation round limit reached; narrow the question.")
