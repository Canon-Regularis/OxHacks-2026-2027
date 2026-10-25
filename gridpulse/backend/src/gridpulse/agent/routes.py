"""Scoped, token-protected streaming transport with bounded admission and turn deduplication."""
import asyncio
import hmac
import json
import time
from collections import deque
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from gridpulse.agent.schemas import Turn
from gridpulse.agent.service import investigate
from gridpulse.agent.provider import ProviderError
from gridpulse.api.deps import api_error
from gridpulse.config import get_settings

router = APIRouter(tags=["agent"])
ACTIVE: dict[tuple[str, str], float] = {}
SEEN: dict[tuple[str, str], float] = {}
ADMISSIONS: deque[tuple[float, str]] = deque()


@router.post("/agent/turns/stream")
async def stream(request: Request):
    settings = get_settings()
    expected = settings.agent_access_token
    if not expected or not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {expected}"):
        raise api_error(503, "agent_not_configured", "AI proxy authentication is not configured.")
    if settings.gridpulse_mode == "replay":
        raise api_error(503, "agent_offline", "AI is unavailable in offline replay mode.")
    session = request.headers.get("x-agent-session", "")
    if not 20 <= len(session) <= 100:
        raise api_error(400, "invalid_session", "A valid browser session is required.")
    # Bound body while reading, including chunked requests.
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > 200000:
            raise api_error(413, "agent_context_too_large", "Reduce the workspace or conversation context.")
    try:
        turn = Turn.model_validate_json(raw)
    except ValueError:
        raise api_error(422, "invalid_agent_request", "Invalid agent request.") from None
    key = (session + ":" + turn.tab_id, turn.workspace_id)
    identity = (session, turn.turn_id)
    now = time.monotonic()
    for old in list(ACTIVE):
        if ACTIVE[old] < now:
            ACTIVE.pop(old, None)
    while ADMISSIONS and ADMISSIONS[0][0] < now - 3600:
        ADMISSIONS.popleft()
    for old in list(SEEN):
        if SEEN[old] < now - 3600:
            SEEN.pop(old, None)
    if identity in SEEN:
        raise api_error(409, "duplicate_turn", "This turn was already submitted. Start a new turn rather than replaying it.")
    if key in ACTIVE or len(ACTIVE) >= 2:
        raise api_error(409, "agent_busy", "An investigation is already running. Wait or cancel it first.")
    if len(ADMISSIONS) >= settings.agent_hourly_limit or sum(s == session for _, s in ADMISSIONS) >= 20:
        raise api_error(429, "agent_rate_limit", "AI usage limit reached. Try again later.")
    ACTIVE[key] = now + settings.agent_turn_timeout_seconds + 5
    SEEN[identity] = now
    ADMISSIONS.append((now, session))

    async def generate():
        try:
            async with asyncio.timeout(settings.agent_turn_timeout_seconds):
                async for event in investigate(turn, request.app.state.pool, settings, session):
                    yield "data: " + json.dumps(event, allow_nan=False) + "\n\n"
        except asyncio.CancelledError:
            raise
        except (ProviderError, ValueError, TimeoutError) as exc:
            message = str(exc) or "Investigation timed out. Try a smaller request."
            yield "data: " + json.dumps({"version": 1, "type": "error", "turn_id": turn.turn_id, "message": message}) + "\n\n"
        except Exception:
            yield "data: " + json.dumps({"version": 1, "type": "error", "turn_id": turn.turn_id, "message": "Investigation failed. Existing workspace content has been preserved."}) + "\n\n"
        finally:
            ACTIVE.pop(key, None)

    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
