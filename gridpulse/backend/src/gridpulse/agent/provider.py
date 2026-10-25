"""Async inference adapter with strict decisions and one bounded schema repair per turn."""
import httpx
from pydantic import ValidationError
from gridpulse.agent.schemas import Decision


class ProviderError(Exception):
    pass


class Provider:
    def __init__(self, settings):
        self.settings = settings
        self.repairs_remaining = 1

    async def decide(self, messages: list[dict]) -> Decision:
        settings = self.settings
        if not settings.cwave_api_key:
            raise ProviderError("AI is not configured on the backend (CWAVE_API_KEY).")
        request_messages = list(messages)
        async with httpx.AsyncClient(timeout=60) as client:
            while True:
                try:
                    response = await client.post(
                        settings.agent_base_url.rstrip("/") + "/chat/completions",
                        headers={"Authorization": f"Bearer {settings.cwave_api_key}"},
                        json={"model": settings.agent_model, "messages": request_messages,
                              "max_tokens": settings.agent_max_tokens, "temperature": 0.2},
                    )
                    response.raise_for_status()
                    choice = response.json()["choices"][0]
                    if choice.get("finish_reason") == "length":
                        raise ProviderError("The model reached its output budget; try a narrower request.")
                    content = choice["message"]["content"]
                    if not isinstance(content, str):
                        raise ValueError("Missing model answer")
                    content = content.strip()
                    if content.startswith("```json") and content.endswith("```"):
                        content = content[7:-3].strip()
                except httpx.HTTPStatusError as exc:
                    code = exc.response.status_code
                    raise ProviderError(f"Inference service returned HTTP {code}. Check model access, credits or rate limits.") from None
                except httpx.HTTPError:
                    raise ProviderError("Inference service could not be reached. Try again.") from None
                except (ValueError, KeyError, IndexError, TypeError):
                    raise ProviderError("The inference service returned an invalid response.") from None
                try:
                    return Decision.model_validate_json(content)
                except ValidationError as exc:
                    if not self.repairs_remaining:
                        raise ProviderError("The model returned an invalid decision after correction; no workspace changes were applied.") from None
                    self.repairs_remaining -= 1
                    # Only schema locations/types are fed back; never relax validation or run partial tools.
                    problems = "; ".join(".".join(map(str, e["loc"])) + ": " + e["type"] for e in exc.errors()[:8])
                    request_messages = request_messages + [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": "Application schema validation failed: " + problems + ". Return a corrected Decision JSON object only. view.widgets describes analytical changes, never chat. If reads are needed, return reads with view=null and no claims of results yet."},
                    ]
