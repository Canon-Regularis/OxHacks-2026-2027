"""Async inference adapter. Strict JSON envelopes work without provider tool extensions."""
import json
import httpx
from gridpulse.agent.schemas import Decision


class ProviderError(Exception):
    pass


class Provider:
    def __init__(self, settings):
        self.settings = settings

    async def decide(self, messages: list[dict]) -> Decision:
        settings = self.settings
        if not settings.cwave_api_key:
            raise ProviderError("AI is not configured on the backend (CWAVE_API_KEY).")
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    settings.agent_base_url.rstrip("/") + "/chat/completions",
                    headers={"Authorization": f"Bearer {settings.cwave_api_key}"},
                    json={"model": settings.agent_model, "messages": messages,
                          "max_tokens": settings.agent_max_tokens, "temperature": 0.2},
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("Missing model answer")
            content = content.strip()
            if content.startswith("```json") and content.endswith("```"):
                content = content[7:-3].strip()
            return Decision.model_validate_json(content)
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            raise ProviderError(f"Inference service returned HTTP {code}. Check model access, credits or rate limits.") from None
        except httpx.HTTPError:
            raise ProviderError("Inference service could not be reached. Try again.") from None
        except (ValueError, KeyError, IndexError, TypeError):
            raise ProviderError("The model returned an invalid response; no workspace changes were applied.") from None
