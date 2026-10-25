"""Server-side settings, read from environment variables or gridpulse/.env.

Nothing here connects to anything; secrets stay on the server.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# The gridpulse/ application directory (backend is installed in editable mode from the repo).
APP_DIR = Path(__file__).resolve().parents[3]
REPO_DIR = APP_DIR.parent
MIGRATIONS_DIR = APP_DIR / "db" / "migrations"

# Later files win: a credentials file downloaded from Tiger Cloud into the repo root,
# then root .env, then gridpulse/.env for local overrides. All are gitignored.
ENV_FILES = (*sorted(REPO_DIR.glob("tiger-cloud-*-credentials.env")), REPO_DIR / ".env", APP_DIR / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILES, extra="ignore")

    # Tiger Cloud's credentials file calls it TIMESCALE_SERVICE_URL.
    database_url: str = Field("", validation_alias=AliasChoices("DATABASE_URL", "TIMESCALE_SERVICE_URL"))
    frontend_origin: str = "http://localhost:5173"

    # live: normal. record: also save API responses to demo_cache_dir.
    # replay: serve saved responses only, no database (offline demo fallback).
    gridpulse_mode: Literal["live", "record", "replay"] = "live"
    demo_cache_dir: Path = APP_DIR / "data" / "demo" / "responses"

    # Response bounds. The database is large; the API never returns raw history.
    max_points: int = 5000
    max_range_days: int = 3660
    query_timeout_seconds: int = 15

    # Terminal agent. Provider credentials and proxy token never leave the server.
    openrouter_api_key: str = ""
    cwave_api_key: str = ""
    agent_provider: Literal["openrouter", "canopywave"] = "openrouter"
    agent_reasoning_effort: Literal["none", "low", "medium"] = "low"
    agent_access_token: str = ""
    agent_base_url: str = "https://openrouter.ai/api/v1"
    agent_model: str = "openai/gpt-5.4-mini"
    agent_max_tokens: int = 5000
    agent_turn_timeout_seconds: int = 180
    agent_job_timeout_seconds: int = 90
    agent_hourly_limit: int = 60

    # Older optional provider fields retained for compatibility.
    gemini_api_key: str = ""
    gemini_model: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
