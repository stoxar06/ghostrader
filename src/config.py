"""Central configuration.

Secrets come from the environment (.env) via `Secrets`.
Non-secret tunables come from config.yaml via `AppConfig`.

Usage:
    from src.config import get_config, get_secrets
    cfg = get_config()
    risk_pct = cfg.risk.risk_per_trade_pct
    api_key = get_secrets().kite_api_key
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"

# Load .env into the process environment (no-op if the file is missing).
load_dotenv(ENV_PATH)


class Secrets(BaseSettings):
    """API keys & secrets from environment / .env. Never commit real values."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH), env_file_encoding="utf-8", extra="ignore"
    )

    kite_api_key: str | None = None
    kite_api_secret: str | None = None
    kite_access_token: str | None = None
    anthropic_api_key: str | None = None
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    openrouter_api_key: str | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None


class _Section(BaseModel):
    """A config.yaml section. Extra keys are allowed and accessible as attributes."""

    model_config = ConfigDict(extra="allow")


class AppConfig(BaseModel):
    """Typed view over config.yaml. Sections stay loose so the YAML is the source of truth."""

    model_config = ConfigDict(extra="allow")

    mode: str = "paper"
    instruments: _Section = Field(default_factory=_Section)
    session: _Section = Field(default_factory=_Section)
    strategy: _Section = Field(default_factory=_Section)
    risk: _Section = Field(default_factory=_Section)
    costs: _Section = Field(default_factory=_Section)
    macro: _Section = Field(default_factory=_Section)
    llm: _Section = Field(default_factory=_Section)
    storage: _Section = Field(default_factory=_Section)
    logging: _Section = Field(default_factory=_Section)


def load_yaml_config(path: Path = CONFIG_PATH) -> AppConfig:
    if not path.exists():
        raise FileNotFoundError(f"config.yaml not found at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return AppConfig(**data)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Cached application config from config.yaml."""
    return load_yaml_config()


@lru_cache(maxsize=1)
def get_secrets() -> Secrets:
    """Cached secrets from environment / .env."""
    return Secrets()
