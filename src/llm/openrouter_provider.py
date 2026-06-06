"""OpenRouter — free models, OpenAI-compatible."""
from __future__ import annotations

from src.llm.base import _OpenAICompatible


class OpenRouterProvider(_OpenAICompatible):
    name = "openrouter"
    is_free = True
    base_url = "https://openrouter.ai/api/v1"
