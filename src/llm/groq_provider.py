"""Groq — free tier, OpenAI-compatible (Llama 3.x)."""
from __future__ import annotations

from src.llm.base import _OpenAICompatible


class GroqProvider(_OpenAICompatible):
    name = "groq"
    is_free = True
    base_url = "https://api.groq.com/openai/v1"
