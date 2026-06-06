"""Anthropic Claude — paid fallback (Haiku -> Sonnet), official SDK."""
from __future__ import annotations

from src.llm.base import LLMProvider, LLMResult


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    is_free = False

    def __init__(self, api_key: str):
        self.api_key = api_key

    def complete(self, system: str, user: str, model: str, max_tokens: int = 512) -> LLMResult:
        import anthropic  # lazy import

        client = anthropic.Anthropic(api_key=self.api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return LLMResult(text, "anthropic", model, resp.usage.input_tokens, resp.usage.output_tokens)
