"""Google Gemini — free tier (gemini-flash) via the google-genai SDK."""
from __future__ import annotations

from src.llm.base import LLMProvider, LLMResult


class GeminiProvider(LLMProvider):
    name = "gemini"
    is_free = True

    def __init__(self, api_key: str):
        self.api_key = api_key

    def complete(self, system: str, user: str, model: str, max_tokens: int = 512) -> LLMResult:
        from google import genai  # lazy import

        client = genai.Client(api_key=self.api_key)
        resp = client.models.generate_content(
            model=model,
            contents=f"{system}\n\n{user}",
            config={"max_output_tokens": max_tokens},
        )
        usage = getattr(resp, "usage_metadata", None)
        return LLMResult(
            getattr(resp, "text", "") or "",
            "gemini",
            model,
            getattr(usage, "prompt_token_count", 0) if usage else 0,
            getattr(usage, "candidates_token_count", 0) if usage else 0,
        )
