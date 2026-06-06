"""LLM provider interface, shared OpenAI-compatible base, and JSON extraction.

Only public market/news text is ever sent to providers (never keys/positions/PII).
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMProvider(ABC):
    name: str = "base"
    is_free: bool = True

    @abstractmethod
    def complete(self, system: str, user: str, model: str, max_tokens: int = 512) -> LLMResult:
        ...


class _OpenAICompatible(LLMProvider):
    """Base for providers that speak the OpenAI chat API (Groq, OpenRouter)."""

    base_url: str = ""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def complete(self, system: str, user: str, model: str, max_tokens: int = 512) -> LLMResult:
        from openai import OpenAI  # lazy import

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return LLMResult(
            text, self.name, model,
            getattr(usage, "prompt_tokens", 0) if usage else 0,
            getattr(usage, "completion_tokens", 0) if usage else 0,
        )


class FakeProvider(LLMProvider):
    """Deterministic provider for offline tests."""

    def __init__(self, name="fake", is_free=True, response="{}", fail=False, tokens=(10, 5)):
        self.name = name
        self.is_free = is_free
        self.response = response
        self.fail = fail
        self.tokens = tokens
        self.calls = 0

    def complete(self, system: str, user: str, model: str, max_tokens: int = 512) -> LLMResult:
        self.calls += 1
        if self.fail:
            raise RuntimeError(f"{self.name} simulated failure")
        return LLMResult(self.response, self.name, model, self.tokens[0], self.tokens[1])


def extract_json(text: str) -> dict | None:
    """Best-effort parse of a JSON object from model output (handles code fences)."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j > i:
        try:
            obj = json.loads(t[i:j + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None
