"""LLM package — free-first multi-provider routing for the macro/analyst layer."""
from src.llm.base import FakeProvider, LLMProvider, LLMResult, extract_json  # noqa: F401
from src.llm.router import Router  # noqa: F401


def build_providers(secrets) -> dict[str, LLMProvider]:
    """Construct only the providers whose API keys are present in `secrets`."""
    from src.llm.anthropic_provider import AnthropicProvider
    from src.llm.gemini_provider import GeminiProvider
    from src.llm.groq_provider import GroqProvider
    from src.llm.openrouter_provider import OpenRouterProvider

    providers: dict[str, LLMProvider] = {}
    if getattr(secrets, "groq_api_key", None):
        providers["groq"] = GroqProvider(secrets.groq_api_key)
    if getattr(secrets, "gemini_api_key", None):
        providers["gemini"] = GeminiProvider(secrets.gemini_api_key)
    if getattr(secrets, "openrouter_api_key", None):
        providers["openrouter"] = OpenRouterProvider(secrets.openrouter_api_key)
    if getattr(secrets, "anthropic_api_key", None):
        providers["anthropic"] = AnthropicProvider(secrets.anthropic_api_key)
    return providers
