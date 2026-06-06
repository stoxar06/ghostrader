"""Free-first LLM router.

Tries providers in order (free tiers first), falls back to the next on error,
invalid JSON, or — for paid providers — when the daily INR cap is already spent.
Records every call to the llm_usage table for the cap. Returns None when every
provider is exhausted, so callers can fall back to deterministic behavior.
"""
from __future__ import annotations

from datetime import date, datetime

from src.llm.base import LLMProvider, LLMResult, extract_json
from src.logutil import get_logger

log = get_logger(__name__)

# USD per 1M tokens (input, output) for the paid fallback models.
PRICING_USD = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-8": (5.0, 25.0),
}
USD_INR = 83.0


class Router:
    def __init__(self, providers: dict[str, LLMProvider], cfg: dict, session_factory):
        self.providers = providers
        self.order = cfg.get("provider_order", list(providers.keys()))
        self.cap_inr = float(cfg.get("daily_paid_cap_inr", 40))
        self.max_tokens = int(cfg.get("max_output_tokens", 512))
        self.classify_model = cfg.get("classify_model", {}) or {}
        self.synthesis_model = cfg.get("synthesis_model", {}) or {}
        self.session_factory = session_factory

    def _cost_inr(self, model: str, res: LLMResult) -> float:
        price = PRICING_USD.get(model)
        if not price:
            return 0.0
        usd = (res.input_tokens * price[0] + res.output_tokens * price[1]) / 1e6
        return usd * USD_INR

    def _paid_spent_today(self) -> float:
        from src.storage.db import LLMUsage

        with self.session_factory() as s:
            rows = s.query(LLMUsage).filter(LLMUsage.day == date.today()).all()
            return float(sum(r.cost_inr for r in rows))

    def _record(self, provider: str, model: str, res: LLMResult, cost_inr: float) -> None:
        from src.storage.db import LLMUsage

        with self.session_factory() as s:
            s.add(LLMUsage(
                ts=datetime.now(), day=date.today(), provider=provider, model=model,
                input_tokens=res.input_tokens, output_tokens=res.output_tokens, cost_inr=cost_inr,
            ))
            s.commit()

    def generate(
        self, system: str, user: str, tier: str = "synthesis",
        required_keys: list[str] | None = None,
    ) -> LLMResult | None:
        models = self.synthesis_model if tier == "synthesis" else self.classify_model

        for name in self.order:
            provider = self.providers.get(name)
            model = models.get(name)
            if provider is None or model is None:
                continue

            if not provider.is_free and self._paid_spent_today() >= self.cap_inr:
                log.info("Paid cap ₹%.2f reached — skipping %s", self.cap_inr, name)
                continue

            try:
                res = provider.complete(system, user, model, self.max_tokens)
            except Exception as exc:  # noqa: BLE001
                log.warning("Provider %s failed: %s", name, exc)
                continue

            cost = 0.0 if provider.is_free else self._cost_inr(model, res)
            self._record(name, model, res, cost)

            if required_keys:
                data = extract_json(res.text)
                if data is None or not all(k in data for k in required_keys):
                    log.warning("Provider %s returned invalid/incomplete JSON", name)
                    continue

            log.info("LLM via %s/%s (in=%d out=%d cost=₹%.3f)",
                     name, model, res.input_tokens, res.output_tokens, cost)
            return res

        log.warning("All LLM providers exhausted — caller should fall back to deterministic.")
        return None
