"""Market regime synthesis.

Combines global cues + recent headlines into a compact market context. Uses the
free-first LLM router when available; falls back to a deterministic regime from
the cues when the LLM is unavailable or over budget. Caches by input hash
(skip-if-unchanged) in the macro_regime table — no LLM call when nothing changed.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime

from src.llm.base import extract_json
from src.logutil import get_logger

log = get_logger(__name__)

REGIME_KEYS = ["regime", "bias"]

SYSTEM_PROMPT = (
    "You are a markets analyst for Indian equities. Given global cues (1-day % "
    "changes) and recent headlines, output ONLY compact JSON with keys: "
    "regime (one of risk_on|neutral|risk_off), bias (bullish|neutral|bearish), "
    "favored_sectors (array of strings), avoid_sectors (array of strings), "
    "summary (one sentence). Base everything on the data provided; do not invent "
    "specific numbers or events."
)


def _deterministic_regime(cues: dict) -> dict:
    """Fallback when no LLM is available: a simple risk score from key cues."""
    nifty = cues.get("nifty", 0.0)
    sp500 = cues.get("sp500", 0.0)
    vix = cues.get("india_vix", 0.0)
    score = nifty + sp500 - 0.5 * vix
    if score > 0.3:
        regime, bias = "risk_on", "bullish"
    elif score < -0.3:
        regime, bias = "risk_off", "bearish"
    else:
        regime, bias = "neutral", "neutral"
    return {
        "regime": regime,
        "bias": bias,
        "favored_sectors": [],
        "avoid_sectors": [],
        "summary": f"Deterministic regime from cues (risk score {score:+.2f}).",
        "source": "deterministic",
    }


def _format_user(cues: dict, headlines: list[str]) -> str:
    cue_lines = "\n".join(f"- {k}: {v:+.2f}%" for k, v in cues.items()) or "- (none)"
    head_lines = "\n".join(f"- {h}" for h in headlines[:25]) or "- (none)"
    return f"GLOBAL CUES (1-day % change):\n{cue_lines}\n\nRECENT HEADLINES:\n{head_lines}"


def build_regime(cues: dict, headlines: list[str], router=None) -> dict:
    """One-shot regime build (no caching). Deterministic if router is None/unavailable."""
    if router is None:
        return _deterministic_regime(cues)

    res = router.generate(SYSTEM_PROMPT, _format_user(cues, headlines),
                          tier="synthesis", required_keys=REGIME_KEYS)
    if res is None:
        return _deterministic_regime(cues)

    data = extract_json(res.text) or {}
    if not all(k in data for k in REGIME_KEYS):
        return _deterministic_regime(cues)

    data.setdefault("favored_sectors", [])
    data.setdefault("avoid_sectors", [])
    data.setdefault("summary", "")
    data["source"] = res.provider
    return data


def input_hash(cues: dict, headlines: list[str]) -> str:
    payload = json.dumps({"cues": cues, "heads": headlines[:25]}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def get_or_build(cues: dict, headlines: list[str], router, session_factory) -> dict:
    """Cached build: reuse the last regime if inputs are unchanged (no LLM call)."""
    from src.storage.db import MacroRegime

    h = input_hash(cues, headlines)
    with session_factory() as s:
        last = s.query(MacroRegime).order_by(MacroRegime.id.desc()).first()
        if last is not None and last.input_hash == h:
            log.info("Macro inputs unchanged — reusing cached regime (no LLM call).")
            return json.loads(last.payload)

    data = build_regime(cues, headlines, router)
    with session_factory() as s:
        s.add(MacroRegime(
            ts=datetime.now(), regime=data.get("regime", "neutral"),
            bias=data.get("bias", "neutral"), payload=json.dumps(data), input_hash=h,
        ))
        s.commit()
    return data
