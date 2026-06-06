"""Phase 7 — macro engine (cues, news dedupe, regime synthesis & caching), offline."""
from __future__ import annotations

import pandas as pd
import pytest

from src.llm.base import FakeProvider
from src.llm.router import Router
from src.macro import globalcues, news, regime

CFG_GROQ = {
    "provider_order": ["groq"], "daily_paid_cap_inr": 40, "max_output_tokens": 256,
    "synthesis_model": {"groq": "g-s"}, "classify_model": {"groq": "g-c"},
}


@pytest.fixture
def session_factory(tmp_path):
    from src.storage.db import get_session, init_db

    init_db(str(tmp_path / "macro.db"))
    return get_session


def test_pct_change_last():
    assert globalcues.pct_change_last(pd.Series([100.0, 110.0])) == pytest.approx(10.0)
    assert globalcues.pct_change_last(pd.Series([100.0])) is None


def test_news_dedupe_collapses_whitespace_and_blanks():
    titles = ["Nifty up", "nifty   up", "Sensex falls", "", "Sensex falls"]
    assert news.dedupe_headlines(titles) == ["Nifty up", "Sensex falls"]


def test_deterministic_regime_risk_on():
    d = regime._deterministic_regime({"nifty": 1.0, "sp500": 1.0, "india_vix": -2.0})
    assert d["regime"] == "risk_on" and d["bias"] == "bullish" and d["source"] == "deterministic"


def test_deterministic_regime_risk_off():
    d = regime._deterministic_regime({"nifty": -1.0, "sp500": -1.0, "india_vix": 5.0})
    assert d["regime"] == "risk_off"


def test_build_regime_none_router_is_deterministic():
    assert regime.build_regime({"nifty": 0.5}, ["x"], router=None)["source"] == "deterministic"


def test_build_regime_uses_llm(session_factory):
    provs = {"groq": FakeProvider("groq", True, '{"regime":"risk_on","bias":"bullish","summary":"ok"}')}
    d = regime.build_regime({"nifty": 0.1}, ["h1"], router=Router(provs, CFG_GROQ, session_factory))
    assert d["regime"] == "risk_on" and d["source"] == "groq"


def test_build_regime_falls_back_when_llm_fails(session_factory):
    provs = {"groq": FakeProvider("groq", True, fail=True)}
    d = regime.build_regime({"nifty": -1.0, "sp500": -1.0}, ["h"],
                            router=Router(provs, CFG_GROQ, session_factory))
    assert d["source"] == "deterministic" and d["regime"] == "risk_off"


def test_get_or_build_skips_llm_when_inputs_unchanged(session_factory):
    provs = {"groq": FakeProvider("groq", True, '{"regime":"neutral","bias":"neutral"}')}
    router = Router(provs, CFG_GROQ, session_factory)
    cues, heads = {"nifty": 0.2}, ["a", "b"]

    regime.get_or_build(cues, heads, router, session_factory)
    assert provs["groq"].calls == 1
    again = regime.get_or_build(cues, heads, router, session_factory)
    assert provs["groq"].calls == 1  # unchanged inputs -> cached, no new LLM call
    assert again["regime"] == "neutral"
