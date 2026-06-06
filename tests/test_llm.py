"""Phase 7 — free-first LLM router (offline, with fake providers)."""
from __future__ import annotations

import pytest

from src.llm.base import FakeProvider, extract_json
from src.llm.router import Router

CFG = {
    "provider_order": ["groq", "gemini", "anthropic"],
    "daily_paid_cap_inr": 40,
    "max_output_tokens": 256,
    "classify_model": {"groq": "g-c", "gemini": "gm-c", "anthropic": "claude-haiku-4-5"},
    "synthesis_model": {"groq": "g-s", "gemini": "gm-s", "anthropic": "claude-sonnet-4-6"},
}


@pytest.fixture
def session_factory(tmp_path):
    from src.storage.db import get_session, init_db

    init_db(str(tmp_path / "llm.db"))
    return get_session


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}


def test_extract_json_embedded_in_prose():
    assert extract_json('Sure! {"a": 3} done') == {"a": 3}


def test_extract_json_invalid():
    assert extract_json("not json at all") is None


def test_free_provider_used_first(session_factory):
    provs = {
        "groq": FakeProvider("groq", True, '{"regime": "x"}'),
        "anthropic": FakeProvider("anthropic", False, '{"regime": "y"}'),
    }
    res = Router(provs, CFG, session_factory).generate("sys", "usr", required_keys=["regime"])
    assert res is not None and res.provider == "groq"
    assert provs["anthropic"].calls == 0  # paid never touched


def test_fallback_to_next_free_on_failure(session_factory):
    provs = {
        "groq": FakeProvider("groq", True, fail=True),
        "gemini": FakeProvider("gemini", True, '{"regime": "x"}'),
        "anthropic": FakeProvider("anthropic", False, '{"regime": "y"}'),
    }
    res = Router(provs, CFG, session_factory).generate("s", "u", required_keys=["regime"])
    assert res.provider == "gemini"
    assert provs["anthropic"].calls == 0


def test_paid_cap_blocks_paid_provider(session_factory):
    cfg = dict(CFG, daily_paid_cap_inr=0.0, provider_order=["anthropic"])
    provs = {"anthropic": FakeProvider("anthropic", False, '{"regime": "y"}')}
    res = Router(provs, cfg, session_factory).generate("s", "u", required_keys=["regime"])
    assert res is None
    assert provs["anthropic"].calls == 0


def test_invalid_json_exhausts_to_none(session_factory):
    cfg = dict(CFG, provider_order=["groq"])
    provs = {"groq": FakeProvider("groq", True, "totally not json")}
    res = Router(provs, cfg, session_factory).generate("s", "u", required_keys=["regime"])
    assert res is None


def test_usage_is_recorded(session_factory):
    from src.storage.db import LLMUsage

    cfg = dict(CFG, provider_order=["groq"])
    provs = {"groq": FakeProvider("groq", True, '{"regime": "x"}')}
    Router(provs, cfg, session_factory).generate("s", "u", required_keys=["regime"])
    with session_factory() as s:
        assert s.query(LLMUsage).count() == 1
