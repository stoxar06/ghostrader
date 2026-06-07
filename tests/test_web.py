"""Phase 11 — web dashboard API (offline endpoints via Flask test client)."""
from __future__ import annotations

from datetime import date, datetime

import pytest


@pytest.fixture
def client(tmp_path):
    pytest.importorskip("flask")
    from src.web.server import create_app

    app = create_app(init_db_path=str(tmp_path / "web.db"))
    app.config.update(TESTING=True)
    return app.test_client()


def test_index_serves_dashboard(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Ghostrader" in r.data and b"dashboard" in r.data.lower()


def test_health_endpoint(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok"
    assert "llm_providers_configured" in body


def test_config_endpoint(client):
    body = client.get("/api/config").get_json()
    assert "instruments" in body and "risk" in body
    assert isinstance(body["research_universe"], list) and body["research_universe"]


def test_db_endpoints_empty_by_default(client):
    assert client.get("/api/daily_pnl").get_json() == []
    assert client.get("/api/trades").get_json() == []
    assert client.get("/api/llm_usage").get_json() == []


def test_daily_pnl_returns_seeded_row(client):
    from src.storage.db import DailyPnL, get_session

    with get_session() as s:
        s.add(DailyPnL(day=date.today(), realized_pnl=42.0, trades_count=1, halted=False, mode="paper"))
        s.commit()
    rows = client.get("/api/daily_pnl").get_json()
    assert len(rows) == 1 and rows[0]["realized_pnl"] == 42.0


def test_backtest_requires_symbol(client):
    r = client.get("/api/backtest")
    assert r.status_code == 400
    assert "error" in r.get_json()
