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


def test_trades_days_filter_anchors_to_latest_recorded_day(client):
    from src.storage.db import Trade, get_session

    def _t(sym, exit_time):
        return Trade(symbol=sym, side="LONG", qty=1, entry_price=100.0, exit_price=101.0,
                     entry_time=exit_time, exit_time=exit_time, pnl=1.0, mode="paper")

    with get_session() as s:
        s.add_all([_t("OLD", datetime(2025, 1, 10, 15)),     # far outside any 7d window
                   _t("IN1", datetime(2025, 3, 5, 15)),
                   _t("IN2", datetime(2025, 3, 10, 15))])    # latest recorded day = 2025-03-10
        s.commit()

    rows = client.get("/api/trades?days=7").get_json()
    assert {r["symbol"] for r in rows} == {"IN1", "IN2"}     # 7d window = Mar 4..10, not today
    assert rows[0]["day"] == "2025-03-10"                    # newest first, day field present
    assert len(client.get("/api/trades?days=7&limit=1").get_json()) == 1
    assert len(client.get("/api/trades").get_json()) == 3    # unfiltered keeps everything


def test_edge_endpoint_serves_research_artifacts(client, tmp_path, monkeypatch):
    import json

    from src.backtest import auto_search, edge_search

    arc = tmp_path / "edge_archive.json"
    auto = tmp_path / "auto_search.json"
    monkeypatch.setattr(edge_search, "ARCHIVE_PATH", str(arc))
    monkeypatch.setattr(auto_search, "ARCHIVE_PATH", str(auto))

    r = client.get("/api/edge").get_json()                   # no artifacts -> explicit nulls
    assert r == {"edge_archive": None, "auto_search": None}

    arc.write_text(json.dumps({"n_winners": 2, "n_real_edge": 0, "n_drift_riders": 2, "winners": []}))
    st = auto_search.fresh_state()
    st.update(configs_tried=276, generations=8,
              hall_of_fame=[{"config": "adaptive_rsi[x]", "family": "adaptive_rsi", "params": {},
                             "fitness_pp": 4.0, "horizon": 3, "oos_accuracy": 0.555,
                             "oos_cond_edge_pp": 4.01, "is_cond_edge_pp": 4.09, "oos_signals": 2192}])
    auto.write_text(json.dumps(st))

    r = client.get("/api/edge").get_json()
    assert r["edge_archive"]["n_winners"] == 2
    assert r["auto_search"]["configs_tried"] == 276
    assert r["auto_search"]["hall_of_fame"][0]["family"] == "adaptive_rsi"
    assert "meaningful" in r["auto_search"]["verdict"]


def test_backtest_requires_symbol(client):
    r = client.get("/api/backtest")
    assert r.status_code == 400
    assert "error" in r.get_json()
