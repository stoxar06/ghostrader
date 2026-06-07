"""Phase 5/8 — paper broker fills & live broker gating (offline)."""
from __future__ import annotations

from datetime import datetime

import pytest

from src.backtest.engine import Costs
from src.execution.live import LiveBroker, order_params
from src.execution.paper import PaperBroker
from src.risk.manager import RiskParams

NOW = datetime(2024, 1, 1, 10, 0)


def _broker(**kw):
    return PaperBroker(RiskParams(capital=100_000, risk_per_trade_pct=0.5, target_pct=1.5),
                       Costs(), **kw)


def test_paper_open_then_target_is_profit():
    b = _broker()
    b.open("X", 1, ref_price=100.0, atr=2.0, ts=NOW)
    assert not b.can_open("X")  # position is held
    r = b.on_bar("X", high=200.0, low=99.5, close=150.0, ts=NOW)
    assert r and r["reason"] == "target" and r["pnl"] > 0
    assert b.can_open("X")      # flat again


def test_paper_stop_is_loss():
    b = _broker()
    b.open("X", 1, ref_price=100.0, atr=2.0, ts=NOW)
    r = b.on_bar("X", high=100.5, low=50.0, close=60.0, ts=NOW)
    assert r and r["reason"] == "stop" and r["pnl"] < 0


def test_paper_persists_trade(tmp_path):
    from src.storage.db import Trade, get_session, init_db

    init_db(str(tmp_path / "p.db"))
    b = _broker(session_factory=get_session)
    b.open("Y", 1, 100.0, 2.0, NOW)
    b.on_bar("Y", 200.0, 99.5, 150.0, NOW)
    with get_session() as s:
        assert s.query(Trade).count() == 1


def test_order_params_buy_sell():
    p = order_params("RELIANCE", 1, 10)
    assert p["transaction_type"] == "BUY" and p["tradingsymbol"] == "RELIANCE" and p["quantity"] == 10
    assert order_params("X", -1, 5)["transaction_type"] == "SELL"


def test_live_broker_refuses_when_disabled():
    lb = LiveBroker(kite=object(), allow_live_orders=False)
    with pytest.raises(RuntimeError):
        lb.place("RELIANCE", 1, 10)


def test_live_broker_places_when_enabled():
    class FakeKite:
        called = None
        def place_order(self, **kw):
            self.called = kw
            return "OID123"

    k = FakeKite()
    lb = LiveBroker(kite=k, allow_live_orders=True)
    assert lb.place("RELIANCE", 1, 10) == "OID123"
    assert k.called["tradingsymbol"] == "RELIANCE" and k.called["transaction_type"] == "BUY"
