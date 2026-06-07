"""Phase 5 — paper session runner + Kite auth helpers (offline)."""
from __future__ import annotations

import pandas as pd

from src.auth import KiteAuth
from src.backtest.engine import Costs
from src.execution.paper import PaperBroker
from src.risk.manager import RiskParams
from src.runner import run_session

STRAT = {
    "indicator_params": {"ema_fast": 9, "ema_slow": 21, "atr_period": 14,
                         "supertrend_period": 10, "supertrend_multiplier": 3.0},
    "enabled_signals": {"trend": ["ema_cross", "supertrend"], "volume": ["vwap"],
                        "candlesticks": [], "momentum": []},
    "confidence_threshold": 0.3,
    "require_higher_tf_agreement": False,
}


def _rising(n=80):
    idx = pd.date_range("2024-01-01 09:15", periods=n, freq="5min", name="datetime")
    c = pd.Series([100 + i for i in range(n)], index=idx, dtype=float)
    return pd.DataFrame({"open": c.shift().fillna(c.iloc[0]), "high": c + 0.5, "low": c - 0.5,
                         "close": c, "volume": pd.Series(1000.0, index=idx)})


def test_run_session_paper_produces_winning_trades():
    broker = PaperBroker(RiskParams(capital=100_000, risk_per_trade_pct=0.5, target_pct=1.5), Costs())
    results = run_session({"UP": _rising()}, STRAT, broker)
    assert isinstance(results, list) and len(results) >= 1
    assert sum(r["pnl"] for r in results) > 0  # longs in a clean uptrend net positive


def test_auth_login_url_and_token_roundtrip(tmp_path):
    a = KiteAuth("APIKEY", "SECRET", token_path=str(tmp_path / "tok.txt"))
    assert a.login_url() == "https://kite.zerodha.com/connect/login?api_key=APIKEY&v=3"
    assert a.load_token() is None
    a.save_token("abc123")
    assert a.load_token() == "abc123"
