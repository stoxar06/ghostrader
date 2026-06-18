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


def test_random_signals_contract_and_determinism():
    from src.runner import random_signals

    df = _rising(300)
    a = random_signals(df, entry_prob=0.1, seed=42)
    b = random_signals(df, entry_prob=0.1, seed=42)
    assert (a["entered"] == b["entered"]).all() and (a["direction"] == b["direction"]).all()
    assert a["entered"].any()
    assert (a.loc[~a["entered"], "direction"] == 0).all()
    assert set(a.loc[a["entered"], "direction"].unique()).issubset({-1, 1})


def test_run_session_with_random_signal_fn():
    from src.runner import random_signals

    broker = PaperBroker(RiskParams(capital=100_000, risk_per_trade_pct=0.5, target_pct=1.5), Costs())
    results = run_session({"A": _rising(200), "B": _rising(150)}, STRAT, broker,
                          signal_fn=lambda df, _p: random_signals(df, 0.1, seed=7))
    assert isinstance(results, list) and len(results) >= 1
    assert all(r["reason"] in ("stop", "target", "square_off") for r in results)


def test_quota_signals_meets_daily_minimum_and_is_deterministic():
    from src.runner import quota_signals

    idx = pd.date_range("2024-01-01", periods=30, freq="D", name="datetime")
    frames = {f"S{i}": pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                                     "volume": 1.0}, index=idx) for i in range(8)}
    sig = quota_signals(frames, per_day=5, seed=0)
    per_day_counts = sum(s["entered"].astype(int) for s in sig.values())
    assert (per_day_counts == 5).all()                      # exactly the quota every day
    assert (quota_signals(frames, per_day=5, seed=0)["S0"]["entered"]   # deterministic
            == sig["S0"]["entered"]).all()
    for d in sig.values():                                   # entered rows carry a ±1 direction
        assert set(d.loc[d["entered"], "direction"].unique()).issubset({-1, 1})
        assert (d.loc[~d["entered"], "direction"] == 0).all()


def test_quota_signals_caps_at_available_symbols():
    from src.runner import quota_signals

    idx = pd.date_range("2024-01-01", periods=10, freq="D", name="datetime")
    frames = {f"S{i}": pd.DataFrame({"close": 1.0}, index=idx) for i in range(3)}
    counts = sum(s["entered"].astype(int) for s in quota_signals(frames, per_day=5).values())
    assert (counts == 3).all()                              # can't pick more than exist


def test_run_session_uses_signals_by_symbol_and_skips_missing():
    broker = PaperBroker(RiskParams(capital=100_000, risk_per_trade_pct=0.5,
                                    max_concurrent_positions=10, max_trades_per_day=50), Costs())
    up = _rising(120)
    sig = pd.DataFrame({"entered": False, "direction": 0}, index=up.index)
    sig.iloc[5, sig.columns.get_loc("entered")] = True
    sig.iloc[5, sig.columns.get_loc("direction")] = 1
    results = run_session({"A": up, "B": _rising(120)}, STRAT, broker,
                          signals_by_symbol={"A": sig})     # "B" has no signals -> skipped
    assert all(r["symbol"] == "A" for r in results)


def test_max_hold_bars_forces_one_day_exit():
    # Flat-ish prices never hit stop/target, so only the hold cap can close the trade.
    idx = pd.date_range("2024-01-01", periods=40, freq="D", name="datetime")
    flat = pd.DataFrame({"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0,
                         "volume": 1000.0}, index=idx)
    sig = pd.DataFrame({"entered": False, "direction": 0}, index=idx)
    sig.iloc[20, sig.columns.get_loc("entered")] = True     # past ATR warmup
    sig.iloc[20, sig.columns.get_loc("direction")] = 1
    broker = PaperBroker(RiskParams(capital=100_000, risk_per_trade_pct=0.5), Costs(),
                         max_hold_bars=1)
    results = run_session({"A": flat}, STRAT, broker, signals_by_symbol={"A": sig})
    assert results and results[0]["reason"] == "max_hold"   # closed by the 1-day cap, not stop/target


def test_paper_persistence_aggregates_and_replaces(tmp_path):
    from datetime import datetime

    from src.runner import persist_daily_pnl, reset_paper_session
    from src.storage.db import DailyPnL, Trade, get_session, init_db

    init_db(str(tmp_path / "runner.db"))

    def _trade(sym, pnl, exit_time, mode="paper"):
        return Trade(symbol=sym, side="LONG", qty=1, entry_price=100.0, exit_price=101.0,
                     entry_time=exit_time, exit_time=exit_time, pnl=pnl, mode=mode)

    with get_session() as s:
        s.add_all([
            _trade("A", 10.0, datetime(2024, 1, 1, 10)),
            _trade("B", -4.0, datetime(2024, 1, 1, 15)),
            _trade("C", 7.5, datetime(2024, 1, 2, 11)),
            _trade("D", 99.0, datetime(2024, 1, 2, 11), mode="backtest"),  # other modes untouched
        ])
        s.commit()

    assert persist_daily_pnl() == 2
    assert persist_daily_pnl() == 2  # rebuilds rather than appends
    with get_session() as s:
        rows = {str(r.day): r for r in s.query(DailyPnL).filter(DailyPnL.mode == "paper")}
        assert len(rows) == 2
        assert rows["2024-01-01"].realized_pnl == 6.0 and rows["2024-01-01"].trades_count == 2
        assert rows["2024-01-02"].realized_pnl == 7.5 and rows["2024-01-02"].trades_count == 1

    reset_paper_session()
    with get_session() as s:
        assert s.query(Trade).filter(Trade.mode == "paper").count() == 0
        assert s.query(DailyPnL).filter(DailyPnL.mode == "paper").count() == 0
        assert s.query(Trade).filter(Trade.mode == "backtest").count() == 1


def test_auth_login_url_and_token_roundtrip(tmp_path):
    a = KiteAuth("APIKEY", "SECRET", token_path=str(tmp_path / "tok.txt"))
    assert a.login_url() == "https://kite.zerodha.com/connect/login?api_key=APIKEY&v=3"
    assert a.load_token() is None
    a.save_token("abc123")
    assert a.load_token() == "abc123"
