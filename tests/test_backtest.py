"""Phase 4 — backtest engine, cost model, metrics, weight calibration (offline)."""
from __future__ import annotations

import pandas as pd

from src.backtest.engine import Costs, calibrate_weights, run_backtest, trade_charges
from src.backtest.metrics import compute_metrics
from src.risk.manager import RiskParams


def make_intraday(closes, day="2024-01-01"):
    idx = pd.date_range(f"{day} 09:15", periods=len(closes), freq="5min", name="datetime")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame(
        {"open": c.shift().fillna(c.iloc[0]), "high": c + 0.5, "low": c - 0.5,
         "close": c, "volume": pd.Series(1000.0, index=idx)}
    )


STRAT = {
    "indicator_params": {"ema_fast": 9, "ema_slow": 21, "atr_period": 14,
                         "supertrend_period": 10, "supertrend_multiplier": 3.0},
    "enabled_signals": {"trend": ["ema_cross", "supertrend"], "volume": ["vwap"],
                        "candlesticks": [], "momentum": []},
    "confidence_threshold": 0.3,
    "require_higher_tf_agreement": False,
}


def test_trade_charges_positive_and_scales_with_qty():
    c = Costs()
    small = trade_charges(100.0, 101.0, 10, 1, c)
    big = trade_charges(100.0, 101.0, 100, 1, c)
    assert small > 0
    assert big > small


def test_metrics_known_values():
    trades = pd.DataFrame({"pnl": [100.0, -50.0, 200.0, -30.0]})
    m = compute_metrics(trades, 100_000)
    assert m["trades"] == 4
    assert m["win_rate"] == 0.5
    assert round(m["expectancy"], 2) == 55.0
    assert round(m["total_pnl"], 2) == 220.0
    assert round(m["max_drawdown"], 2) == -50.0


def test_run_backtest_executes_and_nets_positive_in_uptrend():
    df = make_intraday([100 + i for i in range(80)])  # steady uptrend
    rp = RiskParams(capital=100_000, risk_per_trade_pct=0.5, target_pct=1.5)
    trades = run_backtest(df, STRAT, rp, Costs(), symbol="TEST")
    assert not trades.empty
    assert {"entry_price", "exit_price", "pnl", "reason"}.issubset(trades.columns)
    # Longs in a clean uptrend should net positive after costs + slippage.
    assert trades["pnl"].sum() > 0


def test_empty_for_too_short_series():
    df = make_intraday([100, 101])
    assert run_backtest(df, STRAT, RiskParams(), Costs()).empty


def test_calibrate_weights_returns_winrates_in_unit_interval():
    df = make_intraday([100 + (i % 7) for i in range(120)])
    w = calibrate_weights(df, STRAT, target_pct=1.0, stop_pct=0.5, horizon=8)
    assert "ema_cross" in w.weights
    for v in w.weights.values():
        assert 0.0 <= v <= 1.0
