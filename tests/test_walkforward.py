"""Phase 4b — walk-forward folds & out-of-sample harness (offline)."""
from __future__ import annotations

import pandas as pd

from src.backtest.engine import Costs
from src.backtest.walkforward import split_windows, walk_forward
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


def test_split_windows_ordered_and_disjoint():
    folds = split_windows(1000, n_splits=4)
    assert len(folds) >= 1
    for (tr, te) in folds:
        assert tr[0] < tr[1] <= te[0] < te[1]  # train precedes test, no overlap


def test_split_windows_too_small_returns_empty():
    assert split_windows(10, n_splits=4) == []


def test_walk_forward_returns_oos_frame():
    df = make_intraday([100 + (i % 11) for i in range(400)])
    trades = walk_forward(df, STRAT, RiskParams(), Costs(), htf_df=None, n_splits=3)
    assert isinstance(trades, pd.DataFrame)
    if not trades.empty:
        assert "pnl" in trades.columns
