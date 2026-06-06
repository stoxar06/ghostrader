"""Phase 4c — cross-sectional momentum backtest (offline, synthetic)."""
from __future__ import annotations

import pandas as pd

from src.backtest.momentum import momentum_backtest


def _series(start, factor, n, idx):
    return pd.Series([start * (factor ** i) for i in range(n)], index=idx, dtype=float)


def test_momentum_picks_outperformer_and_beats_equal_weight():
    n = 300
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = {
        "FLAT": pd.Series(100.0, index=idx),
        "DOWN": _series(100.0, 0.999, n, idx),
        "UP": _series(100.0, 1.003, n, idx),
    }
    res = momentum_backtest(prices, lookback=20, hold_top=1, rebalance=20)
    assert "momentum" in res and "buy_hold" in res
    # Holding only the strongest trender should beat the equal-weight basket
    # (which is dragged down by the falling name).
    assert res["momentum"]["total_return_pct"] > res["buy_hold"]["total_return_pct"]


def test_momentum_insufficient_history():
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    res = momentum_backtest({"A": pd.Series(range(10), index=idx, dtype=float)},
                            lookback=126, rebalance=21)
    assert "error" in res
