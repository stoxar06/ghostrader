"""Phase 4b — alternative strategy signal generators (offline)."""
from __future__ import annotations

import pandas as pd

from src.strategy import alt


def make(closes):
    idx = pd.date_range("2024-01-01 09:15", periods=len(closes), freq="5min", name="datetime")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame(
        {"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": pd.Series(1000.0, index=idx)}
    )


def test_mean_reversion_signal_shape():
    s = alt.mean_reversion_signals(make([100 + (i % 5) for i in range(60)]),
                                   {"rsi_period": 14, "bb_period": 20})
    assert {"entered", "direction", "score", "confidence"}.issubset(s.columns)
    assert s["direction"].isin([-1, 0, 1]).all()


def test_orb_breaks_above_opening_range_in_uptrend():
    s = alt.opening_range_breakout_signals(make([100 + i for i in range(60)]), {"or_bars": 6})
    assert s["direction"].isin([-1, 0, 1]).all()
    assert (s["direction"] == 1).any()  # rising series breaks above the opening-range high


def test_rsi_signal_shape_and_extremes():
    s = alt.rsi_signals(make([100 - i for i in range(60)]))  # falling -> oversold
    assert {"entered", "direction", "score", "confidence"}.issubset(s.columns)
    assert s["direction"].isin([-1, 0, 1]).all()
    assert (s["direction"] == 1).any()  # persistent decline drives RSI < 30 -> long
    up = alt.rsi_signals(make([100 + i for i in range(60)]))  # rising -> overbought
    assert (up["direction"] == -1).any()  # RSI > 70 -> short


def test_pullback_in_trend_signal_shape():
    import numpy as np
    # Uptrend with periodic dips so the trend filter is up and RSI dips below 40.
    base = np.linspace(100, 300, 320)
    wobble = 8 * np.sin(np.arange(320) / 3.0)
    s = alt.pullback_in_trend_signals(make(list(base + wobble)))
    assert {"entered", "direction", "score", "confidence"}.issubset(s.columns)
    assert s["direction"].isin([-1, 0, 1]).all()
