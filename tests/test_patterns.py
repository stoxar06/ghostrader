"""Phase 3 — candlestick pattern detection on hand-crafted candles."""
from __future__ import annotations

import pandas as pd
import pytest

from src.indicators import patterns


def df_from_rows(rows):
    idx = pd.date_range("2024-01-01 09:15", periods=len(rows), freq="5min", name="datetime")
    return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close", "volume"])


def test_bullish_engulfing():
    df = df_from_rows(
        [
            [10.0, 10.2, 8.8, 9.0, 1000],    # bearish: close < open
            [8.5, 10.6, 8.4, 10.5, 1000],    # bullish body engulfs the prior body
        ]
    )
    assert patterns.engulfing(df).iloc[1] == 1


def test_bearish_engulfing():
    df = df_from_rows(
        [
            [9.0, 10.2, 8.8, 10.0, 1000],    # bullish
            [10.5, 10.6, 8.4, 8.5, 1000],    # bearish body engulfs the prior body
        ]
    )
    assert patterns.engulfing(df).iloc[1] == -1


def test_hammer():
    # small body near the top, long lower shadow (>= 2x body), tiny upper shadow
    df = df_from_rows([[100.0, 100.7, 98.0, 100.5, 1000]])
    assert patterns.hammer(df).iloc[0] == 1


def test_detect_known_and_unknown():
    df = df_from_rows([[100.0, 100.7, 98.0, 100.5, 1000]])
    assert patterns.detect("hammer", df).iloc[0] == 1
    with pytest.raises(KeyError):
        patterns.detect("nonexistent_pattern_xyz", df)  # TA-Lib not installed -> KeyError
