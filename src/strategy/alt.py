"""Alternative strategies (fixed-rule, so a full-period backtest is not overfitting).

Each returns a DataFrame aligned to df with columns: entered (bool), direction
(+1/-1/0), score, confidence — the same interface run_backtest consumes via its
`signals=` argument.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators import engine


def _frame(direction: pd.Series) -> pd.DataFrame:
    direction = direction.fillna(0).astype(int)
    entered = direction != 0
    return pd.DataFrame(
        {
            "score": direction.astype(float),
            "direction": direction,
            "confidence": entered.astype(float),
            "entered": entered,
        }
    )


def mean_reversion_signals(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Fade extremes: long when oversold (below lower Bollinger & RSI<low),
    short when overbought (above upper Bollinger & RSI>high)."""
    p = params or {}
    rsi = engine.rsi(df["close"], p.get("rsi_period", 14))
    bb = engine.bollinger(df["close"], p.get("bb_period", 20))
    rsi_low = p.get("rsi_low", 30)
    rsi_high = p.get("rsi_high", 70)

    long_cond = (df["close"] < bb["lower"]) & (rsi < rsi_low)
    short_cond = (df["close"] > bb["upper"]) & (rsi > rsi_high)
    direction = pd.Series(np.where(long_cond, 1, np.where(short_cond, -1, 0)), index=df.index)
    return _frame(direction)


def opening_range_breakout_signals(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Long on a break above the day's opening-range high, short below its low."""
    p = params or {}
    or_bars = int(p.get("or_bars", 6))  # e.g. 6 x 5min = first 30 min
    if not isinstance(df.index, pd.DatetimeIndex):
        return _frame(pd.Series(0, index=df.index))

    direction = pd.Series(0, index=df.index, dtype=int)
    for _, day_idx in df.groupby(df.index.normalize()).groups.items():
        day = df.loc[day_idx]
        if len(day) <= or_bars:
            continue
        or_high = day["high"].iloc[:or_bars].max()
        or_low = day["low"].iloc[:or_bars].min()
        after = day.iloc[or_bars:]
        direction.loc[after.index[after["close"] > or_high]] = 1
        direction.loc[after.index[after["close"] < or_low]] = -1
    return _frame(direction)
