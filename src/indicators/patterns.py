"""Candlestick patterns — pure pandas (lean set), with an optional TA-Lib fast path.

Each detector returns an int Series aligned to df: +1 bullish, -1 bearish, 0 none.
The lean set (engulfing, hammer) needs no system dependency. When TA-Lib is
installed, `detect()` will fall back to its CDL* functions for any other named
pattern, unlocking the full 60+ library.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _body(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs()


def engulfing(df: pd.DataFrame) -> pd.Series:
    """Bullish (+1) / bearish (-1) engulfing; 0 otherwise."""
    o, c = df["open"], df["close"]
    po, pc = o.shift(), c.shift()
    prev_bear, prev_bull = pc < po, pc > po
    curr_bull, curr_bear = c > o, c < o

    bull = prev_bear & curr_bull & (o <= pc) & (c >= po)
    bear = prev_bull & curr_bear & (o >= pc) & (c <= po)

    out = pd.Series(0, index=df.index, dtype=int)
    out[bull] = 1
    out[bear] = -1
    return out


def hammer(df: pd.DataFrame) -> pd.Series:
    """Bullish hammer (+1): small body near the top, long lower shadow, tiny upper shadow."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    body_low = pd.concat([o, c], axis=1).min(axis=1)
    body_high = pd.concat([o, c], axis=1).max(axis=1)
    lower_shadow = body_low - l
    upper_shadow = h - body_high

    is_hammer = (body > 0) & (lower_shadow >= 2 * body) & (upper_shadow <= body)
    out = pd.Series(0, index=df.index, dtype=int)
    out[is_hammer] = 1
    return out


def is_doji(df: pd.DataFrame, frac: float = 0.1) -> pd.Series:
    """Boolean: real body <= `frac` of the bar's range (indecision)."""
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    return ((_body(df) / rng).fillna(1.0) <= frac)


# Pure-Python registry (directional votes).
PATTERN_FUNCS = {
    "engulfing": engulfing,
    "hammer": hammer,
}


def talib_available() -> bool:
    try:
        import talib  # noqa: F401

        return True
    except Exception:
        return False


def detect(name: str, df: pd.DataFrame) -> pd.Series:
    """Detect a pattern by name. Pure-Python first; TA-Lib CDL* fallback if installed."""
    if name in PATTERN_FUNCS:
        return PATTERN_FUNCS[name](df)

    if talib_available():  # optional full library
        import talib

        fn_name = "CDL" + name.upper().replace("_", "")
        fn = getattr(talib, fn_name, None)
        if fn is not None:
            raw = fn(df["open"], df["high"], df["low"], df["close"])
            return np.sign(raw).astype(int)

    raise KeyError(
        f"Unknown pattern {name!r}. Built-in: {sorted(PATTERN_FUNCS)}. "
        "Install TA-Lib for the full CDL* library."
    )
