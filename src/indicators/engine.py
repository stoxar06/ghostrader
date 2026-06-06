"""Technical indicators — pure pandas/numpy (no TA-Lib needed for the lean set).

All functions take a DataFrame with lowercase columns: open, high, low, close, volume.
TA-Lib can be layered on later as an optional fast path for the full library.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI in [0, 100]. No-loss windows -> 100."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift()
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR (>= 0)."""
    return true_range(df).ewm(alpha=1 / period, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(close, period)
    sd = close.rolling(period).std(ddof=0)
    return pd.DataFrame({"mid": mid, "upper": mid + num_std * sd, "lower": mid - num_std * sd})


def vwap(df: pd.DataFrame) -> pd.Series:
    """Session VWAP, reset each calendar day for intraday data."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    tpv = tp * df["volume"]
    if isinstance(df.index, pd.DatetimeIndex):
        day = df.index.normalize()
        cum_tpv = tpv.groupby(day).cumsum()
        cum_vol = df["volume"].groupby(day).cumsum()
    else:
        cum_tpv = tpv.cumsum()
        cum_vol = df["volume"].cumsum()
    return cum_tpv / cum_vol.replace(0.0, np.nan)


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Supertrend line + direction (+1 uptrend, -1 downtrend)."""
    atr_ = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2.0
    upper = (hl2 + multiplier * atr_).to_numpy()
    lower = (hl2 - multiplier * atr_).to_numpy()
    close = df["close"].to_numpy()
    n = len(df)

    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    direction = np.ones(n, dtype=int)
    line = np.full(n, np.nan)

    for i in range(n):
        if i == 0:
            final_upper[i], final_lower[i] = upper[i], lower[i]
            direction[i], line[i] = 1, lower[i]
            continue
        final_upper[i] = (
            upper[i] if (upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1])
            else final_upper[i - 1]
        )
        final_lower[i] = (
            lower[i] if (lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1])
            else final_lower[i - 1]
        )
        if close[i] > final_upper[i - 1]:
            direction[i] = 1
        elif close[i] < final_lower[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
        line[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    return pd.DataFrame({"supertrend": line, "direction": direction}, index=df.index)


def compute_indicators(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Attach the lean indicator columns used by the confluence strategy."""
    p = params or {}
    out = df.copy()
    out["ema_fast"] = ema(df["close"], p.get("ema_fast", 9))
    out["ema_slow"] = ema(df["close"], p.get("ema_slow", 21))
    out["atr"] = atr(df, p.get("atr_period", 14))
    out["rsi"] = rsi(df["close"], p.get("rsi_period", 14))
    out["vwap"] = vwap(df)
    st = supertrend(df, p.get("supertrend_period", 10), p.get("supertrend_multiplier", 3.0))
    out["supertrend"] = st["supertrend"]
    out["supertrend_dir"] = st["direction"]
    return out
