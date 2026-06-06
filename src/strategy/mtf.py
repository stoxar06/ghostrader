"""Multi-timeframe confirmation: a base-timeframe entry must agree with the
higher-timeframe trend before it can fire.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators import engine


def htf_trend(df_htf: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """Higher-timeframe trend direction (+1 up / -1 down / 0 flat) via EMA cross."""
    p = params or {}
    fast = engine.ema(df_htf["close"], p.get("ema_fast", 9))
    slow = engine.ema(df_htf["close"], p.get("ema_slow", 21))
    return np.sign(fast - slow).astype(int)


def align_to_base(htf_series: pd.Series, base_index: pd.Index) -> pd.Series:
    """Forward-fill each higher-TF value onto the base-timeframe bars.

    Each base bar gets the most recent *completed* higher-TF trend value.
    """
    return htf_series.reindex(base_index, method="ffill").fillna(0).astype(int)
