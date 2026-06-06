"""Phase 3 — indicator correctness on synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators import engine


def make_df(closes, spread=0.5):
    idx = pd.date_range("2024-01-01 09:15", periods=len(closes), freq="5min", name="datetime")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame(
        {
            "open": c.shift().fillna(c.iloc[0]),
            "high": c + spread,
            "low": c - spread,
            "close": c,
            "volume": pd.Series(1000.0, index=idx),
        }
    )


def test_ema_of_constant_is_constant():
    s = pd.Series([5.0] * 20)
    assert np.allclose(engine.ema(s, 9), 5.0)


def test_atr_nonnegative():
    df = make_df(list(range(100, 140)))
    a = engine.atr(df, 14)
    assert len(a) == len(df)
    assert (a.dropna() >= 0).all()


def test_rsi_within_bounds():
    df = make_df([100 + (i % 5) for i in range(60)])
    r = engine.rsi(df["close"], 14)
    assert r.min() >= 0.0 and r.max() <= 100.0


def test_vwap_within_price_range():
    df = make_df(list(range(100, 160)))
    v = engine.vwap(df).dropna()
    assert v.min() >= df["low"].min() - 1e-9
    assert v.max() <= df["high"].max() + 1e-9


def test_supertrend_detects_uptrend():
    df = make_df(list(range(100, 160)))
    st = engine.supertrend(df, 10, 3.0)
    assert set(st["direction"].unique()).issubset({-1, 1})
    assert st["direction"].iloc[-1] == 1


def test_compute_indicators_adds_columns():
    out = engine.compute_indicators(make_df(list(range(100, 160))), {})
    for col in ["ema_fast", "ema_slow", "atr", "rsi", "vwap", "supertrend", "supertrend_dir"]:
        assert col in out.columns
