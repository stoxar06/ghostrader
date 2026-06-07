"""Phase 3 — confluence strategy & multi-timeframe gating."""
from __future__ import annotations

import pandas as pd

from src.strategy import confluence


def _df(values, falling=False):
    n = len(values)
    idx = pd.date_range("2024-01-01 09:15", periods=n, freq="5min", name="datetime")
    c = pd.Series(values, index=idx, dtype=float)
    open_off = 0.2 if not falling else -0.2
    return pd.DataFrame(
        {"open": c - open_off, "high": c + 0.5, "low": c - 0.5, "close": c,
         "volume": pd.Series(1000.0, index=idx)}
    )


def rising_df(n=60, start=100.0):
    return _df([start + i for i in range(n)])


def falling_df(n=60, start=200.0):
    return _df([start - i for i in range(n)], falling=True)


CFG = {
    "indicator_params": {"ema_fast": 9, "ema_slow": 21, "atr_period": 14,
                         "supertrend_period": 10, "supertrend_multiplier": 3.0},
    "enabled_signals": {"trend": ["ema_cross", "supertrend"], "volume": ["vwap"],
                        "candlesticks": [], "momentum": []},
    "confidence_threshold": 0.6,
    "require_higher_tf_agreement": False,
}


def test_bullish_confluence_enters_long():
    res = confluence.analyze(rising_df(), CFG)
    last = res.iloc[-1]
    assert last["direction"] == 1
    assert last["confidence"] == 1.0   # all 3 votes +1, equal weights
    assert bool(last["entered"]) is True


def test_score_all_bullish_is_one():
    res = confluence.analyze(rising_df(), CFG)
    assert res["score"].iloc[-1] == 1.0


def test_htf_disagreement_blocks_entry():
    cfg = dict(CFG)
    cfg["require_higher_tf_agreement"] = True
    res = confluence.analyze(rising_df(), cfg, htf_df=falling_df())  # HTF trend is down
    last = res.iloc[-1]
    assert last["direction"] == 1           # base is still bullish
    assert bool(last["entered"]) is False   # but opposing HTF blocks the entry


def test_latest_signal_helper():
    sig = confluence.latest_signal(confluence.analyze(rising_df(), CFG))
    assert sig.direction == 1
    assert sig.entered is True
    assert "ema_cross" in sig.votes


def test_volatility_gate_reduces_entries():
    df = rising_df()
    base = confluence.analyze(df, CFG)["entered"].sum()
    cfg = dict(CFG, min_atr_pct=100.0)  # absurdly high ATR% -> nothing qualifies
    gated = confluence.analyze(df, cfg)["entered"].sum()
    assert gated < base
    assert gated == 0


def test_trade_window_filters_out_of_window_bars():
    cfg = dict(CFG, trade_window={"start": "00:00", "end": "00:01"})  # excludes all 09:15+ bars
    assert confluence.analyze(rising_df(), cfg)["entered"].sum() == 0


def test_full_confluence_is_stricter():
    df = rising_df()
    base = confluence.analyze(df, CFG)["entered"].sum()
    strict = confluence.analyze(df, dict(CFG, full_confluence=True))["entered"].sum()
    assert strict <= base


def test_invert_flips_direction_only():
    df = rising_df()
    normal = confluence.analyze(df, CFG)
    inv = confluence.analyze(df, dict(CFG, invert=True))
    assert (inv["direction"] == -normal["direction"]).all()        # opposite direction
    assert (inv["entered"] == normal["entered"]).all()             # same bars traded
