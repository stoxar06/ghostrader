"""Tests for the stock screener (offline, deterministic, synthetic OHLCV)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.invest import screener as sc


def _df(closes):
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range("2023-01-01", periods=len(closes))
    return pd.DataFrame({"open": closes, "high": closes, "low": closes,
                         "close": closes, "volume": 1e6}, index=idx)


def test_metrics_uptrend_is_up_and_near_high():
    df = _df([100 * 1.004 ** i for i in range(300)])     # steady climb, ends at the high
    m = sc.stock_metrics(df)
    assert m["trend_up"] is True
    assert m["ret_1y"] > 0 and m["ret_1m"] > 0
    assert m["from_high_pct"] >= -1.0                    # last bar is at/near the 52w high
    assert m["setup"] == "near-high uptrend"


def test_metrics_downtrend_is_down():
    df = _df([100 * 0.997 ** i for i in range(300)])     # steady decline
    m = sc.stock_metrics(df)
    assert m["trend_up"] is False
    assert m["ret_1y"] < 0
    assert m["from_low_pct"] <= 1.0                       # near its 52w low
    assert m["setup"] in ("downtrend", "oversold downtrend")


def test_metrics_uptrend_pullback_tag():
    up = [100 * 1.005 ** i for i in range(260)]          # long climb...
    pull = [up[-1] * 0.99 ** i for i in range(1, 26)]    # ...then a ~22% pullback, still > SMA200
    m = sc.stock_metrics(_df(up + pull))
    assert m["trend_up"] is True
    assert m["from_high_pct"] <= -12
    assert m["setup"] == "uptrend pullback"


def test_classify_thresholds():
    assert sc.classify({"history": 300, "trend_up": True, "from_high_pct": -1, "rsi": 60}) == "near-high uptrend"
    assert sc.classify({"history": 300, "trend_up": True, "from_high_pct": -20, "rsi": 50}) == "uptrend pullback"
    assert sc.classify({"history": 300, "trend_up": True, "from_high_pct": -8, "rsi": 50}) == "uptrend"
    assert sc.classify({"history": 300, "trend_up": False, "from_high_pct": -40, "rsi": 28}) == "oversold downtrend"
    assert sc.classify({"history": 300, "trend_up": False, "from_high_pct": -40, "rsi": 55}) == "downtrend"
    assert sc.classify({"history": 30, "trend_up": True, "from_high_pct": 0, "rsi": 50}) == "insufficient history"


def test_short_history_is_safe():
    m = sc.stock_metrics(_df([100, 101]))
    assert m["setup"] == "insufficient history"
    assert np.isfinite(m["price"]) and np.isfinite(m["ret_1y"])


def test_screen_groups_and_sorts_by_1y_return():
    strong = _df([100 * 1.004 ** i for i in range(300)])   # big 1y gain
    weak = _df([100 * 1.001 ** i for i in range(300)])     # small 1y gain
    down = _df([100 * 0.998 ** i for i in range(300)])     # loss
    res = sc.screen({"large": {"STRONG.NS": strong, "WEAK.NS": weak, "DOWN.NS": down}})
    block = res["large"]
    assert block["count"] == 3
    assert block["gainers"] == 2 and block["losers"] == 1
    order = [r["symbol"] for r in block["rows"]]
    assert order == ["STRONG.NS", "WEAK.NS", "DOWN.NS"]     # best 1y return first
    assert block["rows"][0]["ret_1y"] >= block["rows"][1]["ret_1y"]


def test_screen_skips_empty_frames():
    res = sc.screen({"mid": {"A.NS": pd.DataFrame(), "B.NS": None}})
    assert res["mid"]["count"] == 0 and res["mid"]["rows"] == []
