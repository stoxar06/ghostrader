"""Multi-indicator directional accuracy harness (offline)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import indicator_accuracy as ia


def _series(n=400, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    close = 100 + 0.02 * t + 6 * np.sin(t / 9.0) + rng.normal(0, 0.6, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", name="date")
    c = pd.Series(close, index=idx)
    return pd.DataFrame({"open": c, "high": c + 0.6, "low": c - 0.6, "close": c,
                         "volume": pd.Series(1000.0, index=idx)})


def test_every_indicator_builder_returns_valid_stance():
    df = _series()
    for name, info, fn in ia.INDICATORS:
        d = fn(df)
        assert len(d) == len(df), name
        assert set(pd.unique(d.fillna(0).astype(int))).issubset({-1, 0, 1}), name
        assert info  # each indicator has an info line


def test_analyze_pools_rows_with_edge_and_base():
    # Monkeypatch the data loader to avoid network; reuse one synthetic frame.
    df = _series()

    class _Hist:
        def __init__(self, **_):
            pass

        def get(self, *_a, **_k):
            return df

    import src.data.historical as hist_mod
    orig = hist_mod.HistoricalData
    hist_mod.HistoricalData = _Hist
    try:
        res = ia.analyze(symbols=["X", "Y"], horizon=5)
    finally:
        hist_mod.HistoricalData = orig

    assert res["horizon"] == 5
    assert 0.0 <= res["base"] <= 1.0
    names = {r["indicator"] for r in res["rows"]}
    assert names == {n for n, _, _ in ia.INDICATORS}
    for r in res["rows"]:
        if r["signals"] > 0:
            assert 0.0 <= r["accuracy"] <= 1.0


def test_confluence_builders_and_analyze():
    df = _series()
    for name, info, fn in ia.CONFLUENCE:
        d = fn(df)
        assert len(d) == len(df), name
        assert set(pd.unique(d.fillna(0).astype(int))).issubset({-1, 0, 1}), name
        assert info

    class _Hist:
        def __init__(self, **_):
            pass

        def get(self, *_a, **_k):
            return df

    import src.data.historical as hist_mod
    orig = hist_mod.HistoricalData
    hist_mod.HistoricalData = _Hist
    try:
        res = ia.analyze_confluence(symbols=["X", "Y"], horizon=5)
    finally:
        hist_mod.HistoricalData = orig
    assert {r["rule"] for r in res["rows"]} == {n for n, _, _ in ia.CONFLUENCE}
    assert 0.0 <= res["base"] <= 1.0


def test_analyze_volume_reports_baseline_and_confirmed():
    df = _series()

    class _Hist:
        def __init__(self, **_):
            pass

        def get(self, *_a, **_k):
            return df

    import src.data.historical as hist_mod
    orig = hist_mod.HistoricalData
    hist_mod.HistoricalData = _Hist
    try:
        res = ia.analyze_volume(symbols=["X", "Y"], horizon=5)
    finally:
        hist_mod.HistoricalData = orig

    assert {r["indicator"] for r in res["rows"]} == {n for n, _, _ in ia.INDICATORS}
    for r in res["rows"]:
        # volume-confirmed signals are a subset of all signals
        assert r["vol_signals"] >= 0
        if r["base_acc"] == r["base_acc"]:
            assert 0.0 <= r["base_acc"] <= 1.0
