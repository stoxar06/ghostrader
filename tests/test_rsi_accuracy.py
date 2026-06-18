"""RSI accuracy harness — directional hit-rate measurement (offline)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import rsi_accuracy as ra


def _mean_reverting(n=400, seed=0):
    """Oscillating series so RSI hits both extremes and there are signals."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    close = 100 + 8 * np.sin(t / 7.0) + rng.normal(0, 0.5, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", name="date")
    c = pd.Series(close, index=idx)
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c,
                         "volume": pd.Series(1000.0, index=idx)})


def test_directional_accuracy_columns_and_ranges():
    out = ra.directional_accuracy(_mean_reverting(), horizons=(1, 3, 5))
    assert list(out["horizon"]) == [1, 3, 5]
    assert {"signals", "accuracy", "base_up_rate", "edge_vs_base"}.issubset(out.columns)
    valid = out[out["signals"] > 0]
    assert (valid["accuracy"].between(0, 1)).all()
    assert (valid["base_up_rate"].between(0, 1)).all()


def test_directional_accuracy_finds_signals_on_oscillator():
    out = ra.directional_accuracy(_mean_reverting(), horizons=(5,))
    assert int(out["signals"].iloc[0]) > 0  # extremes are actually hit
