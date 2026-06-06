"""Phase 2 data-layer tests — offline, using a fake provider (no network)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.historical import HistoricalData, OHLCV_COLUMNS, YFinanceProvider


class FakeProvider:
    """Deterministic in-memory provider so tests never hit the network."""

    def __init__(self):
        self.calls = 0

    def fetch(self, symbol, interval, start=None, end=None, period=None):
        self.calls += 1
        idx = pd.date_range("2024-01-01 09:15", periods=10, freq="5min", name="datetime")
        return pd.DataFrame(
            {
                "open": [100 + i for i in range(10)],
                "high": [101 + i for i in range(10)],
                "low": [99 + i for i in range(10)],
                "close": [100.5 + i for i in range(10)],
                "volume": [1000 + i for i in range(10)],
            },
            index=idx,
        )


def test_get_returns_ohlcv_and_caches(tmp_path):
    fake = FakeProvider()
    hd = HistoricalData(cache_dir=str(tmp_path), provider=fake)

    df1 = hd.get("RELIANCE", "5minute")
    assert list(df1.columns) == OHLCV_COLUMNS
    assert len(df1) == 10
    assert fake.calls == 1
    assert (tmp_path / "RELIANCE_5minute.csv").exists()

    # Second call is served from cache — provider is not hit again.
    df2 = hd.get("RELIANCE", "5minute")
    assert fake.calls == 1
    assert len(df2) == 10


def test_refresh_forces_refetch(tmp_path):
    fake = FakeProvider()
    hd = HistoricalData(cache_dir=str(tmp_path), provider=fake)
    hd.get("INFY", "5minute")
    hd.get("INFY", "5minute", refresh=True)
    assert fake.calls == 2


def test_warmup_returns_tail(tmp_path):
    hd = HistoricalData(cache_dir=str(tmp_path), provider=FakeProvider())
    df = hd.warmup("TCS", "5minute", bars=3)
    assert len(df) == 3
    # tail() keeps the most recent rows
    assert df["close"].iloc[-1] == 109.5


def test_unknown_interval_raises_before_import():
    # Validates fast without needing yfinance installed.
    with pytest.raises(ValueError):
        YFinanceProvider().fetch("RELIANCE", "weekly")
