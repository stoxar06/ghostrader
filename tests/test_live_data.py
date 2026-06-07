"""Phase 5 — tick→bar CandleBuilder (offline)."""
from __future__ import annotations

from datetime import datetime

from src.data.live import CandleBuilder


def test_candle_builder_aggregates_and_completes():
    cb = CandleBuilder(interval_seconds=60)
    assert cb.add_tick("A", 100.0, 10, datetime(2024, 1, 1, 9, 15, 0)) is None
    assert cb.add_tick("A", 102.0, 5, datetime(2024, 1, 1, 9, 15, 30)) is None
    assert cb.add_tick("A", 101.0, 5, datetime(2024, 1, 1, 9, 15, 59)) is None

    done = cb.add_tick("A", 103.0, 1, datetime(2024, 1, 1, 9, 16, 1))  # rolls into next minute
    assert done is not None
    assert done["open"] == 100.0 and done["high"] == 102.0
    assert done["low"] == 100.0 and done["close"] == 101.0 and done["volume"] == 20

    df = cb.frame("A")
    assert len(df) == 1
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_candle_builder_per_symbol_isolation():
    cb = CandleBuilder(interval_seconds=60)
    cb.add_tick("A", 10.0, 1, datetime(2024, 1, 1, 9, 15, 0))
    cb.add_tick("B", 20.0, 1, datetime(2024, 1, 1, 9, 15, 0))
    cb.add_tick("A", 11.0, 1, datetime(2024, 1, 1, 9, 16, 1))  # completes A only
    assert len(cb.frame("A")) == 1
    assert len(cb.frame("B")) == 0  # B's first bar is still in progress
