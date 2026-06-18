"""Tests for the 1-12 day holding-horizon filter (offline, synthetic data)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import horizon as hz
from src.backtest.engine import Costs


def _df(closes):
    idx = pd.bdate_range("2024-01-01", periods=len(closes))
    return pd.DataFrame({"close": np.asarray(closes, dtype=float)}, index=idx)


def _signals(df, directions):
    d = pd.Series(directions, index=df.index, dtype=int)
    return pd.DataFrame({"direction": d, "entered": d != 0,
                         "score": d.astype(float), "confidence": (d != 0).astype(float)})


def test_signal_trades_returns_and_exit_dates():
    df = _df([100, 110, 121, 133.1, 146.41])  # +10% per bar
    sig = _signals(df, [1, 0, 0, 0, 0])
    t = hz.signal_trades(df, sig, horizon=2, cost_pct=0.01)
    assert len(t) == 1
    assert t.iloc[0]["exit_date"] == df.index[2]
    assert t.iloc[0]["gross_ret"] == pytest.approx(0.21)
    assert t.iloc[0]["net_ret"] == pytest.approx(0.20)


def test_signal_trades_drops_incomplete_holds():
    df = _df([100, 101, 102, 103])
    sig = _signals(df, [1, 1, 1, 1])  # signals near the end can't complete a 3-bar hold
    t = hz.signal_trades(df, sig, horizon=3, cost_pct=0.0)
    assert len(t) == 1  # only the first signal has 3 forward bars
    assert t.iloc[0]["entry_date"] == df.index[0]


def test_short_signals_profit_when_price_falls():
    df = _df([100, 90, 81, 72.9])
    sig = _signals(df, [-1, 0, 0, 0])
    t = hz.signal_trades(df, sig, horizon=1, cost_pct=0.0)
    assert t.iloc[0]["gross_ret"] == pytest.approx(0.10)


def test_daily_pnl_totals_reconcile_with_trades():
    df = _df([100, 102, 104, 106, 108, 110, 112])
    sig = _signals(df, [1, 1, 0, 1, 0, 0, 0])
    t = hz.signal_trades(df, sig, horizon=2, cost_pct=0.001)
    daily = hz.daily_pnl(t)
    assert daily["trades"].sum() == len(t)
    assert daily["pnl_pct"].sum() == pytest.approx(t["net_ret"].sum() * 100)


def test_daily_pnl_groups_same_exit_day():
    df = _df([100, 100, 100, 100])
    sig = _signals(df, [1, 1, 0, 0])
    # horizon 2 from bar0 and bar1 exit on bars 2 and 3 -> two separate days
    t = hz.signal_trades(df, sig, horizon=2, cost_pct=0.0)
    assert len(hz.daily_pnl(t)) == 2


def test_horizon_table_drift_rider_shows_zero_edge():
    # Monotonic uptrend: longs win 100% but base up-rate is also 100% ->
    # edge_vs_base must be ~0 (riding drift is not a signal).
    df = _df([100 * 1.01 ** i for i in range(40)])
    sig = _signals(df, [1] * 40)
    tbl = hz.horizon_table(df, sig, horizons=(1, 5), cost_pct=0.0)
    assert (tbl["win_rate"] == 1.0).all()
    assert (tbl["base_up_rate"] == 1.0).all()
    assert (tbl["edge_vs_base"].abs() < 1e-12).all()


def test_horizon_table_costs_can_flip_total_negative():
    df = _df([100, 100.05, 100.1, 100.15, 100.2, 100.25])
    sig = _signals(df, [1, 1, 1, 1, 1, 0])
    cheap = hz.horizon_table(df, sig, horizons=(1,), cost_pct=0.0)
    costly = hz.horizon_table(df, sig, horizons=(1,), cost_pct=0.01)
    assert cheap.iloc[0]["total_pnl_pct"] > 0
    assert costly.iloc[0]["total_pnl_pct"] < 0


def test_horizon_table_counts_profit_and_loss_days():
    df = _df([100, 110, 99, 108.9, 98.01])
    sig = _signals(df, [1, 1, 1, 1, 0])
    tbl = hz.horizon_table(df, sig, horizons=(1,), cost_pct=0.0)
    row = tbl.iloc[0]
    assert row["signals"] == 4
    assert row["profit_days"] == 2 and row["loss_days"] == 2
    assert row["profit_days"] + row["loss_days"] <= row["signals"]


def test_round_trip_cost_pct_positive_and_small():
    pct = hz.round_trip_cost_pct(Costs())
    assert 0 < pct < 0.01  # well under 1% per round trip
