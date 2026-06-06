"""Phase 9 — SIP / buy-and-hold analyzer (offline, synthetic)."""
from __future__ import annotations

import math
from datetime import date

import pandas as pd
import pytest

from src.invest.analyzer import lumpsum, sip, xirr


def test_xirr_simple_one_year():
    cfs = [(date(2020, 1, 1), -100.0), (date(2021, 1, 1), 110.0)]
    assert xirr(cfs) == pytest.approx(0.10, abs=1e-3)


def test_xirr_no_sign_change_is_nan():
    cfs = [(date(2020, 1, 1), 100.0), (date(2021, 1, 1), 110.0)]
    assert math.isnan(xirr(cfs))


def _rising(n=400, factor=1.0008):
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    return pd.Series([100 * (factor ** i) for i in range(n)], index=idx, dtype=float)


def test_sip_on_rising_series_is_profitable():
    s = sip(_rising(), 10_000)
    assert s["invested"] > 0
    assert s["final_value"] > s["invested"]
    assert s["xirr_pct"] > 0
    assert s["months"] > 0


def test_lumpsum_rising_positive_cagr():
    lp = lumpsum(_rising())
    assert lp["total_return_pct"] > 0
    assert lp["cagr_pct"] > 0
    assert lp["years"] > 0
