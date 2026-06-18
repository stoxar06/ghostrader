"""Edge-search harness — offline & deterministic (no network, no data/cache).

These tests pin the *honesty math*: that accuracy, the conditional (own-bars) edge, the
60% flag, and the archive labelling all behave as documented. The headline guard is that a
pure long-only "drift rider" can NEVER show positive conditional edge, no matter how high
its raw accuracy climbs in an up-trend.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.backtest import edge_search as es


def ohlc(closes, volumes=None):
    idx = pd.date_range("2010-01-01", periods=len(closes), freq="D", name="datetime")
    c = pd.Series(closes, index=idx, dtype=float)
    v = pd.Series(volumes if volumes is not None else 1_000_000.0, index=idx, dtype=float)
    return pd.DataFrame({"open": c.shift(1).fillna(c.iloc[0]), "high": c * 1.01,
                         "low": c * 0.99, "close": c, "volume": v})


def uptrend(n=1200):
    i = np.arange(n)
    return ohlc(100.0 * (1.0004 ** i) + 2.0 * np.sin(i / 15.0))


# --- builders only ever emit +1 / -1 / 0, aligned to the frame ------------- #
def test_all_builders_emit_only_valid_stances():
    df = uptrend(400)
    for fid, _human, fn, grid in es.RULES:
        for p in grid:
            st = fn(df, p)
            assert len(st) == len(df), f"{fid} length mismatch"
            assert set(pd.Series(st).dropna().unique()).issubset({-1, 0, 1}), f"{fid} bad stance values"


# --- volume + RSI + MA combo families -------------------------------------- #
def test_tri_ma_vol_fires_only_on_confirming_volume():
    n = 400
    i = np.arange(n)
    vols = np.where(i % 2 == 0, 2_000_000.0, 500_000.0)   # alternating high / low volume
    df = ohlc(100.0 * (1.0006 ** i), vols)
    p = {"fast": 5, "mid": 20, "slow": 50, "vol_mult": 1.0}
    base = es._tri_ma(df, p).to_numpy()
    gated = es._tri_ma_vol(df, p).to_numpy()
    vol_ok = es._vol_confirm(df, p).to_numpy()
    assert (gated[~vol_ok] == 0).all()                    # never acts on weak volume
    assert (gated[vol_ok] == base[vol_ok]).all()          # passes MA3 stance through otherwise
    assert (gated == 1).any()                             # and does fire in a confirmed uptrend


def test_rsi_ma_vol_dip_goes_long_on_uptrend_pullback():
    n = 500
    i = np.arange(n)
    closes = 100.0 * (1.0008 ** i)
    closes[300:308] *= 0.985                              # sharp pullback, trend intact
    vols = np.full(n, 1_000_000.0)
    vols[295:315] = 3_000_000.0                           # ... and it comes on volume
    df = ohlc(closes, vols)
    st = es._rsi_ma_vol(df, {"style": "dip", "n": 50, "dip": 45, "vol_mult": 1.0})
    assert (st == 1).any()                                # buys the dip inside the uptrend
    assert not (st == -1).any()                           # never shorts while above the MA


def test_candidate_signals_fire_long_on_volume_dip_above_sma200():
    n = 800
    i = np.arange(n)
    closes = 100.0 * (1.0008 ** i)
    closes[600:610] *= 0.97                               # sharp dip, still above SMA200
    vols = np.full(n, 1_000_000.0)
    vols[595:615] = 3_000_000.0                           # ... on strong volume
    sig = es.candidate_signals(ohlc(closes, vols))
    assert {"entered", "direction"}.issubset(sig.columns)
    fired = sig.loc[sig["entered"], "direction"]
    assert len(fired) > 0 and (fired == 1).all()          # buys the confirmed dip, never shorts


def test_combo_families_are_causal():
    df = uptrend(400)
    df["volume"] = 1_000_000.0 * (1.0 + (np.arange(400) % 3))  # gate sees both states
    for fid in ("tri_ma", "tri_ma_vol", "rsi_ma_vol"):
        _fid, _human, fn, grid = next(r for r in es.RULES if r[0] == fid)
        for p in grid:
            full = fn(df, p)
            for cut in (250, 320):                        # truncating the future never
                part = fn(df.iloc[:cut], p)               # changes a past stance
                assert (part.to_numpy() == full.iloc[:cut].to_numpy()).all(), (fid, p, cut)


# --- literature families: overnight, turn-of-month, low-vol reversal, 52w -- #
def test_overnight_mode_rides_persistent_overnight_gains():
    # All gains arrive overnight (open = prev close * 1.01, flat intraday).
    n = 60
    idx = pd.date_range("2015-01-01", periods=n, freq="D")
    close = pd.Series(100.0 * 1.01 ** np.arange(n), index=idx)
    df = pd.DataFrame({"open": close, "close": close, "high": close, "low": close,
                       "volume": 1e6}, index=idx)
    st = es._overnight_intraday(df, {"mode": "overnight", "n": 5})
    assert (st.iloc[6:] == 1).all()                       # rides the overnight drift


def test_tug_mode_shorts_when_intraday_leads():
    # All gains arrive INTRADAY (zero overnight gap) -> overnight-minus-intraday < 0.
    n = 60
    idx = pd.date_range("2015-01-01", periods=n, freq="D")
    close = pd.Series(100.0 * 1.01 ** np.arange(n), index=idx)
    df = pd.DataFrame({"open": close.shift(1).fillna(close.iloc[0]), "close": close,
                       "high": close, "low": close * 0.99, "volume": 1e6}, index=idx)
    st = es._overnight_intraday(df, {"mode": "tug", "n": 5})
    assert (st.iloc[6:] == -1).all()


def test_turn_of_month_windows_and_rest_stance():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    df = ohlc(np.full(len(idx), 100.0))
    df.index = idx
    flat = es._turn_of_month(df, {"before": 1, "after": 3, "rest": "flat"})
    short = es._turn_of_month(df, {"before": 1, "after": 3, "rest": "short"})
    for period, g in flat.groupby(pd.PeriodIndex(idx, freq="M")):
        assert (g.iloc[:3] == 1).all(), period            # first 3 trading days long
        assert g.iloc[-1] == 1, period                    # last trading day long
        assert (g.iloc[3:-1] == 0).all(), period          # flat in between
    assert ((short == -1) == (flat == 0)).all()           # 'short' bets the complement


def test_low_vol_reversal_fades_only_low_volume_moves():
    n = 80
    closes = np.full(n, 100.0)
    closes[40:] = 90.0                                    # a down move...
    vols = np.full(n, 1_000_000.0)
    vols[41:60] = 200_000.0                               # ...on drying volume
    df = ohlc(closes, vols)
    st = es._low_vol_reversal(df, {"n": 5, "vol_window": 20, "vol_mult": 1.0})
    assert (st.to_numpy()[vols >= 1_000_000.0] == 0).all()  # silent on normal volume
    assert (st.iloc[41:45] == 1).all()                    # fades the low-volume decline


def test_near_52w_long_at_highs_short_at_lows():
    up = ohlc(100.0 * 1.002 ** np.arange(300))
    dn = ohlc(100.0 * 0.998 ** np.arange(300))
    p = {"n": 50, "band": 0.02}
    assert (es._near_52w_extreme(up, p).iloc[60:] == 1).all()    # hugging prior highs
    assert (es._near_52w_extreme(dn, p).iloc[60:] == -1).all()   # hugging prior lows


def test_new_families_are_causal():
    # Truncating the future must never change a past stance. turn_of_month is excluded
    # by design: its month-end flag comes from the (pre-published) exchange calendar,
    # which a truncated frame cannot represent.
    rng = np.random.default_rng(7)
    n = 400
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, n))
    vols = 1_000_000.0 * (1.0 + (np.arange(n) % 4))
    df = ohlc(closes, vols)
    df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0]) * (1.0 + rng.normal(0, 0.003, n))
    for fid in ("overnight", "low_vol_reversal", "near_52w"):
        _fid, _human, fn, grid = next(r for r in es.RULES if r[0] == fid)
        for p in grid:
            full = fn(df, p)
            for cut in (250, 320):
                part = fn(df.iloc[:cut], p)
                assert (part.to_numpy() == full.iloc[:cut].to_numpy()).all(), (fid, p, cut)


# --- _slice_stats arithmetic ---------------------------------------------- #
def test_slice_stats_long_only_has_zero_conditional_edge():
    # 100 long calls, 70 went up -> accuracy 70%, but best single-direction bet on these
    # same bars is also 70% (always long) -> conditional edge is exactly zero.
    a = {"n": 100, "hits": 70, "up": 70, "dn": 30, "long": 100, "short": 0}
    base = {"valid": 200, "up": 120}  # global up-rate 60% -> global base 60%
    s = es._slice_stats(a, base)
    assert s["accuracy"] == 0.70
    assert s["cond_base"] == 0.70
    assert abs(s["cond_edge_pp"]) < 1e-9          # no real edge despite 70% accuracy
    assert abs(s["global_edge_pp"] - 10.0) < 1e-9  # +10pp vs market drift is the misleading number


def test_slice_stats_perfect_two_sided_rule_shows_real_edge():
    # 100 calls, all correct, evenly split long/short -> 100% acc, cond base 50% -> +50pp edge.
    a = {"n": 100, "hits": 100, "up": 50, "dn": 50, "long": 50, "short": 50}
    base = {"valid": 100, "up": 55}
    s = es._slice_stats(a, base)
    assert s["accuracy"] == 1.0
    assert s["cond_base"] == 0.50
    assert abs(s["cond_edge_pp"] - 50.0) < 1e-9


def test_slice_stats_empty_is_safe():
    assert es._slice_stats({"n": 0, "hits": 0, "up": 0, "dn": 0, "long": 0, "short": 0},
                           {"valid": 0, "up": 0}) == {"signals": 0}


# --- evaluate: the drift-rider invariant ----------------------------------- #
def test_evaluate_drift_rider_never_shows_edge():
    frames = {"UP1": uptrend(1300), "UP2": uptrend(1100)}
    rows = es.evaluate(frames, horizons=(5, 10), min_is=50, min_oos=50)
    assert rows, "expected at least some configs with enough signals"
    drift = [r for r in rows if r["family"] == "drift_rider"]
    assert drift, "long-only drift rider should fire enough to be measured on an up-trend"
    for r in drift:
        # high raw accuracy is fine; positive *conditional* edge must be impossible
        assert r["oos"]["cond_edge_pp"] <= 1e-6, r["config"]
        assert not r["real_edge"]


def test_evaluate_rows_sorted_by_oos_conditional_edge():
    frames = {"UP1": uptrend(1300), "UP2": uptrend(1100)}
    rows = es.evaluate(frames, horizons=(5, 10), min_is=50, min_oos=50)
    edges = [r["oos"]["cond_edge_pp"] for r in rows]
    assert edges == sorted(edges, reverse=True)


# --- archive_winners ------------------------------------------------------- #
def _row(config, oos_acc, oos_edge, real_edge):
    raw60 = oos_acc >= es.ACC_TARGET
    return {"config": config, "family": "x", "rule": "r", "horizon": 5,
            "is": {"accuracy": oos_acc, "cond_edge_pp": oos_edge, "signals": 999},
            "oos": {"accuracy": oos_acc, "cond_edge_pp": oos_edge, "signals": 999},
            "raw_60": raw60, "real_edge": real_edge, "drift_rider": raw60 and not real_edge}


def test_archive_writes_only_winners_with_labels(tmp_path):
    rows = [
        _row("good", 0.62, 5.0, True),     # 60%+ AND real edge
        _row("drift", 0.61, 0.0, False),   # 60%+ but drift
        _row("weak", 0.55, 1.0, False),    # below target -> not archived
    ]
    path = tmp_path / "edge_archive.json"
    payload = es.archive_winners(rows, path=str(path))
    assert payload["n_winners"] == 2
    assert payload["n_real_edge"] == 1
    assert payload["n_drift_riders"] == 1
    on_disk = json.loads(path.read_text())
    assert {w["config"] for w in on_disk["winners"]} == {"good", "drift"}
    labels = {w["config"]: w["label"] for w in on_disk["winners"]}
    assert labels == {"good": "real_edge", "drift": "drift_rider"}


# --- stance -> signals adapter -------------------------------------------- #
def test_cross_sectional_detects_a_persistent_winner_spread():
    # Build a panel where one name always trends hardest (persistent winner) and one
    # always falls (persistent loser): the long-short spread must come out positive.
    n = 800
    i = np.arange(n)
    frames = {
        "WIN": ohlc(100.0 * (1.0010 ** i)),     # strongest momentum -> always long
        "MID1": ohlc(100.0 * (1.0002 ** i)),
        "MID2": ohlc(100.0 * (1.0001 ** i)),
        "MID3": ohlc(100.0 + 0.01 * i),
        "LOSE": ohlc(100.0 * (0.9990 ** i)),     # weakest momentum -> always short
    }
    res = es.analyze_cross_sectional(frames, lookbacks=(20,), horizons=(10,),
                                     quantile=0.2, min_oos=10)
    assert res["available"]
    oos = res["rows"][0]["oos"]
    assert oos["signals"] > 0
    assert oos["long_short_spread_pp"] > 0   # winners out-run losers forward -> positive spread


def test_cross_sectional_unavailable_with_too_few_symbols():
    res = es.analyze_cross_sectional({"A": uptrend(400), "B": uptrend(400)})
    assert res["available"] is False


def test_stance_to_signals_contract():
    stance = pd.Series([1, 0, -1, 0, 1], dtype=int)
    sig = es._stance_to_signals(stance)
    assert {"entered", "direction", "score", "confidence"}.issubset(sig.columns)
    assert sig["entered"].tolist() == [True, False, True, False, True]
    assert sig["direction"].tolist() == [1, 0, -1, 0, 1]
