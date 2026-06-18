"""Self-improving search — offline & deterministic.

Pins the honesty rails: mutation stays in valid regions, a planted lookahead rule
is quarantined (never enshrined), fitness requires edge in BOTH slices, state
resumes across sessions, and the luck-verdict math behaves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import auto_search as a

from .test_edge_search import ohlc, uptrend


def test_mutate_respects_ordering_and_clamps():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = a.mutate("tri_ma", {"fast": 5, "mid": 20, "slow": 50}, rng)
        assert p["fast"] < p["mid"] < p["slow"]
        r = a.mutate("rsi_ma_vol", {"style": "dip", "n": 200, "dip": 35, "vol_mult": 1.5}, rng)
        assert r["style"] == "dip"                       # strings pass through untouched
        assert 5 <= r["dip"] <= 49 and 0.5 <= r["vol_mult"] <= 5.0


def test_mutate_is_deterministic_with_seed():
    p = {"fast": 9, "mid": 21, "slow": 50}
    one = a.mutate("tri_ma", p, np.random.default_rng(7))
    two = a.mutate("tri_ma", p, np.random.default_rng(7))
    assert one == two


def _cheat_rule(df, p):
    """Deliberate lookahead: stance = sign of the NEXT bar's move (must be quarantined)."""
    nxt = np.sign(df["close"].shift(-1) - df["close"])
    return pd.Series(nxt, index=df.index).fillna(0).astype(int)


def test_lookahead_cheat_is_quarantined_not_enshrined():
    frames = {"UP1": uptrend(900), "UP2": uptrend(800)}
    rules = [("cheat", "lookahead cheat", _cheat_rule, [{}])]
    st = a.evolve(frames, generations=1, seed=1, horizons=(1,), min_is=50, min_oos=50,
                  rules=rules, state=a.fresh_state())
    assert any(s["config"].startswith("cheat") for s in st["suspects"])
    assert not any(w["config"].startswith("cheat") for w in st["hall_of_fame"])


def test_evolve_runs_resumes_and_stops_honestly(tmp_path):
    frames = {"UP1": uptrend(900), "UP2": uptrend(800)}
    rules = [r for r in a.RULES if r[0] in ("ema_cross", "drift_rider", "always_long")]
    st = a.evolve(frames, generations=2, seed=3, horizons=(3, 5), min_is=50, min_oos=50,
                  rules=rules, state=a.fresh_state())
    assert st["generations"] >= 1 and st["configs_tried"] > 0
    assert st["last_stop"]                               # always reports why it stopped
    fits = [w["fitness_pp"] for w in st["hall_of_fame"]]
    assert fits == sorted(fits, reverse=True)            # hall of fame is ranked

    path = tmp_path / "auto.json"
    a.save_state(st, path=str(path))
    st2 = a.load_state(path=str(path))
    assert st2["configs_tried"] == st["configs_tried"]   # state round-trips
    st3 = a.evolve(frames, generations=1, seed=4, horizons=(3, 5), min_is=50, min_oos=50,
                   rules=rules, state=st2)
    assert st3["configs_tried"] > st["configs_tried"]    # resumed, not restarted


def test_verdict_flags_luck_for_small_edges():
    st = a.fresh_state()
    st["configs_tried"] = 300
    st["hall_of_fame"] = [{"config": "x", "family": "x", "params": {}, "fitness_pp": 1.0,
                           "horizon": 5, "oos_accuracy": 0.55, "oos_cond_edge_pp": 2.0,
                           "is_cond_edge_pp": 1.0, "oos_signals": 200}]
    v = a.verdict(st)
    assert v["meaningful"] is False                      # +2pp on n=200 after 300 tries = luck
    assert a.verdict(a.fresh_state())["meaningful"] is False
