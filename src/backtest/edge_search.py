"""Out-of-sample edge search — hunt for any rule config that clears 60% accuracy.

This is the honest answer to "get me 60%+ accuracy on each trade". It sweeps a broad
catalog of *causal* (no-lookahead) rule families across a parameter grid and several
forward horizons, on a chronological **train / test split**, and for every config asks
three separate questions that "accuracy" alone hides:

1. **Raw accuracy** — does the next-N-bar move agree with the rule's stance? (the literal
   "60%" the user wants). Reported on the held-out **out-of-sample** slice, because a
   number picked on the same data it was tuned on is meaningless.
2. **Real edge vs drift** — markets drift up, so on its *own signal bars* a rule is
   compared against `max(P(up), P(down))` (the best you'd do by always betting one
   direction on exactly those bars). `cond_edge = accuracy - that`. A long-only rule in a
   bull market has `cond_edge ~= 0`: 60% accuracy with no edge is just the drift, not skill.
3. **Tradeable** — the top candidates are run through the costed backtest engine; a high
   hit-rate that loses money after fees/slippage is not an edge.

Every config that clears the accuracy target on the held-out slice is **archived** to
`data/edge_archive.json`, each tagged `real_edge` or `drift_rider` so the archive can never
be mistaken for "I found a money printer".

Run: `python -m src edgesearch`
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.indicators import engine
from src.logutil import get_logger

log = get_logger(__name__)

HORIZONS = (1, 3, 5, 10, 20, 40, 60)
DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_MIN_IS = 300       # min in-sample signals to even consider a config
DEFAULT_MIN_OOS = 150      # min out-of-sample signals (else the OOS accuracy is noise)
ACC_TARGET = 0.60          # the "60%" bar the user asked for
EDGE_MIN_PP = 2.0          # conditional edge (pp) over its own-bars drift to call it "real"
ARCHIVE_PATH = "data/edge_archive.json"


def _sign(s: pd.Series) -> pd.Series:
    return pd.Series(np.sign(s), index=s.index).fillna(0).astype(int)


# --------------------------------------------------------------------------- #
# Rule families. Each builder returns a +1/-1/0 stance per bar, computed ONLY  #
# from data up to and including that bar (causal — breakout channels are       #
# shifted by 1 so a bar never "breaks out" of a range that includes itself).   #
# --------------------------------------------------------------------------- #
def _ema_cross(df, p):
    return _sign(engine.ema(df["close"], p["fast"]) - engine.ema(df["close"], p["slow"]))


def _momentum(df, p):
    return _sign(df["close"] / df["close"].shift(p["lookback"]) - 1.0)


def _rsi_reversion(df, p):
    r = engine.rsi(df["close"], p.get("period", 14))
    return pd.Series(np.where(r < p["low"], 1, np.where(r > p["high"], -1, 0)), index=df.index)


def _rsi_pro_trend(df, p):
    """RSI used WITH the trend: long strong RSI above SMA200, short weak RSI below."""
    r = engine.rsi(df["close"], p.get("period", 14))
    above = df["close"] > engine.sma(df["close"], 200)
    long = above & (r > p.get("mid", 50))
    short = (~above) & (r < p.get("mid", 50))
    return pd.Series(np.where(long, 1, np.where(short, -1, 0)), index=df.index)


def _boll_reversion(df, p):
    bb = engine.bollinger(df["close"], p.get("period", 20), p.get("num_std", 2.0))
    return pd.Series(np.where(df["close"] < bb["lower"], 1,
                              np.where(df["close"] > bb["upper"], -1, 0)), index=df.index)


def _ret_zscore_fade(df, p):
    """Fade extreme single-bar moves: z-score of 1-bar return over a rolling window."""
    ret = df["close"].pct_change()
    mu = ret.rolling(p["window"]).mean()
    sd = ret.rolling(p["window"]).std(ddof=0)
    z = (ret - mu) / sd.replace(0.0, np.nan)
    k = p["k"]
    return pd.Series(np.where(z < -k, 1, np.where(z > k, -1, 0)), index=df.index).fillna(0).astype(int)


def _streak_fade(df, p):
    """After n consecutive down closes -> long (oversold); n up closes -> short."""
    up = df["close"].diff() > 0
    dn = df["close"].diff() < 0
    up_streak = up.rolling(p["n"]).sum() == p["n"]
    dn_streak = dn.rolling(p["n"]).sum() == p["n"]
    return pd.Series(np.where(dn_streak, 1, np.where(up_streak, -1, 0)), index=df.index)


def _donchian_breakout(df, p):
    """Break above the PRIOR N-bar high -> long; below prior N-bar low -> short."""
    hi = df["high"].rolling(p["n"]).max().shift(1)
    lo = df["low"].rolling(p["n"]).min().shift(1)
    return pd.Series(np.where(df["close"] > hi, 1, np.where(df["close"] < lo, -1, 0)), index=df.index)


def _gap_fade(df, p):
    """Fade large overnight gaps: today's open vs prior close, in ATR units."""
    gap = (df["open"] - df["close"].shift(1)) / engine.atr(df, p.get("atr_period", 14)).replace(0.0, np.nan)
    k = p["k"]
    return pd.Series(np.where(gap < -k, 1, np.where(gap > k, -1, 0)), index=df.index).fillna(0).astype(int)


def _vol_regime_trend(df, p):
    """Trend-follow (EMA cross) only in LOW volatility (ATR% below its rolling median)."""
    atrp = engine.atr(df, 14) / df["close"]
    low_vol = atrp < atrp.rolling(p.get("window", 100)).median()
    trend = _ema_cross(df, {"fast": p.get("fast", 20), "slow": p.get("slow", 50)})
    return pd.Series(np.where(low_vol, trend, 0), index=df.index).astype(int)


def _vol_regime_reversion(df, p):
    """Bollinger reversion only in HIGH volatility (ATR% above its rolling median)."""
    atrp = engine.atr(df, 14) / df["close"]
    high_vol = atrp > atrp.rolling(p.get("window", 100)).median()
    rev = _boll_reversion(df, {"period": p.get("period", 20), "num_std": p.get("num_std", 2.0)})
    return pd.Series(np.where(high_vol, rev, 0), index=df.index).astype(int)


def _above_sma_long_only(df, p):
    """Pure regime/drift rider: long whenever close > SMA(n), else flat. Included on
    purpose to show what 'riding the drift' scores — its cond_edge should be ~0."""
    return pd.Series(np.where(df["close"] > engine.sma(df["close"], p.get("n", 200)), 1, 0),
                     index=df.index).astype(int)


def _always_long(df, p):
    """The ultimate drift control: always long, every bar. Its accuracy IS the base
    up-rate, so it shows the cleanest 'how often is price simply higher later'. By
    construction its conditional edge is <= 0 — never a tradeable signal."""
    return pd.Series(1, index=df.index, dtype=int)


def _ensemble_vote(df, p):
    """K-of-N agreement across weak trend signals; act only on high conviction."""
    close = df["close"]
    votes = [
        _sign(engine.ema(close, 9) - engine.ema(close, 21)),
        _sign(engine.macd(close)["hist"]),
        engine.supertrend(df, 10, 3.0)["direction"].astype(int),
        _sign(close / close.shift(20) - 1.0),
        _sign(close - engine.sma(close, 200)),
    ]
    s = sum(votes)
    k = p["k"]
    return pd.Series(np.where(s >= k, 1, np.where(s <= -k, -1, 0)), index=df.index).astype(int)


def _adaptive_rsi(df, p):
    """Self-calibrating RSI: long when RSI below its own rolling p-low percentile,
    short above its rolling p-high percentile (replaces the fixed 30/70)."""
    r = engine.rsi(df["close"], p.get("period", 14))
    w = p.get("window", 250)
    lo = r.rolling(w).quantile(p.get("q_low", 0.10))
    hi = r.rolling(w).quantile(p.get("q_high", 0.90))
    return pd.Series(np.where(r < lo, 1, np.where(r > hi, -1, 0)), index=df.index).fillna(0).astype(int)


def _range_position_fade(df, p):
    """Stochastic-style: fade the close's location within the recent high-low range."""
    n = p.get("n", 14)
    hi = df["high"].rolling(n).max()
    lo = df["low"].rolling(n).min()
    pos = (df["close"] - lo) / (hi - lo).replace(0.0, np.nan)
    return pd.Series(np.where(pos < p.get("low", 0.1), 1,
                              np.where(pos > p.get("high", 0.9), -1, 0)),
                     index=df.index).fillna(0).astype(int)


def _vol_confirm(df, p):
    """Above-average-volume gate (same definition as the `volume` accuracy harness)."""
    if "volume" not in df:
        return pd.Series(False, index=df.index)
    avg = df["volume"].rolling(p.get("vol_window", 20)).mean()
    return df["volume"] > avg * p.get("vol_mult", 1.0)


def _tri_ma(df, p):
    """'MA3' alignment: long when SMA(fast) > SMA(mid) > SMA(slow), short when inverted."""
    c = df["close"]
    f, m, s = engine.sma(c, p["fast"]), engine.sma(c, p["mid"]), engine.sma(c, p["slow"])
    return pd.Series(np.where((f > m) & (m > s), 1, np.where((f < m) & (m < s), -1, 0)),
                     index=df.index).astype(int)


def _tri_ma_vol(df, p):
    """'MA3' alignment, acted on only when volume runs above its rolling average."""
    return pd.Series(np.where(_vol_confirm(df, p), _tri_ma(df, p), 0),
                     index=df.index).astype(int)


def _rsi_ma_vol(df, p):
    """The retail classic, tested honestly: MA trend filter + RSI gate + volume confirm.

    style='pro': momentum — long above SMA(n) with RSI > mid, short below with RSI < 100-mid.
    style='dip': pullback — long above SMA(n) when RSI < dip, short below when RSI > 100-dip.
    """
    r = engine.rsi(df["close"], p.get("period", 14))
    ma = engine.sma(df["close"], p.get("n", 50))
    above, below = df["close"] > ma, df["close"] < ma   # NaN warmup -> neither side
    if p.get("style") == "dip":
        long_, short_ = above & (r < p["dip"]), below & (r > 100 - p["dip"])
    else:
        long_, short_ = above & (r > p.get("mid", 50)), below & (r < 100 - p.get("mid", 50))
    st = np.where(long_, 1, np.where(short_, -1, 0))
    return pd.Series(np.where(_vol_confirm(df, p), st, 0), index=df.index).astype(int)


def _overnight_intraday(df, p):
    """Overnight/intraday decomposition (Lou, Polk & Skouras 2019 'tug of war'):
    in the literature overnight returns (prev close -> open) persist while intraday
    returns (open -> close) revert.
    mode='overnight': ride the sign of the trailing n-day overnight component.
    mode='tug': long when the overnight component leads intraday, short when it lags."""
    overnight = df["open"] / df["close"].shift(1) - 1.0
    intraday = df["close"] / df["open"] - 1.0
    s = (overnight - intraday) if p.get("mode") == "tug" else overnight
    return _sign(s.rolling(p["n"]).sum())


def _turn_of_month(df, p):
    """Turn-of-month seasonality (Lakonishok & Smidt 1988): monthly returns concentrate
    in the last `before` + first `after` trading days. Long inside that window;
    rest='short' bets the complement (flat-to-weak mid-month), rest='flat' stays out —
    the long-only form exists to show cond_edge ~ 0, like the drift rider.
    Month boundaries come from the exchange calendar, which is published in advance,
    so spotting "last trading day of the month" is causal in real time."""
    pos = pd.Series(0, index=df.index).groupby(pd.PeriodIndex(df.index, freq="M"))
    from_start = pos.cumcount()
    from_end = pos.transform("size") - 1 - from_start
    active = (from_start < p.get("after", 3)) | (from_end < p.get("before", 1))
    rest = -1 if p.get("rest") == "short" else 0
    return pd.Series(np.where(active, 1, rest), index=df.index).astype(int)


def _low_vol_reversal(df, p):
    """Low-volume reversal (Campbell, Grossman & Wang 1993): moves made on LOW volume
    reverse harder. Fade the trailing n-bar return only when volume runs below its
    rolling average — the mirror image of the volume-confirmation gate."""
    if "volume" not in df:
        return pd.Series(0, index=df.index, dtype=int)
    low_vol = df["volume"] < (df["volume"].rolling(p.get("vol_window", 20)).mean()
                              * p.get("vol_mult", 1.0))
    fade = -_sign(df["close"].pct_change(p.get("n", 5)))
    return pd.Series(np.where(low_vol, fade, 0), index=df.index).astype(int)


def _near_52w_extreme(df, p):
    """52-week-high momentum (George & Hwang 2004), time-series form: anchoring makes
    traders under-react near salient extremes — long within `band` of the PRIOR n-bar
    high, short within `band` of the prior n-bar low (shift 1 so a bar never measures
    against itself)."""
    hi = df["high"].rolling(p.get("n", 252)).max().shift(1)
    lo = df["low"].rolling(p.get("n", 252)).min().shift(1)
    band = p.get("band", 0.02)
    return pd.Series(np.where(df["close"] >= hi * (1.0 - band), 1,
                              np.where(df["close"] <= lo * (1.0 + band), -1, 0)),
                     index=df.index).astype(int)


# (family_id, human label, builder, [param grid]) -------------------------- #
RULES = [
    ("ema_cross", "EMA fast/slow cross (trend)", _ema_cross,
     [{"fast": 9, "slow": 21}, {"fast": 20, "slow": 50}, {"fast": 50, "slow": 200}]),
    ("momentum", "N-bar momentum sign", _momentum,
     [{"lookback": 10}, {"lookback": 20}, {"lookback": 60}, {"lookback": 120}]),
    ("rsi_reversion", "RSI fade extremes", _rsi_reversion,
     [{"low": 30, "high": 70}, {"low": 20, "high": 80}, {"low": 10, "high": 90}]),
    ("rsi_pro_trend", "RSI with the SMA200 trend", _rsi_pro_trend,
     [{"mid": 50}, {"mid": 55}]),
    ("boll_reversion", "Bollinger band fade", _boll_reversion,
     [{"period": 20, "num_std": 2.0}, {"period": 20, "num_std": 2.5}]),
    ("ret_z_fade", "Fade N-sigma single-bar move", _ret_zscore_fade,
     [{"window": 20, "k": 1.5}, {"window": 20, "k": 2.0}, {"window": 20, "k": 2.5}, {"window": 60, "k": 2.0}]),
    ("streak_fade", "Fade consecutive-close streaks", _streak_fade,
     [{"n": 3}, {"n": 4}, {"n": 5}]),
    ("donchian", "Donchian channel breakout", _donchian_breakout,
     [{"n": 20}, {"n": 55}, {"n": 120}]),
    ("gap_fade", "Fade overnight gaps (ATR units)", _gap_fade,
     [{"k": 1.0}, {"k": 1.5}, {"k": 2.0}]),
    ("vol_trend", "Trend only in low-vol regime", _vol_regime_trend,
     [{"fast": 20, "slow": 50, "window": 100}, {"fast": 50, "slow": 200, "window": 100}]),
    ("vol_reversion", "Reversion only in high-vol regime", _vol_regime_reversion,
     [{"period": 20, "num_std": 2.0, "window": 100}]),
    ("drift_rider", "Long-only above SMA200 (drift control)", _above_sma_long_only,
     [{"n": 200}, {"n": 50}]),
    ("always_long", "Always long, every bar (= P(price higher later))", _always_long,
     [{}]),
    ("ensemble", "K-of-5 trend ensemble vote", _ensemble_vote,
     [{"k": 3}, {"k": 4}, {"k": 5}]),
    ("adaptive_rsi", "Percentile-adaptive RSI fade", _adaptive_rsi,
     [{"window": 250, "q_low": 0.10, "q_high": 0.90}, {"window": 250, "q_low": 0.05, "q_high": 0.95}]),
    ("range_fade", "Fade close position in range", _range_position_fade,
     [{"n": 14, "low": 0.1, "high": 0.9}, {"n": 14, "low": 0.05, "high": 0.95}]),
    ("tri_ma", "Triple-MA alignment (MA3)", _tri_ma,
     [{"fast": 5, "mid": 20, "slow": 50}, {"fast": 9, "mid": 21, "slow": 50},
      {"fast": 20, "mid": 50, "slow": 200}]),
    ("tri_ma_vol", "Triple-MA alignment + volume confirm", _tri_ma_vol,
     [{"fast": 5, "mid": 20, "slow": 50, "vol_mult": 1.0},
      {"fast": 9, "mid": 21, "slow": 50, "vol_mult": 1.5},
      {"fast": 20, "mid": 50, "slow": 200, "vol_mult": 1.0}]),
    ("rsi_ma_vol", "MA trend + RSI gate + volume confirm", _rsi_ma_vol,
     [{"style": "pro", "n": 50, "mid": 50, "vol_mult": 1.0},
      {"style": "pro", "n": 200, "mid": 55, "vol_mult": 1.0},
      {"style": "dip", "n": 50, "dip": 40, "vol_mult": 1.0},
      {"style": "dip", "n": 200, "dip": 35, "vol_mult": 1.5}]),
    ("overnight", "Overnight/intraday tug-of-war (Lou-Polk-Skouras)", _overnight_intraday,
     [{"mode": "overnight", "n": 5}, {"mode": "overnight", "n": 20},
      {"mode": "tug", "n": 5}, {"mode": "tug", "n": 20}]),
    ("turn_of_month", "Turn-of-month seasonality (Lakonishok-Smidt)", _turn_of_month,
     [{"before": 1, "after": 3, "rest": "flat"}, {"before": 1, "after": 3, "rest": "short"},
      {"before": 2, "after": 2, "rest": "short"}]),
    ("low_vol_reversal", "Fade moves made on LOW volume (Campbell-Grossman-Wang)", _low_vol_reversal,
     [{"n": 5, "vol_window": 20, "vol_mult": 1.0}, {"n": 10, "vol_window": 20, "vol_mult": 0.8}]),
    ("near_52w", "52-week high/low proximity momentum (George-Hwang)", _near_52w_extreme,
     [{"n": 252, "band": 0.02}, {"n": 252, "band": 0.05}]),
]


def _expand():
    """Yield (label, family_id, fn, params) for every config in the grid."""
    for fid, _human, fn, grid in RULES:
        for p in grid:
            tag = ",".join(f"{k}={v}" for k, v in p.items())
            yield (f"{fid}[{tag}]", fid, fn, p)


def _human(fid: str) -> str:
    for f, h, _fn, _g in RULES:
        if f == fid:
            return h
    return fid


def _slice_stats(a: dict, base: dict) -> dict:
    """Turn a raw accumulator into accuracy / edge numbers for one (config, horizon, slice)."""
    n = a["n"]
    if n == 0:
        return {"signals": 0}
    acc = a["hits"] / n
    cond_base = max(a["up"], a["dn"]) / n            # best single-direction bet on these very bars
    g_up = base["up"] / base["valid"] if base["valid"] else float("nan")
    g_base = max(g_up, 1 - g_up)                      # market-wide drift baseline
    return {
        "signals": n,
        "coverage": n / base["valid"] if base["valid"] else float("nan"),
        "accuracy": acc,
        "cond_base": cond_base,
        "cond_edge_pp": (acc - cond_base) * 100,
        "global_base": g_base,
        "global_edge_pp": (acc - g_base) * 100,
        "long": a["long"],
        "short": a["short"],
    }


def evaluate(frames: dict, horizons=HORIZONS, train_frac: float = DEFAULT_TRAIN_FRAC,
             min_is: int = DEFAULT_MIN_IS, min_oos: int = DEFAULT_MIN_OOS,
             configs=None) -> list[dict]:
    """Per-config in-sample/out-of-sample directional stats, pooled across `frames`.

    `frames`: dict[symbol] -> OHLCV DataFrame. Each symbol is split chronologically into
    train (first `train_frac`) and test; stats are pooled within each slice. Returns one
    row per (config, horizon) that meets the minimum-signal bars, richest-edge first.
    `configs` (label, family_id, builder, params) overrides the default full grid.
    """
    configs = list(configs) if configs is not None else list(_expand())
    acc: dict = defaultdict(lambda: {"n": 0, "hits": 0, "up": 0, "dn": 0, "long": 0, "short": 0})
    base: dict = defaultdict(lambda: {"valid": 0, "up": 0})

    for sym, df in frames.items():
        if df is None or len(df) < 260:
            continue
        close = df["close"]
        split = int(len(df) * train_frac)
        sl_arr = np.array(["is"] * split + ["oos"] * (len(df) - split))
        fwds = {h: (close.shift(-h) / close - 1.0).to_numpy() for h in horizons}

        for h in horizons:                                    # base rate per (horizon, slice)
            fwd = fwds[h]
            valid = ~np.isnan(fwd)
            for sl in ("is", "oos"):
                m = valid & (sl_arr == sl)
                base[(h, sl)]["valid"] += int(m.sum())
                base[(h, sl)]["up"] += int((fwd[m] > 0).sum())

        for label, _fid, fn, p in configs:
            try:
                st = fn(df, p).reindex(df.index).fillna(0).astype(int).to_numpy()
            except Exception as exc:  # noqa: BLE001
                log.warning("%s on %s failed: %s", label, sym, exc)
                continue
            for h in horizons:
                fwd = fwds[h]
                valid = ~np.isnan(fwd)
                fsign = np.sign(fwd)
                for sl in ("is", "oos"):
                    sig = valid & (sl_arr == sl) & (st != 0)
                    a = acc[(label, h, sl)]
                    a["n"] += int(sig.sum())
                    a["hits"] += int((sig & (fsign == st)).sum())
                    a["up"] += int((fwd[sig] > 0).sum())
                    a["dn"] += int((fwd[sig] < 0).sum())
                    a["long"] += int((sig & (st == 1)).sum())
                    a["short"] += int((sig & (st == -1)).sum())

    rows = []
    seen = {(lbl, fid) for lbl, fid, _fn, _p in configs}
    for label, fid in seen:
        for h in horizons:
            is_s = _slice_stats(acc[(label, h, "is")], base[(h, "is")])
            oos_s = _slice_stats(acc[(label, h, "oos")], base[(h, "oos")])
            if is_s.get("signals", 0) < min_is or oos_s.get("signals", 0) < min_oos:
                continue
            raw60 = oos_s["accuracy"] >= ACC_TARGET
            real_edge = (oos_s["cond_edge_pp"] >= EDGE_MIN_PP and is_s["cond_edge_pp"] >= 1.0
                         and oos_s["signals"] >= min_oos)
            rows.append({
                "config": label, "family": fid, "rule": _human(fid), "horizon": h,
                "is": is_s, "oos": oos_s,
                "raw_60": bool(raw60),
                "real_edge": bool(real_edge),
                "drift_rider": bool(raw60 and not real_edge),
            })
    rows.sort(key=lambda r: (r["oos"]["cond_edge_pp"], r["oos"]["accuracy"]), reverse=True)
    return rows


def _stance_to_signals(stance: pd.Series) -> pd.DataFrame:
    d = stance.fillna(0).astype(int)
    entered = d != 0
    return pd.DataFrame({"score": d.astype(float), "direction": d,
                         "confidence": entered.astype(float), "entered": entered})


# Best config of the full sweep — UNVALIDATED: positive OOS edge but flat-to-negative
# in-sample at every split point, i.e. a recent-regime artifact. Exposed so the paper
# runner can *watch* it; do not treat it as a proven edge.
BEST_CANDIDATE = ("rsi_ma_vol", {"style": "dip", "n": 200, "dip": 35, "vol_mult": 1.5})


def candidate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """entered/direction signals for `BEST_CANDIDATE` (for paper-watching only)."""
    family, params = BEST_CANDIDATE
    fn = next(f for fid, _h, f, _g in RULES if fid == family)
    return _stance_to_signals(fn(df, params))


def costed(frames: dict, family: str, params: dict, rp, costs) -> dict:
    """Run one rule through the costed backtest engine, pooled across symbols.

    Full-sample net-of-cost lens (does the directional call survive fees at all?).
    Returns metrics incl. scale-free per-trade %. Secondary to the IS/OOS accuracy gate.
    """
    from src.backtest.engine import run_backtest
    from src.backtest.metrics import compute_metrics

    fn = next((f for fid, _h, f, _g in RULES if fid == family), None)
    if fn is None:
        return {"trades": 0}
    trade_frames = []
    for sym, df in frames.items():
        if df is None or len(df) < 260:
            continue
        try:
            sig = _stance_to_signals(fn(df, params))
            t = run_backtest(df, {"indicator_params": {}}, rp, costs, symbol=sym,
                             signals=sig, square_off_eod=False)
            if t is not None and not t.empty:
                trade_frames.append(t)
        except Exception as exc:  # noqa: BLE001
            log.warning("costed %s on %s failed: %s", family, sym, exc)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    m = compute_metrics(trades, rp.capital)
    if not trades.empty:
        notional = (trades["entry_price"] * trades["qty"]).replace(0, np.nan)
        ret = (trades["pnl"] / notional).dropna()
        m["avg_trade_pct"] = float(ret.mean() * 100) if len(ret) else 0.0
        m["median_trade_pct"] = float(ret.median() * 100) if len(ret) else 0.0
    else:
        m["avg_trade_pct"] = m["median_trade_pct"] = 0.0
    return m


def archive_winners(rows: list[dict], path: str = ARCHIVE_PATH, meta: dict | None = None) -> dict:
    """Persist every config that clears the accuracy target on the held-out slice.

    Each entry is tagged real_edge / drift_rider so the archive is self-explanatory.
    Returns the archive payload (also written to `path`).
    """
    winners = [r for r in rows if r["raw_60"]]
    payload = {
        "meta": meta or {},
        "acc_target": ACC_TARGET,
        "edge_min_pp": EDGE_MIN_PP,
        "n_winners": len(winners),
        "n_real_edge": sum(1 for r in winners if r["real_edge"]),
        "n_drift_riders": sum(1 for r in winners if r["drift_rider"]),
        "winners": [{
            "config": r["config"], "rule": r["rule"], "horizon": r["horizon"],
            "oos_accuracy": round(r["oos"]["accuracy"], 4),
            "oos_signals": r["oos"]["signals"],
            "oos_cond_edge_pp": round(r["oos"]["cond_edge_pp"], 2),
            "is_accuracy": round(r["is"]["accuracy"], 4),
            "is_cond_edge_pp": round(r["is"]["cond_edge_pp"], 2),
            "label": "real_edge" if r["real_edge"] else "drift_rider",
        } for r in winners],
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
    return payload


def analyze_cross_sectional(frames: dict, lookbacks=(20, 60, 120), horizons=(5, 10, 20),
                            quantile: float = 0.2, train_frac: float = DEFAULT_TRAIN_FRAC,
                            min_oos: int = DEFAULT_MIN_OOS) -> dict:
    """Cross-sectional relative strength — the one bet a per-symbol sweep can't make.

    Each bar, rank the universe by L-bar momentum; go long the top quantile, short the
    bottom. The honest metric is the **long-short spread** (do past winners out-run past
    losers over the next h bars?), reported out-of-sample. Directional accuracy is shown
    too, but a long-short factor lives or dies by the spread, not the hit-rate.
    """
    closes = {sym: df["close"] for sym, df in frames.items()
              if df is not None and len(df) >= 260}
    if len(closes) < 5:
        return {"available": False, "reason": "need >=5 aligned symbols"}
    panel = pd.DataFrame(closes).dropna(how="any")     # common dates where all names trade
    if len(panel) < 300:
        return {"available": False, "reason": "too few aligned dates"}

    n_dates, n_names = len(panel), panel.shape[1]
    split = int(n_dates * train_frac)
    k = max(1, int(round(n_names * quantile)))
    rows = []
    for L in lookbacks:
        mom = panel.pct_change(L)
        rank = mom.rank(axis=1, method="first")
        long_mask = rank > (n_names - k)               # top-k momentum
        short_mask = rank <= k                          # bottom-k momentum
        for h in horizons:
            fwd = panel.shift(-h) / panel - 1.0
            rec = {"lookback": L, "horizon": h}
            for sl, lo, hi in (("is", 0, split), ("oos", split, n_dates)):
                idx = panel.index[lo:hi]
                fwd_s = fwd.loc[idx]
                lm = long_mask.loc[idx] & fwd_s.notna()
                sm = short_mask.loc[idx] & fwd_s.notna()
                lr = fwd_s.where(lm).to_numpy().ravel()
                sr = fwd_s.where(sm).to_numpy().ravel()
                lr = lr[~np.isnan(lr)]
                sr = sr[~np.isnan(sr)]
                n = len(lr) + len(sr)
                if n == 0:
                    rec[sl] = {"signals": 0}
                    continue
                hits = int((lr > 0).sum()) + int((sr < 0).sum())
                spread_pp = (float(lr.mean()) - float(sr.mean())) * 100 if len(lr) and len(sr) else float("nan")
                rec[sl] = {"signals": n, "accuracy": hits / n,
                           "long_short_spread_pp": spread_pp,
                           "long_mean_pp": float(lr.mean()) * 100 if len(lr) else float("nan"),
                           "short_mean_pp": float(sr.mean()) * 100 if len(sr) else float("nan")}
            rows.append(rec)
    rows.sort(key=lambda r: (r.get("oos", {}).get("long_short_spread_pp", -1e9)
                             if r.get("oos", {}).get("long_short_spread_pp") == r.get("oos", {}).get("long_short_spread_pp")
                             else -1e9), reverse=True)
    return {"available": True, "names": n_names, "dates": n_dates, "top_k": k,
            "min_oos": min_oos, "rows": rows}


def analyze(symbols=None, timeframe: str = "day", train_frac: float = DEFAULT_TRAIN_FRAC,
            top_costed: int = 8) -> dict:
    """Full search over the cached universe: evaluate -> archive -> cost the top edges."""
    from datetime import datetime

    from src.backtest.engine import Costs
    from src.backtest.momentum import DEFAULT_UNIVERSE
    from src.config import get_config
    from src.data.historical import HistoricalData
    from src.risk.manager import RiskParams

    symbols = symbols or [f"{s}.NS" for s in DEFAULT_UNIVERSE]
    hist = HistoricalData(cache_dir="data/cache")
    frames = {}
    for sym in symbols:
        try:
            frames[sym] = hist.get(sym, timeframe)
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch failed for %s: %s", sym, exc)

    rows = evaluate(frames, train_frac=train_frac)
    meta = {"timeframe": timeframe, "symbols": len([f for f in frames.values() if f is not None]),
            "train_frac": train_frac, "horizons": list(HORIZONS),
            "generated": datetime.now().isoformat(timespec="seconds")}
    archive = archive_winners(rows, meta=meta)

    cfg = get_config()
    rp = RiskParams.from_config(cfg.risk.model_dump())
    costs = Costs.from_config(cfg.costs.model_dump())
    # Cost the most promising real-edge candidates (best OOS conditional edge).
    seen_fam: set = set()
    costed_rows = []
    for r in rows:
        if r["family"] in seen_fam:
            continue
        params = next((p for lbl, _fid, _fn, p in _expand() if lbl == r["config"]), {})
        m = costed(frames, r["family"], params, rp, costs)
        costed_rows.append({"config": r["config"], "rule": r["rule"], "horizon": r["horizon"],
                            "oos_accuracy": r["oos"]["accuracy"], "oos_cond_edge_pp": r["oos"]["cond_edge_pp"],
                            "metrics": m})
        seen_fam.add(r["family"])
        if len(costed_rows) >= top_costed:
            break

    cross = analyze_cross_sectional(frames, train_frac=train_frac)
    return {"meta": meta, "rows": rows, "archive": archive, "costed": costed_rows, "cross": cross}


def _main() -> None:  # pragma: no cover - CLI, reads data/cache
    from rich.console import Console
    from rich.table import Table

    res = analyze()
    c = Console()
    m = res["meta"]
    rows = res["rows"]

    c.print(f"[bold]Edge search[/] — {m['symbols']} symbols, {m['timeframe']}, "
            f"train/test {int(m['train_frac']*100)}/{100-int(m['train_frac']*100)}, "
            f"horizons {m['horizons']}, {len(rows)} configs with enough signals.")

    t = Table(title="Top 20 by out-of-sample conditional edge (held-out slice)")
    for col in ("config", "h", "OOS acc", "OOS cond-edge", "IS acc", "OOS n", "verdict"):
        t.add_column(col, overflow="fold")
    for r in rows[:20]:
        verdict = ("[green]REAL EDGE[/]" if r["real_edge"]
                   else "[yellow]drift 60%+[/]" if r["drift_rider"]
                   else "[red]no edge[/]")
        ce = r["oos"]["cond_edge_pp"]
        t.add_row(r["config"], str(r["horizon"]),
                  f"{r['oos']['accuracy']*100:.1f}%",
                  f"[{'green' if ce > 0 else 'red'}]{ce:+.1f} pp[/]",
                  f"{r['is']['accuracy']*100:.1f}%",
                  str(r["oos"]["signals"]), verdict)
    c.print(t)

    arc = res["archive"]
    c.print(f"\n[bold]Archived[/] {arc['n_winners']} configs clearing {int(ACC_TARGET*100)}% OOS accuracy "
            f"-> [cyan]{ARCHIVE_PATH}[/]  "
            f"([green]{arc['n_real_edge']} real edge[/], [yellow]{arc['n_drift_riders']} drift riders[/])")

    t2 = Table(title="Costed (net of fees/slippage) — does the directional call make money?")
    for col in ("config", "trades", "win%", "avg trade%", "profit factor", "tradeable?"):
        t2.add_column(col, overflow="fold")
    for cr in res["costed"]:
        mm = cr["metrics"]
        ok = mm["trades"] >= 20 and mm.get("avg_trade_pct", 0) > 0 and mm["profit_factor"] > 1
        t2.add_row(cr["config"], str(mm["trades"]), f"{mm['win_rate']*100:.1f}",
                   f"{mm.get('avg_trade_pct', 0):+.2f}", f"{mm['profit_factor']:.2f}",
                   "[green]yes[/]" if ok else "[red]no[/]")
    c.print(t2)

    cross = res.get("cross", {})
    if cross.get("available"):
        t3 = Table(title=f"Cross-sectional momentum (long top-{cross['top_k']} / short bottom-{cross['top_k']} "
                         f"of {cross['names']}) — out-of-sample")
        for col in ("lookback", "horizon", "OOS acc", "OOS long-short spread", "edge?"):
            t3.add_column(col, overflow="fold")
        for r in cross["rows"][:6]:
            o = r.get("oos", {})
            if o.get("signals", 0) < cross["min_oos"]:
                continue
            sp = o.get("long_short_spread_pp", float("nan"))
            good = sp == sp and sp > 0
            t3.add_row(str(r["lookback"]), str(r["horizon"]),
                       f"{o['accuracy']*100:.1f}%",
                       f"[{'green' if good else 'red'}]{sp:+.2f} pp[/]" if sp == sp else "—",
                       "[green]positive spread[/]" if good else "[red]no spread[/]")
        c.print(t3)
        c.print("[dim]Cross-sectional edge = the long-short SPREAD (winners minus losers over the "
                "next h bars), before costs. A positive pre-cost spread is the only thing here with "
                "academic support — but it must still clear shorting costs & turnover to be tradeable.[/]")

    c.print("[dim]OOS acc = held-out accuracy (the honest '60%'). cond-edge = accuracy minus the best "
            "single-direction bet on the SAME bars; ~0 means 60% is just market drift, not skill. "
            "A real edge must clear the target out-of-sample AND make money after costs.[/]")


if __name__ == "__main__":  # pragma: no cover
    _main()
