"""Honest accuracy of every directional indicator.

For each indicator we derive a +1/-1 stance per bar, then measure how often the
*next-N-bar* price move agrees with that stance (cost-free directional hit-rate).
The number to beat is the **base rate** — the best you'd get by always guessing the
market's prevailing drift direction. An indicator only has predictive edge if its
accuracy exceeds the base rate (`edge` > 0). Most public indicators don't.

Coverage = share of bars where the indicator takes a stance (trend rules ~always;
threshold rules like RSI only at extremes).

Run: `python -m src indicators`
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators import engine
from src.logutil import get_logger

log = get_logger(__name__)

DEFAULT_HORIZON = 5


def _sign(s: pd.Series) -> pd.Series:
    return pd.Series(np.sign(s), index=s.index).fillna(0).astype(int)


# Each builder returns a +1/-1/0 stance per bar. info = one-line plain description.
def _rsi_meanrev(df):
    r = engine.rsi(df["close"], 14)
    return pd.Series(np.where(r < 30, 1, np.where(r > 70, -1, 0)), index=df.index)


def _ema_cross(df):
    return _sign(engine.ema(df["close"], 9) - engine.ema(df["close"], 21))


def _sma_cross(df):
    return _sign(engine.sma(df["close"], 50) - engine.sma(df["close"], 200))


def _macd(df):
    m = engine.macd(df["close"])
    return _sign(m["macd"] - m["signal"])


def _boll_reversion(df):
    bb = engine.bollinger(df["close"], 20)
    return pd.Series(np.where(df["close"] < bb["lower"], 1,
                              np.where(df["close"] > bb["upper"], -1, 0)), index=df.index)


def _boll_breakout(df):
    bb = engine.bollinger(df["close"], 20)
    return pd.Series(np.where(df["close"] > bb["upper"], 1,
                              np.where(df["close"] < bb["lower"], -1, 0)), index=df.index)


def _supertrend(df):
    return engine.supertrend(df, 10, 3.0)["direction"].astype(int)


def _price_vs_sma200(df):
    return _sign(df["close"] - engine.sma(df["close"], 200))


def _momentum_20(df):
    return _sign(df["close"] / df["close"].shift(20) - 1.0)


INDICATORS = [
    ("RSI(14) reversion", "Mean-reversion: long oversold (<30), short overbought (>70)", _rsi_meanrev),
    ("EMA 9/21 cross", "Trend-follow: long when fast EMA > slow EMA", _ema_cross),
    ("SMA 50/200 cross", "Trend: golden/death cross", _sma_cross),
    ("MACD(12,26,9)", "Trend: MACD line vs signal line", _macd),
    ("Bollinger reversion", "Fade 2σ bands: long below lower, short above upper", _boll_reversion),
    ("Bollinger breakout", "Breakout beyond 2σ bands (opposite of reversion)", _boll_breakout),
    ("Supertrend(10,3)", "Trend: Supertrend flip direction", _supertrend),
    ("Price vs SMA200", "Trend filter: above/below the 200-bar average", _price_vs_sma200),
    ("Momentum (20-bar)", "Long if 20-bar return positive, else short", _momentum_20),
]


def analyze(symbols=None, timeframe: str = "day", horizon: int = DEFAULT_HORIZON) -> dict:
    """Pooled directional hit-rate per indicator across the universe."""
    from src.backtest.momentum import DEFAULT_UNIVERSE
    from src.data.historical import HistoricalData

    symbols = symbols or [f"{s}.NS" for s in DEFAULT_UNIVERSE]
    hist = HistoricalData(cache_dir="data/cache")

    agg = {name: {"signals": 0, "hits": 0} for name, _, _ in INDICATORS}
    up_total = valid_total = 0

    for sym in symbols:
        try:
            df = hist.get(sym, timeframe)
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch failed for %s: %s", sym, exc)
            continue
        if len(df) < 220:  # need room for SMA200 + horizon
            continue
        close = df["close"]
        fwd = close.shift(-horizon) / close - 1.0
        fwd_sign = _sign(fwd)
        valid = fwd.notna()
        up_total += int((fwd[valid] > 0).sum())
        valid_total += int(valid.sum())

        for name, _, fn in INDICATORS:
            try:
                d = fn(df).reindex(df.index).fillna(0).astype(int)
            except Exception as exc:  # noqa: BLE001
                log.warning("%s on %s failed: %s", name, sym, exc)
                continue
            mask = valid & d.isin([-1, 1])
            agg[name]["signals"] += int(mask.sum())
            agg[name]["hits"] += int(((fwd_sign == d) & mask).sum())

    base_up = up_total / valid_total if valid_total else float("nan")
    base = max(base_up, 1 - base_up) if valid_total else float("nan")

    rows = []
    for name, info, _ in INDICATORS:
        n, h = agg[name]["signals"], agg[name]["hits"]
        acc = h / n if n else float("nan")
        rows.append({
            "indicator": name, "info": info, "signals": n,
            "coverage": (n / valid_total) if valid_total else float("nan"),
            "accuracy": acc, "base": base,
            "edge_pp": (acc - base) * 100 if n else float("nan"),
        })
    rows.sort(key=lambda r: (r["edge_pp"] if r["edge_pp"] == r["edge_pp"] else -1e9), reverse=True)
    return {"timeframe": timeframe, "horizon": horizon, "symbols": len(symbols),
            "base_up_rate": base_up, "base": base, "rows": rows}


# --- Confluence + regime-conditioned rules ---------------------------------
# Each builder returns a +1/-1/0 stance combining multiple indicators and/or a
# market regime, to test whether combinations beat the base rate where singles don't.

def _regime(df):
    """+1 trending-up, -1 trending-down, 0 ranging (from SMA50/SMA200 alignment)."""
    close = df["close"]
    sma50, sma200 = engine.sma(close, 50), engine.sma(close, 200)
    up = (close > sma200) & (sma50 > sma200)
    dn = (close < sma200) & (sma50 < sma200)
    return pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=df.index)


def _c_pullback_in_trend(df):
    """Buy the dip in an uptrend, sell the rip in a downtrend (trend + RSI agree)."""
    reg = _regime(df)
    r = engine.rsi(df["close"], 14)
    long = (reg == 1) & (r < 40)
    short = (reg == -1) & (r > 60)
    return pd.Series(np.where(long, 1, np.where(short, -1, 0)), index=df.index)


def _c_triple_trend_agree(df):
    """Long/short only when EMA-cross, MACD and Supertrend all point the same way."""
    e, m, s = _ema_cross(df), _macd(df), _supertrend(df)
    long = (e > 0) & (m > 0) & (s > 0)
    short = (e < 0) & (m < 0) & (s < 0)
    return pd.Series(np.where(long, 1, np.where(short, -1, 0)), index=df.index)


def _c_reversion_in_range(df):
    """Bollinger mean-reversion, but only when the market is ranging (no trend)."""
    reg = _regime(df)
    d = _boll_reversion(df)
    return pd.Series(np.where(reg == 0, d, 0), index=df.index).astype(int)


def _c_rsi_in_range(df):
    """RSI mean-reversion, but only in a ranging regime."""
    reg = _regime(df)
    d = _rsi_meanrev(df)
    return pd.Series(np.where(reg == 0, d, 0), index=df.index).astype(int)


def _c_trend_volume(df):
    """Trend-following (EMA cross) confirmed by both regime alignment and high volume."""
    reg = _regime(df)
    e = _ema_cross(df)
    hv = df["volume"] > df["volume"].rolling(20).mean()
    long = (reg == 1) & (e > 0) & hv
    short = (reg == -1) & (e < 0) & hv
    return pd.Series(np.where(long, 1, np.where(short, -1, 0)), index=df.index)


CONFLUENCE = [
    ("Pullback in trend", "Trend(SMA) + RSI: buy dips up-trend, sell rips down-trend", _c_pullback_in_trend),
    ("Triple trend agree", "EMA cross + MACD + Supertrend all aligned", _c_triple_trend_agree),
    ("Reversion in range", "Bollinger fade, only when ranging (no trend)", _c_reversion_in_range),
    ("RSI in range", "RSI fade, only in a ranging regime", _c_rsi_in_range),
    ("Trend + volume", "EMA cross + regime + above-average volume", _c_trend_volume),
]


def analyze_confluence(symbols=None, timeframe: str = "day", horizon: int = DEFAULT_HORIZON) -> dict:
    """Pooled directional hit-rate for each confluence/regime rule vs the base rate."""
    from src.backtest.momentum import DEFAULT_UNIVERSE
    from src.data.historical import HistoricalData

    symbols = symbols or [f"{s}.NS" for s in DEFAULT_UNIVERSE]
    hist = HistoricalData(cache_dir="data/cache")
    agg = {name: {"signals": 0, "hits": 0} for name, _, _ in CONFLUENCE}
    up_total = valid_total = 0

    for sym in symbols:
        try:
            df = hist.get(sym, timeframe)
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch failed for %s: %s", sym, exc)
            continue
        if len(df) < 220:
            continue
        close = df["close"]
        fwd = close.shift(-horizon) / close - 1.0
        fwd_sign = _sign(fwd)
        valid = fwd.notna()
        up_total += int((fwd[valid] > 0).sum())
        valid_total += int(valid.sum())
        for name, _, fn in CONFLUENCE:
            try:
                d = fn(df).reindex(df.index).fillna(0).astype(int)
            except Exception as exc:  # noqa: BLE001
                log.warning("%s on %s failed: %s", name, sym, exc)
                continue
            mask = valid & d.isin([-1, 1])
            agg[name]["signals"] += int(mask.sum())
            agg[name]["hits"] += int(((fwd_sign == d) & mask).sum())

    base_up = up_total / valid_total if valid_total else float("nan")
    base = max(base_up, 1 - base_up) if valid_total else float("nan")
    rows = []
    for name, info, _ in CONFLUENCE:
        n, h = agg[name]["signals"], agg[name]["hits"]
        acc = h / n if n else float("nan")
        rows.append({"rule": name, "info": info, "signals": n,
                     "coverage": (n / valid_total) if valid_total else float("nan"),
                     "accuracy": acc, "base": base,
                     "edge_pp": (acc - base) * 100 if n else float("nan")})
    rows.sort(key=lambda r: (r["edge_pp"] if r["edge_pp"] == r["edge_pp"] else -1e9), reverse=True)
    return {"timeframe": timeframe, "horizon": horizon, "symbols": len(symbols),
            "base_up_rate": base_up, "base": base, "rows": rows}


def _main_confluence() -> None:  # pragma: no cover - CLI, needs network
    from rich.console import Console
    from rich.table import Table

    res = analyze_confluence()
    c = Console()
    t = Table(title=f"Confluence + regime rules vs base rate — {res['horizon']}-bar, "
                    f"{res['timeframe']}, {res['symbols']} symbols")
    for col in ("rule", "what it combines", "coverage", "accuracy", "base rate", "edge", "verdict"):
        t.add_column(col, overflow="fold")
    for r in res["rows"]:
        edge = r["edge_pp"]
        good = edge == edge and edge > 1.0 and r["signals"] >= 100
        t.add_row(
            r["rule"], r["info"],
            f"{r['coverage']*100:.0f}%" if r["coverage"] == r["coverage"] else "—",
            f"{r['accuracy']*100:.1f}%" if r["accuracy"] == r["accuracy"] else "—",
            f"{r['base']*100:.1f}%",
            f"[{'green' if edge > 0 else 'red'}]{edge:+.1f} pp[/]" if edge == edge else "—",
            f"[{'green' if good else 'red'}]{'EDGE' if good else 'no edge'}[/]",
        )
    c.print(t)
    c.print(f"[dim]Base rate {res['base']*100:.1f}%. A rule only has predictive edge if "
            f"accuracy > base rate with enough signals. Still cost-free.[/]")


def analyze_volume(symbols=None, timeframe: str = "day", horizon: int = DEFAULT_HORIZON,
                   vol_window: int = 20, vol_mult: float = 1.0) -> dict:
    """Same hit-rate test, but compare each indicator's accuracy with and without a
    volume-confirmation filter (signal counts only when volume > vol_mult x its
    rolling average). Quantifies how much volume actually moves accuracy.
    """
    from src.backtest.momentum import DEFAULT_UNIVERSE
    from src.data.historical import HistoricalData

    symbols = symbols or [f"{s}.NS" for s in DEFAULT_UNIVERSE]
    hist = HistoricalData(cache_dir="data/cache")

    agg = {name: {"b_sig": 0, "b_hit": 0, "v_sig": 0, "v_hit": 0} for name, _, _ in INDICATORS}
    up_total = valid_total = 0

    for sym in symbols:
        try:
            df = hist.get(sym, timeframe)
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch failed for %s: %s", sym, exc)
            continue
        if len(df) < 220 or "volume" not in df:
            continue
        close = df["close"]
        fwd = close.shift(-horizon) / close - 1.0
        fwd_sign = _sign(fwd)
        valid = fwd.notna()
        up_total += int((fwd[valid] > 0).sum())
        valid_total += int(valid.sum())
        high_vol = df["volume"] > df["volume"].rolling(vol_window).mean() * vol_mult

        for name, _, fn in INDICATORS:
            try:
                d = fn(df).reindex(df.index).fillna(0).astype(int)
            except Exception as exc:  # noqa: BLE001
                log.warning("%s on %s failed: %s", name, sym, exc)
                continue
            mask = valid & d.isin([-1, 1])
            hit = (fwd_sign == d) & mask
            agg[name]["b_sig"] += int(mask.sum())
            agg[name]["b_hit"] += int(hit.sum())
            mv = mask & high_vol
            agg[name]["v_sig"] += int(mv.sum())
            agg[name]["v_hit"] += int((hit & high_vol).sum())

    base_up = up_total / valid_total if valid_total else float("nan")
    base = max(base_up, 1 - base_up) if valid_total else float("nan")

    rows = []
    for name, info, _ in INDICATORS:
        a = agg[name]
        b_acc = a["b_hit"] / a["b_sig"] if a["b_sig"] else float("nan")
        v_acc = a["v_hit"] / a["v_sig"] if a["v_sig"] else float("nan")
        delta = (v_acc - b_acc) if (a["b_sig"] and a["v_sig"]) else float("nan")
        rows.append({
            "indicator": name, "info": info,
            "base_acc": b_acc, "vol_acc": v_acc, "delta_pp": delta * 100 if delta == delta else float("nan"),
            "vol_signals": a["v_sig"], "vol_edge_pp": (v_acc - base) * 100 if a["v_sig"] else float("nan"),
        })
    rows.sort(key=lambda r: (r["delta_pp"] if r["delta_pp"] == r["delta_pp"] else -1e9), reverse=True)
    return {"timeframe": timeframe, "horizon": horizon, "symbols": len(symbols),
            "vol_window": vol_window, "vol_mult": vol_mult, "base": base, "rows": rows}


def _main_volume() -> None:  # pragma: no cover - CLI, needs network
    from rich.console import Console
    from rich.table import Table

    res = analyze_volume()
    c = Console()
    t = Table(title=f"Does volume confirmation help? — {res['horizon']}-bar, {res['timeframe']}, "
                    f"{res['symbols']} symbols (volume > {res['vol_mult']:g}x {res['vol_window']}-avg)")
    for col in ("indicator", "accuracy", "+volume", "Δ from volume", "vol edge vs base"):
        t.add_column(col, overflow="fold")
    for r in res["rows"]:
        d = r["delta_pp"]
        ve = r["vol_edge_pp"]
        t.add_row(
            r["indicator"],
            f"{r['base_acc']*100:.1f}%" if r["base_acc"] == r["base_acc"] else "—",
            f"{r['vol_acc']*100:.1f}%" if r["vol_acc"] == r["vol_acc"] else "—",
            f"[{'green' if d > 0 else 'red'}]{d:+.1f} pp[/]" if d == d else "—",
            f"[{'green' if ve > 0 else 'red'}]{ve:+.1f} pp[/]" if ve == ve else "—",
        )
    c.print(t)
    c.print(f"[dim]Δ = accuracy change from requiring above-average volume. "
            f"Base rate {res['base']*100:.1f}%. Positive Δ that still can't clear the base "
            f"rate is noise, not edge.[/]")


_ALL_CACHE: dict = {}
_ALL_TTL = 600.0  # seconds


def analyze_all(timeframe: str = "day", horizon: int = DEFAULT_HORIZON) -> dict:
    """Single + volume + confluence results in one payload, cached, for the web UI."""
    import time

    key = (timeframe, horizon)
    hit = _ALL_CACHE.get(key)
    if hit and time.time() - hit[0] < _ALL_TTL:
        return hit[1]
    single = analyze(timeframe=timeframe, horizon=horizon)
    val = {
        "timeframe": timeframe,
        "horizon": horizon,
        "symbols": single["symbols"],
        "base": single["base"],
        "base_up_rate": single["base_up_rate"],
        "single": single["rows"],
        "volume": analyze_volume(timeframe=timeframe, horizon=horizon)["rows"],
        "confluence": analyze_confluence(timeframe=timeframe, horizon=horizon)["rows"],
    }
    _ALL_CACHE[key] = (time.time(), val)
    return val


def _main() -> None:  # pragma: no cover - CLI, needs network
    from rich.console import Console
    from rich.table import Table

    res = analyze()
    c = Console()
    t = Table(title=f"Indicator directional accuracy — {res['horizon']}-bar horizon, "
                    f"{res['timeframe']}, {res['symbols']} symbols")
    for col in ("indicator", "rule", "coverage", "accuracy", "base rate", "edge", "verdict"):
        t.add_column(col, overflow="fold")
    for r in res["rows"]:
        edge = r["edge_pp"]
        good = edge == edge and edge > 1.0 and r["signals"] >= 200
        t.add_row(
            r["indicator"], r["info"],
            f"{r['coverage']*100:.0f}%" if r["coverage"] == r["coverage"] else "—",
            f"{r['accuracy']*100:.1f}%" if r["accuracy"] == r["accuracy"] else "—",
            f"{r['base']*100:.1f}%",
            f"[{'green' if edge > 0 else 'red'}]{edge:+.1f} pp[/]" if edge == edge else "—",
            f"[{'green' if good else 'red'}]{'edge?' if good else 'no edge'}[/]",
        )
    c.print(t)
    c.print(f"[dim]Base rate {res['base']*100:.1f}% = best constant-direction guess "
            f"(market drift). Edge = accuracy − base; only a positive edge is predictive. "
            f"Cost-free — a real strategy must also survive fees & slippage.[/]")


if __name__ == "__main__":  # pragma: no cover
    _main()
