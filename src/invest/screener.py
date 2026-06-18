"""Stock screener — gain/loss + honest positioning, grouped by market-cap bucket.

This answers "show me good stocks with potential, by large/mid/small cap" the only
honest way the rest of the repo allows: it shows **what is true now** (returns over
several windows, distance from the 52-week high/low, trend vs the 200-day average, RSI,
volatility) and a **descriptive setup tag** — never a prediction or a buy call. This
engine has shown no per-trade edge (see `reality`), so "potential" here means *current
positioning*, not a forecast. A stock "near its highs in an uptrend" or "pulled back in
an uptrend" is a description of the chart, not a promise about tomorrow.

Cap buckets are an **indicative static classification** (market caps drift over time).
Large-cap names are cache-backed; mid/small are fetched live on first use (and cached),
so the screener degrades gracefully offline — a bucket with no data simply shows empty.

Run: `python -m src screener`   (or the dashboard's Screener tab / `/api/screener`)
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.indicators import engine
from src.logutil import get_logger

log = get_logger(__name__)

# Indicative cap buckets (NSE). Large-caps are the cached research universe; mid/small are
# fetched live on demand. Classification is approximate and drifts — labelled as such in the UI.
UNIVERSE = {
    "large": ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "BHARTIARTL",
              "ITC", "LT", "KOTAKBANK", "AXISBANK", "HINDUNILVR", "MARUTI", "SUNPHARMA",
              "TITAN", "ONGC", "NTPC", "ASIANPAINT", "BAJFINANCE", "WIPRO"],
    "mid": ["TATAPOWER", "ASHOKLEY", "VOLTAS", "MPHASIS", "PERSISTENT", "CUMMINSIND",
            "FEDERALBNK", "AUROPHARMA", "PAGEIND", "COFORGE"],
    "small": ["RVNL", "IRFC", "SUZLON", "NBCC", "HUDCO", "RAILTEL", "IRCON",
              "KIRLOSENG", "JWL", "MAZDOCK"],
}
CATEGORY_LABEL = {"large": "Large cap", "mid": "Mid cap", "small": "Small cap"}

# Descriptive setup tags — what the chart is doing now, with no predictive claim.
SETUP_LEGEND = {
    "near-high uptrend": "above the 200-day avg and within 3% of its 52-week high (strength/momentum)",
    "uptrend pullback": "above the 200-day avg but ≥12% below its 52-week high (a pullback in an uptrend)",
    "uptrend": "above its 200-day average (long-term up)",
    "oversold downtrend": "below the 200-day avg with RSI < 35 (weak, but stretched down)",
    "downtrend": "below its 200-day average (long-term down)",
    "insufficient history": "not enough bars to classify",
}


def _safe(x, default=0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def stock_metrics(df: pd.DataFrame) -> dict:
    """Descriptive, causal snapshot of one stock from its daily OHLCV (last bar = now)."""
    close = df["close"].astype(float).dropna()
    n = len(close)
    if n < 2:
        return {"price": _safe(close.iloc[-1]) if n else 0.0, "history": n,
                "ret_1m": 0.0, "ret_6m": 0.0, "ret_1y": 0.0, "from_high_pct": 0.0,
                "from_low_pct": 0.0, "trend_up": False, "rsi": 50.0, "vol_pct": 0.0,
                "setup": "insufficient history"}
    px = _safe(close.iloc[-1])

    def ret(bars: int) -> float:
        base = close.iloc[-1 - bars] if n > bars else close.iloc[0]
        return _safe((px / base - 1.0) * 100) if base else 0.0

    win = min(n, 252)
    hi, lo = _safe(close.tail(win).max(), px), _safe(close.tail(win).min(), px)
    sma200 = _safe(engine.sma(close, 200).iloc[-1], _safe(close.mean(), px)) if n >= 200 \
        else _safe(close.mean(), px)
    rsi = _safe(engine.rsi(close, 14).iloc[-1], 50.0) if n >= 15 else 50.0
    rets = close.pct_change().tail(252).dropna()
    vol = _safe(rets.std() * np.sqrt(252) * 100) if len(rets) > 2 else 0.0

    m = {
        "price": round(px, 2), "history": n,
        "ret_1m": round(ret(21), 1), "ret_6m": round(ret(126), 1), "ret_1y": round(ret(252), 1),
        "from_high_pct": round((px / hi - 1.0) * 100, 1) if hi else 0.0,
        "from_low_pct": round((px / lo - 1.0) * 100, 1) if lo else 0.0,
        "trend_up": bool(px > sma200), "rsi": round(rsi, 0), "vol_pct": round(vol, 0),
    }
    m["setup"] = classify(m)
    return m


def classify(m: dict) -> str:
    """Map a metrics snapshot to a transparent, descriptive setup tag (no prediction)."""
    if m.get("history", 0) < 60:
        return "insufficient history"
    up, from_high, rsi = m["trend_up"], m["from_high_pct"], m["rsi"]
    if up and from_high >= -3:
        return "near-high uptrend"
    if up and from_high <= -12:
        return "uptrend pullback"
    if up:
        return "uptrend"
    if rsi < 35:
        return "oversold downtrend"
    return "downtrend"


def screen(frames_by_cat: dict) -> dict:
    """Build per-category rows (sorted by 1-year return, best first) from loaded frames."""
    out = {}
    for cat, frames in frames_by_cat.items():
        rows = []
        for sym, df in frames.items():
            if df is None or df.empty:
                continue
            m = stock_metrics(df)
            m["symbol"] = sym
            rows.append(m)
        rows.sort(key=lambda r: r["ret_1y"], reverse=True)
        gainers = sum(1 for r in rows if r["ret_1y"] > 0)
        out[cat] = {"label": CATEGORY_LABEL.get(cat, cat), "count": len(rows),
                    "gainers": gainers, "losers": len(rows) - gainers, "rows": rows}
    return out


def analyze(categories=None, timeframe: str = "day") -> dict:
    """Load each bucket (cache first, live fetch + cache on miss) and screen it."""
    from src.data.historical import HistoricalData

    cats = categories or list(UNIVERSE)
    hist = HistoricalData(cache_dir="data/cache")
    frames_by_cat: dict = {}
    for cat in cats:
        frames = {}
        for sym in UNIVERSE.get(cat, []):
            try:
                frames[f"{sym}.NS"] = hist.get(f"{sym}.NS", timeframe)
            except Exception as exc:  # noqa: BLE001 — offline / delisted names just drop out
                log.warning("screener: skip %s (%s)", sym, exc)
        frames_by_cat[cat] = frames
    return {
        "timeframe": timeframe,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "disclaimer": ("Descriptive positioning only — NOT predictions or buy/sell signals. "
                       "This engine has shown no per-trade edge (run `reality`). Cap buckets are "
                       "indicative and drift over time."),
        "setup_legend": SETUP_LEGEND,
        "categories": screen(frames_by_cat),
    }


def _main() -> None:  # pragma: no cover - CLI, reads cache + may fetch
    from rich.console import Console
    from rich.table import Table

    res = analyze()
    c = Console()
    c.print(f"[bold]Stock screener[/] — {res['timeframe']}, generated {res['generated']}")
    for cat in ("large", "mid", "small"):
        block = res["categories"].get(cat)
        if not block or not block["rows"]:
            c.print(f"\n[dim]{CATEGORY_LABEL.get(cat, cat)}: no data (offline or uncached).[/]")
            continue
        t = Table(title=f"{block['label']} — {block['gainers']} up / {block['losers']} down "
                        f"(sorted by 1-year return)")
        for col in ("symbol", "price", "1m%", "6m%", "1y%", "from 52w-high", "trend", "RSI", "vol%", "setup"):
            t.add_column(col, overflow="fold")
        for r in block["rows"]:
            g = lambda v: f"[{'green' if v >= 0 else 'red'}]{v:+.1f}[/]"  # noqa: E731
            t.add_row(r["symbol"].replace(".NS", ""), f"{r['price']:.0f}",
                      g(r["ret_1m"]), g(r["ret_6m"]), g(r["ret_1y"]),
                      f"{r['from_high_pct']:+.0f}%",
                      "[green]up[/]" if r["trend_up"] else "[red]down[/]",
                      f"{r['rsi']:.0f}", f"{r['vol_pct']:.0f}", r["setup"])
        c.print(t)
    c.print(f"\n[yellow]{res['disclaimer']}[/]")


if __name__ == "__main__":  # pragma: no cover
    _main()
