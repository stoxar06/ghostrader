"""Global market cues via yfinance (free). The pct-change helper is unit-tested;
`fetch_global_cues` is the thin network wrapper.
"""
from __future__ import annotations

import pandas as pd

from src.logutil import get_logger

log = get_logger(__name__)

# name -> yfinance ticker
TICKERS = {
    "nifty": "^NSEI",
    "sensex": "^BSESN",
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "dow": "^DJI",
    "crude": "CL=F",
    "gold": "GC=F",
    "usdinr": "INR=X",
    "india_vix": "^INDIAVIX",
    "us10y": "^TNX",
}


def pct_change_last(series: pd.Series) -> float | None:
    """Latest close vs prior close, in percent."""
    s = series.dropna()
    if len(s) < 2:
        return None
    return float((s.iloc[-1] / s.iloc[-2] - 1) * 100)


def fetch_global_cues(tickers: dict[str, str] | None = None) -> dict[str, float]:
    """1-day % change for each cue. Best-effort: missing tickers are skipped."""
    import yfinance as yf  # lazy import

    tickers = tickers or TICKERS
    out: dict[str, float] = {}
    for name, tk in tickers.items():
        try:
            df = yf.download(tk, period="5d", interval="1d", auto_adjust=True, progress=False)
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns=str.lower)
            pc = pct_change_last(df["close"])
            if pc is not None:
                out[name] = round(pc, 2)
        except Exception as exc:  # noqa: BLE001
            log.warning("cue %s (%s) failed: %s", name, tk, exc)
    return out
