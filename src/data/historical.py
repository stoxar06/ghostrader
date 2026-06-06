"""Historical OHLCV data: provider abstraction + local CSV cache.

Backends:
  - YFinanceProvider (free): NSE equities via '<SYMBOL>.NS'. `auto_adjust=True`
    returns corporate-action-adjusted prices (handles the splits/bonuses gap).
    Intraday history is capped by yfinance (1m~7d, 5m/15m~60d, 60m~730d).
  - KiteHistoricalProvider: richer history; wired in Phase 5 once auth exists.

All providers return a pandas DataFrame indexed by datetime with columns:
    open, high, low, close, volume
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

from src.logutil import get_logger

log = get_logger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Our (Kite-style) interval names -> yfinance interval codes.
_YF_INTERVAL = {
    "minute": "1m", "3minute": "3m", "5minute": "5m", "10minute": "10m",
    "15minute": "15m", "30minute": "30m", "60minute": "60m", "day": "1d",
}
# yfinance history caps per interval -> a safe default `period` when none given.
_YF_MAX_PERIOD = {
    "1m": "7d", "3m": "60d", "5m": "60d", "10m": "60d", "15m": "60d",
    "30m": "60d", "60m": "730d", "1d": "max",
}


class HistoricalProvider(Protocol):
    def fetch(
        self, symbol: str, interval: str, start=None, end=None, period: str | None = None
    ) -> pd.DataFrame: ...


class YFinanceProvider:
    """Free historical data via yfinance (NSE by default: '<SYMBOL>.NS')."""

    def __init__(self, suffix: str = ".NS"):
        self.suffix = suffix

    def _ticker(self, symbol: str) -> str:
        # Leave indices (^NSEI) and already-qualified tickers (with a dot) untouched.
        if "." in symbol or symbol.startswith("^"):
            return symbol
        return f"{symbol}{self.suffix}"

    def fetch(self, symbol, interval, start=None, end=None, period=None) -> pd.DataFrame:
        # Validate the interval BEFORE importing yfinance so callers/tests fail fast.
        if interval not in _YF_INTERVAL:
            raise ValueError(
                f"Unsupported interval {interval!r}. Known: {sorted(_YF_INTERVAL)}"
            )
        import yfinance as yf  # lazy import — heavy, only needed for a real fetch

        yf_int = _YF_INTERVAL[interval]
        ticker = self._ticker(symbol)

        kwargs = dict(interval=yf_int, auto_adjust=True, progress=False)
        if period:
            kwargs["period"] = period
        elif start or end:
            kwargs["start"], kwargs["end"] = start, end
        else:
            kwargs["period"] = _YF_MAX_PERIOD[yf_int]

        log.info("Fetching %s @ %s from yfinance (%s)", ticker, yf_int,
                 period or f"{start}..{end}")
        df = yf.download(ticker, **kwargs)
        if df is None or df.empty:
            raise RuntimeError(f"No data returned for {ticker} @ {yf_int}")

        # Single-ticker downloads sometimes come back with a MultiIndex column.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)

        missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
        if missing:
            raise RuntimeError(f"yfinance result missing columns {missing} for {ticker}")
        df = df[OHLCV_COLUMNS].copy()
        df.index.name = "datetime"
        return df.dropna()


class KiteHistoricalProvider:
    """Placeholder — wired in Phase 5 (auth). Provides richer history than yfinance."""

    def __init__(self, kite=None):
        self.kite = kite

    def fetch(self, symbol, interval, start=None, end=None, period=None) -> pd.DataFrame:
        raise NotImplementedError("KiteHistoricalProvider is wired in Phase 5 (auth).")


def _bound(x) -> str | None:
    return None if x is None else str(x)


class HistoricalData:
    """Fetch via a provider, cache to CSV, slice to the requested range."""

    def __init__(self, cache_dir: str = "data/cache", provider: HistoricalProvider | None = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.provider: HistoricalProvider = provider or YFinanceProvider()

    def _cache_path(self, symbol: str, interval: str) -> Path:
        return self.cache_dir / f"{symbol.replace('/', '_')}_{interval}.csv"

    def get(self, symbol, interval, start=None, end=None, period=None, refresh=False) -> pd.DataFrame:
        """Return cached data if present, else fetch + cache. Slices to [start, end] if given."""
        path = self._cache_path(symbol, interval)
        if path.exists() and not refresh:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            log.debug("Loaded %s from cache (%d rows)", path.name, len(df))
        else:
            df = self.provider.fetch(symbol, interval, start, end, period)
            df.to_csv(path)
            log.info("Cached %s (%d rows) -> %s", symbol, len(df), path.name)

        if start is not None or end is not None:
            df = df.loc[_bound(start):_bound(end)]
        return df

    def warmup(self, symbol, interval, bars: int = 200, **kw) -> pd.DataFrame:
        """Return the most recent `bars` rows for indicator warmup."""
        return self.get(symbol, interval, **kw).tail(bars)
