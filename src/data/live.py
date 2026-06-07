"""Live market data — tick → bar aggregation + a (thin) KiteTicker feed wrapper.

`CandleBuilder` is pure and unit-tested; the WebSocket feed needs real Kite creds.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.logutil import get_logger

log = get_logger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class CandleBuilder:
    """Aggregate ticks into fixed-interval OHLCV bars, per symbol.

    `add_tick` returns the just-completed bar when a tick rolls into a new bucket,
    else None. Completed bars are also retained and exposed via `frame`.
    """

    def __init__(self, interval_seconds: int = 300):
        self.interval = interval_seconds
        self._current: dict[str, dict] = {}
        self._completed: dict[str, list[dict]] = {}

    def add_tick(self, symbol: str, price: float, volume: float = 0.0,
                 ts: datetime | None = None) -> dict | None:
        ts = ts or datetime.now()
        bucket = int(ts.timestamp() // self.interval) * self.interval
        cur = self._current.get(symbol)
        completed = None

        if cur is None or cur["bucket"] != bucket:
            if cur is not None:
                self._completed.setdefault(symbol, []).append(cur)
                completed = cur
            cur = {"bucket": bucket, "start": datetime.fromtimestamp(bucket),
                   "open": price, "high": price, "low": price, "close": price, "volume": volume}
            self._current[symbol] = cur
        else:
            cur["high"] = max(cur["high"], price)
            cur["low"] = min(cur["low"], price)
            cur["close"] = price
            cur["volume"] += volume
        return completed

    def frame(self, symbol: str, include_current: bool = False) -> pd.DataFrame:
        bars = list(self._completed.get(symbol, []))
        if include_current and symbol in self._current:
            bars = bars + [self._current[symbol]]
        if not bars:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        df = pd.DataFrame(bars).set_index("start")[OHLCV_COLUMNS]
        df.index.name = "datetime"
        return df


class KiteTickerFeed:  # pragma: no cover - needs real Kite credentials
    """Thin wrapper over KiteTicker that pushes ticks into a CandleBuilder."""

    def __init__(self, api_key: str, access_token: str, interval_seconds: int = 300):
        self.api_key = api_key
        self.access_token = access_token
        self.builder = CandleBuilder(interval_seconds)

    def run(self, instrument_tokens, on_bar=None):
        from kiteconnect import KiteTicker  # lazy import

        kws = KiteTicker(self.api_key, self.access_token)

        def on_ticks(ws, ticks):
            for t in ticks:
                bar = self.builder.add_tick(str(t["instrument_token"]),
                                            t["last_price"], t.get("volume_traded", 0))
                if bar and on_bar:
                    on_bar(str(t["instrument_token"]), bar)

        def on_connect(ws, _):
            ws.subscribe(instrument_tokens)
            ws.set_mode(ws.MODE_FULL, instrument_tokens)

        kws.on_ticks = on_ticks
        kws.on_connect = on_connect
        log.info("Connecting KiteTicker for %d instruments", len(instrument_tokens))
        kws.connect()
