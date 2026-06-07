"""Live broker — real Zerodha Kite orders. ⚠️ HARD-GATED OFF BY DEFAULT.

The backtests showed NO edge vs buy-and-hold, so deploying this loses money in
expectation. It exists only to complete the architecture. `place()` refuses unless
`allow_live_orders=True` is explicitly set, and logs a warning on every call.
Untested against the real API (needs a paid Kite Connect subscription + market hours).
"""
from __future__ import annotations

from src.logutil import get_logger

log = get_logger(__name__)


def order_params(symbol: str, direction: int, qty: int,
                 exchange: str = "NSE", product: str = "MIS",
                 order_type: str = "MARKET") -> dict:
    """Build Kite place_order kwargs. Pure function — unit-tested without the API."""
    return {
        "variety": "regular",
        "exchange": exchange,
        "tradingsymbol": symbol,
        "transaction_type": "BUY" if direction > 0 else "SELL",
        "quantity": int(qty),
        "product": product,
        "order_type": order_type,
    }


class LiveBroker:
    def __init__(self, kite, product: str = "MIS", exchange: str = "NSE",
                 allow_live_orders: bool = False):
        self.kite = kite
        self.product = product
        self.exchange = exchange
        self.allow = allow_live_orders
        if allow_live_orders:
            log.warning("⚠️ LIVE ORDERS ENABLED. The strategy has NO proven edge — "
                        "this is expected to lose money. Proceed only if you accept that.")

    def place(self, symbol: str, direction: int, qty: int, order_type: str = "MARKET") -> str:
        if not self.allow:
            raise RuntimeError(
                "LIVE ORDERS DISABLED. The strategy has no proven edge (it loses to "
                "buy-and-hold). Set live.allow_live_orders=true to override — not recommended."
            )
        params = order_params(symbol, direction, qty, self.exchange, self.product, order_type)
        log.warning("Placing REAL order: %s", params)
        return self.kite.place_order(**params)

    def positions(self) -> list:
        return self.kite.positions()
