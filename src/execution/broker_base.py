"""Broker interface shared by paper and live execution."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Order:
    symbol: str
    direction: int          # +1 buy/long, -1 sell/short
    qty: int
    exchange: str = "NSE"
    product: str = "MIS"    # MIS=intraday, CNC=delivery, NRML=F&O carry
    order_type: str = "MARKET"


class Broker(ABC):
    """Minimal execution surface used by the runner."""

    @abstractmethod
    def can_open(self, symbol: str) -> bool: ...

    @abstractmethod
    def open(self, symbol: str, direction: int, ref_price: float, atr: float,
             ts: datetime) -> dict | None: ...

    @abstractmethod
    def on_bar(self, symbol: str, high: float, low: float, close: float,
               ts: datetime, force_exit: bool = False) -> dict | None: ...
