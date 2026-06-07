"""Execution package — broker adapters behind one interface.

PaperBroker (simulated, safe, default) and LiveBroker (real Kite orders, hard-gated
OFF). The same strategy/risk code drives both. ⚠️ The strategy has no proven edge —
live trading is for completeness, not recommended.
"""
from .broker_base import Broker, Order  # noqa: F401
from .paper import PaperBroker  # noqa: F401
