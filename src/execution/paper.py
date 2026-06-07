"""Paper broker — simulated fills with slippage + Zerodha costs. No real money.

Mirrors the backtest's fill/exit logic but as a stateful, bar-by-bar live broker,
so the same runner can drive paper today and (gated) live later. Persists closed
trades to SQLite when a session factory is provided.
"""
from __future__ import annotations

from datetime import datetime

from src.backtest.engine import Costs, trade_charges
from src.logutil import get_logger
from src.risk.manager import (
    RiskManager,
    RiskParams,
    position_size,
    stop_and_target,
    trailing_stop,
)

log = get_logger(__name__)


class PaperBroker:
    def __init__(self, risk_params: RiskParams, costs: Costs,
                 session_factory=None, use_trailing: bool = True, mode: str = "paper"):
        self.p = risk_params
        self.costs = costs
        self.session_factory = session_factory
        self.use_trailing = use_trailing
        self.mode = mode
        self.rm = RiskManager(risk_params)
        self.positions: dict[str, dict] = {}

    def can_open(self, symbol: str) -> bool:
        return symbol not in self.positions and self.rm.can_open()

    def open(self, symbol, direction, ref_price, atr, ts):
        if not self.can_open(symbol) or direction == 0 or atr <= 0:
            return None
        slip = self.costs.slippage_pct
        fill = ref_price * (1 + slip) if direction > 0 else ref_price * (1 - slip)
        stop, target = stop_and_target(fill, atr, direction, self.p)
        qty = position_size(self.p.capital, self.p.risk_per_trade_pct, fill, stop)
        if qty <= 0:
            return None
        self.positions[symbol] = {
            "dir": direction, "entry": fill, "stop": stop, "target": target,
            "qty": qty, "atr": atr, "ts": ts,
        }
        self.rm.register_open()
        log.info("[%s] OPEN %s %s qty=%d @%.2f stop=%.2f target=%.2f",
                 self.mode, "LONG" if direction > 0 else "SHORT", symbol, qty, fill, stop, target)
        return self.positions[symbol]

    def on_bar(self, symbol, high, low, close, ts, force_exit=False):
        pos = self.positions.get(symbol)
        if not pos:
            return None

        exit_price, reason = None, None
        if pos["dir"] > 0:
            if low <= pos["stop"]:
                exit_price, reason = pos["stop"], "stop"
            elif high >= pos["target"]:
                exit_price, reason = pos["target"], "target"
        else:
            if high >= pos["stop"]:
                exit_price, reason = pos["stop"], "stop"
            elif low <= pos["target"]:
                exit_price, reason = pos["target"], "target"
        if exit_price is None and force_exit:
            exit_price, reason = close, "square_off"

        if exit_price is None:
            if self.use_trailing:
                pos["stop"] = trailing_stop(close, pos["atr"], pos["dir"], pos["stop"], self.p)
            return None

        slip = self.costs.slippage_pct
        fill = exit_price * (1 - slip) if pos["dir"] > 0 else exit_price * (1 + slip)
        gross = (fill - pos["entry"]) * pos["qty"] * pos["dir"]
        charges = trade_charges(pos["entry"], fill, pos["qty"], pos["dir"], self.costs)
        net = gross - charges
        self.rm.register_close(net)
        self._record(symbol, pos, fill, net, charges, reason, ts)
        log.info("[%s] CLOSE %s @%.2f pnl=%.2f (%s)", self.mode, symbol, fill, net, reason)
        del self.positions[symbol]
        return {"symbol": symbol, "pnl": round(net, 2), "reason": reason, "exit": round(fill, 2)}

    def _record(self, symbol, pos, fill, net, charges, reason, ts):
        if self.session_factory is None:
            return
        from src.storage.db import Trade
        with self.session_factory() as s:
            s.add(Trade(
                symbol=symbol, side="LONG" if pos["dir"] > 0 else "SHORT", qty=pos["qty"],
                entry_price=round(pos["entry"], 4), exit_price=round(fill, 4),
                entry_time=pos["ts"], exit_time=ts if isinstance(ts, datetime) else datetime.now(),
                stop_loss=pos["stop"], target=pos["target"], pnl=round(net, 2),
                charges=round(charges, 2), mode=self.mode, reason=reason,
            ))
            s.commit()

    def realized_pnl(self) -> float:
        return self.rm.realized_pnl_today
