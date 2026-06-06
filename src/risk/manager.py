"""Risk management — identical in paper and live.

- Fixed-fractional position sizing (risk % of capital ÷ stop distance).
- ATR-based stop; target = max(target_pct, reward:risk · stop) so ~1% nets after costs.
- ATR trailing stop to let winners run.
- Daily loss halt, max concurrent positions, max trades/day.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RiskParams:
    capital: float = 100_000.0
    risk_per_trade_pct: float = 0.5
    target_pct: float = 1.5
    atr_stop_multiplier: float = 1.5
    reward_risk_min: float = 1.5
    trailing_atr_multiplier: float = 2.0
    daily_loss_halt_pct: float = 1.0
    max_concurrent_positions: int = 3
    max_trades_per_day: int = 10
    per_instrument_exposure_pct: float = 30.0

    @classmethod
    def from_config(cls, risk_cfg: dict) -> "RiskParams":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (risk_cfg or {}).items() if k in known})


def position_size(capital: float, risk_pct: float, entry: float, stop: float) -> int:
    """Whole-share quantity such that (entry-stop) loss ≈ risk_pct% of capital."""
    per_share = abs(entry - stop)
    if per_share <= 0:
        return 0
    risk_amount = capital * risk_pct / 100.0
    return int(math.floor(risk_amount / per_share))


def stop_and_target(entry: float, atr_value: float, direction: int, p: RiskParams) -> tuple[float, float]:
    """ATR stop; target is the larger of target_pct and reward:risk · stop distance."""
    stop_dist = atr_value * p.atr_stop_multiplier
    pct_move = entry * p.target_pct / 100.0
    rr_move = stop_dist * p.reward_risk_min
    move = max(pct_move, rr_move)
    if direction > 0:
        return entry - stop_dist, entry + move
    return entry + stop_dist, entry - move


def trailing_stop(current_price: float, atr_value: float, direction: int,
                  prev_stop: float, p: RiskParams) -> float:
    """Ratchet the stop in the trade's favor; never loosen it."""
    dist = atr_value * p.trailing_atr_multiplier
    if direction > 0:
        return max(prev_stop, current_price - dist)
    return min(prev_stop, current_price + dist)


class RiskManager:
    """Tracks per-day state and enforces the trading guardrails."""

    def __init__(self, params: RiskParams):
        self.p = params
        self.realized_pnl_today: float = 0.0
        self.trades_today: int = 0
        self.open_positions: int = 0

    def reset_day(self) -> None:
        self.realized_pnl_today = 0.0
        self.trades_today = 0
        # open_positions persists across the reset only if positions are held overnight;
        # intraday it should be 0 at the start of a session.

    def daily_halt_triggered(self) -> bool:
        limit = -self.p.capital * self.p.daily_loss_halt_pct / 100.0
        return self.realized_pnl_today <= limit

    def can_open(self) -> bool:
        if self.daily_halt_triggered():
            return False
        if self.trades_today >= self.p.max_trades_per_day:
            return False
        if self.open_positions >= self.p.max_concurrent_positions:
            return False
        return True

    def register_open(self) -> None:
        self.open_positions += 1
        self.trades_today += 1

    def register_close(self, pnl: float) -> None:
        self.realized_pnl_today += pnl
        self.open_positions = max(0, self.open_positions - 1)
