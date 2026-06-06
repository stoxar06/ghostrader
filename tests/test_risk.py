"""Phase 3 — risk sizing, stops/targets, trailing, daily halt & caps."""
from __future__ import annotations

from src.risk.manager import (
    RiskManager,
    RiskParams,
    position_size,
    stop_and_target,
    trailing_stop,
)


def test_position_size_fixed_fractional():
    # risk 0.5% of 100000 = 500; stop distance = 2 -> 250 shares
    assert position_size(100_000, 0.5, 100.0, 98.0) == 250


def test_position_size_zero_distance():
    assert position_size(100_000, 0.5, 100.0, 100.0) == 0


def test_stop_and_target_long():
    p = RiskParams(target_pct=1.5, atr_stop_multiplier=1.5, reward_risk_min=1.5)
    stop, target = stop_and_target(100.0, 2.0, 1, p)
    assert stop == 97.0       # 100 - 2*1.5
    assert target == 104.5    # max(1.5, 3*1.5=4.5) -> 4.5 above entry


def test_stop_and_target_short():
    p = RiskParams(target_pct=1.5, atr_stop_multiplier=1.5, reward_risk_min=1.5)
    stop, target = stop_and_target(100.0, 2.0, -1, p)
    assert stop == 103.0
    assert target == 95.5


def test_trailing_stop_only_ratchets():
    p = RiskParams(trailing_atr_multiplier=2.0)
    assert trailing_stop(110.0, 2.0, 1, 100.0, p) == 106.0   # tightens up
    assert trailing_stop(104.0, 2.0, 1, 106.0, p) == 106.0   # never loosens


def test_daily_halt_and_caps():
    p = RiskParams(capital=100_000, daily_loss_halt_pct=1.0,
                   max_trades_per_day=2, max_concurrent_positions=1)
    rm = RiskManager(p)

    assert rm.can_open() is True
    rm.register_open()
    assert rm.open_positions == 1
    assert rm.can_open() is False           # hit max concurrent (1)

    rm.register_close(-1000.0)              # realized -1% of capital
    assert rm.daily_halt_triggered() is True
    assert rm.can_open() is False           # halted for the day


def test_riskparams_from_config_ignores_unknown_keys():
    p = RiskParams.from_config({"capital": 50_000, "target_pct": 2.0, "bogus": 1})
    assert p.capital == 50_000
    assert p.target_pct == 2.0
