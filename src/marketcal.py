"""NSE session & holiday helpers.

Weekends are always non-trading. Full NSE holidays change yearly — supply them as
a set of ISO date strings (e.g. from config) to `holidays=`.
"""
from __future__ import annotations

from datetime import date, datetime, time


def _parse_time(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def is_weekend(d: date) -> bool:
    return d.weekday() >= 5  # Sat=5, Sun=6


def is_trading_day(d: date, holidays: set[str] | None = None) -> bool:
    if is_weekend(d):
        return False
    if holidays and d.isoformat() in holidays:
        return False
    return True


def is_market_open(
    now: datetime,
    open_str: str = "09:15",
    close_str: str = "15:30",
    holidays: set[str] | None = None,
) -> bool:
    if not is_trading_day(now.date(), holidays):
        return False
    return _parse_time(open_str) <= now.time() <= _parse_time(close_str)
