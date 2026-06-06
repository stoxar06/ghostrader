"""Phase 6 — market calendar, briefing formatting, Telegram no-op, scheduler tick."""
from __future__ import annotations

from datetime import date, datetime

from src import marketcal
from src.notify.report import format_briefing
from src.notify.telegram import send_message
from src.scheduler import run_once


def test_is_weekend():
    assert marketcal.is_weekend(date(2024, 1, 6))       # Saturday
    assert not marketcal.is_weekend(date(2024, 1, 8))   # Monday


def test_is_trading_day_respects_holidays():
    assert marketcal.is_trading_day(date(2024, 1, 8))
    assert not marketcal.is_trading_day(date(2024, 1, 8), {"2024-01-08"})


def test_is_market_open_window():
    assert marketcal.is_market_open(datetime(2024, 1, 8, 10, 0))      # Mon 10:00
    assert not marketcal.is_market_open(datetime(2024, 1, 8, 8, 0))   # before open
    assert not marketcal.is_market_open(datetime(2024, 1, 6, 10, 0))  # Saturday


def test_format_briefing_contains_key_fields():
    brief = {
        "regime": {"regime": "risk_off", "bias": "bearish", "summary": "down",
                   "source": "deterministic", "favored_sectors": [], "avoid_sectors": []},
        "cues": {"nifty": -1.0}, "headline_count": 5,
    }
    s = format_briefing(brief)
    assert "risk_off" in s and "bearish" in s and "nifty" in s


def test_telegram_noop_without_credentials():
    assert send_message("hi", token="", chat_id="") is False


def test_run_once_delivers_formatted_briefing():
    captured = {}
    fake = {
        "regime": {"regime": "risk_on", "bias": "bullish", "summary": "ok", "source": "deterministic"},
        "cues": {"nifty": 0.5}, "headline_count": 3, "headlines": ["a"],
    }
    data = run_once(briefing_fn=lambda: fake, deliver_fn=lambda t: captured.update(text=t))
    assert data["regime"]["regime"] == "risk_on"
    assert "risk_on" in captured["text"]
