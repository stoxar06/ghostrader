"""Analyst scheduler — runs the daily briefing pre-market + periodically during
market hours and delivers it (log + optional Telegram).

ANALYSIS DELIVERY ONLY. It does not trade — the backtests showed no edge.

Run: python -m src.scheduler
"""
from __future__ import annotations

from datetime import datetime

from src.logutil import get_logger, setup_logging
from src.mcp import tools
from src.notify import report, telegram

log = get_logger(__name__)


def deliver(text: str) -> None:
    log.info("\n%s", text)
    telegram.send_message(text)


def run_once(briefing_fn=None, deliver_fn=None) -> dict:
    """Fetch a briefing and deliver it. Injectable for testing."""
    briefing_fn = briefing_fn or tools.get_briefing
    deliver_fn = deliver_fn or deliver
    data = briefing_fn()
    deliver_fn(report.format_briefing(data))
    return data


def intraday_tick() -> None:  # pragma: no cover - exercised via scheduler
    from src import marketcal
    from src.config import get_config

    cfg = get_config()
    if marketcal.is_market_open(datetime.now(), cfg.session.market_open, cfg.session.market_close):
        run_once()
    else:
        log.debug("Market closed — skipping intraday briefing.")


def main() -> None:  # pragma: no cover - long-running
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    from src.config import get_config

    cfg = get_config()
    setup_logging(cfg.logging.level, cfg.logging.file)
    refresh = int(getattr(cfg.macro, "refresh_minutes", 20))
    oh, om = str(cfg.session.market_open).split(":")
    pre_minute = max(0, int(om) - 15)

    sched = BlockingScheduler(timezone="Asia/Kolkata")
    sched.add_job(run_once, CronTrigger(hour=int(oh), minute=pre_minute, day_of_week="mon-fri"),
                  id="premarket")
    sched.add_job(intraday_tick, IntervalTrigger(minutes=refresh), id="intraday")

    log.info("Scheduler started: pre-market %02d:%02d + every %d min during market hours (IST).",
             int(oh), pre_minute, refresh)
    run_once()  # immediate first briefing
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopped.")


if __name__ == "__main__":  # pragma: no cover
    main()
