"""Analyst tool logic (plain functions, testable without the MCP runtime).

Network/LLM tools are thin wrappers over the already-tested macro/backtest
modules; DB tools read from the storage layer. All are READ-ONLY analysis.
"""
from __future__ import annotations

from datetime import date, timedelta

from src.logutil import get_logger

log = get_logger(__name__)


def _router_and_session():
    """Build the free-first router (or None if no keys) over the configured DB."""
    from src.config import get_config, get_secrets
    from src.llm import Router, build_providers
    from src.storage.db import get_session, init_db

    cfg = get_config()
    init_db(cfg.storage.db_path)
    providers = build_providers(get_secrets())
    router = Router(providers, cfg.llm.model_dump(), get_session) if providers else None
    return cfg, router, get_session


def get_cues() -> dict:
    """Latest 1-day % change for the global market cues."""
    from src.macro import globalcues

    return globalcues.fetch_global_cues()


def get_regime() -> dict:
    """Current market regime + bias (free-first LLM, deterministic fallback)."""
    from src.macro import globalcues, news, regime

    _, router, sf = _router_and_session()
    return regime.get_or_build(globalcues.fetch_global_cues(), news.fetch_headlines(), router, sf)


def get_briefing() -> dict:
    """Full daily briefing: regime, cues, and recent headlines."""
    from src.macro import globalcues, news, regime

    _, router, sf = _router_and_session()
    cues = globalcues.fetch_global_cues()
    headlines = news.fetch_headlines()
    reg = regime.get_or_build(cues, headlines, router, sf)
    return {"regime": reg, "cues": cues, "headline_count": len(headlines), "headlines": headlines[:10]}


def run_backtest_tool(symbol: str, timeframe: str | None = None) -> dict:
    """Backtest the confluence strategy on one symbol (net of costs) -> metrics."""
    from src.backtest.engine import Costs, run_backtest
    from src.backtest.metrics import compute_metrics
    from src.config import get_config
    from src.data.historical import HistoricalData
    from src.risk.manager import RiskParams

    cfg = get_config()
    strat = cfg.strategy.model_dump()
    rp = RiskParams.from_config(cfg.risk.model_dump())
    costs = Costs.from_config(cfg.costs.model_dump())
    tf = timeframe or cfg.instruments.base_timeframe
    df = HistoricalData(cache_dir="data/cache").get(symbol, tf)
    trades = run_backtest(df, strat, rp, costs, symbol=symbol)
    return compute_metrics(trades, rp.capital)


def get_daily_pnl(days: int = 5) -> list[dict]:
    """Recent daily realized-P&L records from the database."""
    from src.storage.db import DailyPnL, get_session

    cutoff = date.today() - timedelta(days=days)
    with get_session() as s:
        rows = (
            s.query(DailyPnL).filter(DailyPnL.day >= cutoff)
            .order_by(DailyPnL.day.desc()).all()
        )
        return [
            {"day": str(r.day), "realized_pnl": r.realized_pnl, "trades": r.trades_count,
             "halted": r.halted, "mode": r.mode}
            for r in rows
        ]


def explain_last_trades(n: int = 5) -> list[dict]:
    """The last N trades with entry/exit/pnl/reason."""
    from src.storage.db import Trade, get_session

    with get_session() as s:
        rows = s.query(Trade).order_by(Trade.id.desc()).limit(n).all()
        return [
            {"symbol": r.symbol, "side": r.side, "qty": r.qty, "entry": r.entry_price,
             "exit": r.exit_price, "pnl": r.pnl, "reason": r.reason, "mode": r.mode}
            for r in rows
        ]
