"""SQLite persistence (SQLAlchemy 2.0).

Tables: trades, positions, daily_pnl, signals, macro_regime, llm_usage.
Call init_db(db_path) once at startup, then use get_session().
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from sqlalchemy import Date, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Trade(Base):
    """A completed (or open) round-trip trade — the audit record."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(4))  # BUY / SELL (entry direction)
    qty: Mapped[int] = mapped_column(Integer)
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    entry_time: Mapped[datetime] = mapped_column(DateTime)
    exit_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    target: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)  # net of charges
    charges: Mapped[float | None] = mapped_column(Float, nullable=True)
    mode: Mapped[str] = mapped_column(String(8))  # backtest / paper / live
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)  # entry/exit rationale


class Position(Base):
    """A currently-open position (live state)."""

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(4))
    qty: Mapped[int] = mapped_column(Integer)
    avg_price: Mapped[float] = mapped_column(Float)
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    target: Mapped[float | None] = mapped_column(Float, nullable=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime)
    mode: Mapped[str] = mapped_column(String(8))


class DailyPnL(Base):
    """Per-day realized P&L and whether the daily loss halt fired."""

    __tablename__ = "daily_pnl"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    day: Mapped[date] = mapped_column(Date, index=True)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    trades_count: Mapped[int] = mapped_column(Integer, default=0)
    halted: Mapped[bool] = mapped_column(default=False)
    mode: Mapped[str] = mapped_column(String(8))


class Signal(Base):
    """Every generated signal (acted or not) — for post-mortem and 'explain' tooling."""

    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    direction: Mapped[str] = mapped_column(String(8))  # long / short / neutral
    confidence: Mapped[float] = mapped_column(Float)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON of votes
    acted: Mapped[bool] = mapped_column(default=False)


class MacroRegime(Base):
    """Cached macro context produced by the macro engine (delta-skip via input_hash)."""

    __tablename__ = "macro_regime"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    regime: Mapped[str] = mapped_column(String(16))  # risk_on / neutral / risk_off
    bias: Mapped[str] = mapped_column(String(16))  # bullish / bearish / neutral
    payload: Mapped[str] = mapped_column(Text)  # full JSON (sectors, caution symbols, ...)
    input_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)


class LLMUsage(Base):
    """Per-call LLM usage for the daily paid-cost cap (free tiers record cost_inr=0)."""

    __tablename__ = "llm_usage"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    day: Mapped[date] = mapped_column(Date, index=True)
    provider: Mapped[str] = mapped_column(String(16))
    model: Mapped[str] = mapped_column(String(64))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_inr: Mapped[float] = mapped_column(Float, default=0.0)


_engine = None
_SessionLocal: sessionmaker | None = None


def init_db(db_path: str) -> None:
    """Create the engine + tables. Idempotent (create_all skips existing tables)."""
    global _engine, _SessionLocal
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(f"sqlite:///{db_path}", future=True)
    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)


def get_session():
    """Return a new Session. Raises if init_db() hasn't been called."""
    if _SessionLocal is None:
        raise RuntimeError("DB not initialized — call init_db(db_path) first.")
    return _SessionLocal()
