"""Trading session runner — drives data → strategy → risk → broker.

`run_session` replays bars through a Broker (paper or, gated, live); it's the live
execution path, testable with a synthetic feed. `main()` runs a PAPER simulation on
recent free data and persists trades + per-day P&L to SQLite (each run replaces the
previous paper session — the sim is deterministic, so re-runs would only duplicate rows).

⚠️ The strategy has no proven edge. Paper only; do not deploy live.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.execution.broker_base import Broker
from src.indicators import engine as ind
from src.logutil import get_logger
from src.strategy import confluence

log = get_logger(__name__)


def run_session(frames: dict[str, pd.DataFrame], strategy_cfg: dict, broker: Broker,
                weights=None, signal_fn=None, signals_by_symbol=None) -> list[dict]:
    """Replay each symbol's bars through the broker; return closed-trade results.

    `signal_fn(df, params) -> DataFrame[entered, direction]` overrides the default
    confluence strategy (e.g. the 'pullback in trend' rule). `signals_by_symbol` is a
    precomputed {symbol: signals} mapping for cross-symbol-coordinated entries (e.g. a
    daily trade quota) that a per-symbol `signal_fn` can't express. Default keeps confluence.
    """
    results: list[dict] = []
    params = strategy_cfg.get("indicator_params", {}) or {}
    for sym, df in frames.items():
        if len(df) < 3:
            continue
        if signals_by_symbol is not None:
            res = signals_by_symbol.get(sym)
            if res is None:
                continue
        elif signal_fn:
            res = signal_fn(df, params)
        else:
            res = confluence.analyze(df, strategy_cfg, weights)
        atr = ind.compute_indicators(df, params)["atr"].to_numpy()
        entered = res["entered"].to_numpy()
        direction = res["direction"].to_numpy()
        opens, highs = df["open"].to_numpy(), df["high"].to_numpy()
        lows, closes = df["low"].to_numpy(), df["close"].to_numpy()
        times, n = df.index, len(df)
        for i in range(n):
            closed = broker.on_bar(sym, highs[i], lows[i], closes[i], times[i], force_exit=(i == n - 1))
            if closed:
                results.append(closed)
            if i > 0 and entered[i - 1] and atr[i - 1] > 0 and broker.can_open(sym):
                broker.open(sym, int(direction[i - 1]), opens[i], atr[i - 1], times[i])
    return results


def reset_paper_session(mode: str = "paper") -> None:
    """Delete prior trades + daily P&L for `mode` so a re-run replaces the last session."""
    from src.storage.db import DailyPnL, Trade, get_session

    with get_session() as s:
        s.query(Trade).filter(Trade.mode == mode).delete()
        s.query(DailyPnL).filter(DailyPnL.mode == mode).delete()
        s.commit()


def persist_daily_pnl(mode: str = "paper") -> int:
    """Rebuild per-exit-day DailyPnL rows for `mode` from the persisted trades.

    Feeds the dashboard's daily-P&L panel. Returns the number of days written.
    """
    from src.storage.db import DailyPnL, Trade, get_session

    with get_session() as s:
        s.query(DailyPnL).filter(DailyPnL.mode == mode).delete()
        totals: dict[date, tuple[float, int]] = {}
        for t in s.query(Trade).filter(Trade.mode == mode, Trade.exit_time.is_not(None)):
            pnl, count = totals.get(t.exit_time.date(), (0.0, 0))
            totals[t.exit_time.date()] = (pnl + (t.pnl or 0.0), count + 1)
        for day, (pnl, count) in sorted(totals.items()):
            s.add(DailyPnL(day=day, realized_pnl=round(pnl, 2), trades_count=count, mode=mode))
        s.commit()
        return len(totals)


def main(symbols: list[str] | None = None, timeframe: str | None = None,
         signal_fn=None, signals_builder=None, risk_overrides: dict | None = None,
         max_hold_bars: int | None = None, tail: int | None = None,
         label: str = "confluence strategy") -> None:  # pragma: no cover - convenience, needs network
    """PAPER simulation. Defaults replay the confluence strategy on the configured
    universe/timeframe; `paper_candidate` passes overrides for the edgesearch candidate.

    `signals_builder(frames) -> {symbol: signals}` enables cross-symbol-coordinated entries
    (e.g. a daily trade quota); `risk_overrides` patches RiskParams for this run only."""
    from rich.console import Console

    from src.backtest.engine import Costs
    from src.config import get_config
    from src.data.historical import HistoricalData
    from src.execution.paper import PaperBroker
    from src.risk.manager import RiskParams
    from src.storage.db import get_session, init_db

    cfg = get_config()
    init_db(cfg.storage.db_path)
    strat = cfg.strategy.model_dump()
    rp = RiskParams.from_config({**cfg.risk.model_dump(), **(risk_overrides or {})})
    costs = Costs.from_config(cfg.costs.model_dump())

    hist = HistoricalData(cache_dir="data/cache")
    timeframe = timeframe or cfg.instruments.base_timeframe
    frames = {}
    for sym in symbols or cfg.instruments.symbols:
        try:
            df = hist.get(sym, timeframe)
            frames[sym] = df.tail(tail) if tail else df
        except Exception as exc:  # noqa: BLE001
            log.warning("skip %s: %s", sym, exc)

    if not frames:
        log.error("no data fetched — keeping previous paper session intact")
        return

    reset_paper_session()
    broker = PaperBroker(rp, costs, session_factory=get_session, mode="paper",
                         max_hold_bars=max_hold_bars)
    signals_by_symbol = signals_builder(frames) if signals_builder else None
    results = run_session(frames, strat, broker, signal_fn=signal_fn,
                          signals_by_symbol=signals_by_symbol)
    days = persist_daily_pnl()
    total = sum(r["pnl"] for r in results)  # broker.realized_pnl() is per-day, not session
    Console().print(
        f"[bold]Paper session[/bold] ({label}, {timeframe}): {len(results)} trades closed "
        f"over {days} day(s), realized ₹{total:.0f}  "
        f"[dim](persisted to SQLite — replaces prior paper run)[/dim]"
    )
    log.info("Reminder: strategy has no proven edge — paper simulation only.")


def random_signals(df: pd.DataFrame, entry_prob: float = 0.02,
                   seed: int | None = None) -> pd.DataFrame:
    """Coin-flip control: enter at random bars with a random direction.

    Zero information by construction — the honest baseline any strategy must beat
    once it runs through the same risk manager and costs.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    entered = rng.random(n) < entry_prob
    direction = np.where(entered, rng.choice((-1, 1), size=n), 0)
    return pd.DataFrame({"entered": entered, "direction": direction}, index=df.index)


def quota_signals(frames: dict[str, pd.DataFrame], per_day: int = 5,
                  seed: int | None = 0) -> dict[str, pd.DataFrame]:
    """Coordinated random entries with a daily quota: each trading day pick `per_day`
    distinct symbols (that trade that day) to enter, with a random direction.

    A per-symbol signal_fn can't guarantee a universe-wide minimum, so this returns a
    {symbol: signals} mapping for `run_session(signals_by_symbol=...)`. Note: actual fills
    can still be fewer than the quota when a chosen symbol already holds a position or the
    risk caps bite — raise those caps (see `paper_active`) for the quota to actually execute.
    """
    rng = np.random.default_rng(seed)
    syms = list(frames)
    dates = sorted(set().union(*[set(df.index) for df in frames.values()])) if syms else []
    picks: dict[str, dict] = {s: {} for s in syms}
    for d in dates:
        avail = [s for s in syms if d in frames[s].index]
        if not avail:
            continue
        for i in rng.choice(len(avail), size=min(per_day, len(avail)), replace=False):
            picks[avail[int(i)]][d] = int(rng.choice((-1, 1)))
    out = {}
    for s, df in frames.items():
        entered = pd.Series(False, index=df.index)
        direction = pd.Series(0, index=df.index)
        if picks[s]:
            idx = pd.DatetimeIndex(list(picks[s]))
            entered.loc[idx] = True
            direction.loc[idx] = list(picks[s].values())
        out[s] = pd.DataFrame({"entered": entered, "direction": direction})
    return out


def paper_random() -> None:  # pragma: no cover - convenience, needs network
    """PAPER-trade pure coin-flips across the research universe — the zero-skill control."""
    from src.backtest.momentum import DEFAULT_UNIVERSE

    main(symbols=[f"{s}.NS" for s in DEFAULT_UNIVERSE], timeframe="day",
         signal_fn=lambda df, _params: random_signals(df), tail=500,
         label="random coin-flip control")


def paper_active(per_day: int = 5) -> None:  # pragma: no cover - convenience, needs network
    """PAPER-trade an ACTIVE quota: at least `per_day` (default 5) random entries every day.

    To hit the quota the day-scoped risk caps are raised for this run only (more concurrent
    positions, higher trades/day, daily loss-halt off). Old paper data is cleared first (via
    `main`). This is a teaching control: forcing more trades on a no-edge signal just multiplies
    costs — expect a bigger loss than the lighter random control, not a profit.
    """
    from src.backtest.momentum import DEFAULT_UNIVERSE

    main(symbols=[f"{s}.NS" for s in DEFAULT_UNIVERSE], timeframe="day",
         signals_builder=lambda frames: quota_signals(frames, per_day=per_day + 1, seed=0),
         risk_overrides={"max_concurrent_positions": max(3 * per_day, 20),
                         "max_trades_per_day": max(6 * per_day, 60),
                         "daily_loss_halt_pct": 100.0},
         max_hold_bars=1,                       # 1-day holds so symbols free up to meet the daily quota
         tail=500,                              # recent data only — full back-adjusted history blows up sizing
         label=f"active ≥{per_day} trades/day (random, 1-day holds, caps raised)")


def paper_candidate() -> None:  # pragma: no cover - convenience, needs network
    """PAPER-trade the best edge-search candidate on recent daily data.

    ⚠️ The candidate is UNVALIDATED (recent-regime artifact, see CLAUDE.md) — this
    command exists to watch it honestly, not because it is believed to have edge.
    """
    from src.backtest.edge_search import BEST_CANDIDATE, candidate_signals
    from src.backtest.momentum import DEFAULT_UNIVERSE

    log.info("Candidate paper session: %s %s", BEST_CANDIDATE[0], BEST_CANDIDATE[1])
    main(symbols=[f"{s}.NS" for s in DEFAULT_UNIVERSE], timeframe="day",
         signal_fn=lambda df, _params: candidate_signals(df), tail=500,
         label="edgesearch candidate")


if __name__ == "__main__":  # pragma: no cover
    main()
