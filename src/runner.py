"""Trading session runner — drives data → strategy → risk → broker.

`run_session` replays bars through a Broker (paper or, gated, live); it's the live
execution path, testable with a synthetic feed. `main()` runs a PAPER simulation on
recent free data and persists trades to SQLite.

⚠️ The strategy has no proven edge. Paper only; do not deploy live.
"""
from __future__ import annotations

import pandas as pd

from src.execution.broker_base import Broker
from src.indicators import engine as ind
from src.logutil import get_logger
from src.strategy import confluence

log = get_logger(__name__)


def run_session(frames: dict[str, pd.DataFrame], strategy_cfg: dict, broker: Broker,
                weights=None) -> list[dict]:
    """Replay each symbol's bars through the broker; return closed-trade results."""
    results: list[dict] = []
    params = strategy_cfg.get("indicator_params", {}) or {}
    for sym, df in frames.items():
        if len(df) < 3:
            continue
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


def main() -> None:  # pragma: no cover - convenience, needs network
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
    rp = RiskParams.from_config(cfg.risk.model_dump())
    costs = Costs.from_config(cfg.costs.model_dump())

    hist = HistoricalData(cache_dir="data/cache")
    frames = {}
    for sym in cfg.instruments.symbols:
        try:
            frames[sym] = hist.get(sym, cfg.instruments.base_timeframe)
        except Exception as exc:  # noqa: BLE001
            log.warning("skip %s: %s", sym, exc)

    broker = PaperBroker(rp, costs, session_factory=get_session, mode="paper")
    results = run_session(frames, strat, broker)
    Console().print(
        f"[bold]Paper session[/bold]: {len(results)} trades closed, "
        f"realized ₹{broker.realized_pnl():.0f}  [dim](persisted to SQLite)[/dim]"
    )
    log.info("Reminder: strategy has no proven edge — paper simulation only.")


if __name__ == "__main__":  # pragma: no cover
    main()
