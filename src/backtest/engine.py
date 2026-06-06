"""Event-driven intraday backtest.

Decisions are made on bar i-1's close and acted on bar i's open (no look-ahead).
Open positions are managed bar-by-bar against stop/target (using the bar's high/low),
trailed by ATR, and force-squared-off at each session's last bar. Every trade is net
of Zerodha intraday charges + slippage so the 1.5%→~1% net is realistic.

This is the GO/NO-GO gate: only proceed past Phase 4 if the edge proves out
on out-of-sample (walk-forward) data.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.indicators import accuracy, patterns
from src.indicators import engine as ind
from src.logutil import get_logger
from src.risk.manager import (
    RiskManager,
    RiskParams,
    position_size,
    stop_and_target,
    trailing_stop,
)
from src.strategy import confluence

log = get_logger(__name__)


@dataclass
class Costs:
    """Zerodha intraday equity charges (per round trip) + assumed slippage per side."""

    brokerage_per_order: float = 20.0
    brokerage_pct: float = 0.0003
    stt_sell_pct: float = 0.00025
    exchange_txn_pct: float = 0.0000297
    gst_pct: float = 0.18
    sebi_per_crore: float = 10.0
    stamp_buy_pct: float = 0.00003
    slippage_pct: float = 0.0005

    @classmethod
    def from_config(cls, costs_cfg: dict) -> "Costs":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (costs_cfg or {}).items() if k in known})


def trade_charges(entry: float, exit: float, qty: int, direction: int, c: Costs) -> float:
    """Total charges for one round-trip trade (entry + exit legs)."""
    if direction > 0:  # long: buy at entry, sell at exit
        buy_val, sell_val = entry * qty, exit * qty
    else:              # short: sell at entry, buy at exit
        sell_val, buy_val = entry * qty, exit * qty
    turnover = buy_val + sell_val

    brokerage = min(c.brokerage_per_order, c.brokerage_pct * buy_val) + min(
        c.brokerage_per_order, c.brokerage_pct * sell_val
    )
    stt = c.stt_sell_pct * sell_val
    exchange = c.exchange_txn_pct * turnover
    sebi = c.sebi_per_crore * turnover / 1e7
    stamp = c.stamp_buy_pct * buy_val
    gst = c.gst_pct * (brokerage + exchange + sebi)
    return brokerage + stt + exchange + sebi + stamp + gst


def run_backtest(
    df: pd.DataFrame,
    strategy_cfg: dict,
    risk_params: RiskParams,
    costs: Costs,
    weights: accuracy.AccuracyWeights | None = None,
    symbol: str = "?",
    htf_df: pd.DataFrame | None = None,
    use_trailing: bool = True,
    signals: pd.DataFrame | None = None,
    square_off_eod: bool = True,
) -> pd.DataFrame:
    """Replay `df` and return a trades DataFrame (net of costs + slippage).

    `signals` (optional) lets any strategy plug in: a DataFrame aligned to df with
    boolean `entered` and int `direction` columns. If None, the confluence
    strategy is used.
    """
    if len(df) < 3:
        return pd.DataFrame()

    result = signals if signals is not None else confluence.analyze(df, strategy_cfg, weights, htf_df)
    df_ind = ind.compute_indicators(df, strategy_cfg.get("indicator_params", {}) or {})

    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    times = df.index
    atr_arr = df_ind["atr"].to_numpy()
    entered = result["entered"].to_numpy()
    direction = result["direction"].to_numpy()
    n = len(df)

    dates = df.index.normalize().to_numpy() if isinstance(df.index, pd.DatetimeIndex) else None

    def session_end(i: int) -> bool:
        if i == n - 1:
            return True
        return dates is not None and dates[i] != dates[i + 1]

    rm = RiskManager(risk_params)
    pos: dict | None = None
    trades: list[dict] = []
    slip = costs.slippage_pct

    for i in range(n):
        if i > 0 and dates is not None and dates[i] != dates[i - 1]:
            rm.reset_day()

        # 1) Manage an open position against this bar's range.
        if pos is not None:
            exit_price = None
            reason = None
            if pos["dir"] > 0:
                if lows[i] <= pos["stop"]:
                    exit_price, reason = pos["stop"], "stop"
                elif highs[i] >= pos["target"]:
                    exit_price, reason = pos["target"], "target"
            else:
                if highs[i] >= pos["stop"]:
                    exit_price, reason = pos["stop"], "stop"
                elif lows[i] <= pos["target"]:
                    exit_price, reason = pos["target"], "target"

            if exit_price is None and square_off_eod and session_end(i):
                exit_price, reason = closes[i], "square_off"
            if exit_price is None and i == n - 1:  # close any open position at end of data
                exit_price, reason = closes[i], "end_of_data"

            if exit_price is not None:
                fill = exit_price * (1 - slip) if pos["dir"] > 0 else exit_price * (1 + slip)
                gross = (fill - pos["entry_fill"]) * pos["qty"] * pos["dir"]
                ch = trade_charges(pos["entry_fill"], fill, pos["qty"], pos["dir"], costs)
                net = gross - ch
                trades.append(
                    {
                        "symbol": symbol,
                        "side": "LONG" if pos["dir"] > 0 else "SHORT",
                        "qty": pos["qty"],
                        "entry_time": pos["entry_time"],
                        "entry_price": round(pos["entry_fill"], 4),
                        "exit_time": times[i],
                        "exit_price": round(fill, 4),
                        "charges": round(ch, 2),
                        "pnl": round(net, 2),
                        "reason": reason,
                    }
                )
                rm.register_close(net)
                pos = None
            elif use_trailing:
                pos["stop"] = trailing_stop(closes[i], atr_arr[i], pos["dir"], pos["stop"], risk_params)

        # 2) Open a new position at this bar's open, using the prior bar's signal.
        block_eod = square_off_eod and session_end(i)
        if pos is None and 0 < i < n - 1 and entered[i - 1] and rm.can_open() and not block_eod:
            d = int(direction[i - 1])
            atr_val = atr_arr[i - 1]
            if d != 0 and atr_val > 0:
                raw = opens[i]
                entry_fill = raw * (1 + slip) if d > 0 else raw * (1 - slip)
                stop, target = stop_and_target(entry_fill, atr_val, d, risk_params)
                qty = position_size(
                    risk_params.capital, risk_params.risk_per_trade_pct, entry_fill, stop
                )
                if qty > 0:
                    pos = {
                        "dir": d, "entry_fill": entry_fill, "stop": stop,
                        "target": target, "qty": qty, "entry_time": times[i],
                    }
                    rm.register_open()

    return pd.DataFrame(trades)


def calibrate_weights(
    df: pd.DataFrame,
    strategy_cfg: dict,
    target_pct: float = 1.5,
    stop_pct: float = 0.75,
    horizon: int = 12,
) -> accuracy.AccuracyWeights:
    """Per-signal vote weight = that signal's standalone forward-outcome win rate.

    Run this on a TRAIN window only, then backtest on the unseen TEST window
    (walk-forward) so weights aren't fit to the data they're evaluated on.
    """
    params = strategy_cfg.get("indicator_params", {}) or {}
    enabled = strategy_cfg.get("enabled_signals", {}) or {}
    df_ind = ind.compute_indicators(df, params)
    pattern_series = {
        name: patterns.detect(name, df) for name in (enabled.get("candlesticks", []) or [])
    }
    votes = confluence.compute_votes(df_ind, enabled, pattern_series)

    weights = {
        col: accuracy.forward_outcome_winrate(df, votes[col], horizon, target_pct, stop_pct)
        for col in votes.columns
    }
    return accuracy.AccuracyWeights(weights)


def _main() -> None:  # pragma: no cover - CLI convenience, needs network
    from rich.console import Console
    from rich.table import Table

    from src.config import get_config
    from src.data.historical import HistoricalData
    from src.backtest.metrics import compute_metrics

    cfg = get_config()
    strat = cfg.strategy.model_dump()
    rp = RiskParams.from_config(cfg.risk.model_dump())
    costs = Costs.from_config(cfg.costs.model_dump())
    base_tf = cfg.instruments.base_timeframe
    symbols = cfg.instruments.symbols

    hist = HistoricalData(cache_dir="data/cache")
    all_trades = []
    for sym in symbols:
        try:
            df = hist.get(sym, base_tf)
        except Exception as exc:  # noqa: BLE001
            log.warning("Skipping %s: %s", sym, exc)
            continue
        trades = run_backtest(df, strat, rp, costs, symbol=sym)
        if not trades.empty:
            all_trades.append(trades)
        log.info("%s: %d trades", sym, 0 if trades.empty else len(trades))

    combined = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    m = compute_metrics(combined, rp.capital)

    table = Table(title="Ghostrader backtest (net of costs)")
    for k in ("trades", "win_rate", "expectancy", "total_pnl", "profit_factor",
              "max_drawdown_pct", "return_pct", "sharpe"):
        table.add_column(k)
    table.add_row(
        str(m["trades"]), f"{m['win_rate']*100:.1f}%", f"{m['expectancy']:.2f}",
        f"{m['total_pnl']:.0f}", f"{m['profit_factor']:.2f}",
        f"{m['max_drawdown_pct']:.1f}%", f"{m['return_pct']:.2f}%", f"{m['sharpe']:.2f}",
    )
    Console().print(table)


if __name__ == "__main__":  # pragma: no cover
    _main()
