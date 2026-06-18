"""Honest RSI accuracy check.

Two complementary measurements, because "accuracy" alone is misleading:

1. Directional hit-rate (`directional_accuracy`): when RSI says oversold, does
   price actually rise over the next N bars? Overbought -> fall? We compare the
   hit-rate against the *base rate* (the unconditional up-rate of the same bars).
   RSI only has edge if its hit-rate beats the base rate — otherwise it's just
   riding market drift, not predicting. Cost-free, so it's the kindest possible test.

2. Tradeable edge (`backtest_edge`): the same RSI rule run through the trade engine
   *net of costs* -> win% and expectancy. A high hit-rate that loses money after
   costs is not an edge.

Run: `python -m src rsi`
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators import engine
from src.logutil import get_logger
from src.strategy import alt

log = get_logger(__name__)

HORIZONS = (1, 3, 5, 10)


def directional_accuracy(
    df: pd.DataFrame, horizons=HORIZONS, rsi_period: int = 14, low: int = 30, high: int = 70
) -> pd.DataFrame:
    """Forward-return hit-rate of RSI extremes vs the base rate, per horizon.

    A 'hit' = oversold then price up after `h` bars, or overbought then down.
    `base_up_rate` is the unconditional P(up) over the same horizon — the bar to beat.
    """
    close = df["close"]
    rsi = engine.rsi(close, rsi_period)
    oversold = rsi < low
    overbought = rsi > high

    rows = []
    for h in horizons:
        fwd = close.shift(-h) / close - 1.0  # forward return over h bars
        valid = fwd.notna()
        base_up = float((fwd[valid] > 0).mean())  # market drift baseline

        os_hits = (fwd > 0) & oversold & valid       # predicted up, went up
        ob_hits = (fwd < 0) & overbought & valid      # predicted down, went down
        n_os = int((oversold & valid).sum())
        n_ob = int((overbought & valid).sum())
        n = n_os + n_ob
        hits = int(os_hits.sum() + ob_hits.sum())
        acc = hits / n if n else float("nan")
        rows.append({
            "horizon": h,
            "signals": n,
            "accuracy": acc,
            "base_up_rate": base_up,
            # edge vs a naive "always predict the base-rate direction" guess
            "edge_vs_base": (acc - max(base_up, 1 - base_up)) if n else float("nan"),
        })
    return pd.DataFrame(rows)


def backtest_edge(
    df: pd.DataFrame, symbol: str, params: dict, rp, costs, square_off_eod: bool = True
) -> pd.DataFrame:
    """RSI mean-reversion trades, net of costs, as a trades DataFrame.

    On daily data pass square_off_eod=False, else every position is closed on its
    own entry bar (end-of-day square-off) and no swing trade can form.
    """
    from src.backtest.engine import run_backtest

    sig = alt.rsi_signals(df, params)
    return run_backtest(df, {"indicator_params": params}, rp, costs, symbol=symbol,
                        signals=sig, square_off_eod=square_off_eod)


def analyze(symbols=None, timeframe: str = "day") -> dict:
    """Run both measurements across the universe. Returns a dict for CLI/web."""
    from src.backtest.engine import Costs
    from src.backtest.metrics import compute_metrics
    from src.backtest.momentum import DEFAULT_UNIVERSE
    from src.config import get_config
    from src.data.historical import HistoricalData
    from src.risk.manager import RiskParams

    cfg = get_config()
    params = (cfg.strategy.model_dump().get("indicator_params", {}) or {})
    rp = RiskParams.from_config(cfg.risk.model_dump())
    costs = Costs.from_config(cfg.costs.model_dump())
    symbols = symbols or [f"{s}.NS" for s in DEFAULT_UNIVERSE]

    hist = HistoricalData(cache_dir="data/cache")
    acc_frames, trade_frames = [], []
    for sym in symbols:
        try:
            df = hist.get(sym, timeframe)
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch failed for %s: %s", sym, exc)
            continue
        if len(df) < 50:
            continue
        acc_frames.append(directional_accuracy(df))
        trade_frames.append(backtest_edge(df, sym, params, rp, costs,
                                          square_off_eod=(timeframe != "day")))

    # Pool hit-rates across symbols by horizon (signal-weighted).
    acc_by_h = {}
    if acc_frames:
        allf = pd.concat(acc_frames, ignore_index=True)
        for h, g in allf.groupby("horizon"):
            sig = g["signals"].sum()
            acc_by_h[int(h)] = {
                "signals": int(sig),
                "accuracy": float((g["accuracy"] * g["signals"]).sum() / sig) if sig else float("nan"),
                "base_up_rate": float((g["base_up_rate"] * g["signals"]).sum() / sig) if sig else float("nan"),
                "edge_vs_base": float((g["edge_vs_base"] * g["signals"]).sum() / sig) if sig else float("nan"),
            }

    trades = pd.concat([t for t in trade_frames if t is not None and not t.empty], ignore_index=True) \
        if any(t is not None and not t.empty for t in trade_frames) else pd.DataFrame()
    metrics = compute_metrics(trades, rp.capital)
    # Scale-free per-trade return — robust to position sizing / pooling across symbols,
    # unlike pooled absolute P&L which depends on a single shared capital base.
    if not trades.empty:
        notional = (trades["entry_price"] * trades["qty"]).replace(0, np.nan)
        ret = (trades["pnl"] / notional).dropna()
        metrics["avg_trade_pct"] = float(ret.mean() * 100) if len(ret) else 0.0
        metrics["median_trade_pct"] = float(ret.median() * 100) if len(ret) else 0.0
    else:
        metrics["avg_trade_pct"] = metrics["median_trade_pct"] = 0.0
    return {"timeframe": timeframe, "symbols": len(symbols),
            "directional": acc_by_h, "backtest": metrics}


def _main() -> None:  # pragma: no cover - CLI, needs network
    from rich.console import Console
    from rich.table import Table

    res = analyze()
    c = Console()

    t1 = Table(title=f"RSI directional accuracy ({res['timeframe']}, {res['symbols']} symbols)")
    for col in ("horizon (bars)", "signals", "RSI accuracy", "base up-rate", "edge vs base"):
        t1.add_column(col)
    for h, d in sorted(res["directional"].items()):
        edge = d["edge_vs_base"]
        t1.add_row(str(h), str(d["signals"]), f"{d['accuracy']*100:.1f}%",
                   f"{d['base_up_rate']*100:.1f}%",
                   f"[{'green' if edge > 0 else 'red'}]{edge*100:+.1f} pp[/]")
    c.print(t1)

    m = res["backtest"]
    t2 = Table(title="RSI mean-reversion, net of costs (scale-free per-trade return)")
    for col in ("trades", "win%", "avg trade%", "median trade%", "profit factor", "verdict"):
        t2.add_column(col)
    verdict = ("PLAUSIBLE" if (m["avg_trade_pct"] > 0 and m["profit_factor"] > 1 and m["trades"] >= 20)
               else "no tradeable edge")
    t2.add_row(str(m["trades"]), f"{m['win_rate']*100:.1f}", f"{m['avg_trade_pct']:+.2f}",
               f"{m['median_trade_pct']:+.2f}", f"{m['profit_factor']:.2f}",
               f"[{'green' if verdict == 'PLAUSIBLE' else 'red'}]{verdict}[/]")
    c.print(t2)
    c.print("[dim]Accuracy only matters if it beats the base up-rate; a positive "
            "hit-rate that loses money after costs is not an edge.[/]")


if __name__ == "__main__":  # pragma: no cover
    _main()
