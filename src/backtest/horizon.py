"""Holding-horizon filter: profit/loss of the strategy's signals at 1..12-day holds.

For every horizon h in 1..12 days, each entry signal is held exactly h bars and
exited at close — no stops, no compounding, one measurement per signal. Per
horizon we report win-rate, profit days vs loss days, and total P&L (scale-free
% of notional, net of round-trip costs), plus the same honesty guards as the
other harnesses: directional accuracy is only an edge if it beats the base
up-rate, and a positive hit-rate that loses money after costs is not an edge.

Run: `python -m src horizon`
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.logutil import get_logger
from src.strategy import alt

log = get_logger(__name__)

HORIZONS = tuple(range(1, 13))


def round_trip_cost_pct(c) -> float:
    """Percentage cost of one round trip (both legs), ignoring fixed per-order fees
    so the figure stays scale-free."""
    pct = (2 * c.brokerage_pct + c.stt_sell_pct + 2 * c.exchange_txn_pct
           + c.stamp_buy_pct + 2 * c.slippage_pct)
    pct += c.gst_pct * (2 * c.brokerage_pct + 2 * c.exchange_txn_pct)
    return float(pct)


def signal_trades(df: pd.DataFrame, signals: pd.DataFrame, horizon: int,
                  cost_pct: float = 0.0) -> pd.DataFrame:
    """One row per signal held exactly `horizon` bars: entry/exit date, direction,
    gross and net (after `cost_pct` round-trip) return. Signals too close to the
    end of the data to complete the hold are dropped, not guessed."""
    close = df["close"]
    fwd = close.shift(-horizon) / close - 1.0
    direction = signals["direction"].reindex(df.index).fillna(0).astype(int)
    mask = (direction != 0) & fwd.notna()
    pos = np.flatnonzero(mask.to_numpy())
    gross = direction.to_numpy()[pos] * fwd.to_numpy()[pos]
    return pd.DataFrame({
        "entry_date": df.index[pos],
        "exit_date": df.index[pos + horizon],
        "direction": direction.to_numpy()[pos],
        "gross_ret": gross,
        "net_ret": gross - cost_pct,
    })


def daily_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    """Day-by-day realized P&L: trades grouped by exit date, equal-weight per trade.
    Columns: pnl_pct (sum of net returns realized that day, in %), trades."""
    if trades.empty:
        return pd.DataFrame(columns=["pnl_pct", "trades"])
    g = trades.groupby("exit_date")["net_ret"]
    return pd.DataFrame({"pnl_pct": g.sum() * 100, "trades": g.count()})


def horizon_table(df: pd.DataFrame, signals: pd.DataFrame,
                  horizons=HORIZONS, cost_pct: float = 0.0) -> pd.DataFrame:
    """Per-horizon summary: signals, win-rate, profit/loss days, total net P&L,
    and accuracy vs the base up-rate (the bar a signal must beat to be an edge)."""
    close = df["close"]
    rows = []
    for h in horizons:
        fwd = close.shift(-h) / close - 1.0
        valid = fwd.notna()
        base_up = float((fwd[valid] > 0).mean()) if valid.any() else float("nan")
        t = signal_trades(df, signals, h, cost_pct)
        daily = daily_pnl(t)
        n = len(t)
        acc = float((t["gross_ret"] > 0).mean()) if n else float("nan")
        rows.append({
            "horizon": h,
            "signals": n,
            "win_rate": acc,
            "profit_days": int((daily["pnl_pct"] > 0).sum()),
            "loss_days": int((daily["pnl_pct"] < 0).sum()),
            "avg_ret_pct": float(t["net_ret"].mean() * 100) if n else 0.0,
            "total_pnl_pct": float(t["net_ret"].sum() * 100),
            "base_up_rate": base_up,
            "edge_vs_base": (acc - max(base_up, 1 - base_up)) if n else float("nan"),
        })
    return pd.DataFrame(rows)


def analyze(symbols=None, timeframe: str = "day", horizons=HORIZONS) -> dict:
    """Pool the horizon sweep across the universe. Returns a dict for CLI/web."""
    from src.backtest.engine import Costs
    from src.backtest.momentum import DEFAULT_UNIVERSE
    from src.config import get_config
    from src.data.historical import HistoricalData

    cfg = get_config()
    params = (cfg.strategy.model_dump().get("indicator_params", {}) or {})
    cost_pct = round_trip_cost_pct(Costs.from_config(cfg.costs.model_dump()))
    symbols = symbols or [f"{s}.NS" for s in DEFAULT_UNIVERSE]

    hist = HistoricalData(cache_dir="data/cache")
    frames, trade_frames = [], {h: [] for h in horizons}
    for sym in symbols:
        try:
            df = hist.get(sym, timeframe)
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch failed for %s: %s", sym, exc)
            continue
        if len(df) < 50:
            continue
        sig = alt.rsi_signals(df, params)
        frames.append(horizon_table(df, sig, horizons, cost_pct))
        for h in horizons:
            trade_frames[h].append(signal_trades(df, sig, h, cost_pct))

    by_h = {}
    if frames:
        allf = pd.concat(frames, ignore_index=True)
        for h, g in allf.groupby("horizon"):
            n = int(g["signals"].sum())
            w = g["signals"].replace(0, np.nan)
            by_h[int(h)] = {
                "signals": n,
                "win_rate": float((g["win_rate"] * w).sum() / n) if n else float("nan"),
                "avg_ret_pct": float((g["avg_ret_pct"] * w).sum() / n) if n else 0.0,
                "total_pnl_pct": float(g["total_pnl_pct"].sum()),
                "base_up_rate": float((g["base_up_rate"] * w).sum() / n) if n else float("nan"),
                "edge_vs_base": float((g["edge_vs_base"] * w).sum() / n) if n else float("nan"),
            }

    # Daily P&L (pooled across symbols) for every horizon; totals must reconcile.
    daily_by_h = {}
    for h in horizons:
        parts = [t for t in trade_frames[h] if not t.empty]
        trades = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        daily = daily_pnl(trades)
        daily_by_h[int(h)] = {
            "days": [{"date": str(d)[:10], "pnl_pct": float(r["pnl_pct"]),
                      "trades": int(r["trades"])} for d, r in daily.iterrows()],
            "profit_days": int((daily["pnl_pct"] > 0).sum()) if not daily.empty else 0,
            "loss_days": int((daily["pnl_pct"] < 0).sum()) if not daily.empty else 0,
            "total_pnl_pct": float(daily["pnl_pct"].sum()) if not daily.empty else 0.0,
        }
        if h in by_h:
            by_h[h]["profit_days"] = daily_by_h[h]["profit_days"]
            by_h[h]["loss_days"] = daily_by_h[h]["loss_days"]
    return {"timeframe": timeframe, "symbols": len(symbols),
            "cost_pct": cost_pct, "horizons": by_h, "daily": daily_by_h}


def _main() -> None:  # pragma: no cover - CLI, needs network
    from rich.console import Console
    from rich.table import Table

    res = analyze()
    c = Console()

    t1 = Table(title=f"Holding-horizon filter, net of costs "
                     f"({res['timeframe']}, {res['symbols']} symbols, "
                     f"round trip {res['cost_pct']*100:.2f}%)")
    for col in ("hold (days)", "signals", "win%", "profit days", "loss days",
                "avg trade%", "total P&L%", "base up-rate", "edge vs base"):
        t1.add_column(col)
    for h, d in sorted(res["horizons"].items()):
        edge = d["edge_vs_base"]
        tot = d["total_pnl_pct"]
        t1.add_row(str(h), str(d["signals"]), f"{d['win_rate']*100:.1f}",
                   str(d.get("profit_days", 0)), str(d.get("loss_days", 0)),
                   f"{d['avg_ret_pct']:+.2f}",
                   f"[{'green' if tot > 0 else 'red'}]{tot:+.1f}[/]",
                   f"{d['base_up_rate']*100:.1f}%",
                   f"[{'green' if edge > 0 else 'red'}]{edge*100:+.1f} pp[/]")
    c.print(t1)

    if res["horizons"]:
        best = max(res["horizons"], key=lambda h: res["horizons"][h]["total_pnl_pct"])
        d = res["daily"][best]
        t2 = Table(title=f"Daily P&L at the best horizon ({best}d hold) — last 15 days")
        for col in ("date", "trades closed", "day P&L%"):
            t2.add_column(col)
        for row in d["days"][-15:]:
            p = row["pnl_pct"]
            t2.add_row(row["date"], str(row["trades"]),
                       f"[{'green' if p > 0 else 'red'}]{p:+.2f}[/]")
        t2.add_row("TOTAL (full sample)",
                   f"{d['profit_days']} profit / {d['loss_days']} loss days",
                   f"{d['total_pnl_pct']:+.1f}")
        c.print(t2)
    c.print("[dim]Edge only exists where accuracy beats the base up-rate AND total "
            "P&L is positive after costs. Long horizons riding market drift are "
            "the equity risk premium, not a signal.[/]")


if __name__ == "__main__":  # pragma: no cover
    _main()
