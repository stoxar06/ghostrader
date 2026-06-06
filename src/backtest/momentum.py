"""Cross-sectional momentum factor backtest.

Rank a universe by trailing return, hold the top N equal-weight, rebalance every
`rebalance` bars. Weights are shifted one bar before being applied (no look-ahead).
Compared against an equal-weight buy-and-hold benchmark. Momentum is the most
academically-supported anomaly — this is the honest final edge test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# A liquid Nifty large/mid-cap universe for the CLI run.
DEFAULT_UNIVERSE = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "SBIN", "BHARTIARTL",
    "ITC", "LT", "KOTAKBANK", "AXISBANK", "HINDUNILVR", "BAJFINANCE", "MARUTI",
    "SUNPHARMA", "TITAN", "WIPRO", "ONGC", "NTPC", "ASIANPAINT",
]


def _curve_stats(daily_ret: pd.Series, periods_per_year: int = 252) -> dict:
    curve = (1 + daily_ret).cumprod()
    n = len(daily_ret)
    final = float(curve.iloc[-1]) if n else 1.0
    years = n / periods_per_year if n else 0.0
    cagr = (final ** (1 / years) - 1) if years > 0 and final > 0 else float("nan")
    std = float(daily_ret.std())
    sharpe = float(daily_ret.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0
    peak = curve.cummax()
    max_dd = float((curve / peak - 1).min()) if n else 0.0
    return {
        "total_return_pct": (final - 1) * 100,
        "cagr_pct": cagr * 100 if cagr == cagr else 0.0,  # NaN-safe
        "sharpe": sharpe,
        "max_dd_pct": max_dd * 100,
    }


def momentum_backtest(
    prices: dict[str, pd.Series],
    lookback: int = 126,
    hold_top: int = 5,
    rebalance: int = 21,
) -> dict:
    """Compare a momentum-rotation portfolio to equal-weight buy-and-hold."""
    px = pd.DataFrame(prices).dropna()
    if len(px) <= lookback + rebalance:
        return {"error": "not enough overlapping history"}

    returns = px.pct_change().fillna(0.0)
    weights = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    rebal_dates = list(px.index[::rebalance])

    for i, d in enumerate(rebal_dates):
        loc = px.index.get_loc(d)
        if loc < lookback:
            continue
        trailing = px.iloc[loc] / px.iloc[loc - lookback] - 1.0
        top = trailing.nlargest(min(hold_top, len(px.columns))).index
        end = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else px.index[-1]
        weights.loc[d:end, :] = 0.0
        weights.loc[d:end, top] = 1.0 / len(top)

    # Shift weights by one bar so today's allocation can't use today's return.
    port_ret = (weights.shift(1).fillna(0.0) * returns).sum(axis=1)
    bh_ret = returns.mean(axis=1)  # equal-weight buy & hold

    momentum = _curve_stats(port_ret)
    buy_hold = _curve_stats(bh_ret)
    beats = momentum["cagr_pct"] > buy_hold["cagr_pct"] and momentum["sharpe"] >= buy_hold["sharpe"]
    return {
        "momentum": momentum,
        "buy_hold": buy_hold,
        "verdict": "momentum beats buy-and-hold" if beats else "no edge vs buy-and-hold",
        "bars": len(px),
        "names": len(px.columns),
    }


def _main() -> None:  # pragma: no cover - CLI, needs network
    from rich.console import Console
    from rich.table import Table

    from src.data.historical import HistoricalData

    hist = HistoricalData(cache_dir="data/cache")
    prices: dict[str, pd.Series] = {}
    for sym in DEFAULT_UNIVERSE:
        try:
            prices[sym] = hist.get(sym, "day")["close"]
        except Exception:  # noqa: BLE001
            continue

    res = momentum_backtest(prices, lookback=126, hold_top=5, rebalance=21)
    con = Console()
    if "error" in res:
        con.print(f"[red]{res['error']}[/red]")
        return

    table = Table(title=f"Momentum factor vs buy-and-hold ({res['names']} names, {res['bars']} bars)")
    for c in ("strategy", "total_return%", "CAGR%", "sharpe", "maxDD%"):
        table.add_column(c)
    for name in ("momentum", "buy_hold"):
        m = res[name]
        table.add_row(name, f"{m['total_return_pct']:.0f}", f"{m['cagr_pct']:.1f}",
                      f"{m['sharpe']:.2f}", f"{m['max_dd_pct']:.0f}")
    con.print(table)
    con.print(f"[bold]Verdict:[/bold] {res['verdict']}")


if __name__ == "__main__":  # pragma: no cover
    _main()
