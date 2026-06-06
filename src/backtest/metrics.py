"""Backtest performance metrics."""
from __future__ import annotations

import pandas as pd

_EMPTY = {
    "trades": 0, "win_rate": 0.0, "expectancy": 0.0, "total_pnl": 0.0,
    "avg_win": 0.0, "avg_loss": 0.0, "profit_factor": 0.0,
    "max_drawdown": 0.0, "max_drawdown_pct": 0.0, "return_pct": 0.0, "sharpe": 0.0,
}


def compute_metrics(trades: pd.DataFrame, starting_capital: float) -> dict:
    """Win rate, expectancy, profit factor, max drawdown, return %, naive Sharpe."""
    if trades is None or trades.empty:
        return dict(_EMPTY)

    pnl = trades["pnl"].astype(float).reset_index(drop=True)
    n = len(pnl)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]

    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else 0.0

    equity = starting_capital + pnl.cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    max_dd = float(drawdown.min())
    max_dd_pct = float((drawdown / peak).min() * 100) if (peak > 0).all() else 0.0

    std = float(pnl.std(ddof=0))
    sharpe = float(pnl.mean() / std) if n > 1 and std > 0 else 0.0

    total = float(pnl.sum())
    return {
        "trades": n,
        "win_rate": len(wins) / n,
        "expectancy": float(pnl.mean()),
        "total_pnl": total,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "return_pct": total / starting_capital * 100 if starting_capital else 0.0,
        "sharpe": sharpe,
    }


def format_metrics(m: dict) -> str:
    return (
        f"trades={m['trades']}  win_rate={m['win_rate']*100:.1f}%  "
        f"expectancy=₹{m['expectancy']:.2f}  total=₹{m['total_pnl']:.0f}  "
        f"PF={m['profit_factor']:.2f}  maxDD={m['max_drawdown_pct']:.1f}%  "
        f"return={m['return_pct']:.2f}%  sharpe={m['sharpe']:.2f}"
    )
