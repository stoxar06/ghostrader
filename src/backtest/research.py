"""Honest research sweep — test several strategy archetypes through the gate.

`sweep()` returns the metrics dict (reused by the web dashboard); `main()` prints it.
Fixed-rule strategies (mean-reversion, ORB, equal-weight trend) are evaluated
full-period (no fitting = no overfitting); the calibrated trend strategy uses
walk-forward OOS.
"""
from __future__ import annotations

import pandas as pd

from src.backtest.engine import Costs, run_backtest
from src.backtest.metrics import compute_metrics
from src.backtest.walkforward import walk_forward
from src.logutil import get_logger
from src.risk.manager import RiskParams
from src.strategy import alt

log = get_logger(__name__)


def _agg(frames) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def sweep() -> dict[str, dict]:
    """Run all strategy archetypes on the configured basket; return {name: metrics}."""
    from src.config import get_config
    from src.data.historical import HistoricalData

    cfg = get_config()
    strat = cfg.strategy.model_dump()
    params = strat.get("indicator_params", {}) or {}
    rp = RiskParams.from_config(cfg.risk.model_dump())
    costs = Costs.from_config(cfg.costs.model_dump())
    base_tf = cfg.instruments.base_timeframe
    htf_tf = (cfg.instruments.higher_timeframes or [None])[0]
    symbols = cfg.instruments.symbols

    hist = HistoricalData(cache_dir="data/cache")
    base_data, htf_data, daily_data = {}, {}, {}
    for sym in symbols:
        try:
            base_data[sym] = hist.get(sym, base_tf)
            if htf_tf:
                htf_data[sym] = hist.get(sym, htf_tf)
            daily_data[sym] = hist.get(sym, "day")
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch failed for %s: %s", sym, exc)

    mr_cfg = {"indicator_params": params}
    rows: dict[str, dict] = {}
    rows["trend_intraday_wf_oos"] = compute_metrics(
        _agg([walk_forward(df, strat, rp, costs, htf_df=htf_data.get(s), symbol=s)
              for s, df in base_data.items()]), rp.capital)
    rows["mean_reversion_intraday"] = compute_metrics(
        _agg([run_backtest(df, mr_cfg, rp, costs, symbol=s,
                           signals=alt.mean_reversion_signals(df, params))
              for s, df in base_data.items()]), rp.capital)
    rows["orb_intraday"] = compute_metrics(
        _agg([run_backtest(df, mr_cfg, rp, costs, symbol=s,
                           signals=alt.opening_range_breakout_signals(df, params))
              for s, df in base_data.items()]), rp.capital)
    swing_cfg = dict(strat)
    swing_cfg["require_higher_tf_agreement"] = False
    rows["trend_swing_daily"] = compute_metrics(
        _agg([run_backtest(df, swing_cfg, rp, costs, symbol=s, square_off_eod=False)
              for s, df in daily_data.items()]), rp.capital)
    return rows


def main() -> None:  # pragma: no cover - CLI, needs network
    from rich.console import Console
    from rich.table import Table

    rows = sweep()
    table = Table(title="Strategy research sweep (net of costs)")
    table.add_column("strategy")
    for k in ("trades", "win%", "expectancy", "total_pnl", "PF", "maxDD%", "return%", "sharpe", "verdict"):
        table.add_column(k)
    for name, m in rows.items():
        verdict = ("PLAUSIBLE" if (m["expectancy"] > 0 and m["profit_factor"] > 1 and m["trades"] >= 20)
                   else "no edge")
        table.add_row(
            name, str(m["trades"]), f"{m['win_rate']*100:.1f}", f"{m['expectancy']:.1f}",
            f"{m['total_pnl']:.0f}", f"{m['profit_factor']:.2f}", f"{m['max_drawdown_pct']:.0f}",
            f"{m['return_pct']:.1f}", f"{m['sharpe']:.2f}", verdict,
        )
    Console().print(table)


if __name__ == "__main__":  # pragma: no cover
    main()
