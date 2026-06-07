"""Baseline vs tuned strategy — honest walk-forward comparison.

'baseline' strips the cost-aware filters; 'tuned' keeps them (from config). Both
run walk-forward out-of-sample, net of costs, so we can see whether the filters
actually help. Expectation: they cut the loss (fewer, higher-quality trades) but
do NOT manufacture an edge.
"""
from __future__ import annotations

import copy

import pandas as pd

from src.backtest.engine import Costs
from src.backtest.metrics import compute_metrics
from src.backtest.walkforward import walk_forward
from src.logutil import get_logger
from src.risk.manager import RiskParams

log = get_logger(__name__)

_FILTER_KEYS = ("min_atr_pct", "trade_window", "full_confluence")


def _agg(frames) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_variants() -> dict[str, dict]:
    """Return {variant_name: metrics} for baseline (no filters) and tuned (config filters)."""
    from src.config import get_config
    from src.data.historical import HistoricalData

    cfg = get_config()
    tuned = cfg.strategy.model_dump()
    baseline = copy.deepcopy(tuned)
    for k in _FILTER_KEYS:
        baseline.pop(k, None)

    rp = RiskParams.from_config(cfg.risk.model_dump())
    costs = Costs.from_config(cfg.costs.model_dump())
    base_tf = cfg.instruments.base_timeframe
    htf_tf = (cfg.instruments.higher_timeframes or [None])[0]

    hist = HistoricalData(cache_dir="data/cache")
    base_data, htf_data = {}, {}
    for sym in cfg.instruments.symbols:
        try:
            base_data[sym] = hist.get(sym, base_tf)
            if htf_tf:
                htf_data[sym] = hist.get(sym, htf_tf)
        except Exception as exc:  # noqa: BLE001
            log.warning("skip %s: %s", sym, exc)

    out: dict[str, dict] = {}
    for name, scfg in (("baseline (no filters)", baseline), ("tuned (vol + time filters)", tuned)):
        trades = _agg([walk_forward(df, scfg, rp, costs, htf_df=htf_data.get(s), symbol=s)
                       for s, df in base_data.items()])
        out[name] = compute_metrics(trades, rp.capital)
    return out


def main() -> None:  # pragma: no cover - CLI, needs network
    from rich.console import Console
    from rich.table import Table

    rows = run_variants()
    table = Table(title="Strategy improvement — baseline vs tuned (walk-forward OOS, net of costs)")
    table.add_column("variant")
    for c in ("trades", "win%", "PF", "return%", "maxDD%", "sharpe", "verdict"):
        table.add_column(c)
    for name, m in rows.items():
        edge = (m["expectancy"] > 0 and m["profit_factor"] > 1 and m["trades"] >= 20)
        table.add_row(name, str(m["trades"]), f"{m['win_rate']*100:.1f}", f"{m['profit_factor']:.2f}",
                      f"{m['return_pct']:.1f}", f"{m['max_drawdown_pct']:.0f}", f"{m['sharpe']:.2f}",
                      "edge?" if edge else "no edge")
    Console().print(table)


if __name__ == "__main__":  # pragma: no cover
    main()
