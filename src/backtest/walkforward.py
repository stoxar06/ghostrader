"""Walk-forward validation — the honest way to test the strategy.

Weights are calibrated on a TRAIN window and evaluated on the following, unseen
TEST window (expanding/anchored folds). The higher-timeframe filter is applied.
Only the concatenated OUT-OF-SAMPLE trades count toward the verdict, so we never
grade the strategy on data its weights were fit to.
"""
from __future__ import annotations

import pandas as pd

from src.backtest.engine import Costs, calibrate_weights, run_backtest
from src.backtest.metrics import compute_metrics
from src.logutil import get_logger
from src.risk.manager import RiskParams

log = get_logger(__name__)


def split_windows(n: int, n_splits: int = 4, min_train: int = 50, min_test: int = 20):
    """Anchored walk-forward folds as ((train_start, train_end), (test_start, test_end))."""
    bounds = [round(n * i / (n_splits + 1)) for i in range(n_splits + 2)]
    folds = []
    for k in range(1, n_splits + 1):
        train = (0, bounds[k])
        test = (bounds[k], bounds[k + 1])
        if (train[1] - train[0]) >= min_train and (test[1] - test[0]) >= min_test:
            folds.append((train, test))
    return folds


def walk_forward(
    df_base: pd.DataFrame,
    strategy_cfg: dict,
    risk_params: RiskParams,
    costs: Costs,
    htf_df: pd.DataFrame | None = None,
    n_splits: int = 4,
    target_pct: float | None = None,
    stop_pct: float | None = None,
    horizon: int = 12,
    symbol: str = "?",
) -> pd.DataFrame:
    """Return concatenated out-of-sample trades across all folds."""
    tgt = target_pct if target_pct is not None else risk_params.target_pct
    stp = stop_pct if stop_pct is not None else risk_params.target_pct / 2.0

    oos = []
    for (tr, te) in split_windows(len(df_base), n_splits):
        train_df = df_base.iloc[tr[0]:tr[1]]
        test_df = df_base.iloc[te[0]:te[1]]
        weights = calibrate_weights(train_df, strategy_cfg, tgt, stp, horizon)
        trades = run_backtest(
            test_df, strategy_cfg, risk_params, costs, weights=weights,
            symbol=symbol, htf_df=htf_df,
        )
        if not trades.empty:
            oos.append(trades)
    return pd.concat(oos, ignore_index=True) if oos else pd.DataFrame()


def _main() -> None:  # pragma: no cover - CLI, needs network
    from rich.console import Console
    from rich.table import Table

    from src.config import get_config
    from src.data.historical import HistoricalData

    cfg = get_config()
    strat = cfg.strategy.model_dump()
    rp = RiskParams.from_config(cfg.risk.model_dump())
    costs = Costs.from_config(cfg.costs.model_dump())
    base_tf = cfg.instruments.base_timeframe
    htf_tf = (cfg.instruments.higher_timeframes or [None])[0]

    hist = HistoricalData(cache_dir="data/cache")
    oos_all = []
    for sym in cfg.instruments.symbols:
        try:
            base = hist.get(sym, base_tf)
            htf = hist.get(sym, htf_tf) if htf_tf else None
        except Exception as exc:  # noqa: BLE001
            log.warning("Skipping %s: %s", sym, exc)
            continue
        trades = walk_forward(base, strat, rp, costs, htf_df=htf, symbol=sym)
        log.info("%s: %d out-of-sample trades", sym, 0 if trades.empty else len(trades))
        if not trades.empty:
            oos_all.append(trades)

    combined = pd.concat(oos_all, ignore_index=True) if oos_all else pd.DataFrame()
    m = compute_metrics(combined, rp.capital)

    table = Table(title="Walk-forward OUT-OF-SAMPLE (calibrated weights + MTF filter, net of costs)")
    for k in ("trades", "win_rate", "expectancy", "total_pnl", "profit_factor",
              "max_drawdown_pct", "return_pct", "sharpe"):
        table.add_column(k)
    table.add_row(
        str(m["trades"]), f"{m['win_rate']*100:.1f}%", f"{m['expectancy']:.2f}",
        f"{m['total_pnl']:.0f}", f"{m['profit_factor']:.2f}",
        f"{m['max_drawdown_pct']:.1f}%", f"{m['return_pct']:.2f}%", f"{m['sharpe']:.2f}",
    )
    Console().print(table)
    verdict = "PLAUSIBLE EDGE (validate further)" if m["expectancy"] > 0 and m["profit_factor"] > 1 \
        else "NO EDGE — do not go live"
    Console().print(f"[bold]Verdict:[/bold] {verdict}")


if __name__ == "__main__":  # pragma: no cover
    _main()
