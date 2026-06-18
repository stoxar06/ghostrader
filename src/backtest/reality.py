"""Backtest-overfitting reality check — the audit `edgesearch` can't do on itself.

`edgesearch` sweeps ~400 causal configs and tags the best one `REAL EDGE` whenever
its held-out conditional edge clears a threshold. But picking the best of 400 trials
is *itself* a way to manufacture a number: with that many tries, the luckiest config
shows a few points of "edge" even when none exists. The honest question is not "does
the winner look good on the held-out slice?" but "is the winner distinguishable from
the best config that pure noise would have produced after the same number of tries?"

This module answers that with the institutional anti-overfitting toolkit — the part
retail backtesters skip and quant funds never do:

1. **PBO — Probability of Backtest Overfitting** (Bailey, Borwein, López de Prado &
   Zhu 2017), via **Combinatorially Symmetric Cross-Validation (CSCV)**. Split the
   timeline into S blocks; over every balanced in-/out-of-sample partition, pick the
   config that ranks best in-sample and see where it lands out-of-sample. PBO is the
   fraction of partitions where the in-sample champion falls below the OOS median —
   ~0.5 means "the selection process is fitting noise", ~0 means "the winner is robust".

2. **Deflated Sharpe Ratio** (Bailey & López de Prado 2014). Deflate the best config's
   Sharpe by the number of trials, the spread of Sharpes across trials, the sample
   length, and the returns' skew/kurtosis. DSR is the probability the true Sharpe is
   > 0 *after* admitting how many configs were tried. DSR < 0.95 ⇒ not significant.

3. **Multiple-testing-corrected edge** — takes the literal `edgesearch` headline
   (e.g. "58.6%, +6.7pp") and reports its family-wise-error p-value: given N trials,
   the chance noise alone beats it. This is the direct translation of "is the 60% real?"

No new dependencies: the normal CDF/quantile are implemented inline (erf + Acklam).

Run: `python -m src reality`
"""
from __future__ import annotations

import math
from functools import reduce
from itertools import combinations

import numpy as np
import pandas as pd

from src.backtest.edge_search import HORIZONS, _expand, evaluate
from src.logutil import get_logger

log = get_logger(__name__)

EULER_GAMMA = 0.5772156649015329
DEFAULT_BLOCKS = 14          # S in CSCV; C(14,7)=3432 balanced partitions
DEFAULT_MAX_COMBOS = 4000    # cap enumerated partitions (sampled if more), for speed
MATRIX_HORIZON = 1           # hold for the daily returns matrix (clean, non-overlapping)


# --------------------------------------------------------------------------- #
# Normal CDF / inverse-CDF (no scipy): erf for the CDF, Acklam for the quantile #
# --------------------------------------------------------------------------- #
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation, |err| < 1.2e-9)."""
    if not 0.0 < p < 1.0:
        return float("-inf") if p <= 0.0 else float("inf")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)


def _sharpe(returns: np.ndarray) -> float:
    """Per-observation Sharpe (mean/std, ddof=1). Flat/degenerate series score 0."""
    r = returns[~np.isnan(returns)]
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 1e-12 else 0.0


# --------------------------------------------------------------------------- #
# Per-config daily strategy returns, pooled across symbols -> the T x N matrix  #
# --------------------------------------------------------------------------- #
def strategy_returns_matrix(frames: dict, configs=None, horizon: int = MATRIX_HORIZON,
                            market_neutral: bool = True) -> pd.DataFrame:
    """Build a dates x configs matrix of pooled daily strategy returns (the CSCV input).

    For each config and symbol, the per-bar reward is the (causal) stance at t times the
    forward `horizon`-bar return per bar; pooled across symbols by equal weight (a config
    flat on a day contributes 0, so columns stay dense and comparable). Rows are time
    observations, columns are the strategies tried.

    With `market_neutral` (default), each symbol's forward return is measured **relative to
    the equal-weight universe that day** before applying the stance, so the columns reflect
    skill *above the market drift* rather than the equity risk premium. This matters: a
    long-biased drift-rider looks "robust" under PBO only because the drift itself persists;
    netting the market out makes PBO, the Deflated Sharpe and the edge test all speak to the
    same thing — selection/timing skill — instead of one of them being fooled by drift.
    """
    configs = list(configs) if configs is not None else list(_expand())
    usable = {s: d for s, d in frames.items() if d is not None and len(d) >= 260}
    if not usable:
        return pd.DataFrame()
    fwd_by_sym = {s: (d["close"].shift(-horizon) / d["close"] - 1.0) / horizon
                  for s, d in usable.items()}
    market = pd.DataFrame(fwd_by_sym).mean(axis=1) if market_neutral else None  # equal-wt drift

    per_sym, counts = [], []
    for sym, df in usable.items():
        fwd = fwd_by_sym[sym]
        if market_neutral:
            fwd = fwd - market.reindex(fwd.index)       # active return vs the universe
        cols = {}
        for label, _fid, fn, p in configs:
            try:
                stance = fn(df, p).reindex(df.index).fillna(0).astype(int)
            except Exception as exc:  # noqa: BLE001
                log.warning("%s on %s failed: %s", label, sym, exc)
                continue
            cols[label] = stance * fwd
        if not cols:
            continue
        r = pd.DataFrame(cols, index=df.index)
        per_sym.append(r)
        counts.append(r.notna().astype(int))
    if not per_sym:
        return pd.DataFrame()
    total = reduce(lambda a, b: a.add(b, fill_value=0.0), per_sym)
    cnt = reduce(lambda a, b: a.add(b, fill_value=0), counts).replace(0, np.nan)
    matrix = (total / cnt).dropna(how="all").sort_index()
    return matrix.loc[:, matrix.std(axis=0) > 1e-12]    # drop never-trading columns


# --------------------------------------------------------------------------- #
# CSCV -> Probability of Backtest Overfitting                                   #
# --------------------------------------------------------------------------- #
def pbo_cscv(matrix: pd.DataFrame, n_blocks: int = DEFAULT_BLOCKS,
             max_combos: int = DEFAULT_MAX_COMBOS, seed: int = 0) -> dict:
    """Probability of Backtest Overfitting via Combinatorially Symmetric CV.

    Slice the rows into `n_blocks` contiguous blocks; for every balanced split of the
    blocks into in-sample / out-of-sample halves, find the in-sample-best column and
    record its *relative rank* out-of-sample. PBO = P(logit of that rank <= 0) = the
    chance the in-sample winner is below the OOS median. Also returns how often the
    chosen config actually loses money OOS, and the IS->OOS performance-degradation slope.
    """
    M = matrix.to_numpy()
    T, N = M.shape
    if N < 2 or T < n_blocks * 2:
        return {"available": False, "reason": f"need >=2 configs and >={n_blocks*2} rows "
                                              f"(have {N} configs, {T} rows)"}
    n_blocks -= n_blocks % 2                              # CSCV needs an even block count
    bounds = np.linspace(0, T, n_blocks + 1).astype(int)
    blocks = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_blocks)]
    half = n_blocks // 2

    all_splits = list(combinations(range(n_blocks), half))
    rng = np.random.default_rng(seed)
    if len(all_splits) > max_combos:                     # deterministic sub-sample
        idx = rng.choice(len(all_splits), size=max_combos, replace=False)
        all_splits = [all_splits[i] for i in sorted(idx)]

    logits, is_perf, oos_perf, oos_neg = [], [], [], 0
    for is_blocks in all_splits:
        is_set = set(is_blocks)
        is_rows = np.concatenate([blocks[b] for b in range(n_blocks) if b in is_set])
        oos_rows = np.concatenate([blocks[b] for b in range(n_blocks) if b not in is_set])
        is_sr = np.array([_sharpe(M[is_rows, c]) for c in range(N)])
        oos_sr = np.array([_sharpe(M[oos_rows, c]) for c in range(N)])
        best = int(np.argmax(is_sr))
        # relative OOS rank of the IS-best config in (0,1); 1 == OOS-best
        rank = float((oos_sr <= oos_sr[best]).sum()) / (N + 1)
        rank = min(max(rank, 1.0 / (N + 1)), N / (N + 1))
        logits.append(math.log(rank / (1.0 - rank)))
        is_perf.append(is_sr[best])
        oos_perf.append(oos_sr[best])
        oos_neg += int(oos_sr[best] < 0)

    logits = np.array(logits)
    n = len(logits)
    # IS->OOS degradation: slope < 0 means in-sample skill predicts OOS *under*-performance
    slope = float(np.polyfit(is_perf, oos_perf, 1)[0]) if n >= 2 and np.ptp(is_perf) > 0 else float("nan")
    return {
        "available": True,
        "pbo": float((logits <= 0).mean()),
        "n_splits": n, "n_blocks": n_blocks, "n_configs": N, "n_obs": T,
        "median_logit": float(np.median(logits)),
        "prob_oos_loss": oos_neg / n,
        "degradation_slope": slope,
    }


# --------------------------------------------------------------------------- #
# Deflated Sharpe Ratio                                                         #
# --------------------------------------------------------------------------- #
def _skew_kurt(r: np.ndarray) -> tuple[float, float]:
    r = r[~np.isnan(r)]
    n = r.size
    if n < 3:
        return 0.0, 3.0
    m = r.mean()
    s = r.std(ddof=0)
    if s < 1e-12:
        return 0.0, 3.0
    z = (r - m) / s
    return float((z ** 3).mean()), float((z ** 4).mean())   # non-excess kurtosis


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """Expected maximum of `n_trials` independent Sharpe estimates under the null
    (true SR == 0), given the cross-trial Sharpe variance (Bailey & López de Prado)."""
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    return math.sqrt(sr_variance) * (
        (1.0 - EULER_GAMMA) * norm_ppf(1.0 - 1.0 / n_trials)
        + EULER_GAMMA * norm_ppf(1.0 - 1.0 / (n_trials * math.e)))


def deflated_sharpe(best_returns: np.ndarray, trial_sharpes: np.ndarray,
                    n_trials: int | None = None) -> dict:
    """Deflated Sharpe Ratio of the selected strategy.

    `best_returns`: per-observation returns of the chosen (best-Sharpe) strategy.
    `trial_sharpes`: the Sharpe of every config tried (gives the cross-trial variance).
    DSR = P(true SR > 0) after deflating for the number of trials and the return shape;
    DSR > 0.95 is the usual 'survives' bar.
    """
    r = best_returns[~np.isnan(best_returns)]
    T = r.size
    n_trials = int(n_trials or len(trial_sharpes))
    sr = _sharpe(r)
    sr_var = float(np.var(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else 0.0
    sr0 = expected_max_sharpe(sr_var, n_trials)
    skew, kurt = _skew_kurt(r)
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    if T < 2 or denom <= 0:
        dsr = float("nan")
    else:
        dsr = norm_cdf((sr - sr0) * math.sqrt(T - 1) / math.sqrt(denom))
    return {
        "sharpe": sr, "sharpe0_expected_max": sr0, "dsr": dsr,
        "n_trials": n_trials, "n_obs": T, "skew": skew, "kurtosis": kurt,
        "sr_trial_std": math.sqrt(sr_var), "significant": bool(dsr == dsr and dsr > 0.95),
    }


# --------------------------------------------------------------------------- #
# Multiple-testing correction for the edgesearch headline directional edge      #
# --------------------------------------------------------------------------- #
def multiple_testing_edge(rows: list[dict], n_trials: int | None = None) -> dict:
    """Family-wise correction for the best directional claim `edgesearch` reports.

    Each (config, horizon) row is one trial. For the row with the largest OOS conditional
    edge, the per-trial one-sided p-value comes from z = edge / SE(p̂ on its own signals).
    The honest p-value admits all N trials: p_fwe = 1 - (1 - p)^N (Šidák). Also reports the
    edge the *luckiest of N* configs would show under the null, as a tangible bar to clear.
    """
    scored = [r for r in rows if r.get("oos", {}).get("signals", 0) > 0]
    if not scored:
        return {"available": False, "reason": "no scored configs"}
    n_trials = int(n_trials or len(rows))
    best = max(scored, key=lambda r: r["oos"]["cond_edge_pp"])
    n = best["oos"]["signals"]
    edge_pp = best["oos"]["cond_edge_pp"]
    se_pp = 100.0 * math.sqrt(0.25 / max(n, 1))          # SE of a hit-rate, in pp
    z = edge_pp / se_pp if se_pp else 0.0
    p_single = 1.0 - norm_cdf(z)
    p_fwe = 1.0 - (1.0 - p_single) ** n_trials
    # edge the best of N pure-noise trials is expected to show, same units (pp)
    luck_edge_pp = se_pp * expected_max_sharpe(1.0, n_trials)  # E[max of N std-normals]*SE
    return {
        "available": True,
        "best_config": best["config"], "horizon": best["horizon"],
        "oos_accuracy": best["oos"]["accuracy"], "oos_cond_edge_pp": edge_pp,
        "oos_signals": n, "z": z, "p_single": p_single, "p_fwe": p_fwe,
        "n_trials": n_trials, "luck_edge_pp": luck_edge_pp,
        "survives": bool(p_fwe < 0.05 and edge_pp > luck_edge_pp),
    }


def verdict(pbo: dict, dsr: dict, mte: dict) -> dict:
    """Single honest read across the three lenses."""
    robust = (pbo.get("available") and pbo["pbo"] < 0.5
              and dsr.get("significant") and mte.get("survives"))
    parts = []
    if pbo.get("available"):
        parts.append(f"PBO {pbo['pbo']*100:.0f}% ({'overfit' if pbo['pbo'] >= 0.5 else 'robust'})")
    if dsr.get("dsr") == dsr.get("dsr"):
        parts.append(f"DSR {dsr['dsr']*100:.0f}% ({'sig' if dsr.get('significant') else 'not sig'})")
    if mte.get("available"):
        parts.append(f"FWE p {mte['p_fwe']:.2f} ({'survives' if mte['survives'] else 'noise'})")
    return {
        "robust": bool(robust),
        "summary": " | ".join(parts),
        "note": ("the selected edge survives multiple-testing, deflation AND CSCV — "
                 "re-test on NEW data before believing it"
                 if robust else
                 "the best config is indistinguishable from the luckiest of N noise trials "
                 "— consistent with no edge"),
    }


def audit(frames: dict, rows: list[dict] | None = None, timeframe: str = "day",
          horizon: int = MATRIX_HORIZON, n_blocks: int = DEFAULT_BLOCKS,
          train_frac: float = 0.70) -> dict:
    """Run the full overfitting audit on already-loaded frames: matrix -> PBO + DSR + FWE.

    `rows` (from `edge_search.evaluate`) may be passed in to avoid recomputing the sweep
    when the caller already has it; otherwise it is computed here.
    """
    from datetime import datetime

    matrix = strategy_returns_matrix(frames, horizon=horizon)
    if matrix.empty:
        return {"available": False, "reason": "no usable data"}

    pbo = pbo_cscv(matrix, n_blocks=n_blocks)
    trial_sr = np.array([_sharpe(matrix[c].to_numpy()) for c in matrix.columns])
    best_col = matrix.columns[int(np.argmax(trial_sr))]
    # n_trials = the full edgesearch sweep (configs x horizons), not just the matrix columns
    full_trials = len(list(_expand())) * len(HORIZONS)
    dsr = deflated_sharpe(matrix[best_col].to_numpy(), trial_sr, n_trials=full_trials)

    rows = rows if rows is not None else evaluate(frames, train_frac=train_frac)
    mte = multiple_testing_edge(rows, n_trials=len(rows))

    meta = {"timeframe": timeframe, "symbols": len([f for f in frames.values() if f is not None]),
            "matrix_horizon": horizon, "matrix_obs": int(matrix.shape[0]),
            "matrix_configs": int(matrix.shape[1]), "full_trials": full_trials,
            "best_by_sharpe": best_col,
            "generated": datetime.now().isoformat(timespec="seconds")}
    return {"available": True, "meta": meta, "pbo": pbo, "dsr": dsr,
            "edge": mte, "verdict": verdict(pbo, dsr, mte)}


def analyze(symbols=None, timeframe: str = "day", horizon: int = MATRIX_HORIZON,
            n_blocks: int = DEFAULT_BLOCKS, train_frac: float = 0.70) -> dict:
    """Load the cached universe, then run the overfitting `audit`."""
    from src.backtest.momentum import DEFAULT_UNIVERSE
    from src.data.historical import HistoricalData

    symbols = symbols or [f"{s}.NS" for s in DEFAULT_UNIVERSE]
    hist = HistoricalData(cache_dir="data/cache")
    frames = {}
    for sym in symbols:
        try:
            frames[sym] = hist.get(sym, timeframe)
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch failed for %s: %s", sym, exc)
    return audit(frames, timeframe=timeframe, horizon=horizon, n_blocks=n_blocks,
                 train_frac=train_frac)


def _main() -> None:  # pragma: no cover - CLI, reads data/cache
    from rich.console import Console
    from rich.table import Table

    res = analyze()
    c = Console()
    if not res.get("available"):
        c.print(f"[red]Reality check unavailable:[/] {res.get('reason')}")
        return
    m, pbo, dsr, mte, v = res["meta"], res["pbo"], res["dsr"], res["edge"], res["verdict"]

    c.print(f"[bold]Backtest-overfitting reality check[/] — {m['symbols']} symbols, {m['timeframe']}, "
            f"{m['matrix_configs']} configs x {m['matrix_obs']} days in the CSCV matrix; "
            f"full sweep = {m['full_trials']} trials.")

    if pbo.get("available"):
        t = Table(title="CSCV — Probability of Backtest Overfitting")
        for col in ("PBO", "splits", "blocks", "P(OOS loss)", "IS→OOS slope", "read"):
            t.add_column(col)
        overfit = pbo["pbo"] >= 0.5
        t.add_row(f"[{'red' if overfit else 'green'}]{pbo['pbo']*100:.1f}%[/]",
                  str(pbo["n_splits"]), str(pbo["n_blocks"]),
                  f"{pbo['prob_oos_loss']*100:.0f}%",
                  f"{pbo['degradation_slope']:+.2f}",
                  "[red]selection fits noise[/]" if overfit else "[green]winner is robust[/]")
        c.print(t)

    t2 = Table(title=f"Deflated Sharpe Ratio — best config by Sharpe ({m['best_by_sharpe']})")
    for col in ("Sharpe", "E[max] under null", "trials", "skew", "kurt", "DSR", "verdict"):
        t2.add_column(col)
    sig = dsr.get("significant")
    t2.add_row(f"{dsr['sharpe']:+.3f}", f"{dsr['sharpe0_expected_max']:+.3f}",
               str(dsr["n_trials"]), f"{dsr['skew']:+.2f}", f"{dsr['kurtosis']:.2f}",
               f"[{'green' if sig else 'red'}]{dsr['dsr']*100:.1f}%[/]" if dsr["dsr"] == dsr["dsr"] else "—",
               "[green]significant[/]" if sig else "[red]not significant[/]")
    c.print(t2)

    if mte.get("available"):
        t3 = Table(title="Multiple-testing correction — the edgesearch headline edge")
        for col in ("best config", "h", "OOS acc", "OOS edge", "luck bar", "FWE p-value", "real?"):
            t3.add_column(col, overflow="fold")
        t3.add_row(mte["best_config"], str(mte["horizon"]),
                   f"{mte['oos_accuracy']*100:.1f}%", f"{mte['oos_cond_edge_pp']:+.1f} pp",
                   f"{mte['luck_edge_pp']:+.1f} pp", f"{mte['p_fwe']:.3f}",
                   "[green]survives[/]" if mte["survives"] else "[red]noise[/]")
        c.print(t3)

    color = "green" if v["robust"] else "red"
    c.print(f"\n[bold]Verdict[/]: {v['summary']} → [{color}]{v['note']}[/]")
    c.print("[dim]The matrix is market-neutral (each config's return is netted against the equal-weight "
            "universe), so all three lenses speak to skill above the drift. PBO tests whether the "
            "in-sample winner keeps its RANK out-of-sample; the Deflated Sharpe tests whether its "
            "MAGNITUDE beats the best Sharpe N random trials would yield; the FWE p-value tests the "
            "directional edge. A low PBO with a low DSR = persistent but economically insignificant. "
            "All three must pass to call it real.[/]")


if __name__ == "__main__":  # pragma: no cover
    _main()
