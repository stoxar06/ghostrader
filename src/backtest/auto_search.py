"""Self-improving strategy search — evolutionary, honest, and self-aware.

The "keep improving until perfect" loop, built the only honest way: each run
seeds a population from the rule grids + the all-time hall of fame, evaluates
everything out-of-sample (`edge_search.evaluate`), keeps the elites, mutates
them, and repeats — persisting its memory to `data/auto_search.json` so the
next run resumes where this one stopped.

Three rails keep it from lying to itself:
- **Fitness = min(in-sample, out-of-sample) conditional edge** — a config only
  scores if its edge exists in BOTH halves of history. This kills the
  recent-regime artifacts that raw OOS accuracy rewards.
- **Lookahead tripwire** — OOS accuracy >= 80% is recorded as a suspected bug,
  never as a discovery.
- **Luck-aware verdict** — after N configs tried, the best z-score expected
  from pure noise is ~sqrt(2·ln N); the report says whether the champion
  actually clears it.

100% accuracy is NOT a reachable target (markets are mostly noise; this repo's
own sweeps showed even "60%" is drift). The loop therefore stops on stall or
budget — never on a fantasy threshold.

Run: `python -m src evolve` (re-run anytime; it continues the search).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from src.backtest.edge_search import RULES, evaluate
from src.logutil import get_logger

log = get_logger(__name__)

ARCHIVE_PATH = "data/auto_search.json"
SEED_HORIZONS = (3, 5, 10, 20)
SUSPECT_ACC = 0.80          # nothing honest scores this high — treat as lookahead/bug
STALL_PATIENCE = 2          # stop after this many non-improving generations

# family -> {param: (lo, hi)} sanity clamps applied after mutation
_CLAMPS = {
    "rsi_reversion": {"low": (2, 45), "high": (55, 98), "period": (3, 50)},
    "rsi_pro_trend": {"mid": (40, 70)},
    "rsi_ma_vol": {"dip": (5, 49), "mid": (50, 95), "n": (20, 400),
                   "vol_mult": (0.5, 5.0), "period": (3, 50)},
    "tri_ma_vol": {"vol_mult": (0.5, 5.0)},
    "boll_reversion": {"num_std": (0.5, 4.0)},
    "ret_z_fade": {"k": (0.5, 4.0)},
    "gap_fade": {"k": (0.5, 4.0)},
    "ensemble": {"k": (2, 5)},
    "adaptive_rsi": {"q_low": (0.01, 0.45), "q_high": (0.55, 0.99)},
    "range_fade": {"low": (0.01, 0.45), "high": (0.55, 0.99)},
    "overnight": {"n": (2, 60)},
    "turn_of_month": {"before": (1, 5), "after": (1, 5)},
    "low_vol_reversal": {"n": (1, 30), "vol_window": (5, 120), "vol_mult": (0.3, 1.5)},
    "near_52w": {"n": (60, 400), "band": (0.005, 0.15)},
}
# params that must stay strictly increasing
_ORDERED = {
    "ema_cross": ("fast", "slow"), "vol_trend": ("fast", "slow"),
    "tri_ma": ("fast", "mid", "slow"), "tri_ma_vol": ("fast", "mid", "slow"),
    "rsi_reversion": ("low", "high"), "adaptive_rsi": ("q_low", "q_high"),
    "range_fade": ("low", "high"),
}


def _sane(family: str, params: dict) -> dict:
    """Clamp a mutated config back into a valid region."""
    q = dict(params)
    for k, v in q.items():
        if isinstance(v, bool) or isinstance(v, str):
            continue
        lo, hi = _CLAMPS.get(family, {}).get(k, (2, 500) if isinstance(v, int) else (0.01, 500.0))
        q[k] = type(v)(min(max(v, lo), hi))
    keys = _ORDERED.get(family)
    if keys:
        vals = sorted(q[k] for k in keys)
        for i in range(1, len(vals)):                 # break ties, keep strict order
            if vals[i] <= vals[i - 1]:
                vals[i] = vals[i - 1] + (1 if isinstance(vals[i], int) else 0.05)
        q.update(zip(keys, vals))
    return q


def mutate(family: str, params: dict, rng: np.random.Generator) -> dict:
    """Jitter numeric params by ±30%; strings/bools pass through; result is sanitized."""
    q = {}
    for k, v in params.items():
        if isinstance(v, bool) or isinstance(v, str):
            q[k] = v
        elif isinstance(v, int):
            nv = int(round(v * float(rng.uniform(0.7, 1.4))))
            q[k] = nv if nv != v else v + int(rng.choice((-1, 1)))
        elif isinstance(v, float):
            q[k] = round(float(v) * float(rng.uniform(0.7, 1.4)), 4)
        else:
            q[k] = v
    return _sane(family, q)


def _label(fid: str, p: dict) -> str:
    return f"{fid}[{','.join(f'{k}={v}' for k, v in sorted(p.items()))}]"


def _fitness(row: dict) -> float:
    """Edge must exist on BOTH slices to count — min() punishes regime artifacts."""
    return min(row["is"]["cond_edge_pp"], row["oos"]["cond_edge_pp"])


def fresh_state() -> dict:
    return {"configs_tried": 0, "generations": 0, "hall_of_fame": [],
            "suspects": [], "history": [], "last_stop": ""}


def verdict(state: dict) -> dict:
    """Is the champion distinguishable from the best config luck alone would produce?"""
    hof = state["hall_of_fame"]
    if not hof:
        return {"meaningful": False, "note": "no config produced enough signals to score"}
    champ = hof[0]
    se_pp = 100.0 * math.sqrt(0.25 / max(champ["oos_signals"], 1))
    z = champ["oos_cond_edge_pp"] / se_pp if se_pp else 0.0
    z_luck = math.sqrt(2.0 * math.log(max(state["configs_tried"], 2)))
    return {
        "champion": champ["config"], "z": round(z, 2),
        "z_expected_by_luck": round(z_luck, 2),
        "meaningful": z > z_luck + 1.0,
        "note": ("clears the luck bar — re-test on NEW data before believing it"
                 if z > z_luck + 1.0 else
                 "indistinguishable from the best of N random configs — not an edge"),
    }


def evolve(frames: dict, generations: int = 4, pop_size: int = 24, elite: int = 8,
           seed: int | None = None, state: dict | None = None,
           horizons=SEED_HORIZONS, train_frac: float = 0.7,
           min_is: int = 200, min_oos: int = 100, rules=None) -> dict:
    """Run one bounded evolution session; returns the updated (resumable) state."""
    rules = rules if rules is not None else RULES
    rng = np.random.default_rng(seed)
    state = state if state is not None else fresh_state()
    for k, v in fresh_state().items():
        state.setdefault(k, v)

    fns = {fid: fn for fid, _h, fn, _g in rules}
    seeds = [(fid, dict(p)) for fid, _h, _fn, grid in rules for p in grid]

    pop = [(w["family"], dict(w["params"])) for w in state["hall_of_fame"] if w["family"] in fns]
    pop += seeds
    best_fit, stall = float("-inf"), 0
    state["last_stop"] = f"budget ({generations} generations)"

    for _g in range(generations):
        cfgs, seen = [], set()
        for fid, p in pop:
            key = (fid, tuple(sorted(p.items())))
            if key in seen or fid not in fns:
                continue
            seen.add(key)
            cfgs.append((_label(fid, p), fid, fns[fid], p))
        cfgs = cfgs[: pop_size + len(seeds)]          # cap a runaway population
        rows = evaluate(frames, horizons=horizons, train_frac=train_frac,
                        min_is=min_is, min_oos=min_oos, configs=cfgs)
        params_by_label = {lbl: p for lbl, _fid, _fn, p in cfgs}
        state["configs_tried"] += len(cfgs)

        best_rows: dict[str, tuple[float, dict]] = {}
        for r in rows:
            if r["oos"]["accuracy"] >= SUSPECT_ACC:   # tripwire: that's a bug, not alpha
                log.warning("suspect lookahead: %s OOS acc %.1f%%", r["config"],
                            r["oos"]["accuracy"] * 100)
                state["suspects"].append({"config": r["config"], "horizon": r["horizon"],
                                          "oos_accuracy": round(r["oos"]["accuracy"], 4)})
                continue
            f = _fitness(r)
            if f > best_rows.get(r["config"], (float("-inf"), None))[0]:
                best_rows[r["config"]] = (f, r)

        ranked = sorted(best_rows.items(), key=lambda kv: kv[1][0], reverse=True)
        elites = ranked[:elite]
        hof = {w["config"]: w for w in state["hall_of_fame"]}
        for lbl, (f, r) in elites:
            entry = {"config": lbl, "family": r["family"], "params": params_by_label[lbl],
                     "fitness_pp": round(f, 2), "horizon": r["horizon"],
                     "oos_accuracy": round(r["oos"]["accuracy"], 4),
                     "oos_cond_edge_pp": round(r["oos"]["cond_edge_pp"], 2),
                     "is_cond_edge_pp": round(r["is"]["cond_edge_pp"], 2),
                     "oos_signals": r["oos"]["signals"]}
            if lbl not in hof or entry["fitness_pp"] > hof[lbl]["fitness_pp"]:
                hof[lbl] = entry
        state["hall_of_fame"] = sorted(hof.values(), key=lambda w: w["fitness_pp"],
                                       reverse=True)[:12]

        gen_best = ranked[0][1][0] if ranked else float("-inf")
        state["history"].append(round(gen_best, 3) if ranked else None)
        state["generations"] += 1
        if gen_best <= best_fit + 0.05:
            stall += 1
            if stall >= STALL_PATIENCE:
                state["last_stop"] = (f"stalled — no improvement for {stall} generations "
                                      f"(best fitness {best_fit:+.2f}pp)")
                break
        else:
            best_fit, stall = gen_best, 0

        pop = []                                       # elites survive + breed
        for lbl, (_f, r) in elites:
            p = params_by_label[lbl]
            pop.append((r["family"], dict(p)))
            for _ in range(2):
                pop.append((r["family"], mutate(r["family"], p, rng)))
        if seeds:                                      # a little immigration for diversity
            for idx in rng.choice(len(seeds), size=min(4, len(seeds)), replace=False):
                pop.append(seeds[int(idx)])

    return state


def load_state(path: str = ARCHIVE_PATH) -> dict:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            log.warning("could not read %s (%s) — starting fresh", path, exc)
    return fresh_state()


def save_state(state: dict, path: str = ARCHIVE_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


def _main() -> None:  # pragma: no cover - CLI, reads data/cache
    from rich.console import Console
    from rich.table import Table

    from src.backtest.momentum import DEFAULT_UNIVERSE
    from src.data.historical import HistoricalData

    hist = HistoricalData(cache_dir="data/cache")
    frames = {}
    for sym in [f"{s}.NS" for s in DEFAULT_UNIVERSE]:
        try:
            frames[sym] = hist.get(sym, "day")
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch failed for %s: %s", sym, exc)

    state = load_state()
    state = evolve(frames, state=state)
    save_state(state)

    c = Console()
    c.print(f"[bold]Self-improving search[/bold] — session stopped: {state['last_stop']}; "
            f"lifetime: {state['generations']} generations, {state['configs_tried']} configs tried, "
            f"{len(state['suspects'])} lookahead suspects quarantined.")
    t = Table(title="Hall of fame (fitness = min(IS, OOS) conditional edge — must exist in BOTH halves)")
    for col in ("config", "h", "fitness", "IS edge", "OOS edge", "OOS acc", "OOS n"):
        t.add_column(col, overflow="fold")
    for w in state["hall_of_fame"][:10]:
        t.add_row(w["config"], str(w["horizon"]), f"{w['fitness_pp']:+.2f}pp",
                  f"{w['is_cond_edge_pp']:+.2f}pp", f"{w['oos_cond_edge_pp']:+.2f}pp",
                  f"{w['oos_accuracy']*100:.1f}%", str(w["oos_signals"]))
    c.print(t)

    v = verdict(state)
    color = "green" if v.get("meaningful") else "red"
    c.print(f"[bold]Verdict[/bold]: champion z={v.get('z')} vs z≈{v.get('z_expected_by_luck')} "
            f"expected from luck alone after {state['configs_tried']} tries → "
            f"[{color}]{v['note']}[/]")
    c.print("[dim]100% accuracy is not a reachable target — the loop stops on stall/budget and "
            "resumes next run from data/auto_search.json. Anything scoring ≥80% OOS is treated "
            "as a lookahead bug, not a discovery.[/]")


if __name__ == "__main__":  # pragma: no cover
    _main()
