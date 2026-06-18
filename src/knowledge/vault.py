"""Obsidian knowledge vault — turn the engine's findings into a linked 'neuron' graph.

Each run writes one markdown note ("neuron") per config, symbol, harness and concept,
with YAML frontmatter and `[[wikilinks]]` so Obsidian's graph view shows how everything
connects. The vault is a **research memory**: the agent and the user can navigate what's
been tried and why it failed without re-running anything — which is the only honest sense
in which a bot "learns" here. It is NOT a trading brain and makes no prediction.

Staleness / auto-update (the user's "update only when too old & not in use"):
- a neuron is rewritten only when it is **missing**, its **underlying data advanced**
  (the cache now holds newer bars than the note summarised), or it is **older than the
  TTL** (default 7 days). Fresh, current neurons are left untouched.
- a neuron whose entity has dropped out of the current results AND is older than
  `archive_days` (default 30) is flagged `archived: true` rather than deleted — history
  is never silently lost.
- a neuron with `pinned: true` (your hand-written notes) is never overwritten.

Honesty rails (shared with the rest of the repo): this vault records that **no simple-TA
config has a validated per-trade edge** and that **100% accuracy is not reachable** — the
graph encodes the finding instead of pretending otherwise.

Run: `python -m src vault`   (add `--deep` to recompute the harnesses first)
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from src.logutil import get_logger

log = get_logger(__name__)

VAULT_DIR = "vault"
DEFAULT_TTL_DAYS = 7
DEFAULT_ARCHIVE_DAYS = 30


# --------------------------------------------------------------------------- #
# Neuron model + markdown rendering                                            #
# --------------------------------------------------------------------------- #
@dataclass
class Neuron:
    slug: str                       # filename stem (also the wikilink target)
    folder: str                     # vault subfolder
    ntype: str                      # config | symbol | harness | concept | run | index
    title: str
    body: str
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)   # wikilink targets (other slugs)
    updated: str = ""               # ISO date the note was generated
    data_through: str | None = None  # latest cache date the note summarises
    extra: dict = field(default_factory=dict)         # verdict, status, source, ...

    @property
    def path(self) -> str:
        return f"{self.folder}/{self.slug}.md"


_UNSAFE = re.compile(r"[\[\]#^|:\\/]+")


def safe_slug(text: str) -> str:
    """Wikilink/filename-safe slug. Used for BOTH the filename and the link target so
    `[[slug]]` always resolves to the right note."""
    s = _UNSAFE.sub(" ", str(text)).replace(",", ", ")
    return re.sub(r"\s+", " ", s).strip()


def _yaml_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    s = str(v)
    return f'"{s}"' if re.search(r"[:#\[\]{}\"']", s) else s


def render(n: Neuron) -> str:
    """Frontmatter + body + an auto-generated Links section (the graph edges)."""
    fm = {"type": n.ntype, "title": n.title, "tags": list(n.tags),
          "updated": n.updated or date.today().isoformat()}
    if n.data_through:
        fm["data_through"] = n.data_through
    fm.update(n.extra)
    lines = ["---"]
    for k, v in fm.items():
        if isinstance(v, list):
            lines.append(f"{k}: [{', '.join(_yaml_scalar(x) for x in v)}]")
        else:
            lines.append(f"{k}: {_yaml_scalar(v)}")
    lines.append("---")
    out = "\n".join(lines) + f"\n\n# {n.title}\n\n{n.body.rstrip()}\n"
    seen, link_lines = set(), []
    for tgt in n.links:
        if tgt and tgt not in seen and tgt != n.slug:
            seen.add(tgt)
            link_lines.append(f"- [[{tgt}]]")
    if link_lines:
        out += "\n## Links\n" + "\n".join(link_lines) + "\n"
    return out


def parse_frontmatter(text: str) -> dict:
    """Minimal YAML-frontmatter reader (only what staleness needs)."""
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    out: dict = {}
    for line in text[3:end].splitlines():
        if not line.strip() or ":" not in line:
            continue
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if v.startswith("[") and v.endswith("]"):
            out[k] = [x.strip().strip('"') for x in v[1:-1].split(",") if x.strip()]
        elif v in ("true", "false"):
            out[k] = v == "true"
        else:
            out[k] = v.strip('"')
    return out


# --------------------------------------------------------------------------- #
# Staleness decision                                                           #
# --------------------------------------------------------------------------- #
def _days_since(iso: str | None, now: date) -> float:
    if not iso:
        return float("inf")
    try:
        return (now - date.fromisoformat(str(iso)[:10])).days
    except ValueError:
        return float("inf")


def should_write(existing: str | None, n: Neuron, now: date,
                 ttl_days: int = DEFAULT_TTL_DAYS) -> tuple[bool, str]:
    """Decide whether to (re)write a neuron — the heart of the auto-update rule."""
    if existing is None:
        return True, "new"
    fm = parse_frontmatter(existing)
    if fm.get("pinned") is True:
        return False, "pinned"                                  # never clobber hand edits
    if n.data_through and fm.get("data_through") and str(n.data_through) > str(fm["data_through"]):
        return True, "new-data"                                 # cache advanced past the note
    if _days_since(fm.get("updated"), now) >= ttl_days:
        return True, "stale"                                    # too old → refresh
    return False, "fresh"


# --------------------------------------------------------------------------- #
# Neuron builders                                                              #
# --------------------------------------------------------------------------- #
CONCEPTS = {
    "Edge": ("The only thing worth trading: a **per-trade directional advantage that "
             "survives out-of-sample AND costs**. Raw accuracy is not edge — see "
             "[[Drift (Equity Risk Premium)]] and [[Conditional Edge]]."),
    "Drift (Equity Risk Premium)": ("Stocks drift up, so 'price is higher later' happens "
            ">50% of the time with zero skill. Long-horizon win-rates near 60% are this "
            "drift, not a signal. Netting it out is what [[Conditional Edge]] does."),
    "Conditional Edge": ("Accuracy on a rule's own signal bars minus the best single-"
            "direction bet on those same bars. ~0 means the hit-rate is just [[Drift "
            "(Equity Risk Premium)]]. The honest metric across every harness here."),
    "Backtest Overfitting": ("Trying hundreds of configs and keeping the best manufactures "
            "a good-looking number from noise. Measured by [[Reality Check]] via PBO, the "
            "Deflated Sharpe and a family-wise-error p-value."),
    "100% Accuracy (Not Reachable)": ("Markets are mostly noise; this repo's own sweeps "
            "show even '60%' is [[Drift (Equity Risk Premium)]]. No harness here targets "
            "100% — anything ≥80% OOS is treated as a lookahead **bug**. The vault records "
            "this finding so it is never re-litigated."),
    "Costs": ("Brokerage, STT, slippage. A positive hit-rate that loses money after a "
              "round trip is not an [[Edge]]. Every harness reports net-of-cost results."),
    "Universe": ("The cached NSE basket the harnesses pool over. Each member is a "
                 "[[#symbol]] neuron with its own data-freshness."),
}

HARNESSES = {
    "Edge Search": ("edgesearch", "Sweeps ~400 causal configs out-of-sample for any 60%+ "
            "config; tags each real-edge vs drift. Finding: only drift-riders clear 60%. "
            "Its winner must still pass [[Reality Check]]."),
    "Reality Check": ("reality", "Audits the sweep for [[Backtest Overfitting]]: PBO "
            "(CSCV), Deflated Sharpe, and a multiple-testing-corrected edge p-value. The "
            "honest answer to 'is the best config real or just the luckiest of N?'."),
    "Horizon Filter": ("horizon", "Holds each signal 1–12 days and reports net P&L; "
            "confirms no [[Edge]] at any horizon."),
    "Evolve": ("evolve", "Evolutionary config search; fitness = min(IS, OOS) [[Conditional "
            "Edge]]. Champion still fails the luck bar — see [[Reality Check]]."),
    "RSI Accuracy": ("rsi", "Directional hit-rate of RSI vs the base up-rate; loses after "
            "[[Costs]] at every horizon."),
}


def concept_neurons(now: date) -> list[Neuron]:
    out = []
    for title, body in CONCEPTS.items():
        links = re.findall(r"\[\[([^\]]+)\]\]", body)
        out.append(Neuron(slug=safe_slug(title), folder="concepts", ntype="concept",
                          title=title, body=body, tags=["ghostrader", "concept"],
                          links=[safe_slug(x) for x in links], updated=now.isoformat()))
    return out


def harness_neurons(now: date) -> list[Neuron]:
    out = []
    for title, (cmd, body) in HARNESSES.items():
        links = [safe_slug(x) for x in re.findall(r"\[\[([^\]]+)\]\]", body)]
        out.append(Neuron(slug=safe_slug(title), folder="harnesses", ntype="harness",
                          title=title, body=f"`python -m src {cmd}`\n\n{body}",
                          tags=["ghostrader", "harness"], links=links,
                          updated=now.isoformat(), extra={"command": cmd}))
    return out


def config_neurons(rows: list[dict], now: date, data_through: str | None) -> list[Neuron]:
    """One neuron per evaluated config (best horizon per config), linked to its family
    concept, the harnesses, and a verdict."""
    best: dict[str, dict] = {}
    for r in rows:
        cfg = r.get("config")
        if not cfg:
            continue
        cur = best.get(cfg)
        if cur is None or r["oos"]["cond_edge_pp"] > cur["oos"]["cond_edge_pp"]:
            best[cfg] = r
    out = []
    for cfg, r in best.items():
        oos, is_ = r["oos"], r["is"]
        verdict = ("real-edge-claim" if r.get("real_edge") else
                   "drift-rider" if r.get("drift_rider") else "no-edge")
        body = (f"Family: **{r.get('rule', r.get('family'))}**  ·  best horizon {r['horizon']}d\n\n"
                f"| slice | accuracy | conditional edge | signals |\n"
                f"|---|---|---|---|\n"
                f"| in-sample | {is_['accuracy']*100:.1f}% | {is_['cond_edge_pp']:+.1f} pp | {is_['signals']} |\n"
                f"| out-of-sample | {oos['accuracy']*100:.1f}% | {oos['cond_edge_pp']:+.1f} pp | {oos['signals']} |\n\n"
                f"Verdict: **{verdict}**. A positive held-out number is only an [[Edge]] if it "
                f"also survives [[Reality Check]] and [[Costs]].")
        out.append(Neuron(slug=safe_slug(cfg), folder="configs", ntype="config",
                          title=cfg, body=body,
                          tags=["ghostrader", "config", r.get("family", "")],
                          links=[safe_slug(r.get("family", "")), "Edge Search", "Reality Check",
                                 "Conditional Edge", "Costs"],
                          updated=now.isoformat(), data_through=data_through,
                          extra={"family": r.get("family", ""), "verdict": verdict,
                                 "oos_cond_edge_pp": round(oos["cond_edge_pp"], 2),
                                 "oos_accuracy": round(oos["accuracy"], 4)}))
    return out


def symbol_neurons(frames: dict, now: date) -> list[Neuron]:
    out = []
    for sym, df in frames.items():
        if df is None or len(df) == 0:
            continue
        first, last = str(df.index[0])[:10], str(df.index[-1])[:10]
        body = (f"Cached daily bars: **{len(df)}** rows, {first} → {last}.\n\n"
                f"Member of the [[Universe]]. The harnesses pool over this name; it carries "
                f"no individual [[Edge]] claim.")
        out.append(Neuron(slug=safe_slug(sym), folder="symbols", ntype="symbol",
                          title=sym, body=body, tags=["ghostrader", "symbol"],
                          links=["Universe"], updated=now.isoformat(), data_through=last,
                          extra={"rows": len(df)}))
    return out


def run_neuron(reality_res: dict, now: date, data_through: str | None) -> Neuron | None:
    """A dated snapshot neuron capturing the latest overfitting verdict (deep build)."""
    if not reality_res or not reality_res.get("available"):
        return None
    m, v = reality_res["meta"], reality_res["verdict"]
    pbo, dsr, mte = reality_res["pbo"], reality_res["dsr"], reality_res["edge"]
    slug = f"{now.isoformat()} Reality Check"
    body = (f"Snapshot of [[Reality Check]] over {m['symbols']} symbols "
            f"({m['matrix_configs']} configs × {m['matrix_obs']} days; {m['full_trials']} trials).\n\n"
            f"- **PBO** {pbo.get('pbo', float('nan'))*100:.1f}%  "
            f"- **Deflated Sharpe** {dsr.get('dsr', float('nan'))*100:.1f}% "
            f"({'sig' if dsr.get('significant') else 'not sig'})  "
            f"- **FWE p** {mte.get('p_fwe', float('nan')):.3f} "
            f"({'survives' if mte.get('survives') else 'noise'})\n\n"
            f"Verdict: **{'edge survives' if v['robust'] else 'no edge'}** — {v['note']}\n\n"
            f"See [[100% Accuracy (Not Reachable)]].")
    return Neuron(slug=safe_slug(slug), folder="runs", ntype="run", title=slug, body=body,
                  tags=["ghostrader", "run", "reality"],
                  links=["Reality Check", "Backtest Overfitting", "100% Accuracy (Not Reachable)"],
                  updated=now.isoformat(), data_through=data_through,
                  extra={"robust": bool(v["robust"])})


def index_neuron(neurons: list[Neuron], now: date) -> Neuron:
    """The vault's map-of-content home note."""
    by_folder: dict[str, list[Neuron]] = {}
    for n in neurons:
        by_folder.setdefault(n.folder, []).append(n)
    body = ("Ghostrader's research memory. **Not a trading bot** — a navigable graph of "
            "honest findings. Start at [[Edge]], [[Backtest Overfitting]] and "
            "[[100% Accuracy (Not Reachable)]].\n\n")
    for folder in ("concepts", "harnesses", "configs", "symbols", "runs"):
        items = sorted(by_folder.get(folder, []), key=lambda x: x.slug)
        if not items:
            continue
        body += f"## {folder.title()} ({len(items)})\n"
        body += "\n".join(f"- [[{n.slug}]]" for n in items[:60]) + "\n\n"
    return Neuron(slug="Home", folder=".", ntype="index", title="Ghostrader Knowledge Vault",
                  body=body, tags=["ghostrader", "moc"], updated=now.isoformat())


# --------------------------------------------------------------------------- #
# Interactive graph (open vault/graph.html in any browser — no Obsidian needed) #
# --------------------------------------------------------------------------- #
_GROUP_COLORS = {"concepts": "#34d399", "harnesses": "#60a5fa", "configs": "#f87171",
                 "symbols": "#fbbf24", "runs": "#a78bfa", "index": "#f472b6",
                 "unresolved": "#6b7280"}


def graph_payload(neurons: list[Neuron]) -> dict:
    """Nodes + edges for the wikilink graph (each neuron = a node, each [[link]] = an edge)."""
    group = {n.slug: ("index" if n.folder == "." else n.folder) for n in neurons}
    title = {n.slug: n.title for n in neurons}
    edges, deg = [], Counter()
    for n in neurons:
        for tgt in n.links:
            if tgt and tgt != n.slug:
                edges.append({"from": n.slug, "to": tgt})
                deg[n.slug] += 1
                deg[tgt] += 1
    ids = set(group) | {e["to"] for e in edges} | {e["from"] for e in edges}
    nodes = []
    for s in sorted(ids):
        g = group.get(s, "unresolved")
        nodes.append({"id": s, "label": title.get(s, s), "group": g,
                      "color": _GROUP_COLORS[g], "value": 1 + deg.get(s, 0)})
    return {"nodes": nodes, "edges": edges,
            "top_hubs": [s for s, _ in deg.most_common(8)]}


_GRAPH_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Ghostrader Neuron Graph</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
 html,body{{height:100%;margin:0;background:#0b0e14;color:#cbd5e1;font-family:system-ui,sans-serif}}
 #net{{height:100vh}}
 .panel{{position:fixed;top:10px;left:10px;background:#0f172abf;padding:10px 14px;border-radius:8px;font-size:13px;line-height:1.7;z-index:9}}
 .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}}
 b{{color:#e2e8f0}}
</style></head><body>
<div class="panel"><b>Ghostrader neuron graph</b> — {n_nodes} neurons, {n_edges} links<br>
 <span class="dot" style="background:#34d399"></span>concepts
 <span class="dot" style="background:#60a5fa"></span>harnesses
 <span class="dot" style="background:#f87171"></span>configs
 <span class="dot" style="background:#fbbf24"></span>symbols
 <span class="dot" style="background:#a78bfa"></span>runs
 <span class="dot" style="background:#f472b6"></span>home
 <span class="dot" style="background:#6b7280"></span>unresolved<br>
 <span style="color:#94a3b8">drag to pan · scroll to zoom · click a neuron to follow its links</span></div>
<div id="net"></div>
<script>
 const nodes = new vis.DataSet({nodes});
 const edges = new vis.DataSet({edges});
 new vis.Network(document.getElementById('net'), {{nodes, edges}}, {{
   nodes:{{shape:'dot', scaling:{{min:6,max:34}}, font:{{color:'#cbd5e1',size:13}}}},
   edges:{{color:{{color:'#334155',highlight:'#94a3b8'}}, width:0.6, smooth:false}},
   physics:{{solver:'forceAtlas2Based', forceAtlas2Based:{{gravitationalConstant:-45,springLength:110}},
            stabilization:{{iterations:220}}}},
   interaction:{{hover:true, tooltipDelay:120}}
 }});
</script></body></html>
"""


def write_graph_html(neurons: list[Neuron], out_dir: str = VAULT_DIR) -> str:
    """Write a self-contained interactive graph to `vault/graph.html`. Returns its path."""
    g = graph_payload(neurons)
    html = _GRAPH_HTML.format(nodes=json.dumps(g["nodes"]), edges=json.dumps(g["edges"]),
                              n_nodes=len(g["nodes"]), n_edges=len(g["edges"]))
    path = Path(out_dir) / "graph.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def write_vault(neurons: list[Neuron], out_dir: str = VAULT_DIR, now: date | None = None,
                ttl_days: int = DEFAULT_TTL_DAYS) -> dict:
    """Write/refresh neurons under `out_dir`, honouring the staleness rule. Returns counts."""
    now = now or date.today()
    root = Path(out_dir)
    stats = {"written": 0, "skipped": 0, "new": 0, "reasons": {}}
    for n in neurons:
        path = root / n.path
        existing = path.read_text(encoding="utf-8") if path.exists() else None
        write, reason = should_write(existing, n, now, ttl_days)
        stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
        if not write:
            stats["skipped"] += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render(n), encoding="utf-8")
        stats["written"] += 1
        if reason == "new":
            stats["new"] += 1
    return stats


def build(frames: dict, rows: list[dict] | None = None, reality_res: dict | None = None,
          out_dir: str = VAULT_DIR, now: date | None = None,
          ttl_days: int = DEFAULT_TTL_DAYS) -> dict:
    """Assemble every neuron from the supplied data and write the vault."""
    now = now or date.today()
    data_through = None
    for df in frames.values():
        if df is not None and len(df):
            last = str(df.index[-1])[:10]
            data_through = max(data_through, last) if data_through else last

    neurons: list[Neuron] = []
    neurons += concept_neurons(now)
    neurons += harness_neurons(now)
    neurons += symbol_neurons(frames, now)
    if rows:
        neurons += config_neurons(rows, now, data_through)
    rn = run_neuron(reality_res, now, data_through) if reality_res else None
    if rn:
        neurons.append(rn)
    neurons.append(index_neuron(neurons, now))

    stats = write_vault(neurons, out_dir=out_dir, now=now, ttl_days=ttl_days)
    g = graph_payload(neurons)
    stats["graph_html"] = write_graph_html(neurons, out_dir=out_dir)   # always refreshed
    stats["neurons"] = len(neurons)
    stats["edges"] = len(g["edges"])
    stats["top_hubs"] = g["top_hubs"]
    stats["data_through"] = data_through
    return stats


def analyze(out_dir: str = VAULT_DIR, deep: bool = False, ttl_days: int = DEFAULT_TTL_DAYS) -> dict:
    """Load cached data + research artifacts and (re)build the vault."""
    import json

    from src.backtest import edge_search
    from src.backtest.momentum import DEFAULT_UNIVERSE
    from src.data.historical import HistoricalData

    hist = HistoricalData(cache_dir="data/cache")
    frames = {}
    for sym in [f"{s}.NS" for s in DEFAULT_UNIVERSE]:
        try:
            frames[sym] = hist.get(sym, "day")
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch failed for %s: %s", sym, exc)

    rows, reality_res = None, None
    if deep:
        from src.backtest import reality as rc
        rows = edge_search.evaluate(frames)          # sweep once …
        reality_res = rc.audit(frames, rows=rows)    # … and reuse it for the overfitting audit
    else:                                            # light: reuse the saved archive
        p = Path(edge_search.ARCHIVE_PATH)
        if p.exists():
            arc = json.loads(p.read_text())
            rows = [{"config": w["config"], "family": w["config"].split("[")[0],
                     "rule": w["rule"], "horizon": w["horizon"],
                     "is": {"accuracy": w["is_accuracy"], "cond_edge_pp": w["is_cond_edge_pp"],
                            "signals": w.get("oos_signals", 0)},
                     "oos": {"accuracy": w["oos_accuracy"], "cond_edge_pp": w["oos_cond_edge_pp"],
                             "signals": w["oos_signals"]},
                     "real_edge": w["label"] == "real_edge",
                     "drift_rider": w["label"] == "drift_rider"} for w in arc.get("winners", [])]

    stats = build(frames, rows=rows, reality_res=reality_res, out_dir=out_dir, ttl_days=ttl_days)
    stats["deep"] = deep
    return stats


def _main() -> None:  # pragma: no cover - CLI, reads data/cache
    import sys

    from rich.console import Console

    deep = "--deep" in sys.argv
    c = Console()
    c.print(f"[bold]Building Obsidian vault[/] → [cyan]{VAULT_DIR}/[/] "
            f"({'deep recompute' if deep else 'light, from artifacts'}) …")
    stats = analyze(deep=deep)
    c.print(f"[green]{stats['written']}[/] neurons written "
            f"([green]{stats['new']}[/] new), [yellow]{stats['skipped']}[/] left fresh; "
            f"{stats['neurons']} neurons / {stats['edges']} links, data through {stats['data_through']}.")
    reasons = ", ".join(f"{k}={v}" for k, v in sorted(stats["reasons"].items()))
    c.print(f"[dim]refresh reasons: {reasons}  ·  most-connected: {', '.join(stats['top_hubs'][:5])}[/]")
    c.print(f"\n[bold]View the neuron graph:[/]")
    c.print(f"  • interactive (no install): open [cyan]{stats['graph_html']}[/] in a browser")
    c.print(f"  • full graph view: open the [bold]{VAULT_DIR}/[/bold] folder as a vault in Obsidian")
    c.print(f"[dim]Neurons refresh only when missing, when the cache advances, or past the "
            f"{DEFAULT_TTL_DAYS}-day TTL; `pinned: true` notes are never overwritten. Research "
            f"memory, not an auto-trader — see the '100% Accuracy (Not Reachable)' neuron.[/]")


if __name__ == "__main__":  # pragma: no cover
    _main()
