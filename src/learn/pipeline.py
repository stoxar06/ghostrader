"""learn — transcribe a strategy video (or take a transcript), then HONESTLY test it.

Flow: media → local Whisper transcript (or `--text`) → extract the TA rules described →
run them through the same out-of-sample sweep (`edge_search.evaluate`) and the
multiple-testing edge check (`reality.multiple_testing_edge`) as every other harness →
a plain verdict. The point is not to "learn to win" from a video — it's to fact-check it.

Run: `python -m src learn <video-or-audio>`        (needs faster-whisper + ffmpeg)
     `python -m src learn --text "buy when RSI < 30 above the 200-day MA on volume"`
"""
from __future__ import annotations

from src.backtest.edge_search import RULES
from src.learn.strategy_extract import extract_rules
from src.logutil import get_logger

log = get_logger(__name__)

_FN = {fid: fn for fid, _h, fn, _g in RULES}


def _configs(rules: list[dict]) -> list[tuple]:
    """Turn extracted {family, params} rules into edge_search (label, fid, fn, params) configs."""
    out = []
    for r in rules:
        fn = _FN.get(r["family"])
        if fn is None:
            continue
        label = f"{r['family']}[{','.join(f'{k}={v}' for k, v in r['params'].items())}]"
        out.append((label, r["family"], fn, r["params"]))
    return out


def _load_frames(timeframe: str = "day") -> dict:
    from src.backtest.momentum import DEFAULT_UNIVERSE
    from src.data.historical import HistoricalData

    hist = HistoricalData(cache_dir="data/cache")
    frames = {}
    for sym in [f"{s}.NS" for s in DEFAULT_UNIVERSE]:
        try:
            frames[sym] = hist.get(sym, timeframe)
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch failed for %s: %s", sym, exc)
    return frames


def analyze(media: str | None = None, text: str | None = None, transcriber=None,
            frames: dict | None = None, model_size: str = "base",
            min_is: int = 150, min_oos: int = 75) -> dict:
    """Transcribe (unless `text` is given), extract rules, and test them out-of-sample."""
    if text is None:
        if not media:
            raise ValueError("provide a media path or text=")
        from src.learn.transcribe import transcribe
        text = transcribe(media, model_size=model_size, backend=transcriber)

    rules = extract_rules(text)
    configs = _configs(rules)
    res = {"transcript": text, "n_chars": len(text or ""), "rules": rules,
           "tested": [], "edge_test": {"available": False}, "verdict": {}}
    if not configs:
        res["verdict"] = {"edge": False,
                          "note": "No testable TA rule was detected in the transcript."}
        return res

    if frames is None:
        frames = _load_frames()
    from src.backtest import reality as rc
    from src.backtest.edge_search import evaluate

    rows = evaluate(frames, configs=configs, min_is=min_is, min_oos=min_oos)
    best: dict = {}
    for r in rows:
        c = r["config"]
        if c not in best or r["oos"]["cond_edge_pp"] > best[c]["oos"]["cond_edge_pp"]:
            best[c] = r
    tested = sorted(({"config": c, "horizon": r["horizon"],
                      "oos_accuracy": r["oos"]["accuracy"],
                      "oos_cond_edge_pp": r["oos"]["cond_edge_pp"],
                      "oos_signals": r["oos"]["signals"]} for c, r in best.items()),
                    key=lambda x: x["oos_cond_edge_pp"], reverse=True)
    mte = rc.multiple_testing_edge(rows, n_trials=max(len(rows), 1)) if rows else {"available": False}
    survives = bool(mte.get("survives"))
    res["tested"] = tested
    res["edge_test"] = mte
    if not rows:
        res["verdict"] = {"edge": False,
                          "note": "Rules were detected but produced too few signals to test on the cached universe."}
    else:
        res["verdict"] = {"edge": survives, "note": (
            "The best rule clears the bar in-sample — re-test on NEW data before believing it."
            if survives else
            "The claimed edge does NOT survive out-of-sample + multiple-testing — consistent "
            "with this repo's finding that simple TA has no per-trade edge.")}
    return res


def _main() -> None:  # pragma: no cover - CLI, reads data/cache (+ optional media)
    import sys

    from rich.console import Console
    from rich.table import Table

    argv = sys.argv[1:]
    if argv and argv[0] == "learn":
        argv = argv[1:]
    text = None
    if "--text" in argv:
        i = argv.index("--text")
        text = argv[i + 1] if i + 1 < len(argv) else ""
    model = "base"
    if "--model" in argv:
        model = argv[argv.index("--model") + 1]
    media = next((a for a in argv if not a.startswith("-") and a != text), None)

    c = Console()
    if not text and not media:
        c.print("[yellow]Usage:[/] python -m src learn <video|audio>   "
                "or   python -m src learn --text \"...strategy description...\"")
        return
    try:
        res = analyze(media=media, text=text, model_size=model)
    except (RuntimeError, FileNotFoundError) as exc:
        c.print(f"[red]{exc}[/]")
        return

    snippet = (res["transcript"] or "")[:300]
    c.print(f"[bold]Transcript[/] ({res['n_chars']} chars): [dim]{snippet}"
            f"{'…' if res['n_chars'] > 300 else ''}[/]")

    if res["rules"]:
        tr = Table(title="Rules detected in the video → mapped to testable configs")
        for col in ("described as", "family", "params"):
            tr.add_column(col, overflow="fold")
        for r in res["rules"]:
            tr.add_row(r["phrase"], r["family"],
                       ", ".join(f"{k}={v}" for k, v in r["params"].items()))
        c.print(tr)
    else:
        c.print("[yellow]No testable TA rule detected in the transcript.[/]")

    if res["tested"]:
        tt = Table(title="Honest out-of-sample test of what the video claims")
        for col in ("config", "h", "OOS acc", "OOS cond-edge", "OOS signals"):
            tt.add_column(col, overflow="fold")
        for r in res["tested"]:
            ce = r["oos_cond_edge_pp"]
            tt.add_row(r["config"], str(r["horizon"]), f"{r['oos_accuracy']*100:.1f}%",
                       f"[{'green' if ce > 0 else 'red'}]{ce:+.1f} pp[/]", str(r["oos_signals"]))
        c.print(tt)
        mte = res["edge_test"]
        if mte.get("available"):
            c.print(f"[dim]Best directional edge {mte['oos_cond_edge_pp']:+.1f}pp on "
                    f"{mte['oos_signals']} signals · luck bar {mte['luck_edge_pp']:+.1f}pp · "
                    f"family-wise-error p={mte['p_fwe']:.2f}[/]")

    v = res["verdict"]
    color = "green" if v.get("edge") else "red"
    c.print(f"\n[bold]Verdict[/]: [{color}]{v['note']}[/]")
    c.print("[dim]Transcribing a strategy gives you its words, not an edge. This ran the claim "
            "through the same OOS + multiple-testing gate as every other rule here — run "
            "`reality` for the full overfitting audit.[/]")


if __name__ == "__main__":  # pragma: no cover
    _main()
