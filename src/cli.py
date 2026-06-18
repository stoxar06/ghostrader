"""Unified command-line interface:  python -m src <command>

Lazily dispatches to each subcommand's entry point so heavy imports only happen
for the command actually run.
"""
from __future__ import annotations

import argparse
import importlib

# command -> (module, function)
COMMANDS = {
    "brief": ("src.macro.brief", "main"),
    "serve": ("src.mcp.server", "main"),
    "schedule": ("src.scheduler", "main"),
    "backtest": ("src.backtest.engine", "_main"),
    "walkforward": ("src.backtest.walkforward", "_main"),
    "research": ("src.backtest.research", "main"),
    "momentum": ("src.backtest.momentum", "_main"),
    "rsi": ("src.backtest.rsi_accuracy", "_main"),
    "indicators": ("src.backtest.indicator_accuracy", "_main"),
    "volume": ("src.backtest.indicator_accuracy", "_main_volume"),
    "confluence": ("src.backtest.indicator_accuracy", "_main_confluence"),
    "horizon": ("src.backtest.horizon", "_main"),
    "edgesearch": ("src.backtest.edge_search", "_main"),
    "reality": ("src.backtest.reality", "_main"),
    "vault": ("src.knowledge.vault", "_main"),
    "evolve": ("src.backtest.auto_search", "_main"),
    "compare": ("src.backtest.compare", "main"),
    "sip": ("src.invest.analyzer", "_main"),
    "screener": ("src.invest.screener", "_main"),
    "learn": ("src.learn.pipeline", "_main"),
    "web": ("src.web.server", "_main"),
    "paper": ("src.runner", "main"),
    "candidate": ("src.runner", "paper_candidate"),
    "random": ("src.runner", "paper_random"),
    "active": ("src.runner", "paper_active"),
}

DESCRIPTIONS = {
    "brief": "One-off market briefing (cues + news + regime).",
    "serve": "Run the MCP analyst server (stdio) for Claude Desktop etc.",
    "schedule": "Pre-market + intraday briefing delivery (Telegram).",
    "backtest": "Single-pass basket backtest (net of costs).",
    "walkforward": "Walk-forward out-of-sample backtest gate.",
    "research": "Multi-strategy sweep vs buy-and-hold.",
    "momentum": "Cross-sectional momentum factor vs buy-and-hold.",
    "rsi": "RSI accuracy: directional hit-rate vs base rate + costed backtest.",
    "indicators": "Directional accuracy of every indicator vs the base rate.",
    "volume": "Measure how much volume confirmation changes indicator accuracy.",
    "confluence": "Accuracy of multi-indicator + regime-conditioned rules vs base rate.",
    "horizon": "1-12 day holding-horizon filter: daily profit/loss + totals, net of costs.",
    "edgesearch": "Out-of-sample hunt for 60%+ accuracy configs; archives real-edge vs drift.",
    "reality": "Overfitting audit of the sweep: PBO (CSCV) + Deflated Sharpe + multiple-testing edge.",
    "vault": "Export findings to an Obsidian 'neuron' vault (auto-refresh stale notes; --deep recomputes).",
    "evolve": "Self-improving config search (evolutionary, honest OOS fitness, resumes each run).",
    "compare": "Baseline vs tuned (cost-aware filters), walk-forward OOS.",
    "sip": "SIP / buy-and-hold analyzer (XIRR, CAGR) incl. Nifty benchmark.",
    "screener": "Gain/loss + honest positioning (trend, drawdown, RSI) by large/mid/small cap.",
    "learn": "Transcribe a strategy video (local Whisper) and HONESTLY test the claimed rules OOS. Use --text to skip transcription.",
    "web": "Launch the local web dashboard (http://127.0.0.1:5000).",
    "paper": "Run a PAPER trading simulation on recent data (no real money).",
    "candidate": "PAPER-trade the unvalidated edgesearch candidate (RSI dip + SMA200 + volume), daily.",
    "random": "PAPER-trade random coin-flip entries across the universe (zero-skill control).",
    "active": "PAPER-trade an ACTIVE quota (>=5 random entries/day, risk caps raised) — more trades = more cost.",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ghostrader",
        description="Ghostrader — market analyst & research toolkit (not an auto-trader; "
                    "backtests showed no edge vs buy-and-hold).",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    for name in COMMANDS:
        sub.add_parser(name, help=DESCRIPTIONS.get(name, ""))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    # tolerate subcommand-specific flags (e.g. `vault --deep`): each subcommand reads
    # its own options from sys.argv, so unknown args pass through rather than erroring.
    args, _ = parser.parse_known_args(argv)
    if not args.command:
        parser.print_help()
        return 1
    module_name, fn_name = COMMANDS[args.command]
    getattr(importlib.import_module(module_name), fn_name)()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
