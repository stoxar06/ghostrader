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
    "compare": ("src.backtest.compare", "main"),
    "sip": ("src.invest.analyzer", "_main"),
    "web": ("src.web.server", "_main"),
    "paper": ("src.runner", "main"),
}

DESCRIPTIONS = {
    "brief": "One-off market briefing (cues + news + regime).",
    "serve": "Run the MCP analyst server (stdio) for Claude Desktop etc.",
    "schedule": "Pre-market + intraday briefing delivery (Telegram).",
    "backtest": "Single-pass basket backtest (net of costs).",
    "walkforward": "Walk-forward out-of-sample backtest gate.",
    "research": "Multi-strategy sweep vs buy-and-hold.",
    "momentum": "Cross-sectional momentum factor vs buy-and-hold.",
    "compare": "Baseline vs tuned (cost-aware filters), walk-forward OOS.",
    "sip": "SIP / buy-and-hold analyzer (XIRR, CAGR) incl. Nifty benchmark.",
    "web": "Launch the local web dashboard (http://127.0.0.1:5000).",
    "paper": "Run a PAPER trading simulation on recent data (no real money).",
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
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1
    module_name, fn_name = COMMANDS[args.command]
    getattr(importlib.import_module(module_name), fn_name)()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
