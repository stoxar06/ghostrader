"""Phase 10 — unified CLI parsing and index/equity ticker mapping (offline)."""
from __future__ import annotations

from src.cli import COMMANDS, build_parser, main
from src.data.historical import YFinanceProvider


def test_cli_parses_known_commands():
    parser = build_parser()
    assert parser.parse_args(["sip"]).command == "sip"
    for cmd in ("brief", "serve", "research", "momentum", "backtest", "walkforward", "schedule"):
        assert cmd in COMMANDS


def test_cli_no_command_returns_1():
    assert main([]) == 1


def test_index_and_equity_ticker_mapping():
    yp = YFinanceProvider()
    assert yp._ticker("INFY") == "INFY.NS"          # NSE equity gets .NS
    assert yp._ticker("^NSEI") == "^NSEI"           # index untouched
    assert yp._ticker("RELIANCE.NS") == "RELIANCE.NS"  # already-qualified untouched
