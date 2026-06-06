"""Phase 7 — MCP analyst tools (DB-backed ones offline; server builds if mcp installed)."""
from __future__ import annotations

from datetime import date, datetime

import pytest


def test_daily_pnl_and_explain_last_trades(tmp_path):
    from src.storage.db import DailyPnL, Trade, get_session, init_db

    init_db(str(tmp_path / "mcp.db"))
    with get_session() as s:
        s.add(DailyPnL(day=date.today(), realized_pnl=123.0, trades_count=2, halted=False, mode="paper"))
        s.add(Trade(symbol="TCS", side="LONG", qty=5, entry_price=100.0, exit_price=101.5,
                    entry_time=datetime.now(), exit_time=datetime.now(), pnl=7.5,
                    mode="paper", reason="target"))
        s.commit()

    from src.mcp import tools

    pnl = tools.get_daily_pnl(7)
    assert len(pnl) == 1 and pnl[0]["realized_pnl"] == 123.0

    last = tools.explain_last_trades(5)
    assert len(last) == 1
    assert last[0]["symbol"] == "TCS" and last[0]["reason"] == "target"


def test_explain_last_trades_empty(tmp_path):
    from src.storage.db import init_db

    init_db(str(tmp_path / "empty.db"))
    from src.mcp import tools

    assert tools.explain_last_trades(5) == []


def test_server_builds_with_all_tools():
    pytest.importorskip("mcp")  # skip if the MCP SDK isn't installed
    from src.mcp.server import build_server

    server = build_server()
    assert server is not None  # all @mcp.tool() decorators applied without error
