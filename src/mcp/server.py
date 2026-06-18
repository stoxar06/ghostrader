"""MCP server — exposes the analyst engine as read-only tools.

Run for an MCP client (e.g. Claude Desktop) over stdio:
    python -m src.mcp.server

Configure in Claude Desktop's mcpServers with command="python", args=["-m","src.mcp.server"]
and cwd set to this project (and the .venv active).
"""
from __future__ import annotations

from src.logutil import get_logger
from src.mcp import tools

log = get_logger(__name__)


def build_server():
    """Construct the FastMCP server with all analyst tools registered."""
    from mcp.server.fastmcp import FastMCP  # lazy import (optional dependency)

    mcp = FastMCP("ghostrader-analyst")

    @mcp.tool()
    def get_cues() -> dict:
        """Latest 1-day % change for global market cues (Nifty, US indices, crude, gold, USDINR, VIX, US10Y)."""
        return tools.get_cues()

    @mcp.tool()
    def get_regime() -> dict:
        """Current market regime (risk_on/neutral/risk_off) and bias, synthesized from cues + news."""
        return tools.get_regime()

    @mcp.tool()
    def get_briefing() -> dict:
        """Full daily market briefing: regime, all cues, and recent headlines."""
        return tools.get_briefing()

    @mcp.tool()
    def run_backtest(symbol: str, timeframe: str = "") -> dict:
        """Backtest the confluence strategy on a symbol (net of costs). Returns performance metrics.

        Note: backtests showed no edge vs buy-and-hold — this is for analysis, not a trade signal.
        """
        return tools.run_backtest_tool(symbol, timeframe or None)

    @mcp.tool()
    def get_daily_pnl(days: int = 5) -> list:
        """The most recent N recorded daily realized-P&L rows (replay days may be historical)."""
        return tools.get_daily_pnl(days)

    @mcp.tool()
    def explain_last_trades(n: int = 5) -> list:
        """The last N trades with entry, exit, P&L, and the reason for each."""
        return tools.explain_last_trades(n)

    @mcp.tool()
    def sip_return(symbol: str, monthly_amount: float = 10000.0) -> dict:
        """SIP return (invested, final value, XIRR) for a monthly SIP in a stock or index (e.g. ^NSEI)."""
        return tools.sip_tool(symbol, monthly_amount)

    @mcp.tool()
    def buy_and_hold_return(symbol: str) -> dict:
        """Buy-and-hold total return and CAGR for a stock or index over its history."""
        return tools.lumpsum_tool(symbol)

    return mcp


def main() -> None:  # pragma: no cover - long-running stdio server
    from src.config import get_config
    from src.storage.db import init_db

    init_db(get_config().storage.db_path)
    log.info("Starting ghostrader-analyst MCP server (stdio)")
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
