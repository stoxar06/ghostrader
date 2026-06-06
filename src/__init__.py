"""Ghostrader — indicator + candlestick based stock analyst & auto-trading bot for Zerodha Kite.

Layers:
  - Deterministic technical engine (indicators, candlesticks, strategy, risk) — zero LLM, fast.
  - Macro/sentiment engine (FII/DII, news, sector impact) — free-first LLM, slow path.
  - MCP control layer — conversational analyst over the proven core.

Runs backtest -> paper -> (gated) live by swapping only the execution adapter.
"""

__version__ = "0.1.0"
