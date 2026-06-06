"""Macro/analyst package — global cues, news, and LLM-synthesized market regime.

Reframed as an ANALYSIS tool (decision aid), not a trade trigger: the backtests
showed no technical edge, so this layer informs rather than auto-trades.
"""
from . import globalcues, news, regime  # noqa: F401
