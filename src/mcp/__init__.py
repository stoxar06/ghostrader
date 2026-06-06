"""MCP package — conversational analyst layer over the engine (read-only analysis).

Tool logic lives in `tools.py` (testable without the MCP runtime); `server.py`
wraps it with FastMCP for clients like Claude Desktop.
"""
from . import tools  # noqa: F401
