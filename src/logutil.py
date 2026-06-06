"""Logging setup: rich console + rotating file handler. Idempotent."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

_configured = False


def setup_logging(level: str = "INFO", logfile: str | None = None) -> None:
    """Configure root logging once. Safe to call multiple times."""
    global _configured
    if _configured:
        return

    handlers: list[logging.Handler] = [
        RichHandler(rich_tracebacks=True, show_path=False, markup=False)
    ]
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(logfile, maxBytes=5_000_000, backupCount=5)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
