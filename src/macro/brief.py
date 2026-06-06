"""Daily market briefing CLI: global cues + news -> regime, printed nicely.

Runs with zero LLM keys (deterministic regime); add Groq/Gemini/OpenRouter keys
in .env for an LLM-synthesized briefing under the free-first router.
"""
from __future__ import annotations

from src.logutil import get_logger, setup_logging

log = get_logger(__name__)


def main() -> None:  # pragma: no cover - CLI, needs network
    from rich.console import Console
    from rich.panel import Panel

    from src.config import get_config, get_secrets
    from src.llm import Router, build_providers
    from src.macro import globalcues, news, regime
    from src.storage.db import get_session, init_db

    cfg = get_config()
    sec = get_secrets()
    setup_logging(cfg.logging.level, cfg.logging.file)
    init_db(cfg.storage.db_path)

    providers = build_providers(sec)
    router = Router(providers, cfg.llm.model_dump(), get_session) if providers else None
    if router is None:
        log.info("No LLM keys found — using deterministic regime. Add keys to .env for AI synthesis.")

    cues = globalcues.fetch_global_cues()
    headlines = news.fetch_headlines()
    data = regime.get_or_build(cues, headlines, router, get_session)

    cue_str = "  ".join(f"{k} {v:+.2f}%" for k, v in cues.items()) or "(none)"
    body = (
        f"[bold]Regime:[/bold] {data['regime']}    [bold]Bias:[/bold] {data['bias']}\n"
        f"[bold]Favored sectors:[/bold] {', '.join(data.get('favored_sectors') or ['-'])}\n"
        f"[bold]Avoid sectors:[/bold] {', '.join(data.get('avoid_sectors') or ['-'])}\n"
        f"[bold]Summary:[/bold] {data.get('summary', '')}\n\n"
        f"[dim]source={data.get('source')} | {len(headlines)} headlines | cues: {cue_str}[/dim]"
    )
    Console().print(Panel(body, title="Ghostrader — Daily Market Briefing", expand=False))


if __name__ == "__main__":  # pragma: no cover
    main()
