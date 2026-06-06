"""Plain-text formatting of a briefing dict (for console + Telegram)."""
from __future__ import annotations


def format_briefing(brief: dict) -> str:
    reg = brief.get("regime", {}) or {}
    cues = brief.get("cues", {}) or {}
    cue_str = "  ".join(f"{k} {v:+.2f}%" for k, v in cues.items()) or "(none)"
    lines = [
        "Ghostrader — Market Briefing",
        f"Regime: {reg.get('regime', '?')} | Bias: {reg.get('bias', '?')}",
        f"Favored: {', '.join(reg.get('favored_sectors') or ['-'])}",
        f"Avoid: {', '.join(reg.get('avoid_sectors') or ['-'])}",
        f"Summary: {reg.get('summary', '')}",
        f"Source: {reg.get('source', '?')} | {brief.get('headline_count', 0)} headlines",
        f"Cues: {cue_str}",
    ]
    return "\n".join(lines)
