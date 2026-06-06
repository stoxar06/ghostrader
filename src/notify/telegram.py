"""Optional Telegram delivery. No-ops gracefully when credentials are absent."""
from __future__ import annotations

from src.logutil import get_logger

log = get_logger(__name__)


def send_message(text: str, token: str | None = None, chat_id: str | None = None,
                 timeout: int = 10) -> bool:
    """Send `text` to Telegram. Returns False (no-op) if not configured."""
    if token is None or chat_id is None:
        from src.config import get_secrets

        sec = get_secrets()
        token = token if token is not None else sec.telegram_bot_token
        chat_id = chat_id if chat_id is not None else sec.telegram_chat_id

    if not token or not chat_id:
        log.info("Telegram not configured — skipping delivery.")
        return False

    import requests  # lazy import

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=timeout,
        )
        return bool(resp.ok)
    except Exception as exc:  # noqa: BLE001
        log.warning("Telegram send failed: %s", exc)
        return False
