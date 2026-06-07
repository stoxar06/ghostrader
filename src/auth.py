"""Zerodha Kite Connect authentication.

Daily flow: open `login_url()` → log in → Kite redirects with a `request_token` →
`generate_session(request_token)` exchanges it for an `access_token` (saved to disk).
Needs a paid Kite Connect subscription (api_key/secret in .env).
"""
from __future__ import annotations

from pathlib import Path

from src.logutil import get_logger

log = get_logger(__name__)


class KiteAuth:
    LOGIN_BASE = "https://kite.zerodha.com/connect/login"

    def __init__(self, api_key: str, api_secret: str, token_path: str = "data/kite_token.txt"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.token_path = Path(token_path)

    def login_url(self) -> str:
        """URL to open in a browser to start the daily login."""
        return f"{self.LOGIN_BASE}?api_key={self.api_key}&v=3"

    def generate_session(self, request_token: str) -> str:  # pragma: no cover - needs API
        from kiteconnect import KiteConnect

        kite = KiteConnect(api_key=self.api_key)
        data = kite.generate_session(request_token, api_secret=self.api_secret)
        token = data["access_token"]
        self.save_token(token)
        return token

    def client(self, access_token: str | None = None):  # pragma: no cover - needs API
        from kiteconnect import KiteConnect

        kite = KiteConnect(api_key=self.api_key)
        kite.set_access_token(access_token or self.load_token())
        return kite

    def save_token(self, token: str) -> None:
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(token, encoding="utf-8")

    def load_token(self) -> str | None:
        return self.token_path.read_text(encoding="utf-8").strip() if self.token_path.exists() else None
