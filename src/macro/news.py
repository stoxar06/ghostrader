"""Free market-news headlines via RSS (feedparser). Only public headlines are
collected — nothing private is ever sent onward. Dedupe is unit-tested.
"""
from __future__ import annotations

from src.logutil import get_logger

log = get_logger(__name__)

FEEDS = [
    "https://www.moneycontrol.com/rss/marketsnews.xml",
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.business-standard.com/rss/markets-106.rss",
]


def dedupe_headlines(titles, limit: int = 40) -> list[str]:
    seen, out = set(), []
    for t in titles:
        key = " ".join((t or "").lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(t.strip())
        if len(out) >= limit:
            break
    return out


def fetch_headlines(feeds: list[str] | None = None, limit: int = 40) -> list[str]:
    import feedparser  # lazy import

    feeds = feeds or FEEDS
    titles: list[str] = []
    for url in feeds:
        try:
            parsed = feedparser.parse(url)
            titles.extend(e.get("title", "") for e in parsed.entries)
        except Exception as exc:  # noqa: BLE001
            log.warning("feed %s failed: %s", url, exc)
    return dedupe_headlines(titles, limit)
