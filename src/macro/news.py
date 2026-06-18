"""Free market-news headlines via RSS (feedparser). Only public headlines are
collected — nothing private is ever sent onward. Dedupe is unit-tested.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote_plus

from src.logutil import get_logger

log = get_logger(__name__)

FEEDS = [
    "https://www.moneycontrol.com/rss/marketsnews.xml",
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.business-standard.com/rss/markets-106.rss",
]

# NSE ticker -> readable company name, for company-specific news search.
COMPANY_NAMES = {
    "RELIANCE": "Reliance Industries", "HDFCBANK": "HDFC Bank",
    "ICICIBANK": "ICICI Bank", "INFY": "Infosys",
    "TCS": "Tata Consultancy Services", "SBIN": "State Bank of India",
    "BHARTIARTL": "Bharti Airtel", "ITC": "ITC Ltd", "LT": "Larsen & Toubro",
    "KOTAKBANK": "Kotak Mahindra Bank", "AXISBANK": "Axis Bank",
    "HINDUNILVR": "Hindustan Unilever", "BAJFINANCE": "Bajaj Finance",
    "MARUTI": "Maruti Suzuki", "SUNPHARMA": "Sun Pharma", "TITAN": "Titan Company",
    "WIPRO": "Wipro", "ONGC": "ONGC", "NTPC": "NTPC", "ASIANPAINT": "Asian Paints",
}

_UA = "Mozilla/5.0 (compatible; Ghostrader/1.0; +https://localhost)"
_CACHE: dict[str, tuple[float, object]] = {}
_TTL = 300.0  # seconds — RSS is slow; cache per process for 5 min.


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


# --- Structured news (title + link + source + time) for the news page ------

def _split_source(title: str) -> tuple[str, str]:
    """Google News titles look like 'Headline - Publisher'. Split the publisher out."""
    if " - " in title:
        head, _, src = title.rpartition(" - ")
        if head and len(src) < 60:
            return head.strip(), src.strip()
    return title.strip(), ""


def _parse_feed(url: str):
    """Fetch a feed with a hard timeout + UA, then parse. Returns feedparser entries."""
    import feedparser  # lazy import
    import requests

    try:
        resp = requests.get(url, timeout=6, headers={"User-Agent": _UA})
        resp.raise_for_status()
        return feedparser.parse(resp.content).entries
    except Exception as exc:  # noqa: BLE001
        log.warning("feed %s failed: %s", url, exc)
        return []


def _entry_to_item(e, *, default_source: str = "") -> dict:
    import html

    title, src = _split_source(html.unescape(e.get("title", "")))
    feed_src = ""
    if isinstance(e.get("source"), dict):
        feed_src = e["source"].get("title", "")
    return {
        "title": title,
        "link": e.get("link", ""),
        "source": src or feed_src or default_source,
        "published": e.get("published", "") or e.get("updated", ""),
        "published_ts": _to_ts(e),
    }


def _to_ts(e) -> float:
    import calendar

    for key in ("published_parsed", "updated_parsed"):
        t = e.get(key)
        if t:
            try:
                return calendar.timegm(t)
            except Exception:  # noqa: BLE001
                pass
    return 0.0


def _cached(key: str, builder):
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < _TTL:
        return hit[1]
    val = builder()
    _CACHE[key] = (now, val)
    return val


def fetch_news_items(feeds: list[str] | None = None, limit: int = 40) -> list[dict]:
    """General market news as structured items, newest first."""
    feeds = feeds or FEEDS

    def build():
        items: list[dict] = []
        for url in feeds:
            domain = url.split("/")[2].replace("www.", "")
            items.extend(_entry_to_item(e, default_source=domain) for e in _parse_feed(url))
        return _dedupe_items(items, limit)

    return _cached(f"market:{limit}:{tuple(feeds)}", build)


def _gnews_url(query: str) -> str:
    q = quote_plus(f"{query} stock when:7d")
    return f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"


def fetch_company_news(
    symbols: list[str] | None = None, per_company: int = 4, limit: int = 60
) -> list[dict]:
    """Per-company breaking news for the universe, fetched concurrently, newest first.

    Each item carries `symbol` and `company` so the UI can group/badge it.
    """
    symbols = symbols or list(COMPANY_NAMES.keys())

    def build():
        def one(sym: str) -> list[dict]:
            name = COMPANY_NAMES.get(sym, sym)
            entries = _parse_feed(_gnews_url(name))
            out = []
            for e in entries[:per_company]:
                item = _entry_to_item(e)
                item["symbol"], item["company"] = sym, name
                out.append(item)
            return out

        items: list[dict] = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            for res in ex.map(one, symbols):
                items.extend(res)
        items.sort(key=lambda i: i["published_ts"], reverse=True)
        return _dedupe_items(items, limit)

    return _cached(f"company:{tuple(symbols)}:{per_company}:{limit}", build)


def _dedupe_items(items: list[dict], limit: int) -> list[dict]:
    seen, out = set(), []
    for it in items:
        key = " ".join((it.get("title") or "").lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(it)
        if len(out) >= limit:
            break
    return out
