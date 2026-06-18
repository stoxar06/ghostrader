"""Extract testable TA rules from a free-text strategy description (a transcript).

Deterministic and offline — no LLM. It scans for the indicator families this repo can
already test (RSI, moving-average crosses, Bollinger, breakouts, momentum, MACD, the
RSI+MA+volume combo) and the numbers near them, and maps each to an `edge_search` config.
Extraction is best-effort: clear phrasings ("RSI below 30", "50-day and 200-day moving
average cross") map cleanly; vague ones may be missed. The value isn't perfect parsing —
it's that whatever IS extracted then gets tested honestly (see `pipeline.py`).
"""
from __future__ import annotations

import re


def _ints_near(text: str, pattern: str, before: int = 16, after: int = 40) -> list[int]:
    """Integers appearing just before/after each match of `pattern`."""
    out: list[int] = []
    for m in re.finditer(pattern, text):
        seg = text[max(0, m.start() - before): m.end() + after]
        out += [int(x) for x in re.findall(r"\d+", seg)]
    return out


def _ma_periods(text: str) -> list[int]:
    """Distinct moving-average lengths mentioned (e.g. '50 day', 'sma 200', '9-ema')."""
    nums = re.findall(r"(\d+)[\s-]*(?:day|period|bar|ema|sma|wma|ma\b)", text)
    nums += re.findall(r"(?:ema|sma|wma|ma)[\s-]*(\d+)", text)
    return sorted({int(n) for n in nums if 2 <= int(n) <= 400})


def extract_rules(text: str) -> list[dict]:
    """Return a list of {family, params, phrase} rules detected in `text`."""
    t = " " + (text or "").lower() + " "
    rules: list[dict] = []

    def add(family, params, phrase):
        rules.append({"family": family, "params": params, "phrase": phrase})

    if "rsi" in t or "relative strength" in t:
        # precise: the number right after below/under is the oversold level, after above/over the overbought
        lows = [int(n) for n in re.findall(r"(?:below|under)\s+(\d{1,2})\b", t)]
        highs = [int(n) for n in re.findall(r"(?:above|over)\s+(\d{1,2})\b", t)]
        low = next((n for n in lows if n < 50), 30)
        high = next((n for n in highs if n > 50), 70)
        ma_ctx = any(k in t for k in ("moving average", "sma", "200", "50 day", "trend"))
        if ma_ctx and "volume" in t:
            add("rsi_ma_vol", {"style": "dip", "n": 200 if "200" in t else 50,
                               "dip": low if low < 50 else 35, "vol_mult": 1.0},
                "RSI + moving-average + volume")
        elif ma_ctx:
            add("rsi_pro_trend", {"mid": 50}, "RSI in the direction of the trend")
        else:
            add("rsi_reversion", {"low": low, "high": high}, f"RSI fade ({low}/{high})")

    if any(k in t for k in ("moving average", "ema", "sma", "crossover", "cross above",
                            "cross over", "golden cross", "death cross")):
        periods = _ma_periods(t)
        if len(periods) >= 3:
            add("tri_ma", {"fast": periods[0], "mid": periods[1], "slow": periods[2]},
                "triple moving-average alignment")
        elif len(periods) == 2:
            add("ema_cross", {"fast": periods[0], "slow": periods[1]}, "moving-average crossover")
        elif "golden cross" in t or "death cross" in t:
            add("ema_cross", {"fast": 50, "slow": 200}, "golden/death cross")

    if "bollinger" in t:
        add("boll_reversion", {"period": 20, "num_std": 2.0}, "Bollinger-band fade")

    if any(k in t for k in ("breakout", "donchian", "new high", "day high",
                            "52 week", "52-week")):
        ns = [n for n in _ints_near(t, r"breakout|donchian|high|week") if 5 <= n <= 400]
        add("donchian", {"n": max(ns) if ns else 20}, "breakout / Donchian channel")

    if "momentum" in t or "rate of change" in t or " roc " in t:
        ns = [n for n in _ints_near(t, r"momentum|roc|day|month|period") if 2 <= n <= 300]
        add("momentum", {"lookback": ns[0] if ns else 20}, "price momentum")

    if "macd" in t:
        add("ensemble", {"k": 3}, "MACD (tested via the trend-ensemble vote)")

    seen, out = set(), []
    for r in rules:
        key = (r["family"], tuple(sorted(r["params"].items())))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out
