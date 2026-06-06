"""Accuracy-weighted confluence.

Indicators and candlestick patterns each cast a vote (+1/-1/0). Votes are
weighted by measured accuracy, summed to a score in [-1, 1]; a trade fires only
when |score| >= threshold AND (optionally) the higher timeframe agrees.
Candlestick patterns confirm — they never trade alone (they're just more votes).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.indicators import engine, patterns
from src.indicators.accuracy import AccuracyWeights
from src.strategy import mtf


@dataclass
class SignalResult:
    direction: int            # +1 long, -1 short, 0 none
    confidence: float         # |score| in [0, 1]
    score: float              # signed score in [-1, 1]
    entered: bool             # passed threshold + (optional) HTF agreement
    votes: dict = field(default_factory=dict)


def compute_votes(
    df_ind: pd.DataFrame, enabled: dict, pattern_series: dict[str, pd.Series]
) -> pd.DataFrame:
    """Per-bar vote (+1/-1/0) for each enabled signal."""
    votes = pd.DataFrame(index=df_ind.index)

    trend = enabled.get("trend", []) or []
    if "ema_cross" in trend:
        votes["ema_cross"] = np.sign(df_ind["ema_fast"] - df_ind["ema_slow"])
    if "supertrend" in trend:
        votes["supertrend"] = df_ind["supertrend_dir"]

    volume = enabled.get("volume", []) or []
    if "vwap" in volume:
        votes["vwap"] = np.sign(df_ind["close"] - df_ind["vwap"])

    momentum = enabled.get("momentum", []) or []
    if "rsi" in momentum:
        r = df_ind["rsi"]
        votes["rsi"] = np.where((r > 50) & (r < 70), 1, np.where((r > 30) & (r < 50), -1, 0))

    for name in enabled.get("candlesticks", []) or []:
        if name in pattern_series:
            votes[name] = pattern_series[name]

    return votes.fillna(0).astype(int)


def score_votes(votes: pd.DataFrame, weights: AccuracyWeights) -> pd.Series:
    """Weighted, normalized score in [-1, 1]."""
    if votes.empty or votes.shape[1] == 0:
        return pd.Series(0.0, index=votes.index)
    w = np.array([weights.weight(c) for c in votes.columns], dtype=float)
    total = w.sum() or 1.0
    return pd.Series((votes.to_numpy() * w).sum(axis=1) / total, index=votes.index)


def analyze(
    base_df: pd.DataFrame,
    strategy_cfg: dict,
    weights: AccuracyWeights | None = None,
    htf_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Full per-bar evaluation. Returns columns: score, direction, confidence, entered, + votes."""
    params = strategy_cfg.get("indicator_params", {}) or {}
    enabled = strategy_cfg.get("enabled_signals", {}) or {}
    threshold = float(strategy_cfg.get("confidence_threshold", 0.6))
    require_htf = bool(strategy_cfg.get("require_higher_tf_agreement", True))

    df_ind = engine.compute_indicators(base_df, params)
    pattern_series = {
        name: patterns.detect(name, base_df) for name in (enabled.get("candlesticks", []) or [])
    }

    votes = compute_votes(df_ind, enabled, pattern_series)
    if weights is None:
        weights = AccuracyWeights.equal(list(votes.columns))

    score = score_votes(votes, weights)
    direction = np.sign(score).astype(int)
    confidence = score.abs()
    entered = confidence >= threshold

    if require_htf and htf_df is not None:
        # Shift the HTF trend by one higher-TF bar so a base-bar decision uses the
        # last *completed* higher-TF bar (the in-progress bar's close isn't known yet).
        htf_trend = mtf.htf_trend(htf_df, params).shift(1)
        htf_dir = mtf.align_to_base(htf_trend, base_df.index)
        entered = entered & (direction == htf_dir) & (direction != 0)

    out = pd.DataFrame(
        {
            "score": score,
            "direction": direction,
            "confidence": confidence,
            "entered": entered.astype(bool),
        }
    )
    return pd.concat([out, votes.add_prefix("vote_")], axis=1)


def latest_signal(result: pd.DataFrame) -> SignalResult:
    """Convenience: collapse the most recent bar of `analyze()` into a SignalResult (live use)."""
    row = result.iloc[-1]
    votes = {c.removeprefix("vote_"): int(row[c]) for c in result.columns if c.startswith("vote_")}
    return SignalResult(
        direction=int(row["direction"]),
        confidence=float(row["confidence"]),
        score=float(row["score"]),
        entered=bool(row["entered"]),
        votes=votes,
    )
