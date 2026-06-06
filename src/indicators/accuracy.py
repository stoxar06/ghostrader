"""Signal accuracy -> vote weights.

`forward_outcome_winrate` measures, for each non-zero signal bar, whether price
reached +target before -stop within `horizon` bars. Phase 4 uses this on a
training window (walk-forward) to set per-signal weights; the strategy defaults
to equal weights until calibrated.
"""
from __future__ import annotations

import pandas as pd


def forward_outcome_winrate(
    df: pd.DataFrame,
    signals: pd.Series,
    horizon: int = 12,
    target_pct: float = 1.5,
    stop_pct: float = 0.75,
) -> float:
    """Fraction of resolved signals that hit target before stop within `horizon` bars."""
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    sig = signals.reindex(df.index).fillna(0).to_numpy()
    n = len(df)

    wins = losses = 0
    for i in range(n):
        s = sig[i]
        if s == 0:
            continue
        entry = close[i]
        if s > 0:
            tp, sl = entry * (1 + target_pct / 100), entry * (1 - stop_pct / 100)
        else:
            tp, sl = entry * (1 - target_pct / 100), entry * (1 + stop_pct / 100)

        outcome = None
        for j in range(i + 1, min(i + 1 + horizon, n)):
            if s > 0:
                if low[j] <= sl:
                    outcome = "loss"; break
                if high[j] >= tp:
                    outcome = "win"; break
            else:
                if high[j] >= sl:
                    outcome = "loss"; break
                if low[j] <= tp:
                    outcome = "win"; break
        if outcome == "win":
            wins += 1
        elif outcome == "loss":
            losses += 1

    total = wins + losses
    return wins / total if total else 0.0


class AccuracyWeights:
    """Per-signal vote weights. Defaults to 1.0 for any unseen signal."""

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = dict(weights or {})

    def weight(self, name: str) -> float:
        return self.weights.get(name, 1.0)

    @classmethod
    def equal(cls, names) -> "AccuracyWeights":
        return cls({n: 1.0 for n in names})

    def __repr__(self) -> str:
        return f"AccuracyWeights({self.weights})"
