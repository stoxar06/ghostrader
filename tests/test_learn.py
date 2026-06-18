"""Tests for the learn-from-video pipeline (offline, deterministic, no Whisper/network)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.learn import pipeline as pl
from src.learn import transcribe as tr
from src.learn.strategy_extract import extract_rules


# ------------------------------- rule extraction ----------------------------- #
def test_extract_rsi_reversion_with_thresholds():
    rules = extract_rules("Buy when the RSI drops below 25 and sell when it rises above 75.")
    rsi = [r for r in rules if r["family"] == "rsi_reversion"]
    assert rsi and rsi[0]["params"] == {"low": 25, "high": 75}


def test_extract_moving_average_cross():
    rules = extract_rules("Go long when the 50 day moving average crosses above the 200 day moving average.")
    cross = [r for r in rules if r["family"] == "ema_cross"]
    assert cross and cross[0]["params"] == {"fast": 50, "slow": 200}


def test_extract_rsi_ma_volume_combo():
    txt = "Long when RSI is below 35 while price holds above the 200 day moving average on high volume."
    fams = {r["family"] for r in extract_rules(txt)}
    assert "rsi_ma_vol" in fams


def test_extract_breakout_and_momentum_and_bollinger():
    assert any(r["family"] == "donchian" for r in extract_rules("Trade the breakout above the 20 day high."))
    assert any(r["family"] == "momentum" for r in extract_rules("Ride 60 day price momentum."))
    assert any(r["family"] == "boll_reversion" for r in extract_rules("Fade the lower Bollinger band."))


def test_extract_nothing_from_non_strategy_text():
    assert extract_rules("Remember to like and subscribe to the channel!") == []


# --------------------------------- transcribe -------------------------------- #
def test_transcribe_uses_injected_backend():
    out = tr.transcribe("anything.mp4", backend=lambda p: f"transcript of {p}")
    assert out == "transcript of anything.mp4"


def test_transcribe_missing_backend_raises_actionable_error():
    import pytest
    with pytest.raises(RuntimeError) as e:  # faster-whisper not installed in CI
        tr.faster_whisper_transcribe(__file__)  # real file, but engine missing
    assert "pip install faster-whisper" in str(e.value)


# --------------------------- pipeline (mock transcriber) --------------------- #
def _frames(n=400):
    out = {}
    for i in range(6):
        rng = np.random.default_rng(i)
        close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
        idx = pd.bdate_range("2022-01-01", periods=n)
        out[f"S{i}.NS"] = pd.DataFrame({"open": close, "high": close * 1.01,
                                        "low": close * 0.99, "close": close,
                                        "volume": 1e6}, index=idx)
    return out


def test_pipeline_extracts_and_tests_from_text():
    res = pl.analyze(text="Buy when RSI falls below 30, sell above 70.",
                     frames=_frames(), min_is=50, min_oos=30)
    assert any(r["family"] == "rsi_reversion" for r in res["rules"])
    assert "edge" in res["verdict"]                       # a verdict was produced
    # random synthetic data has no edge -> the claim should not survive
    assert res["verdict"]["edge"] is False


def test_pipeline_handles_no_rule_text():
    res = pl.analyze(text="Thanks for watching, smash that like button.", frames=_frames())
    assert res["rules"] == []
    assert res["verdict"]["edge"] is False
    assert "No testable" in res["verdict"]["note"]


def test_pipeline_transcribes_via_backend_then_tests():
    res = pl.analyze(media="video.mp4", frames=_frames(), min_is=50, min_oos=30,
                     transcriber=lambda p: "Short when RSI is above 80.")
    assert res["transcript"] == "Short when RSI is above 80."
    assert any(r["family"] == "rsi_reversion" for r in res["rules"])
