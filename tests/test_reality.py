"""Tests for the backtest-overfitting reality check (offline, synthetic, deterministic).

The maths is verified directly (normal CDF/quantile, expected-max Sharpe, DSR, FWE edge)
and the CSCV PBO is checked on two extremes: a planted real edge must score PBO ~ 0, and
pure noise must score PBO ~ 0.5 (selection is fitting noise).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import reality as rc


# --------------------------- normal CDF / quantile --------------------------- #
def test_norm_cdf_known_values():
    assert rc.norm_cdf(0.0) == pytest.approx(0.5)
    assert rc.norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
    assert rc.norm_cdf(-1.96) == pytest.approx(0.025, abs=1e-3)


def test_norm_ppf_inverts_cdf():
    for x in (-2.5, -1.0, -0.3, 0.4, 1.0, 2.5):
        assert rc.norm_ppf(rc.norm_cdf(x)) == pytest.approx(x, abs=1e-6)


def test_norm_ppf_out_of_domain():
    assert rc.norm_ppf(0.0) == float("-inf")
    assert rc.norm_ppf(1.0) == float("inf")


# ------------------------------ Sharpe helpers ------------------------------- #
def test_sharpe_flat_series_is_zero():
    assert rc._sharpe(np.zeros(50)) == 0.0
    assert rc._sharpe(np.array([0.01])) == 0.0  # too few points


def test_expected_max_sharpe_grows_with_trials():
    e10 = rc.expected_max_sharpe(1.0, 10)
    e100 = rc.expected_max_sharpe(1.0, 100)
    e1000 = rc.expected_max_sharpe(1.0, 1000)
    assert 0 < e10 < e100 < e1000
    assert rc.expected_max_sharpe(0.0, 100) == 0.0  # no spread -> no expected max
    assert rc.expected_max_sharpe(1.0, 1) == 0.0


# ------------------------------- CSCV -> PBO --------------------------------- #
def test_pbo_near_half_on_pure_noise():
    rng = np.random.default_rng(7)
    M = pd.DataFrame(rng.standard_normal((240, 20)))
    out = rc.pbo_cscv(M, n_blocks=14, seed=0)
    assert out["available"]
    # selecting the best of exchangeable noise columns is fitting noise -> PBO ~ 0.5
    assert 0.30 < out["pbo"] < 0.70


def test_pbo_near_zero_with_a_planted_edge():
    rng = np.random.default_rng(11)
    noise = rng.standard_normal((300, 11)) * 0.01           # eleven zero-mean configs
    edge = rng.standard_normal((300, 1)) * 0.001 + 0.01     # one genuinely positive config
    M = pd.DataFrame(np.hstack([edge, noise]))
    out = rc.pbo_cscv(M, n_blocks=14, seed=0)
    assert out["available"]
    assert out["pbo"] < 0.05            # the real edge wins IS and holds up OOS
    assert out["prob_oos_loss"] < 0.05  # and rarely loses out-of-sample


def test_pbo_unavailable_when_too_few_configs():
    M = pd.DataFrame(np.random.default_rng(0).standard_normal((240, 1)))
    assert rc.pbo_cscv(M)["available"] is False


# ---------------------------- Deflated Sharpe -------------------------------- #
def test_dsr_decreases_as_more_configs_are_tried():
    rng = np.random.default_rng(3)
    best = rng.standard_normal(500) * 0.01 + 0.004          # a positive-Sharpe series
    trials = rng.standard_normal(50) * 0.5                  # spread of trial Sharpes
    few = rc.deflated_sharpe(best, trials, n_trials=2)
    many = rc.deflated_sharpe(best, trials, n_trials=1000)
    assert few["dsr"] > many["dsr"]                         # admitting more tries deflates it
    assert few["n_obs"] == 500


def test_dsr_high_for_strong_lone_signal_few_trials():
    rng = np.random.default_rng(5)
    best = rng.standard_normal(1000) * 0.005 + 0.01         # ~2 Sharpe, very strong
    trials = np.array([rc._sharpe(best), 0.1, -0.1, 0.0])
    out = rc.deflated_sharpe(best, trials, n_trials=4)
    assert out["dsr"] > 0.95 and out["significant"] is True


# ----------------------- multiple-testing edge p-value ----------------------- #
def _row(config, h, signals, edge_pp, acc):
    return {"config": config, "horizon": h,
            "oos": {"signals": signals, "cond_edge_pp": edge_pp, "accuracy": acc}}


def test_marginal_edge_among_many_trials_is_noise():
    # mirrors the real finding: +6.7pp on 239 signals, but 406 configs were tried.
    rows = [_row("rsi_ma_vol", 3, 239, 6.7, 0.586)]
    out = rc.multiple_testing_edge(rows, n_trials=406)
    assert out["available"]
    assert out["survives"] is False
    assert out["p_fwe"] > 0.5                 # luck almost certainly beats it
    assert out["luck_edge_pp"] > out["oos_cond_edge_pp"]


def test_strong_edge_few_trials_survives():
    rows = [_row("planted", 1, 6000, 9.0, 0.59)]
    out = rc.multiple_testing_edge(rows, n_trials=5)
    assert out["survives"] is True
    assert out["p_fwe"] < 0.05


def test_multiple_testing_picks_largest_edge():
    rows = [_row("a", 1, 500, 1.0, 0.51), _row("b", 3, 500, 4.0, 0.54),
            _row("c", 5, 500, 2.0, 0.52)]
    out = rc.multiple_testing_edge(rows, n_trials=3)
    assert out["best_config"] == "b" and out["horizon"] == 3


# --------------------------- returns matrix builder -------------------------- #
def _frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    idx = pd.bdate_range("2022-01-01", periods=n)
    return pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": 1e6}, index=idx)


def _always_long(df, p):
    return pd.Series(1, index=df.index)


def _never(df, p):
    return pd.Series(0, index=df.index)


def test_matrix_drops_never_trading_columns():
    frames = {"A": _frame(seed=1), "B": _frame(seed=2)}
    configs = [("long", "long", _always_long, {}), ("flat", "flat", _never, {})]
    M = rc.strategy_returns_matrix(frames, configs=configs, horizon=1, market_neutral=False)
    assert "long" in M.columns
    assert "flat" not in M.columns          # zero-variance column is dropped
    assert not M["long"].isna().any()


def test_market_neutral_nets_out_an_always_long_config():
    # Long the whole universe == the universe, so its active return is ~0 and it drops out:
    # PBO/DSR then speak to skill above the drift, not the equity risk premium.
    frames = {"A": _frame(seed=1), "B": _frame(seed=2), "C": _frame(seed=3)}
    configs = [("long", "long", _always_long, {})]
    M = rc.strategy_returns_matrix(frames, configs=configs, horizon=1, market_neutral=True)
    assert "long" not in M.columns


# --------------------------------- verdict ----------------------------------- #
def test_verdict_flags_overfitting():
    pbo = {"available": True, "pbo": 0.52}
    dsr = {"dsr": 0.40, "significant": False}
    mte = {"available": True, "p_fwe": 0.99, "survives": False}
    v = rc.verdict(pbo, dsr, mte)
    assert v["robust"] is False
    assert "overfit" in v["summary"]
