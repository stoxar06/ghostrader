"""Tests for the Obsidian knowledge vault (offline, deterministic, no network)."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.knowledge import vault as v


def _neuron(**kw):
    base = dict(slug="X", folder="concepts", ntype="concept", title="X", body="body",
                updated="2026-06-10")
    base.update(kw)
    return v.Neuron(**base)


# --------------------------------- rendering --------------------------------- #
def test_render_has_frontmatter_and_links():
    n = _neuron(slug="Edge", title="Edge", tags=["ghostrader", "concept"],
                links=["Costs", "Drift", "Edge"])  # self-link must be dropped
    text = v.render(n)
    assert text.startswith("---\n")
    assert "type: concept" in text
    assert "# Edge" in text
    assert "- [[Costs]]" in text and "- [[Drift]]" in text
    assert "[[Edge]]" not in text.split("## Links")[1]   # no self-link in the graph


def test_render_parse_roundtrip():
    n = _neuron(slug="Reality Check", title="Reality Check", data_through="2026-06-12",
                extra={"verdict": "noise", "pinned": False})
    fm = v.parse_frontmatter(v.render(n))
    assert fm["type"] == "concept"
    assert fm["data_through"] == "2026-06-12"
    assert fm["verdict"] == "noise"
    assert fm["pinned"] is False
    assert "ghostrader" not in fm.get("tags", []) or isinstance(fm["tags"], list)


def test_safe_slug_strips_unsafe_chars():
    assert v.safe_slug("rsi_ma_vol[style=dip,n=200]") == "rsi_ma_vol(style=dip, n=200)" \
        or "[" not in v.safe_slug("rsi_ma_vol[style=dip]")
    assert ":" not in v.safe_slug("a:b")
    assert "/" not in v.safe_slug("a/b")


# ------------------------------ staleness rule ------------------------------- #
def test_should_write_new_when_missing():
    assert v.should_write(None, _neuron(), date(2026, 6, 14)) == (True, "new")


def test_should_write_skips_pinned():
    existing = v.render(_neuron(extra={"pinned": True}, updated="2020-01-01"))
    write, reason = v.should_write(existing, _neuron(updated="2026-06-14"), date(2026, 6, 14))
    assert write is False and reason == "pinned"


def test_should_write_refreshes_on_new_data():
    existing = v.render(_neuron(data_through="2026-06-01", updated="2026-06-14"))
    n = _neuron(data_through="2026-06-10", updated="2026-06-14")
    assert v.should_write(existing, n, date(2026, 6, 14)) == (True, "new-data")


def test_should_write_refreshes_when_stale():
    existing = v.render(_neuron(updated="2026-06-01"))
    write, reason = v.should_write(existing, _neuron(updated="2026-06-14"),
                                   date(2026, 6, 14), ttl_days=7)
    assert write is True and reason == "stale"


def test_should_write_leaves_fresh_alone():
    existing = v.render(_neuron(updated="2026-06-13"))
    write, reason = v.should_write(existing, _neuron(updated="2026-06-14"),
                                   date(2026, 6, 14), ttl_days=7)
    assert write is False and reason == "fresh"


# --------------------------------- writing ----------------------------------- #
def test_write_vault_skips_fresh_on_second_run(tmp_path):
    n = _neuron(slug="Edge", title="Edge", updated="2026-06-14")
    first = v.write_vault([n], out_dir=str(tmp_path), now=date(2026, 6, 14))
    assert first["written"] == 1 and first["new"] == 1
    second = v.write_vault([n], out_dir=str(tmp_path), now=date(2026, 6, 14))
    assert second["written"] == 0 and second["skipped"] == 1


def test_write_vault_preserves_pinned_edits(tmp_path):
    path = tmp_path / "concepts" / "Edge.md"
    path.parent.mkdir(parents=True)
    path.write_text("---\ntype: concept\npinned: true\nupdated: 2020-01-01\n---\n\n"
                    "# Edge\n\nMY HAND-WRITTEN NOTE\n", encoding="utf-8")
    v.write_vault([_neuron(slug="Edge", title="Edge", updated="2026-06-14")],
                  out_dir=str(tmp_path), now=date(2026, 6, 14))
    assert "MY HAND-WRITTEN NOTE" in path.read_text(encoding="utf-8")


# ------------------------------ neuron builders ------------------------------ #
def test_config_neuron_picks_best_horizon_and_links():
    rows = [
        {"config": "rsi_ma_vol[x]", "family": "rsi_ma_vol", "rule": "MA+RSI+vol",
         "horizon": 1, "real_edge": False, "drift_rider": False,
         "is": {"accuracy": 0.54, "cond_edge_pp": 1.0, "signals": 240},
         "oos": {"accuracy": 0.57, "cond_edge_pp": 3.0, "signals": 239}},
        {"config": "rsi_ma_vol[x]", "family": "rsi_ma_vol", "rule": "MA+RSI+vol",
         "horizon": 3, "real_edge": True, "drift_rider": False,
         "is": {"accuracy": 0.556, "cond_edge_pp": 5.6, "signals": 240},
         "oos": {"accuracy": 0.586, "cond_edge_pp": 6.7, "signals": 239}},
    ]
    out = v.config_neurons(rows, date(2026, 6, 14), data_through="2026-06-12")
    assert len(out) == 1
    n = out[0]
    assert n.extra["oos_cond_edge_pp"] == 6.7        # kept the richer-edge horizon
    assert "Reality Check" in n.links and "Edge Search" in n.links
    assert n.data_through == "2026-06-12"


def test_concepts_include_no_100pct_neuron():
    titles = {n.title for n in v.concept_neurons(date(2026, 6, 14))}
    assert "100% Accuracy (Not Reachable)" in titles


def _frame(n=120, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    idx = pd.bdate_range("2026-01-01", periods=n)
    return pd.DataFrame({"close": close}, index=idx)


def test_build_writes_full_graph(tmp_path):
    frames = {"RELIANCE.NS": _frame(seed=1), "TCS.NS": _frame(seed=2)}
    stats = v.build(frames, rows=None, out_dir=str(tmp_path), now=date(2026, 6, 14))
    assert (tmp_path / "Home.md").exists()
    assert (tmp_path / "concepts" / "Edge.md").exists()
    assert (tmp_path / "symbols" / "RELIANCE.NS.md").exists()
    assert (tmp_path / "graph.html").exists()         # interactive graph always written
    assert stats["written"] == stats["neurons"]      # all fresh on first build
    assert stats["edges"] > 0
    home = (tmp_path / "Home.md").read_text(encoding="utf-8")
    assert "[[Edge]]" in home and "Not a trading bot" in home


# ----------------------------- interactive graph ----------------------------- #
def test_graph_payload_links_harness_to_concept():
    now = date(2026, 6, 14)
    neurons = v.concept_neurons(now) + v.harness_neurons(now)
    g = v.graph_payload(neurons)
    ids = {n["id"] for n in g["nodes"]}
    assert "Reality Check" in ids and "Backtest Overfitting" in ids
    assert any(e["from"] == "Reality Check" and e["to"] == "Backtest Overfitting"
               for e in g["edges"])
    assert all("color" in n and "value" in n for n in g["nodes"])


def test_write_graph_html_is_self_contained(tmp_path):
    path = v.write_graph_html(v.concept_neurons(date(2026, 6, 14)), out_dir=str(tmp_path))
    html = Path(path).read_text(encoding="utf-8")
    assert "vis-network" in html and '"id":' in html and "<html" in html
