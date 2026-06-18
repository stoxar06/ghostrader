# CLAUDE.md

Ghostrader — indicator + candlestick stock **analyst / research tool** for Zerodha Kite
(not a deployable auto-trader). This file is the navigation map. Read it first; **don't
re-explore the tree** unless a path below is wrong. Read `README.md` only for setup/run details.

> **Established finding (don't re-litigate):** no simple-TA strategy has a per-trade directional
> edge on this data. `edgesearch` swept ~250 causal configs OOS (incl. volume/RSI/MA-combo
> families; best combo = RSI-dip above SMA200 on volume — recent-regime artifact, IS edge ≤ 0,
> not validated) + cross-sectional momentum —
> the only configs clearing 60% accuracy are drift-riders (a 40–60d hold) with **~0 conditional
> edge** and no profit after costs. "60% of the time price is higher later" = the equity risk
> premium (holding), not a signal. Don't fabricate edge via in-sample fitting or lookahead.
> Extended June 2026: literature anomalies added (overnight/intraday tug-of-war, turn-of-month,
> low-volume reversal, 52-week-high proximity) — all ≤0pp conditional edge OOS; `evolve` champion
> (adaptive_rsi, +4pp min(IS,OOS)) fails the luck bar (z 3.75 < 3.47+1) AND loses after costs
> (PF 0.93). 1–12d `horizon` sweep: RSI signals lose at every hold, net of costs.
> `reality` (the overfitting audit) formalises this: the lone config `edgesearch` still tags
> `REAL EDGE` (rsi_ma_vol dip, 58.6% / +6.7pp OOS at h=3) is a multiple-testing artifact — its
> family-wise-error p≈1.0, the best-Sharpe config's **Deflated Sharpe ≈5%** (below the best-of-N
> noise bar), and PBO confirms the magnitude is economically insignificant. Treat any single-config
> "edge" from a sweep as unproven until it clears `reality`.

## Orient fast (don't grep for these)
- **CLI dispatch**: `src/cli.py` — a `COMMANDS` dict maps `name -> (module, func)`. To add a
  command, add one line there + a `DESCRIPTIONS` entry. Entry point: `python -m src` (`src/__main__.py`).
- **Config**: tunables in `config.yaml` (loaded by `src/config.py`); secrets in `.env`
  (gitignored, see `.env.example`). Briefing runs with zero keys.
- **3 layers**: technical engine (`src/indicators/`, `src/strategy/`, deterministic, **0 LLM tokens**)
  → macro engine (`src/macro/`, `src/llm/` free-first router) → MCP layer (`src/mcp/`, never in trade loop).
- **Same code, 3 modes** via execution adapter swap (`src/execution/`): backtest → paper → gated live.

## Where things live
| Need | Path |
|------|------|
| Indicators / candlestick patterns | `src/indicators/` |
| Strategy (confluence, MTF, alt) | `src/strategy/` |
| Backtest / walkforward / research | `src/backtest/` |
| Honest accuracy / edge harnesses (+ archive) | `src/backtest/{rsi_accuracy,indicator_accuracy,edge_search}.py`; 60%+ configs → `data/edge_archive.json` (gitignored) |
| 1–12 day holding-horizon filter (daily P&L + totals) | `src/backtest/horizon.py` (`horizon` cmd, `/api/horizon`, frontend `/horizon` page w/ GitHub-style heatmap) — confirms no edge at any horizon |
| Self-improving (evolutionary) config search | `src/backtest/auto_search.py` (`evolve` cmd); resumable state → `data/auto_search.json`; fitness = min(IS,OOS) edge, ≥80% acc = lookahead tripwire |
| **Overfitting audit** (PBO + Deflated Sharpe + multiple-testing) | `src/backtest/reality.py` (`reality` cmd, `/api/reality`); CSCV→PBO, Deflated Sharpe, family-wise-error edge p-value over the `edge_search` catalog; market-neutral returns matrix so all 3 lenses measure skill, not drift. `audit(frames, rows=...)` reuses a sweep |
| Obsidian knowledge vault ("neurons") | `src/knowledge/vault.py` (`vault` cmd, `--deep` recomputes); markdown notes + `[[wikilinks]]` graph → `vault/` (gitignored); staleness rule: rewrite only if missing / cache advanced / past TTL; `pinned: true` notes never clobbered. Research memory, **not** an auto-trader |
| Execution adapters (paper/live/base) | `src/execution/` |
| Live + historical data | `src/data/` |
| Macro: cues, news, regime, brief | `src/macro/` |
| LLM providers + router | `src/llm/` |
| MCP server + tools | `src/mcp/` |
| Risk sizing / stops | `src/risk/manager.py` |
| Stock screener (gain/loss + positioning by cap) | `src/invest/screener.py` (`screener` cmd, `/api/screener`, dashboard **Screener** tab); descriptive metrics (returns, 52w-high drawdown, SMA200 trend, RSI) + setup tags — NOT predictions. Large-cap cached; mid/small fetched live + cached |
| Learn-from-video (transcribe → honestly test) | `src/learn/` (`learn` cmd): `transcribe.py` (local faster-whisper, pluggable), `strategy_extract.py` (transcript → TA rule configs, deterministic), `pipeline.py` (extract → `edge_search.evaluate` + `reality` verdict). `--text` skips STT. Tests a video's *claim*; does NOT make the bot profitable |
| Web dashboard (Flask + dashboard.html) | `src/web/` |
| Next.js + shadcn UI (needs **Node 20**, calls Flask `/api/*`) | `frontend/` |
| Notifications (telegram, report) | `src/notify/` |
| Persistence (SQLite) | `src/storage/db.py` |

## Commands (`python -m src <cmd>`)
`brief serve schedule backtest walkforward research momentum rsi indicators horizon edgesearch reality vault evolve compare sip screener learn web paper`
(paper variants in `src/runner.py`: `candidate` `random` `active` — `active` forces a ≥5-trades/day quota via
`quota_signals` + `PaperBroker(max_hold_bars=1)` + raised risk caps; it demonstrates more trades = more cost, not profit)
(`rsi` / `indicators` / `edgesearch` = honest directional-accuracy harnesses in `src/backtest/`;
`edgesearch` = OOS sweep that archives any 60%+ config to `data/edge_archive.json`, labeled
real-edge vs drift-rider — finding: only drift-riders clear 60%, with ~0 conditional edge.
`reality` = multiple-testing audit that debunks any single-config "edge" the sweep surfaces.
`vault [--deep]` = export findings to an Obsidian neuron-graph under `vault/`, auto-refreshing
stale notes. Subcommand flags pass through `cli.py` via `parse_known_args`.)
Authoritative list + help text: `COMMANDS`/`DESCRIPTIONS` in `src/cli.py`.

## Conventions
- Tests are **offline & deterministic** — `pytest -q`, all under `tests/`. Keep them that way:
  no network/API/clock dependence; mock external data.
- TA-Lib is an optional native dep (needed from Phase 3 on); scaffold + most tests run without it.
- Module entry funcs are inconsistent: some `main`, some `_main` (see `COMMANDS`). Match the existing one.
- Runtime artifacts (`/data/`, `/logs/`, `*.db`) are gitignored — never commit them.

## Token-saving rules for the agent
- **Trust this map** — go straight to the relevant `src/<area>/` file; avoid full-tree `ls`/`grep` sweeps.
- The technical/indicator/strategy path is **intentionally LLM-free**; don't add LLM calls there.
- Read narrowly: target the one file for the task, not whole packages. Don't re-read files you just edited.
- Live trading is OFF by default (`config.yaml → live.allow_live_orders: false`) and the strategy has
  no proven edge — never flip it on or wire live execution without explicit user instruction.
