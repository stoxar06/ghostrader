# Ghostrader

Indicator + candlestick based stock analyst & auto-trading bot for **Zerodha Kite**.

> **Reality check:** there is **no guaranteed daily %**. The target is **~1.5% gross per
> trade** (so ~1% nets after charges), with a hard stop-loss on every trade and positive
> expectancy proven by backtest → paper trading **before any real money**. Manipulation
> flags are *suspicion*, not proof; macro/sector signals are probabilistic.

## Architecture (3 layers)
1. **Technical engine** — indicators + all candlestick patterns + multi-timeframe, in an
   accuracy-weighted confluence model. Deterministic, fast, **zero LLM tokens**.
2. **Macro engine** — FII/DII flows, global cues, govt/war news, sector impact, manipulation
   flags → a *soft bias* into the technical engine. **Free-first multi-LLM** (Groq/Gemini/
   OpenRouter free → paid Claude fallback under a tight daily cap).
3. **MCP layer** — conversational analyst/control over the proven core (never in the trade loop).

Same code runs **backtest → paper → (gated) live** by swapping only the execution adapter.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # see TA-Lib note below
cp .env.example .env                      # then fill in keys
```

### TA-Lib (needed from Phase 3 on)
TA-Lib needs its C library first:
```bash
sudo apt-get install -y ta-lib            # Ubuntu/Debian (or build from source)
pip install TA-Lib
```
The scaffold (config, DB, logging) and tests run **without** TA-Lib.

## Run
```bash
pytest -q                          # 142 tests, all offline & deterministic

python -m src                      # list every command
python -m src web                  # 🖥️  web dashboard at http://127.0.0.1:5000
python -m src brief                # market briefing (cues + news + regime)
python -m src sip                  # SIP/buy-hold analyzer (XIRR) + realistic Nifty benchmark
python -m src screener             # gain/loss + positioning (trend, drawdown, RSI) by large/mid/small cap
python -m src learn <video>        # transcribe a strategy video (local Whisper) + honestly test its claim
python -m src learn --text "..."   #   …or skip transcription and test a pasted strategy description
python -m src research             # strategy sweep vs buy-and-hold (proves no edge)
python -m src edgesearch           # OOS hunt for 60%+ accuracy; archives real-edge vs drift
python -m src reality              # overfitting audit: PBO + Deflated Sharpe + multiple-testing edge
python -m src vault                # export findings to an Obsidian neuron-graph (vault/); --deep recomputes
python -m src horizon              # 1-12 day hold filter: daily P&L + totals, net of costs
python -m src momentum             # momentum factor vs buy-and-hold
python -m src backtest             # single-pass backtest (net of costs)
python -m src walkforward          # out-of-sample gate
python -m src paper                # PAPER trading sim on recent data (no real money)
python -m src active               # PAPER ACTIVE quota: >=5 random entries/day, 1-day holds (more trades = more cost)
python -m src schedule             # auto briefing pre-market + intraday (+Telegram)
python -m src serve                # MCP analyst server (Claude Desktop)
```

> ⚠️ **Live trading is built but OFF by default.** `config.yaml → live.allow_live_orders`
> defaults to `false`. The strategy has no proven edge, so real-money trading is expected
> to lose; paper mode is the safe, recommended path.

## Config
- **Secrets** → `.env` (gitignored). **Tunables** → `config.yaml`.
- The briefing runs with **zero keys** (deterministic regime). Add free
  `GROQ_API_KEY` / `GEMINI_API_KEY` / `OPENROUTER_API_KEY` for LLM synthesis;
  `ANTHROPIC_API_KEY` is the paid fallback under a daily INR cap.

## Status & honest findings
Built & tested: data layer, indicators + candlestick patterns, confluence strategy,
risk, backtest + walk-forward gate, free-first multi-LLM router, macro engine, daily
briefing, and the MCP analyst server.

**The auto-trading thesis was tested and failed the gate.** Across trend, mean-reversion,
breakout (intraday) and swing (daily), no approach beat buy-and-hold after costs (e.g.
swing made +71.5% vs buy-and-hold +71,689% over ~28y). **Conclusion: do not deploy the
auto-trader.** The project is therefore an **analyst/decision tool**, not a trading bot.
Live/paper trading (Phases 5/8) are intentionally NOT pursued — they would deploy a
no-edge strategy. The infrastructure remains ready for a genuinely better strategy/data.

**On "just get me 60% accuracy":** `python -m src edgesearch` sweeps ~190 causal rule configs
(15+ families × params × 7 horizons) on a train/test split, plus a cross-sectional momentum
factor. The only configs that clear 60% *out-of-sample* are pure drift-riders — "always long"
over a 40–60 day hold is right ~60–62% of the time, but its **conditional edge is +0.0pp**: that
is the market's upward drift, not a signal, and it loses to simply buying the index. No rule
shows real directional edge that survives costs. The honest path to "right 60%+ of the time" is
**holding** (the equity risk premium), which `sip` already quantifies — not a per-trade signal.

**The overfitting audit (`reality`).** Sweeping hundreds of configs and keeping the best is itself
a way to manufacture a number. `python -m src reality` runs the institutional anti-overfitting
toolkit over the same catalog: **PBO** (Probability of Backtest Overfitting, via combinatorially
symmetric cross-validation), the **Deflated Sharpe Ratio** (deflates the best config's Sharpe by
the number of trials and the return shape), and a **family-wise-error edge p-value**. The one
config `edgesearch` still flags `REAL EDGE` (rsi_ma_vol dip, 58.6% / +6.7pp OOS) does **not**
survive: its FWE p ≈ 1.0 and the best Sharpe deflates to ~5% significance — exactly the luckiest
of N noise trials. This is the rigorous "is the 60% real?" and the answer stays no.

**Knowledge vault (`vault`).** `python -m src vault` exports every finding — concepts, configs,
symbols, harness verdicts — as an [Obsidian](https://obsidian.md) "neuron" graph under `vault/`
(markdown + `[[wikilinks]]`; gitignored). Notes auto-refresh only when missing, when the cached
data advances, or past a 7-day TTL, and your `pinned: true` notes are never overwritten (`--deep`
recomputes the harnesses first). It is a **research memory** so accumulated findings survive across
sessions and are never re-derived — **not** an auto-trader, and it makes no prediction. (100%
accuracy is not reachable; the vault encodes that finding as a first-class neuron.)

**Learn from a video (`learn`).** Point it at a strategy video and it transcribes the audio
locally (faster-whisper, offline), extracts the TA rules described (RSI levels, moving-average
crosses, breakouts, volume combos), and runs them through the same out-of-sample sweep +
multiple-testing check as everything else — so a "this setup is 90% accurate!" video becomes a
verdict, not a belief. (It needs `faster-whisper` + `ffmpeg` for real media; `--text "<description>"`
tests a transcript with no install.) It does **not** teach the bot to win — a clear transcript of a
no-edge strategy is still no-edge. In testing, even a flashy 68% / +7pp claim failed the luck bar.
