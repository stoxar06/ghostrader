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
pytest -q                                 # unit/smoke tests
# Later phases:
# python -m src.backtest.engine           # backtest (no API key needed)
# python -m src.runner                    # paper trading (needs Kite key)
```

## Config
- **Secrets** → `.env` (gitignored). **Tunables** → `config.yaml`.
- Start in `mode: paper`. Live is gated behind ≥100 paper trades over ~2–3 months.

## Status
Phase 1 (scaffold & config) complete. See `~/.claude/plans/` for the full build plan and
the 8-phase roadmap.
