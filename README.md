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
pytest -q                                 # 59 tests (all offline, deterministic)

# Backtest gate (free data, no API key):
python -m src.backtest.engine             # single-pass basket backtest
python -m src.backtest.walkforward        # walk-forward out-of-sample gate
python -m src.backtest.research           # multi-strategy sweep vs buy-and-hold

# Analyst tool (free data; LLM optional):
python -m src.macro.brief                 # daily market briefing
python -m src.mcp.server                  # MCP server (for Claude Desktop etc.)
```

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
