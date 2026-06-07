"""Local web dashboard (Flask) — view all the engine's data in a browser.

    python -m src web      # then open http://127.0.0.1:5000

Read-only analysis UI. It never trades. Network endpoints (cues, briefing, sip,
backtest, research, momentum) fetch live free data and may take a few seconds.
"""
from __future__ import annotations

from pathlib import Path

from src.logutil import get_logger

log = get_logger(__name__)
_HTML = Path(__file__).parent / "dashboard.html"


def create_app(init_db_path: str | None = None):
    from flask import Flask, Response, jsonify, request

    from src.backtest.momentum import DEFAULT_UNIVERSE
    from src.config import get_config, get_secrets
    from src.storage.db import init_db

    cfg = get_config()
    init_db(init_db_path or cfg.storage.db_path)
    app = Flask(__name__)

    @app.get("/")
    def index():
        return Response(_HTML.read_text(encoding="utf-8"), mimetype="text/html")

    @app.get("/api/health")
    def health():
        sec = get_secrets()
        providers = [p for p, k in [
            ("groq", sec.groq_api_key), ("gemini", sec.gemini_api_key),
            ("openrouter", sec.openrouter_api_key), ("anthropic", sec.anthropic_api_key),
        ] if k]
        return jsonify({
            "status": "ok",
            "mode": cfg.mode,
            "llm_providers_configured": providers,
            "kite_configured": bool(sec.kite_api_key),
            "telegram_configured": bool(sec.telegram_bot_token and sec.telegram_chat_id),
        })

    @app.get("/api/config")
    def config():
        llm = cfg.llm.model_dump()
        return jsonify({
            "mode": cfg.mode,
            "instruments": cfg.instruments.model_dump(),
            "risk": cfg.risk.model_dump(),
            "llm": {"provider_order": llm.get("provider_order"),
                    "daily_paid_cap_inr": llm.get("daily_paid_cap_inr")},
            "research_universe": DEFAULT_UNIVERSE,
        })

    @app.get("/api/cues")
    def cues():
        from src.macro import globalcues
        return jsonify(globalcues.fetch_global_cues())

    @app.get("/api/briefing")
    def briefing():
        from src.mcp import tools
        return jsonify(tools.get_briefing())

    @app.get("/api/prices")
    def prices():
        from src.data.historical import HistoricalData
        sym = request.args.get("symbol", "^NSEI")
        tf = request.args.get("tf", "day")
        s = HistoricalData(cache_dir="data/cache").get(sym, tf)["close"].dropna()
        if len(s) > 1500:  # downsample for the chart
            s = s.iloc[:: len(s) // 1500]
        return jsonify({
            "symbol": sym,
            "dates": [d.strftime("%Y-%m-%d") for d in s.index],
            "close": [round(float(x), 2) for x in s.values],
        })

    @app.get("/api/sip")
    def sip():
        from src.mcp import tools
        sym = request.args.get("symbol", "^NSEI")
        amt = float(request.args.get("amount", 10000))
        return jsonify(tools.sip_tool(sym, amt))

    @app.get("/api/lumpsum")
    def lumpsum():
        from src.mcp import tools
        return jsonify(tools.lumpsum_tool(request.args.get("symbol", "^NSEI")))

    @app.get("/api/backtest")
    def backtest():
        from src.mcp import tools
        sym = request.args.get("symbol")
        if not sym:
            return jsonify({"error": "symbol required"}), 400
        return jsonify(tools.run_backtest_tool(sym, request.args.get("tf") or None))

    @app.get("/api/research")
    def research():
        from src.backtest.research import sweep
        return jsonify(sweep())

    @app.get("/api/momentum")
    def momentum():
        from src.backtest.momentum import DEFAULT_UNIVERSE as U, momentum_backtest
        from src.data.historical import HistoricalData
        hist = HistoricalData(cache_dir="data/cache")
        prices_ = {}
        for sym in U:
            try:
                prices_[sym] = hist.get(sym, "day")["close"]
            except Exception:  # noqa: BLE001
                pass
        return jsonify(momentum_backtest(prices_))

    @app.get("/api/daily_pnl")
    def daily_pnl():
        from src.mcp import tools
        return jsonify(tools.get_daily_pnl(30))

    @app.get("/api/trades")
    def trades():
        from src.mcp import tools
        return jsonify(tools.explain_last_trades(50))

    @app.get("/api/llm_usage")
    def llm_usage():
        from src.storage.db import LLMUsage, get_session
        with get_session() as s:
            rows = s.query(LLMUsage).order_by(LLMUsage.id.desc()).limit(50).all()
            return jsonify([
                {"ts": str(r.ts), "provider": r.provider, "model": r.model,
                 "in": r.input_tokens, "out": r.output_tokens, "cost_inr": round(r.cost_inr, 3)}
                for r in rows
            ])

    @app.errorhandler(Exception)
    def on_error(exc):  # network/data failures -> JSON, so the UI can show them
        log.warning("API error: %s", exc)
        return jsonify({"error": str(exc)}), 500

    return app


def _main() -> None:  # pragma: no cover - long-running server
    import os
    app = create_app()
    host = os.environ.get("GHOST_HOST", "127.0.0.1")
    port = int(os.environ.get("GHOST_PORT", "5000"))
    log.info("Dashboard at http://%s:%s  (Ctrl-C to stop)", host, port)
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":  # pragma: no cover
    _main()
