"""Phase 1 smoke tests — config loads, secrets build, DB initializes and round-trips."""
from __future__ import annotations

from datetime import datetime


def test_config_loads():
    from src.config import get_config

    cfg = get_config()
    assert cfg.mode in {"backtest", "paper", "live"}
    # Extra keys from config.yaml are accessible as attributes on the section.
    assert float(cfg.risk.target_pct) == 1.5
    assert float(cfg.risk.risk_per_trade_pct) > 0
    assert cfg.instruments.exchange == "NSE"
    assert isinstance(cfg.instruments.symbols, list) and cfg.instruments.symbols


def test_secrets_build():
    from src.config import get_secrets

    # Keys may be unset in dev/CI; we only assert the object constructs.
    secrets = get_secrets()
    assert secrets is not None
    assert hasattr(secrets, "kite_api_key")
    assert hasattr(secrets, "groq_api_key")


def test_llm_provider_order_present():
    from src.config import get_config

    order = get_config().llm.provider_order
    assert order[-1] == "anthropic"  # paid fallback comes last
    assert "groq" in order


def test_db_init_and_roundtrip(tmp_path):
    from src.storage.db import Trade, get_session, init_db

    db_file = tmp_path / "test.db"
    init_db(str(db_file))

    with get_session() as session:
        session.add(
            Trade(
                symbol="TEST",
                side="BUY",
                qty=1,
                entry_price=100.0,
                entry_time=datetime.now(),
                mode="paper",
                reason="smoke-test",
            )
        )
        session.commit()
        assert session.query(Trade).count() == 1
        row = session.query(Trade).first()
        assert row.symbol == "TEST"
        assert row.mode == "paper"
