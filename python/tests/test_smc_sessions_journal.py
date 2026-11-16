"""Unit tests for deterministic SMC, trading session kill-zones, and SQLite journal."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python.src.journal import TradeJournal
from python.src.schemas import (
    AiTradeDecision,
    AnalysisReport,
    EaPosition,
    EaSignalPayload,
    MarketContext,
    OhlcvBar,
    SessionStatus,
)
from python.src.sessions import evaluate_session
from python.src.smc import build_smc_snapshot, find_unmitigated_fvgs


def _make_bars(prices: list[tuple[float, float, float, float]], start_t: int = 1760000000, step: int = 3600) -> list[OhlcvBar]:
    return [
        OhlcvBar(t=start_t + i * step, o=o, h=h, l=l, c=c, v=1000)
        for i, (o, h, l, c) in enumerate(prices)
    ]


def test_session_killzones_and_blocks() -> None:
    # Wednesday 08:30 UTC -> London Kill Zone
    london = evaluate_session(datetime(2026, 11, 11, 8, 30, tzinfo=timezone.utc))
    assert london.in_killzone is True
    assert london.market_open is True
    assert london.session_name == "LONDON_KILLZONE"

    # Wednesday 13:30 UTC -> NY / London-NY Overlap
    ny = evaluate_session(datetime(2026, 11, 11, 13, 30, tzinfo=timezone.utc))
    assert ny.in_killzone is True
    assert ny.session_name == "NY_OVERLAP_KILLZONE"

    # Wednesday 02:00 UTC -> Asian Session (outside kill zone)
    asian = evaluate_session(datetime(2026, 11, 11, 2, 0, tzinfo=timezone.utc))
    assert asian.in_killzone is False
    assert asian.market_open is True
    assert asian.session_name == "ASIAN_SESSION"

    # Wednesday 21:30 UTC -> Daily Rollover block
    roll = evaluate_session(datetime(2026, 11, 11, 21, 30, tzinfo=timezone.utc))
    assert roll.in_killzone is False
    assert roll.session_name == "DAILY_ROLLOVER"

    # Saturday -> Weekend closed
    sat = evaluate_session(datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc))
    assert sat.market_open is False
    assert sat.in_killzone is False


def test_smc_detects_bullish_structure_ob_and_fvg() -> None:
    # Construct bullish impulse leaving an unmitigated bullish OB and FVG below 2660
    bars_h1 = _make_bars([
        (2620, 2623, 2618, 2621),
        (2621, 2626, 2620, 2625),
        (2625, 2630, 2624, 2628),  # Swing high ~2630
        (2628, 2629, 2622, 2623),
        (2623, 2625, 2619, 2620),  # Swing low ~2619
        (2620, 2624, 2619, 2622),
        (2622, 2624, 2620, 2621),  # Bearish OB candle [2620-2624]
        (2621, 2636, 2621, 2635),  # Strong bullish displacement
        (2635, 2648, 2628, 2646),  # Leaves Bullish FVG [2624 - 2628]
        (2646, 2655, 2644, 2653),
        (2653, 2660, 2650, 2658),
    ])
    bull_fvg, _ = find_unmitigated_fvgs(bars_h1, "H1", current_price=2658.0, min_gap_usd=0.5)
    assert len(bull_fvg) >= 1
    assert any(z.low == 2624.0 and z.high == 2628.0 for z in bull_fvg)

    market = MarketContext(
        updated_at_unix=1760000000,
        symbol="XAUUSDm",
        bid=2657.8,
        ask=2658.1,
        spread_usd=0.3,
        atr_h1_14=8.0,
        h4_bars=bars_h1,
        h1_bars=bars_h1,
        m15_bars=bars_h1,
    )
    smc = build_smc_snapshot(market)
    assert smc.h1.trend == "BULLISH"
    assert len(smc.bullish_obs) >= 1
    assert smc.anchored_buy_limit is not None
    assert smc.anchored_buy_limit.entry < 2658.0
    assert smc.anchored_buy_limit.sl < smc.anchored_buy_limit.entry < smc.anchored_buy_limit.tp


def test_journal_records_signals_and_trade_lifecycle(tmp_path: Path) -> None:
    journal = TradeJournal(tmp_path / "journal.db")
    dec = AiTradeDecision(
        bias="BULLISH",
        macro_summary="Macro",
        news_sentiment_score=0.45,
        news_sentiment_summary="Positive",
        technical_summary="H1 OB",
        order_intent="BUY_LIMIT",
        limit_price=2640.0,
        sl_price=2630.0,
        tp_price=2660.0,
        confidence=0.82,
        reason="Test setup",
    )
    sig = EaSignalPayload(
        signal_id="sig_abc",
        generated_at_unix=100,
        expires_at_unix=200,
        bias="BULLISH",
        order_intent="BUY_LIMIT",
        trade_allowed=True,
        limit_price=2640.0,
        sl_price=2630.0,
        tp_price=2660.0,
        confidence=0.82,
    )
    report = AnalysisReport(
        decision=dec,
        signal=sig,
        hmr_blackout_active=False,
        session_status=SessionStatus(
            session_name="NY_OVERLAP_KILLZONE", in_killzone=True, market_open=True, reason="NY"
        ),
        confluence_score=5,
        validation_note="OK",
    )
    journal.record_analysis(report)

    pos = EaPosition(
        ticket=9001,
        type="BUY",
        volume=0.10,
        open_price=2640.0,
        sl=2640.25,
        tp=2660.0,
        profit=42.50,
        be_applied=True,
    )
    journal.upsert_open_position("XAUUSDm", pos)
    journal.mark_position_closed(9001)

    stats = journal.get_stats()
    assert stats.total_analyses == 1
    assert stats.actionable_signals == 1
    assert stats.trades_closed == 1
    assert stats.be_triggered_count == 1
    assert stats.wins == 1
    assert stats.win_rate_pct == 100.0
    assert stats.net_realized_pnl == 42.50

    learning_txt = journal.format_learning_context()
    assert "1W/0L (100.0%)" in learning_txt
    assert "BUY: 1/1 wins" in learning_txt
    assert "#9001 BUY" in learning_txt
    assert "BUY_LIMIT (BULLISH)" in learning_txt
