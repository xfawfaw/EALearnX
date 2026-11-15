"""Unit tests for Telegram inline callback handling and resilient message delivery."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from telegram.error import BadRequest, TimedOut

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python.src.bridge import Mt5Bridge
from python.src.chart_card import render_analysis_chart
from python.src.config import Settings
from python.src.journal import TradeJournal
from python.src.schemas import (
    AiTradeDecision,
    AnalysisReport,
    EaSignalPayload,
    MarketContext,
    OhlcvBar,
    SessionStatus,
)
from python.src.telegram_bot import TelegramController


def _make_controller(tmp_path: Path, default_mode: str = "auto") -> TelegramController:
    settings = Settings(
        gemini_api_key="test",
        telegram_bot_token="123:test",
        telegram_allowed_user_ids=[5462088246],
        mt5_files_dir=tmp_path,
        default_execution_mode=default_mode,  # type: ignore[arg-type]
    )
    bridge = Mt5Bridge(tmp_path)
    journal = TradeJournal(tmp_path / "journal.db")
    analyst = MagicMock()
    return TelegramController(settings=settings, bridge=bridge, analyst=analyst, journal=journal)


def test_mode_button_when_already_active_handles_not_modified(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path, default_mode="auto")

    reply_text = AsyncMock()
    query = SimpleNamespace(
        data="menu:mode:auto",
        answer=AsyncMock(),
        edit_message_text=AsyncMock(side_effect=BadRequest("Message is not modified")),
        message=SimpleNamespace(reply_text=reply_text),
    )
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=5462088246),
        callback_query=query,
        message=None,
    )
    context = SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock()))

    asyncio.run(ctrl.handle_callback(update, context))  # type: ignore[arg-type]

    query.answer.assert_awaited_once()
    assert "Already in AUTO" in query.answer.call_args.args[0]
    reply_text.assert_awaited_once()
    assert "already <b>AUTO</b>" in reply_text.call_args.args[0]
    cmd = ctrl.bridge.command_path.read_text(encoding="utf-8")
    assert '"RESUME"' in cmd


def test_broadcast_retries_on_timeout_and_chunks_long_messages(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path, default_mode="auto")
    send_mock = AsyncMock(side_effect=[TimedOut("timed out"), None, None])
    context = SimpleNamespace(bot=SimpleNamespace(send_message=send_mock))

    long_text = "A" * 4500
    asyncio.run(ctrl._broadcast(context, long_text))  # type: ignore[arg-type]

    # First 4000-char chunk fails once then succeeds; second 500-char chunk succeeds.
    assert send_mock.await_count == 3


def _sample_market_and_report() -> tuple[MarketContext, AnalysisReport]:
    bars = [
        OhlcvBar(t=1760000000 + i * 3600, o=3870.0 + i * 0.4, h=3875.0 + i * 0.4, l=3868.0 + i * 0.4, c=3873.0 + i * 0.4, v=500)
        for i in range(30)
    ]
    market = MarketContext(
        updated_at_unix=1760100000,
        symbol="XAUUSDm",
        bid=3883.20,
        ask=3883.44,
        spread_usd=0.24,
        atr_h1_14=14.8,
        h1_bars=bars,
    )
    decision = AiTradeDecision(
        bias="BULLISH",
        order_intent="BUY_LIMIT",
        limit_price=3878.50,
        sl_price=3865.00,
        tp_price=3905.50,
        confidence=0.82,
        macro_summary="Fed speakers cautious; real yields eased.",
        news_sentiment_score=0.45,
        news_sentiment_summary="Safe-haven bids steady ahead of US CPI.",
        technical_summary="H4 & H1 bullish BOS; price pulling back into H1 Bull OB.",
        reason="Pullback to H1 Bull OB",
        should_execute=True,
    )
    signal = EaSignalPayload(
        signal_id="sig_test_1",
        generated_at_unix=1760100000,
        expires_at_unix=1760103600,
        bias="BULLISH",
        order_intent="BUY_LIMIT",
        trade_allowed=True,
        limit_price=3878.50,
        sl_price=3865.00,
        tp_price=3905.50,
        confidence=0.82,
        reason="Test setup",
    )
    report = AnalysisReport(
        decision=decision,
        signal=signal,
        hmr_blackout_active=False,
        session_status=SessionStatus(
            session_name="NY_KILLZONE",
            in_killzone=True,
            market_open=True,
            reason="NY Open Kill-Zone",
        ),
        confluence_score=4,
        confluence_notes=["H4+H1 aligned", "Discount PD", "RR 2.0 >= 1.8"],
        validation_note="Validated setup BUY_LIMIT @ 3878.50",
        grounded_sources=["Reuters"],
    )
    return market, report


def test_chart_card_renders_valid_png_and_photo_bubble_has_interactive_buttons(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path, default_mode="auto")
    market, report = _sample_market_and_report()

    png_bytes = render_analysis_chart(market=market, report=report)
    assert png_bytes is not None
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png_bytes) > 5000

    caption = ctrl._format_analysis_caption(report, "⚡ AUTO SIGNAL SENT TO MT5")
    assert len(caption) <= 1024
    assert "blockquote expandable" in caption

    send_photo_mock = AsyncMock()
    context = SimpleNamespace(bot=SimpleNamespace(send_photo=send_photo_mock, send_message=AsyncMock()))
    asyncio.run(ctrl._send_analysis_card(context, report, market, "⚡ AUTO SIGNAL SENT TO MT5"))  # type: ignore[arg-type]

    send_photo_mock.assert_awaited_once()
    kwargs = send_photo_mock.call_args.kwargs
    assert kwargs["reply_markup"] is not None
    btn_callbacks = [btn.callback_data for row in kwargs["reply_markup"].inline_keyboard for btn in row]
    assert "menu:brief" in btn_callbacks
    assert "menu:smc_levels" in btn_callbacks
    assert "menu:analyze" in btn_callbacks


def test_menu_brief_callback_sends_full_expandable_html_report(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path, default_mode="auto")
    _, report = _sample_market_and_report()
    ctrl.runtime.last_report = report

    reply_text = AsyncMock()
    query = SimpleNamespace(
        data="menu:brief",
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
        message=SimpleNamespace(reply_text=reply_text),
    )
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=5462088246),
        callback_query=query,
        message=None,
    )
    context = SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock()))

    asyncio.run(ctrl.handle_callback(update, context))  # type: ignore[arg-type]

    reply_text.assert_awaited_once()
    sent_html = reply_text.call_args.args[0]
    assert "FULL AI &amp; SMC BRIEF" in sent_html
    assert "Macro:" in sent_html
    assert "News Sentiment:" in sent_html

