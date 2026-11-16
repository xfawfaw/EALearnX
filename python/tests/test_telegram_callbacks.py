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
    EaPosition,
    EaSignalPayload,
    EaState,
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


def test_analyze_button_continues_even_when_callback_answer_times_out(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path, default_mode="auto")
    market, report = _sample_market_and_report()
    ctrl.analyst.analyze = AsyncMock(return_value=report)
    ctrl.bridge.write_market_context = MagicMock()
    ctrl.bridge.read_market_context = MagicMock(return_value=market)
    ctrl.bridge.read_ea_state = MagicMock(return_value=None)

    progress_msg = SimpleNamespace(delete=AsyncMock())
    reply_text = AsyncMock(return_value=progress_msg)
    query = SimpleNamespace(
        data="menu:analyze",
        answer=AsyncMock(side_effect=TimedOut("Timed out")),
        edit_message_text=AsyncMock(),
        message=SimpleNamespace(reply_text=reply_text),
    )
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=5462088246),
        callback_query=query,
        message=None,
    )
    send_photo_mock = AsyncMock()
    context = SimpleNamespace(bot=SimpleNamespace(send_photo=send_photo_mock, send_message=AsyncMock()))

    asyncio.run(ctrl.handle_callback(update, context))  # type: ignore[arg-type]

    ctrl.analyst.analyze.assert_awaited_once()
    send_photo_mock.assert_awaited_once()


def test_hold_and_unhold_position_buttons_send_ticket_commands(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path, default_mode="auto")
    state = EaState(
        updated_at_unix=1794809005,
        symbol="XAUUSDm",
        digits=3,
        bid=3840.66,
        ask=3840.91,
        spread_usd=0.25,
        balance=1000000.0,
        equity=1040000.0,
        free_margin=1020000.0,
        daily_pnl_usd=40000.0,
        paused=False,
        daily_halt=False,
        hmr_blackout=False,
        positions=[
            EaPosition(
                ticket=5056757525,
                type="BUY",
                volume=0.01,
                open_price=3834.81,
                sl=0.0,
                tp=0.0,
                profit=50000.0,
                be_applied=False,
                held=False,
            )
        ],
    )
    ctrl.bridge.read_ea_state = MagicMock(return_value=state)

    kb = ctrl._status_keyboard(state)
    callbacks = [b.callback_data for row in kb.inline_keyboard for b in row]
    assert "hold:5056757525" in callbacks

    reply_text = AsyncMock()
    query = SimpleNamespace(
        data="hold:5056757525",
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

    cmd_raw = ctrl.bridge.command_path.read_text(encoding="utf-8")
    assert '"HOLD_TICKET"' in cmd_raw
    assert '5056757525' in cmd_raw


def _sample_ml_prediction(mode: str = "M15_PRECISION", verified: bool = True):
    from python.src.schemas import MlPrediction

    return MlPrediction(
        mode=mode,  # type: ignore[arg-type]
        bar_time_unix=1760100000,
        timeframe="M15" if mode == "M15_PRECISION" else "M5",
        regime="TRENDING",
        regime_probs={"TRENDING": 0.81, "RANGING": 0.14, "VOLATILE": 0.05},
        prob_long=0.69,
        prob_short=0.21,
        threshold=0.55,
        order_intent="MARKET_BUY",
        confidence=69,
        entry_price=3883.44,
        sl_price=3876.00,
        tp_price=3895.30,
        max_hold_until_unix=1760114400,
        reason="TRENDING | P(L)=0.69/P(S)=0.21",
        verified_for_live=verified,
    )


def test_strategy_mode_switch_and_unverified_auto_block(tmp_path: Path) -> None:
    settings = Settings(
        gemini_api_key="test",
        telegram_bot_token="123:test",
        telegram_allowed_user_ids=[5462088246],
        mt5_files_dir=tmp_path,
        default_execution_mode="confirm",
    )
    bridge = Mt5Bridge(tmp_path)
    journal = TradeJournal(tmp_path / "journal.db")
    analyst = MagicMock()
    ml_runner = MagicMock()
    ml_runner.is_mode_verified = MagicMock(return_value=False)
    ml_runner.get_bundle = MagicMock(return_value=None)

    ctrl = TelegramController(
        settings=settings,
        bridge=bridge,
        analyst=analyst,
        journal=journal,
        ml_runner=ml_runner,
    )
    assert ctrl.runtime.strategy_mode == "M15_PRECISION"

    def _run_cb(cb_data: str):
        reply_text = AsyncMock()
        q = SimpleNamespace(
            data=cb_data,
            answer=AsyncMock(),
            edit_message_text=AsyncMock(side_effect=BadRequest("Message is not modified")),
            message=SimpleNamespace(reply_text=reply_text),
        )
        upd = SimpleNamespace(
            effective_user=SimpleNamespace(id=5462088246),
            callback_query=q,
            message=None,
        )
        ctx = SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock()))
        asyncio.run(ctrl.handle_callback(upd, ctx))  # type: ignore[arg-type]
        return q, reply_text

    # Switch strategy M15 -> M5 -> M15
    _run_cb("strategy:M5_INTRADAY")
    assert ctrl.runtime.strategy_mode == "M5_INTRADAY"
    _run_cb("strategy:M15_PRECISION")
    assert ctrl.runtime.strategy_mode == "M15_PRECISION"

    # Attempt menu:mode:auto when M15_PRECISION is unverified -> blocked!
    q_auto, reply_auto = _run_cb("menu:mode:auto")
    assert ctrl.runtime.mode != "auto"
    reply_auto.assert_awaited()
    assert "unverified" in reply_auto.call_args.args[0].lower() or "pf" in reply_auto.call_args.args[0].lower()


def test_analyze_now_is_strictly_read_only_and_never_writes_signal(tmp_path: Path) -> None:
    ctrl = _make_controller(tmp_path, default_mode="auto")
    market, report = _sample_market_and_report()
    ctrl.analyst.analyze = AsyncMock(return_value=report)
    ctrl.bridge.read_market_context = MagicMock(return_value=market)
    ctrl.bridge.read_ea_state = MagicMock(return_value=None)
    ctrl.bridge.write_signal = MagicMock()

    context = SimpleNamespace(bot=SimpleNamespace(send_photo=AsyncMock(), send_message=AsyncMock()))
    asyncio.run(ctrl._execute_analysis_cycle(context, manual=True))  # type: ignore[arg-type]

    ctrl.analyst.analyze.assert_awaited_once()
    ctrl.bridge.write_signal.assert_not_called()


def test_chart_card_renders_with_ml_prediction_pills() -> None:
    from python.src.chart_card import render_chart_card

    market, report = _sample_market_and_report()
    pred = _sample_ml_prediction()
    png = render_chart_card(market=market, report=report, ml_prediction=pred)
    assert png is not None
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png) > 5000


def test_approve_confirm_signal_mints_fresh_signal_id_for_mt5(tmp_path: Path) -> None:
    import time as _time

    ctrl = _make_controller(tmp_path, default_mode="confirm")
    orig_id = "ml_m15_precision_1760100000_abc123"
    pending_sig = EaSignalPayload(
        signal_id=orig_id,
        generated_at_unix=int(_time.time()),
        expires_at_unix=int(_time.time()) + 900,
        bias="BULLISH",
        order_intent="MARKET_BUY",
        trade_allowed=True,
        sl_price=3876.0,
        tp_price=3895.0,
        confidence=0.69,
        max_hold_until_unix=int(_time.time()) + 14400,
        strategy_mode="M15_PRECISION",
        regime="TRENDING",
    )
    ctrl.runtime.pending_signals[orig_id] = pending_sig

    reply_text = AsyncMock()
    query = SimpleNamespace(
        data=f"approve:{orig_id}",
        answer=AsyncMock(),
        edit_message_reply_markup=AsyncMock(),
        message=SimpleNamespace(reply_text=reply_text),
    )
    upd = SimpleNamespace(
        effective_user=SimpleNamespace(id=5462088246),
        callback_query=query,
        message=None,
    )
    ctx = SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock()))

    asyncio.run(ctrl.handle_callback(upd, ctx))  # type: ignore[arg-type]

    written = EaSignalPayload.model_validate_json(ctrl.bridge.signal_path.read_text(encoding="utf-8"))
    assert written.trade_allowed is True
    assert written.signal_id != orig_id
    assert written.signal_id.startswith("ml_app_")





