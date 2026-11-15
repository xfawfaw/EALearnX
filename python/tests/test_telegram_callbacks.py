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
from python.src.config import Settings
from python.src.journal import TradeJournal
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
    assert "already AUTO" in reply_text.call_args.args[0]
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
