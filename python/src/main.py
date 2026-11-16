"""Main entrypoint for the EA-LearningX Python AI & Telegram control service."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from .analyst import GoldAnalyst
from .bridge import Mt5Bridge
from .config import PROJECT_ROOT, Settings
from .data_feeds import MarketDataFeeds
from .journal import TradeJournal
from .telegram_bot import TelegramController


def _configure_logging() -> None:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "bot.log"

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler = RotatingFileHandler(
        log_file, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(stream_handler)


logger = logging.getLogger("ea_learningx")


def main() -> None:
    """Initialize bridge, feeds, journal, Gemini 4 Hydrogen analyst, and Telegram bot."""
    _configure_logging()
    settings = Settings()
    if not settings.telegram_allowed_user_ids:
        logger.warning(
            "TELEGRAM_ALLOWED_USER_IDS is empty; all Telegram commands will be rejected for safety."
        )

    bridge = Mt5Bridge(settings.bridge_dir)
    bridge.ensure_dir()
    if settings.default_execution_mode == "pause":
        bridge.send_command("PAUSE")
    else:
        bridge.send_command("RESUME")

    journal = TradeJournal(settings.journal_db_path)

    logger.info("MT5 Common Files bridge directory: %s", settings.bridge_dir)
    logger.info("Journal DB: %s", settings.journal_db_path)
    logger.info(
        "Model: %s | Mode: %s | Interval=%dm | KillZone=%s | MinConfluence=%d/5 | MinRR=1:%.1f",
        settings.gemini_model,
        settings.default_execution_mode,
        settings.analysis_interval_minutes,
        settings.require_killzone,
        settings.min_confluence_score,
        settings.min_rr_ratio,
    )

    feeds = MarketDataFeeds(finnhub_api_key=settings.finnhub_api_key)
    analyst = GoldAnalyst(settings=settings, feeds=feeds, journal=journal)
    controller = TelegramController(
        settings=settings,
        bridge=bridge,
        analyst=analyst,
        journal=journal,
    )

    app = controller.build_application()
    logger.info("EA-LearningX Telegram & AI service running. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
