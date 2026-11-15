"""Main entrypoint for the EA-LearningX Python AI & Telegram control service."""

from __future__ import annotations

import logging

from .analyst import GoldAnalyst
from .bridge import Mt5Bridge
from .config import Settings
from .data_feeds import MarketDataFeeds
from .journal import TradeJournal
from .telegram_bot import TelegramController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ea_learningx")


def main() -> None:
    """Initialize bridge, feeds, journal, Gemini 4 Hydrogen analyst, and Telegram bot."""
    settings = Settings()
    if not settings.telegram_allowed_user_ids:
        logger.warning(
            "TELEGRAM_ALLOWED_USER_IDS is empty; all Telegram commands will be rejected for safety."
        )

    bridge = Mt5Bridge(settings.bridge_dir)
    bridge.ensure_dir()
    journal = TradeJournal(settings.journal_db_path)

    logger.info("MT5 Common Files bridge directory: %s", settings.bridge_dir)
    logger.info("Journal DB: %s", settings.journal_db_path)
    logger.info(
        "Model: %s | Mode: %s | KillZone=%s | MinConfluence=%d/5 | MinRR=1:%.1f",
        settings.gemini_model,
        settings.default_execution_mode,
        settings.require_killzone,
        settings.min_confluence_score,
        settings.min_rr_ratio,
    )

    feeds = MarketDataFeeds(finnhub_api_key=settings.finnhub_api_key)
    analyst = GoldAnalyst(settings=settings, feeds=feeds)
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
