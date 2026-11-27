"""Configuration management for EA-LearningX."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from .schemas import StrategyMode

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_mt5_common_dir() -> Path:
    """Resolve the default MT5 Common\\Files\\EA-LearningX path on Windows."""
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "MetaQuotes" / "Terminal" / "Common" / "Files" / "EA-LearningX"
    return Path.home() / "AppData" / "Roaming" / "MetaQuotes" / "Terminal" / "Common" / "Files" / "EA-LearningX"


class Settings(BaseSettings):
    """Application settings loaded from .env and environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Gemini AI
    gemini_api_key: str = ""
    gemini_model: str = "gemini-4-hydrogen"
    gemini_fallback_models: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
    )
    gemini_use_search_grounding: bool = False

    # Telegram Bot
    telegram_bot_token: str = ""
    telegram_allowed_user_ids: Annotated[list[int], NoDecode] = Field(default_factory=list)

    # MT5 Bridge & Symbol
    mt5_symbol: str = Field(default="XAUUSDm")
    mt5_common_files_dir: Path | None = None

    # Execution & Risk Controls
    default_execution_mode: Literal["auto", "confirm", "pause"] = "confirm"
    analysis_interval_minutes: int = Field(default=15, ge=5, le=240)
    signal_ttl_minutes: int = Field(default=45, ge=5, le=480)
    min_confidence: float = Field(default=0.70, ge=0.0, le=1.0)
    min_rr_ratio: float = Field(default=1.6, ge=1.0, le=10.0)
    min_confluence_score: int = Field(default=4, ge=1, le=5)
    require_killzone: bool = True
    allow_counter_h4: bool = False

    # ML & HMM / XGBoost Engine
    default_strategy_mode: StrategyMode = "M15_PRECISION"
    ml_models_dir: Path = Field(default=PROJECT_ROOT / "data" / "models")
    ml_spread_cost_usd: float = Field(default=0.30, ge=0.0)
    ml_min_profit_factor: float = Field(default=1.05, ge=1.0)
    ml_min_oos_trades: int = Field(default=20, ge=1)
    ml_m15_bars: int = Field(default=6000, ge=500)
    ml_m5_bars: int = Field(default=12000, ge=1000)
    ml_poll_interval_seconds: int = Field(default=15, ge=1, le=300)
    ml_retrain_hour_utc: int = Field(default=22, ge=0, le=23)
    ml_max_sl_usd: float = Field(default=15.0, gt=0.0)
    ml_cent_lot_min: float = Field(default=0.55, gt=0.0)
    ml_cent_lot_max: float = Field(default=0.90, gt=0.0)

    # Exness HMR Guard
    hmr_pre_news_minutes: int = Field(default=15, ge=0, le=120)
    hmr_post_news_minutes: int = Field(default=10, ge=0, le=120)

    # Journal DB
    journal_db_path: Path = Field(default=PROJECT_ROOT / "data" / "journal.db")

    # Optional external API
    finnhub_api_key: str = ""

    @field_validator("telegram_allowed_user_ids", mode="before")
    @classmethod
    def _parse_allowed_ids(cls, value: object) -> list[int]:
        if value is None or value == "":
            return []
        if isinstance(value, int):
            return [value]
        if isinstance(value, list):
            return [int(v) for v in value]
        if isinstance(value, str):
            return [int(part.strip()) for part in value.split(",") if part.strip()]
        raise ValueError("Invalid TELEGRAM_ALLOWED_USER_IDS format")

    @field_validator("gemini_fallback_models", mode="before")
    @classmethod
    def _parse_fallback_models(cls, value: object) -> list[str]:
        if value is None or value == "":
            return ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        raise ValueError("Invalid GEMINI_FALLBACK_MODELS format")

    @field_validator("mt5_common_files_dir", mode="before")
    @classmethod
    def _parse_bridge_dir(cls, value: object) -> Path | None:
        if value is None or str(value).strip() == "":
            return None
        return Path(str(value).strip())

    @field_validator("ml_models_dir", mode="before")
    @classmethod
    def _parse_models_dir(cls, value: object) -> Path:
        if value is None or str(value).strip() == "":
            return PROJECT_ROOT / "data" / "models"
        return Path(str(value).strip())

    @property
    def bridge_dir(self) -> Path:
        """Return the resolved MT5 Common Files bridge directory."""
        return self.mt5_common_files_dir or default_mt5_common_dir()
