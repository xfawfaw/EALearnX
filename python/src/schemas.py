"""Pydantic data contracts shared across AI analyst, MT5 bridge, and Telegram bot."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ExecutionMode = Literal["auto", "confirm", "pause"]
Bias = Literal["BULLISH", "BEARISH", "NEUTRAL"]
OrderIntent = Literal["NONE", "MARKET_BUY", "MARKET_SELL", "BUY_LIMIT", "SELL_LIMIT"]
RemoteAction = Literal[
    "CLOSE_ALL",
    "CANCEL_PENDING",
    "MOVE_BE_NOW",
    "PAUSE",
    "RESUME",
    "HOLD_TICKET",
    "UNHOLD_TICKET",
]


class AiTradeDecision(BaseModel):
    """Structured output returned by Gemini 4 Hydrogen."""

    bias: Bias = Field(description="Overall directional bias for XAUUSD.")
    macro_summary: str = Field(
        description="Concise synthesis of Fed rates, DXY, US10Y yields, and macro drivers."
    )
    news_sentiment_score: float = Field(
        ge=-1.0,
        le=1.0,
        description="Gold news sentiment score from -1.0 (strongly bearish) to +1.0 (strongly bullish).",
    )
    news_sentiment_summary: str = Field(
        description="Summary of recent gold/USD news headlines and market sentiment."
    )
    technical_summary: str = Field(
        description="Summary of H4/H1/M15 price structure, key support/resistance, and volatility."
    )
    order_intent: OrderIntent = Field(
        description="Proposed trade action. Use NONE when conditions are unclear or risk/reward < 1:1.5."
    )
    limit_price: float = Field(
        default=0.0,
        ge=0.0,
        description="Exact entry price for BUY_LIMIT or SELL_LIMIT; 0.0 for MARKET_* or NONE.",
    )
    sl_price: float = Field(
        default=0.0,
        ge=0.0,
        description="Stop loss price level (required if order_intent != NONE).",
    )
    tp_price: float = Field(
        default=0.0,
        ge=0.0,
        description="Take profit price level (required if order_intent != NONE).",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0.",
    )
    reason: str = Field(
        description="Short 1-sentence rationale (max 140 chars) displayed on MT5 chart."
    )


class EaSignalPayload(BaseModel):
    """Flat JSON payload written to ai_signal.json for MQL5 JsonLite.mqh."""

    signal_id: str
    generated_at_unix: int
    expires_at_unix: int
    bias: Bias
    order_intent: OrderIntent
    trade_allowed: bool
    limit_price: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    confidence: float = 0.0
    blackout_until_unix: int = 0
    reason: str = ""


class EaCommandPayload(BaseModel):
    """Flat JSON command written to ea_command.json for MQL5 EA."""

    command_id: str
    action: RemoteAction
    issued_at_unix: int
    ticket: int = 0


class EaPosition(BaseModel):
    """Open MT5 position reported by the EA."""

    ticket: int
    type: Literal["BUY", "SELL"]
    volume: float
    open_price: float
    sl: float
    tp: float
    profit: float
    be_applied: bool
    held: bool = False


class EaPendingOrder(BaseModel):
    """Pending MT5 order reported by the EA."""

    ticket: int
    type: str
    volume: float
    price: float
    sl: float
    tp: float


class EaState(BaseModel):
    """Live EA & account snapshot read from ea_state.json."""

    updated_at_unix: int
    symbol: str
    digits: int
    bid: float
    ask: float
    spread_usd: float
    balance: float
    equity: float
    free_margin: float
    daily_pnl_usd: float
    paused: bool
    daily_halt: bool
    hmr_blackout: bool
    last_signal_id: str = ""
    last_command_id: str = ""
    positions: list[EaPosition] = Field(default_factory=list)
    pending_orders: list[EaPendingOrder] = Field(default_factory=list)


class OhlcvBar(BaseModel):
    """Single OHLCV candle exported from MT5."""

    t: int
    o: float
    h: float
    l: float
    c: float
    v: int


class MarketContext(BaseModel):
    """Exness symbol market data snapshot read from market_context.json."""

    updated_at_unix: int
    symbol: str
    bid: float
    ask: float
    spread_usd: float
    atr_h1_14: float
    h4_bars: list[OhlcvBar] = Field(default_factory=list)
    h1_bars: list[OhlcvBar] = Field(default_factory=list)
    m15_bars: list[OhlcvBar] = Field(default_factory=list)


class CalendarEvent(BaseModel):
    """Normalized economic calendar event."""

    title: str
    country: str
    impact: str
    event_time_utc: datetime
    forecast: str = ""
    previous: str = ""


class NewsHeadline(BaseModel):
    """Normalized news headline from RSS/API feeds."""

    title: str
    source: str
    published: str = ""
    link: str = ""


class SessionStatus(BaseModel):
    """Active XAUUSD trading session & kill-zone evaluation."""

    session_name: str
    in_killzone: bool
    market_open: bool
    reason: str


class FvgZone(BaseModel):
    """Unmitigated Fair Value Gap zone."""

    direction: Literal["BULLISH", "BEARISH"]
    low: float
    high: float
    mid: float
    timeframe: str


class OrderBlockZone(BaseModel):
    """Unmitigated institutional Order Block zone."""

    direction: Literal["BULLISH", "BEARISH"]
    low: float
    high: float
    entry_optimal: float
    timeframe: str


class TimeframeStructure(BaseModel):
    """Market structure summary for a single timeframe."""

    timeframe: str
    trend: Literal["BULLISH", "BEARISH", "RANGING"]
    last_event: str  # e.g. BOS_UP, CHOCH_DOWN, NONE
    swing_high: float
    swing_low: float
    premium_discount_ratio: float  # 0.0 = low, 1.0 = high


class AnchoredSetup(BaseModel):
    """Deterministic Limit setup derived from nearest unmitigated OB/FVG."""

    direction: Literal["BUY_LIMIT", "SELL_LIMIT"]
    entry: float
    sl: float
    tp: float
    rr: float
    zone_label: str


class SmcSnapshot(BaseModel):
    """Deterministic Smart Money Concepts state computed from Exness bars."""

    symbol: str
    current_price: float
    atr_h1_14: float
    h4: TimeframeStructure
    h1: TimeframeStructure
    m15: TimeframeStructure
    htf_aligned_bias: Bias
    bullish_obs: list[OrderBlockZone] = Field(default_factory=list)
    bearish_obs: list[OrderBlockZone] = Field(default_factory=list)
    bullish_fvgs: list[FvgZone] = Field(default_factory=list)
    bearish_fvgs: list[FvgZone] = Field(default_factory=list)
    bsl_levels: list[float] = Field(default_factory=list)
    ssl_levels: list[float] = Field(default_factory=list)
    anchored_buy_limit: AnchoredSetup | None = None
    anchored_sell_limit: AnchoredSetup | None = None


class AnalysisReport(BaseModel):
    """Full result bundling AI decision, SMC, session, blackout status, and bridge payload."""

    decision: AiTradeDecision
    signal: EaSignalPayload
    hmr_blackout_active: bool
    hmr_event_title: str | None = None
    session_status: SessionStatus | None = None
    smc_snapshot: SmcSnapshot | None = None
    confluence_score: int = 0
    confluence_notes: list[str] = Field(default_factory=list)
    validation_note: str = ""
    grounded_sources: list[str] = Field(default_factory=list)
