"""Unit tests for schemas, bridge atomic I/O, and confluence-gated trade validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python.src.analyst import validate_trade_decision
from python.src.bridge import Mt5Bridge
from python.src.config import Settings
from python.src.schemas import (
    AiTradeDecision,
    EaPosition,
    EaSignalPayload,
    EaState,
    MarketContext,
    OrderBlockZone,
    SessionStatus,
    SmcSnapshot,
    TimeframeStructure,
)


def _sample_market(bid: float = 2650.0, ask: float = 2650.3) -> MarketContext:
    return MarketContext(
        updated_at_unix=1760000000,
        symbol="XAUUSDm",
        bid=bid,
        ask=ask,
        spread_usd=round(ask - bid, 2),
        atr_h1_14=8.5,
    )


def _sample_state() -> EaState:
    return EaState(
        updated_at_unix=1760000000,
        symbol="XAUUSDm",
        digits=2,
        bid=2650.0,
        ask=2650.3,
        spread_usd=0.3,
        balance=1000.0,
        equity=1005.0,
        free_margin=950.0,
        daily_pnl_usd=5.0,
        paused=False,
        daily_halt=False,
        hmr_blackout=False,
    )


def _active_session() -> SessionStatus:
    return SessionStatus(
        session_name="NY_OVERLAP_KILLZONE",
        in_killzone=True,
        market_open=True,
        reason="Active NY Overlap",
    )


def _bullish_smc() -> SmcSnapshot:
    return SmcSnapshot(
        symbol="XAUUSDm",
        current_price=2650.15,
        atr_h1_14=8.5,
        h4=TimeframeStructure(
            timeframe="H4",
            trend="BULLISH",
            last_event="BOS_UP",
            swing_high=2675.0,
            swing_low=2625.0,
            premium_discount_ratio=0.50,
        ),
        h1=TimeframeStructure(
            timeframe="H1",
            trend="BULLISH",
            last_event="BOS_UP",
            swing_high=2665.0,
            swing_low=2635.0,
            premium_discount_ratio=0.45,
        ),
        m15=TimeframeStructure(
            timeframe="M15",
            trend="BULLISH",
            last_event="BOS_UP",
            swing_high=2655.0,
            swing_low=2638.0,
            premium_discount_ratio=0.40,
        ),
        htf_aligned_bias="BULLISH",
        bullish_obs=[
            OrderBlockZone(
                direction="BULLISH",
                low=2640.0,
                high=2643.5,
                entry_optimal=2642.0,
                timeframe="H1",
            )
        ],
    )


def test_bridge_atomic_write_and_read(tmp_path: Path) -> None:
    bridge = Mt5Bridge(tmp_path)
    sig = EaSignalPayload(
        signal_id="sig123",
        generated_at_unix=100,
        expires_at_unix=200,
        bias="BULLISH",
        order_intent="BUY_LIMIT",
        trade_allowed=True,
        limit_price=2642.0,
        sl_price=2632.0,
        tp_price=2662.0,
        confidence=0.82,
        blackout_until_unix=0,
        reason="Pullback to H1 OB + dovish USD macro",
    )
    out_path = bridge.write_signal(sig)
    assert out_path.exists()
    raw = json.loads(out_path.read_text(encoding="utf-8"))
    assert all(not isinstance(v, (dict, list)) for v in raw.values())
    assert raw["order_intent"] == "BUY_LIMIT"

    cmd = bridge.send_command("MOVE_BE_NOW")
    assert cmd.action == "MOVE_BE_NOW"
    assert bridge.command_path.exists()

    bridge.state_path.write_text(_sample_state().model_dump_json(), encoding="utf-8")
    loaded = bridge.read_ea_state()
    assert loaded is not None
    assert loaded.symbol == "XAUUSDm"


def test_validate_buy_limit_valid_and_invalid() -> None:
    market = _sample_market(bid=2650.0, ask=2650.3)
    state = _sample_state()
    sess = _active_session()
    smc = _bullish_smc()

    valid_dec = AiTradeDecision(
        bias="BULLISH",
        macro_summary="Soft US yields support gold",
        news_sentiment_score=0.5,
        news_sentiment_summary="Bullish ETF inflows",
        technical_summary="H1 Bullish OB at 2642",
        order_intent="BUY_LIMIT",
        limit_price=2642.0,
        sl_price=2632.0,
        tp_price=2662.0,
        confidence=0.80,
        reason="Buy limit at H1 demand zone",
    )
    dec, ok, _, score, _ = validate_trade_decision(
        valid_dec, market, state, 0.70, False, None, session=sess, smc=smc
    )
    assert ok is True
    assert score == 5
    assert dec.order_intent == "BUY_LIMIT"

    # Invalid geometry: BUY_LIMIT above ask
    bad_dec = valid_dec.model_copy(update={"limit_price": 2655.0})
    dec2, ok2, note2, _, _ = validate_trade_decision(
        bad_dec, market, state, 0.70, False, None, session=sess, smc=smc
    )
    assert ok2 is False
    assert dec2.order_intent == "NONE"
    assert "BUY_LIMIT invalid" in note2


def test_validate_blocks_hmr_killzone_counter_h4_and_duplicate() -> None:
    market = _sample_market()
    state = _sample_state()
    sess = _active_session()
    smc = _bullish_smc()
    base = AiTradeDecision(
        bias="BULLISH",
        macro_summary="Macro",
        news_sentiment_score=0.4,
        news_sentiment_summary="Positive",
        technical_summary="Uptrend",
        order_intent="MARKET_BUY",
        limit_price=0.0,
        sl_price=2638.0,
        tp_price=2675.0,
        confidence=0.85,
        reason="Breakout",
    )

    # HMR blackout
    dec_hmr, ok_hmr, _, _, _ = validate_trade_decision(
        base, market, state, 0.70, True, "US CPI m/m", session=sess, smc=smc
    )
    assert ok_hmr is False
    assert dec_hmr.order_intent == "NONE"

    # Outside kill zone: blocked when require_killzone=True, allowed when flexible (require_killzone=False)
    off_sess = SessionStatus(
        session_name="ASIAN_SESSION", in_killzone=False, market_open=True, reason="Asian session"
    )
    dec_kz, ok_kz, _, _, _ = validate_trade_decision(
        base, market, state, 0.70, False, None, session=off_sess, smc=smc, require_killzone=True
    )
    assert ok_kz is False
    assert dec_kz.order_intent == "NONE"

    dec_flex, ok_flex, _, score_flex, _ = validate_trade_decision(
        base, market, state, 0.70, False, None, session=off_sess, smc=smc, require_killzone=False
    )
    assert ok_flex is True
    assert score_flex == 4

    # Rollover is always blocked even in flexible mode
    roll_sess = SessionStatus(
        session_name="DAILY_ROLLOVER", in_killzone=False, market_open=True, reason="Rollover"
    )
    dec_roll, ok_roll, _, _, _ = validate_trade_decision(
        base, market, state, 0.70, False, None, session=roll_sess, smc=smc, require_killzone=False
    )
    assert ok_roll is False
    assert dec_roll.order_intent == "NONE"

    # Counter-H4 trend (SELL when H4 is BULLISH)
    sell_dec = base.model_copy(
        update={
            "bias": "BEARISH",
            "news_sentiment_score": -0.5,
            "order_intent": "MARKET_SELL",
            "sl_price": 2662.0,
            "tp_price": 2625.0,
        }
    )
    dec_ct, ok_ct, _, _, _ = validate_trade_decision(
        sell_dec, market, state, 0.70, False, None, session=sess, smc=smc, allow_counter_h4=False
    )
    assert ok_ct is False
    assert dec_ct.order_intent == "NONE"

    # Duplicate BUY position already open
    state_with_buy = state.model_copy(
        update={
            "positions": [
                EaPosition(
                    ticket=101,
                    type="BUY",
                    volume=0.05,
                    open_price=2645.0,
                    sl=2635.0,
                    tp=2670.0,
                    profit=25.0,
                    be_applied=False,
                )
            ]
        }
    )
    dec_dup, ok_dup, _, _, _ = validate_trade_decision(
        base, market, state_with_buy, 0.70, False, None, session=sess, smc=smc
    )
    assert ok_dup is False
    assert dec_dup.order_intent == "NONE"


def test_settings_parses_allowed_user_ids_from_dotenv(tmp_path: Path) -> None:
    env_empty = tmp_path / ".env.empty"
    env_empty.write_text("TELEGRAM_ALLOWED_USER_IDS=\n", encoding="utf-8")
    assert Settings(_env_file=str(env_empty)).telegram_allowed_user_ids == []

    env_multi = tmp_path / ".env.multi"
    env_multi.write_text("TELEGRAM_ALLOWED_USER_IDS=111111, 222222\n", encoding="utf-8")
    assert Settings(_env_file=str(env_multi)).telegram_allowed_user_ids == [111111, 222222]
