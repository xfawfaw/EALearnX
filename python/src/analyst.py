"""Two-stage Gemini 4 Hydrogen analyst + deterministic SMC, session, and confluence gate."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone

from google import genai
from google.genai import types

from .config import Settings
from .data_feeds import MarketDataFeeds, compute_hmr_blackout, format_upcoming_calendar
from .schemas import (
    AiTradeDecision,
    AnalysisReport,
    EaSignalPayload,
    EaState,
    MarketContext,
    OhlcvBar,
    SessionStatus,
    SmcSnapshot,
)
from .sessions import evaluate_session
from .smc import build_smc_snapshot, format_smc_summary

logger = logging.getLogger(__name__)


def _format_bars(bars: list[OhlcvBar], label: str, max_bars: int = 12) -> str:
    if not bars:
        return f"{label}: unavailable"
    subset = bars[-max_bars:]
    lines = [f"{label} (last {len(subset)} closed bars, oldest->newest):"]
    for b in subset:
        ts = datetime.fromtimestamp(b.t, tz=timezone.utc).strftime("%m-%d %H:%M")
        lines.append(f"  {ts} | O={b.o:.2f} H={b.h:.2f} L={b.l:.2f} C={b.c:.2f}")
    return "\n".join(lines)


def _score_confluence(
    decision: AiTradeDecision,
    market: MarketContext,
    smc: SmcSnapshot | None,
    session: SessionStatus,
    is_blackout: bool,
    min_rr_ratio: float,
) -> tuple[int, list[str]]:
    """Score a candidate trade across 5 deterministic institutional factors."""
    want_buy = decision.order_intent in {"MARKET_BUY", "BUY_LIMIT"}
    score = 0
    notes: list[str] = []

    # 1. Session Kill-Zone
    if session.in_killzone:
        score += 1
        notes.append(f"[+1] KillZone: {session.session_name}")
    else:
        notes.append(f"[0] Off-Session: {session.session_name}")

    # 2. HTF Structure Alignment (H4 + H1)
    if smc is not None:
        expected = "BULLISH" if want_buy else "BEARISH"
        if smc.htf_aligned_bias == expected:
            score += 1
            notes.append(f"[+1] HTF Aligned: H4={smc.h4.trend}, H1={smc.h1.trend}")
        elif smc.h4.trend == "RANGING" and smc.h1.trend == expected:
            score += 1
            notes.append(f"[+1] H1 Aligned in H4 Range: H1={smc.h1.trend}")
        else:
            notes.append(f"[0] HTF Misaligned: H4={smc.h4.trend}, H1={smc.h1.trend}")
    else:
        notes.append("[0] SMC unavailable")

    # 3. Macro & News Sentiment Alignment
    sent = decision.news_sentiment_score
    if (want_buy and sent >= 0.15) or (not want_buy and sent <= -0.15):
        score += 1
        notes.append(f"[+1] Sentiment Aligned ({sent:+.2f})")
    else:
        notes.append(f"[0] Weak/Opposing Sentiment ({sent:+.2f})")

    # 4. SMC Zone / Discount-Premium Confluence
    entry_ref = (
        decision.limit_price
        if decision.order_intent in {"BUY_LIMIT", "SELL_LIMIT"}
        else (market.ask if want_buy else market.bid)
    )
    zone_ok = False
    if smc is not None:
        if want_buy:
            in_discount = smc.h1.premium_discount_ratio <= 0.55 or smc.h4.premium_discount_ratio <= 0.55
            near_ob = any(z.low - 2.0 <= entry_ref <= z.high + 2.0 for z in smc.bullish_obs)
            near_fvg = any(z.low - 2.0 <= entry_ref <= z.high + 2.0 for z in smc.bullish_fvgs)
            zone_ok = in_discount and (near_ob or near_fvg or decision.order_intent == "MARKET_BUY")
        else:
            in_premium = smc.h1.premium_discount_ratio >= 0.45 or smc.h4.premium_discount_ratio >= 0.45
            near_ob = any(z.low - 2.0 <= entry_ref <= z.high + 2.0 for z in smc.bearish_obs)
            near_fvg = any(z.low - 2.0 <= entry_ref <= z.high + 2.0 for z in smc.bearish_fvgs)
            zone_ok = in_premium and (near_ob or near_fvg or decision.order_intent == "MARKET_SELL")

    if zone_ok:
        score += 1
        notes.append("[+1] SMC Discount/Premium + Zone Confluence")
    else:
        notes.append("[0] Not anchored to Discount/Premium OB/FVG")

    # 5. Clean R:R & ATR Stop Quality & No HMR Blackout
    risk = abs(entry_ref - decision.sl_price)
    reward = abs(decision.tp_price - entry_ref)
    rr = reward / risk if risk > 0 else 0.0
    atr = market.atr_h1_14 if market.atr_h1_14 > 0 else 8.0
    atr_ok = (0.5 * atr) <= risk <= (3.5 * atr)
    if (not is_blackout) and rr >= min_rr_ratio and atr_ok:
        score += 1
        notes.append(f"[+1] R:R 1:{rr:.2f} & ATR SL OK (${risk:.2f})")
    else:
        notes.append(f"[0] R:R/ATR check failed (rr={rr:.2f}, sl_dist=${risk:.2f}, atr=${atr:.2f})")

    return score, notes


def validate_trade_decision(
    decision: AiTradeDecision,
    market: MarketContext | None,
    ea_state: EaState | None,
    min_confidence: float,
    is_blackout: bool,
    blackout_title: str | None,
    session: SessionStatus | None = None,
    smc: SmcSnapshot | None = None,
    require_killzone: bool = True,
    allow_counter_h4: bool = False,
    min_rr_ratio: float = 1.8,
    min_confluence_score: int = 4,
) -> tuple[AiTradeDecision, bool, str, int, list[str]]:
    """Enforce deterministic safety, session, SMC, R:R, and confluence checks.

    Returns:
        (decision_or_downgraded, trade_allowed, validation_note, confluence_score, confluence_notes)
    """
    def _reject(short_reason: str, note: str, score: int = 0, cnotes: list[str] | None = None):
        downgraded = decision.model_copy(
            update={
                "order_intent": "NONE",
                "limit_price": 0.0,
                "reason": short_reason[:140],
            }
        )
        return downgraded, False, note, score, (cnotes or [])

    if is_blackout:
        return _reject(
            f"HMR blackout: {blackout_title or 'High-impact USD news'}",
            f"Blocked by Exness HMR news blackout ({blackout_title}).",
        )

    if session is not None and not session.market_open:
        return _reject(session.reason, session.reason)

    if session is not None and session.session_name in {"DAILY_ROLLOVER", "FRIDAY_LATE_GUARD"}:
        return _reject(session.reason, session.reason)

    if decision.order_intent == "NONE":
        return decision, False, "Analyst recommended waiting (NONE).", 0, []

    if session is not None and require_killzone and not session.in_killzone:
        return _reject(
            f"Outside kill zone ({session.session_name})",
            f"Blocked: {session.reason}",
        )

    if decision.confidence < min_confidence:
        return _reject(
            f"Low confidence ({decision.confidence:.2f} < {min_confidence:.2f})",
            f"Confidence {decision.confidence:.2f} below threshold {min_confidence:.2f}.",
        )

    if market is None or market.bid <= 0 or market.ask <= 0:
        return _reject("Missing Exness market_context.json", "Missing live Exness market_context.json.")

    want_buy = decision.order_intent in {"MARKET_BUY", "BUY_LIMIT"}

    # Enforce H4 structure alignment unless counter-trend explicitly enabled
    if smc is not None and not allow_counter_h4:
        if want_buy and smc.h4.trend == "BEARISH":
            return _reject(
                "Blocked: Counter-H4 BEARISH trend",
                "Rejected BUY setup against BEARISH H4 structure.",
            )
        if (not want_buy) and smc.h4.trend == "BULLISH":
            return _reject(
                "Blocked: Counter-H4 BULLISH trend",
                "Rejected SELL setup against BULLISH H4 structure.",
            )

    # Prevent duplicate stacking in the same direction
    if ea_state is not None:
        has_same_pos = any(
            (p.type == "BUY" and want_buy) or (p.type == "SELL" and not want_buy)
            for p in ea_state.positions
        )
        has_same_pending = any(
            ("BUY" in o.type and want_buy) or ("SELL" in o.type and not want_buy)
            for o in ea_state.pending_orders
        )
        if has_same_pos or has_same_pending:
            return _reject(
                "Skipped: existing position/pending in same direction",
                "Skipped duplicate order in same direction.",
            )

    bid, ask = market.bid, market.ask
    sl, tp, lim = decision.sl_price, decision.tp_price, decision.limit_price

    if sl <= 0 or tp <= 0:
        return _reject("Invalid SL/TP", "Invalid SL/TP (must be > 0).")

    valid = True
    err = ""
    if decision.order_intent == "MARKET_BUY":
        if not (sl < bid and tp > ask):
            valid, err = False, f"MARKET_BUY invalid SL={sl} TP={tp} vs bid={bid}/ask={ask}"
    elif decision.order_intent == "MARKET_SELL":
        if not (sl > ask and tp < bid):
            valid, err = False, f"MARKET_SELL invalid SL={sl} TP={tp} vs bid={bid}/ask={ask}"
    elif decision.order_intent == "BUY_LIMIT":
        if not (0 < lim < ask and sl < lim < tp):
            valid, err = False, f"BUY_LIMIT invalid limit={lim} SL={sl} TP={tp} vs ask={ask}"
    elif decision.order_intent == "SELL_LIMIT":
        if not (lim > bid and tp < lim < sl):
            valid, err = False, f"SELL_LIMIT invalid limit={lim} SL={sl} TP={tp} vs bid={bid}"

    if not valid:
        return _reject("Invalid geometry", err)

    entry_ref = lim if decision.order_intent in {"BUY_LIMIT", "SELL_LIMIT"} else (ask if want_buy else bid)
    risk = abs(entry_ref - sl)
    reward = abs(tp - entry_ref)
    if risk <= 0 or (reward / risk) < min_rr_ratio:
        return _reject(
            f"R:R below 1:{min_rr_ratio:.1f}",
            f"Rejected setup: risk/reward {reward / max(risk, 1e-6):.2f} < {min_rr_ratio:.2f}.",
        )

    # ATR stop-loss sanity guard
    atr = market.atr_h1_14 if market.atr_h1_14 > 0 else 8.0
    if risk < (0.45 * atr) or risk > (3.5 * atr):
        return _reject(
            f"SL distance ${risk:.2f} out of ATR bounds",
            f"Rejected setup: SL distance ${risk:.2f} outside [0.45x, 3.5x] H1 ATR (${atr:.2f}).",
        )

    eff_session = session or evaluate_session()
    c_score, c_notes = _score_confluence(
        decision=decision,
        market=market,
        smc=smc,
        session=eff_session,
        is_blackout=is_blackout,
        min_rr_ratio=min_rr_ratio,
    )
    required_score = (
        max(3, min_confluence_score - 1)
        if (not require_killzone and not eff_session.in_killzone)
        else min_confluence_score
    )
    if c_score < required_score:
        return _reject(
            f"Confluence {c_score}/5 < {required_score}/5",
            f"Rejected setup: Confluence {c_score}/5 below required {required_score}/5.",
            score=c_score,
            cnotes=c_notes,
        )

    return decision, True, f"Validated setup (Confluence {c_score}/5, R:R 1:{reward / risk:.2f}).", c_score, c_notes


class GoldAnalyst:
    """Runs Gemini 4 Hydrogen macro/news/sentiment + deterministic SMC & session confluence."""

    def __init__(self, settings: Settings, feeds: MarketDataFeeds) -> None:
        self.settings = settings
        self.feeds = feeds
        self._client = (
            genai.Client(api_key=settings.gemini_api_key)
            if settings.gemini_api_key
            else None
        )

    def _run_macro_grounding_sync(
        self,
        now_utc: datetime,
        calendar_text: str,
        headlines_text: str,
    ) -> tuple[str, list[str]]:
        """Stage 1: Gather real-time macro/news/sentiment synthesis with Google Search."""
        assert self._client is not None
        prompt = (
            f"Current UTC time: {now_utc.strftime('%Y-%m-%d %H:%M UTC')}\n"
            "You are a senior institutional XAUUSD (Gold) macroeconomist and news sentiment analyst.\n"
            "Search for the latest real-time developments today on:\n"
            "1. US Dollar Index (DXY), US 10-Year Treasury yields, and Federal Reserve rate expectations.\n"
            "2. Today's breaking gold (XAUUSD) news, geopolitical risk, central bank gold demand, and ETF flows.\n"
            "3. Market reaction to the latest US economic releases.\n\n"
            f"Local Economic Calendar (-4h/+24h):\n{calendar_text}\n\n"
            f"Local RSS Headlines:\n{headlines_text}\n\n"
            "Provide a concise, factual 250-word intelligence brief covering Macro Drivers, News Sentiment, "
            "and Event Risk for Gold right now."
        )

        sources: list[str] = []
        if self.settings.gemini_use_search_grounding:
            try:
                resp = self._client.models.generate_content(
                    model=self.settings.gemini_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearch())],
                    ),
                )
                text = (resp.text or "").strip()
                try:
                    cands = resp.candidates or []
                    if cands and cands[0].grounding_metadata:
                        chunks = cands[0].grounding_metadata.grounding_chunks or []
                        for ch in chunks[:5]:
                            if ch.web and ch.web.title:
                                sources.append(ch.web.title)
                except Exception:  # noqa: BLE001
                    pass
                if text:
                    return text, sources
            except Exception as exc:  # noqa: BLE001
                logger.warning("Google Search grounding fallback to local feeds: %s", exc)

        resp = self._client.models.generate_content(
            model=self.settings.gemini_model,
            contents=prompt,
        )
        return (resp.text or "No macro summary available.").strip(), sources

    def _run_structured_decision_sync(
        self,
        now_utc: datetime,
        macro_brief: str,
        calendar_text: str,
        market: MarketContext | None,
        ea_state: EaState | None,
        smc: SmcSnapshot | None,
        session: SessionStatus,
        is_blackout: bool,
        blackout_title: str | None,
    ) -> AiTradeDecision:
        """Stage 2: Produce strict Pydantic JSON trade decision anchored to deterministic SMC."""
        assert self._client is not None

        if market is not None:
            market_block = (
                f"Exness Symbol: {market.symbol}\n"
                f"Live Bid={market.bid:.2f} | Ask={market.ask:.2f} | Spread=${market.spread_usd:.2f} | H1 ATR(14)=${market.atr_h1_14:.2f}\n\n"
                f"{_format_bars(market.h4_bars, 'H4')}\n\n"
                f"{_format_bars(market.h1_bars, 'H1')}\n\n"
                f"{_format_bars(market.m15_bars, 'M15')}"
            )
        else:
            market_block = "Exness market_context.json not yet exported by MT5 EA."

        smc_block = format_smc_summary(smc) if smc is not None else "SMC snapshot unavailable."

        if ea_state is not None:
            pos_str = (
                ", ".join(
                    f"#{p.ticket} {p.type} {p.volume}@{p.open_price:.2f} SL={p.sl:.2f} TP={p.tp:.2f} PnL={p.profit:+.2f} BE={p.be_applied}"
                    for p in ea_state.positions
                )
                or "None"
            )
            ord_str = (
                ", ".join(
                    f"#{o.ticket} {o.type} {o.volume}@{o.price:.2f} SL={o.sl:.2f} TP={o.tp:.2f}"
                    for o in ea_state.pending_orders
                )
                or "None"
            )
            state_block = (
                f"Balance=${ea_state.balance:.2f} | Equity=${ea_state.equity:.2f} | DailyPnL=${ea_state.daily_pnl_usd:+.2f}\n"
                f"Open Positions: {pos_str}\n"
                f"Pending Orders: {ord_str}"
            )
        else:
            state_block = "EA state unavailable."

        prompt = (
            f"Current UTC: {now_utc.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"Session: {session.session_name} (KillZone={session.in_killzone}, MarketOpen={session.market_open}) - {session.reason}\n"
            f"HMR News Blackout Active: {is_blackout} ({blackout_title or 'None'})\n\n"
            "=== STAGE 1 MACRO & NEWS SENTIMENT BRIEF ===\n"
            f"{macro_brief}\n\n"
            "=== DETERMINISTIC SMART MONEY CONCEPTS (SMC) ===\n"
            f"{smc_block}\n\n"
            "=== UPCOMING USD CALENDAR ===\n"
            f"{calendar_text}\n\n"
            "=== EXNESS MT5 LIVE PRICE ACTION ===\n"
            f"{market_block}\n\n"
            "=== CURRENT ACCOUNT & POSITIONS ===\n"
            f"{state_block}\n\n"
            "HIGH-WIN-RATE RULES:\n"
            "1. Only take A+ setups where Macro/Sentiment, H4/H1 SMC trend, and Liquidity/OB/FVG align.\n"
            "2. For BUY_LIMIT / SELL_LIMIT, snap entry/SL/TP to the Anchored OB/FVG levels above whenever valid.\n"
            "3. Never trade against H4 trend unless structure has a confirmed CHoCH + strong macro catalyst.\n"
            f"4. Minimum Reward-to-Risk is 1:{self.settings.min_rr_ratio:.1f}, and SL distance must be 0.5x-3.5x H1 ATR(14).\n"
            "5. If outside London/NY kill zone, during HMR blackout, or if macro conflicts with SMC, return order_intent='NONE'."
        )

        resp = self._client.models.generate_content(
            model=self.settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AiTradeDecision,
            ),
        )
        if resp.parsed is not None and isinstance(resp.parsed, AiTradeDecision):
            return resp.parsed
        return AiTradeDecision.model_validate_json(resp.text or "{}")

    async def analyze(
        self,
        market: MarketContext | None,
        ea_state: EaState | None,
        publish_allowed: bool,
    ) -> AnalysisReport:
        """Run full macro + news + sentiment + deterministic SMC + confluence pipeline."""
        if self._client is None:
            raise RuntimeError("GEMINI_API_KEY is not configured in .env")

        now_utc = datetime.now(timezone.utc)
        session = evaluate_session(now_utc)
        smc = build_smc_snapshot(market) if market is not None and market.h1_bars else None

        events, headlines = await asyncio.gather(
            self.feeds.fetch_economic_calendar(),
            self.feeds.fetch_gold_news(),
        )

        is_blackout, blackout_until_unix, blackout_title = compute_hmr_blackout(
            events,
            now_utc,
            self.settings.hmr_pre_news_minutes,
            self.settings.hmr_post_news_minutes,
        )
        calendar_text = format_upcoming_calendar(events, now_utc)
        headlines_text = (
            "\n".join(f"- [{h.source}] {h.title}" for h in headlines)
            if headlines
            else "No RSS headlines fetched."
        )

        macro_brief, sources = await asyncio.to_thread(
            self._run_macro_grounding_sync, now_utc, calendar_text, headlines_text
        )
        raw_decision = await asyncio.to_thread(
            self._run_structured_decision_sync,
            now_utc,
            macro_brief,
            calendar_text,
            market,
            ea_state,
            smc,
            session,
            is_blackout,
            blackout_title,
        )

        validated_decision, trade_valid, validation_note, c_score, c_notes = validate_trade_decision(
            raw_decision,
            market,
            ea_state,
            min_confidence=self.settings.min_confidence,
            is_blackout=is_blackout,
            blackout_title=blackout_title,
            session=session,
            smc=smc,
            require_killzone=self.settings.require_killzone,
            allow_counter_h4=self.settings.allow_counter_h4,
            min_rr_ratio=self.settings.min_rr_ratio,
            min_confluence_score=self.settings.min_confluence_score,
        )

        now_unix = int(time.time())
        signal = EaSignalPayload(
            signal_id=uuid.uuid4().hex[:12],
            generated_at_unix=now_unix,
            expires_at_unix=now_unix + (self.settings.signal_ttl_minutes * 60),
            bias=validated_decision.bias,
            order_intent=validated_decision.order_intent,
            trade_allowed=(trade_valid and publish_allowed),
            limit_price=round(validated_decision.limit_price, 2),
            sl_price=round(validated_decision.sl_price, 2),
            tp_price=round(validated_decision.tp_price, 2),
            confidence=round(validated_decision.confidence, 3),
            blackout_until_unix=blackout_until_unix,
            reason=validated_decision.reason[:140],
        )

        return AnalysisReport(
            decision=validated_decision,
            signal=signal,
            hmr_blackout_active=is_blackout,
            hmr_event_title=blackout_title,
            session_status=session,
            smc_snapshot=smc,
            confluence_score=c_score,
            confluence_notes=c_notes,
            validation_note=validation_note,
            grounded_sources=sources,
        )
