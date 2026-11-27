"""Telegram bot remote controller & real-time alert notifier for Android."""

from __future__ import annotations

import asyncio
import html
import io
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import time as dt_time, timezone

from telegram import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest, NetworkError, TimedOut
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

from .analyst import GoldAnalyst
from .bridge import Mt5Bridge
from .chart_card import render_analysis_chart
from .config import Settings
from .journal import TradeJournal
from .ml_engine import MlSignalRunner
from .ml_trainer import load_bundle
from .schemas import (
    AnalysisReport,
    EaSignalPayload,
    EaState,
    ExecutionMode,
    MarketContext,
    MlPrediction,
    StrategyMode,
)
from .sessions import evaluate_session
from .smc import build_smc_snapshot, format_smc_summary

logger = logging.getLogger(__name__)


@dataclass
class BotRuntimeState:
    """Mutable runtime state shared across Telegram handlers and scheduled jobs."""

    mode: ExecutionMode
    strategy_mode: StrategyMode = "M15_PRECISION"
    pending_signals: dict[str, EaSignalPayload] = field(default_factory=dict)
    known_position_tickets: set[int] = field(default_factory=set)
    known_be_tickets: set[int] = field(default_factory=set)
    known_pending_tickets: set[int] = field(default_factory=set)
    state_initialized: bool = False
    trade_alerts_enabled: bool = True
    quota_alert_sent: bool = False
    last_report: AnalysisReport | None = None
    last_ml_prediction: MlPrediction | None = None
    account_currency: str = "USD"


class TelegramController:
    """Manages Telegram commands, inline approvals, scheduled analysis, and alerts."""

    def __init__(
        self,
        settings: Settings,
        bridge: Mt5Bridge,
        analyst: GoldAnalyst,
        journal: TradeJournal,
        ml_runner: MlSignalRunner | None = None,
    ) -> None:
        self.settings = settings
        self.bridge = bridge
        self.analyst = analyst
        self.journal = journal
        self.ml_runner = ml_runner
        initial_mode: ExecutionMode = settings.default_execution_mode
        if (
            ml_runner is not None
            and initial_mode == "auto"
            and not ml_runner.is_mode_verified(settings.default_strategy_mode)
        ):
            initial_mode = "confirm"
        self.runtime = BotRuntimeState(
            mode=initial_mode,
            strategy_mode=settings.default_strategy_mode,
        )

    def _is_authorized(self, update: Update) -> bool:
        user = update.effective_user
        if user is None or not self.settings.telegram_allowed_user_ids:
            return False
        return user.id in self.settings.telegram_allowed_user_ids

    async def _reject_unauthorized(self, update: Update) -> None:
        if update.callback_query:
            await update.callback_query.answer("Unauthorized user.", show_alert=True)
        elif update.message:
            uid = update.effective_user.id if update.effective_user else "unknown"
            await update.message.reply_text(
                f"Unauthorized. Add your Telegram ID ({uid}) to TELEGRAM_ALLOWED_USER_IDS in .env."
            )

    async def _broadcast(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        text: str,
        reply_markup: InlineKeyboardMarkup | None = None,
        parse_mode: str | None = ParseMode.HTML,
    ) -> None:
        chunks = [text[i : i + 4000] for i in range(0, len(text), 4000)] or [""]
        for chat_id in self.settings.telegram_allowed_user_ids:
            for idx, chunk in enumerate(chunks):
                markup = reply_markup if idx == len(chunks) - 1 else None
                for attempt in range(3):
                    try:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=chunk,
                            reply_markup=markup,
                            parse_mode=parse_mode,
                        )
                        break
                    except (TimedOut, NetworkError) as exc:
                        if attempt < 2:
                            logger.warning(
                                "Transient Telegram send error to %s (retry %d): %s",
                                chat_id,
                                attempt + 1,
                                exc,
                            )
                            await asyncio.sleep(1.5 * (attempt + 1))
                            continue
                        logger.warning("Failed sending Telegram message to %s: %s", chat_id, exc)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Failed sending Telegram message to %s: %s", chat_id, exc)
                        break

    async def _broadcast_photo(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        photo_bytes: bytes,
        caption: str,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> bool:
        """Send rendered chart card PNG with HTML caption to all allowed users."""
        safe_caption = caption[:1020]
        sent_any = False
        for chat_id in self.settings.telegram_allowed_user_ids:
            for attempt in range(3):
                try:
                    await context.bot.send_photo(
                        chat_id=chat_id,
                        photo=io.BytesIO(photo_bytes),
                        caption=safe_caption,
                        reply_markup=reply_markup,
                        parse_mode=ParseMode.HTML,
                    )
                    sent_any = True
                    break
                except (TimedOut, NetworkError) as exc:
                    if attempt < 2:
                        logger.warning(
                            "Transient Telegram photo send error to %s (retry %d): %s",
                            chat_id,
                            attempt + 1,
                            exc,
                        )
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    logger.warning("Failed sending Telegram photo to %s: %s", chat_id, exc)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed sending Telegram photo to %s: %s", chat_id, exc)
                    break
        return sent_any

    async def _safe_answer(
        self,
        query: CallbackQuery,
        text: str | None = None,
        show_alert: bool = False,
    ) -> None:
        """Answer a Telegram callback query without letting transient timeouts abort the action."""
        try:
            await query.answer(text, show_alert=show_alert)
        except (TimedOut, NetworkError, BadRequest) as exc:
            logger.warning("Callback answer skipped (%s): %s", query.data, exc)

    @staticmethod
    def _format_money(amount: float, currency: str = "USD", signed: bool = False) -> str:
        """Format account currency values dynamically (e.g., IDR -> Rp 1,551,503, USD -> $1,551.50)."""
        ccy = (currency or "USD").strip().upper()
        sign = ""
        if signed:
            sign = "+" if amount >= 0 else "-"
        elif amount < 0:
            sign = "-"
        mag = abs(amount)
        if ccy == "IDR":
            return f"{sign}Rp {mag:,.0f}"
        if ccy == "USD":
            return f"{sign}${mag:,.2f}"
        return f"{sign}{mag:,.2f} {ccy}"

    def _resolve_account_currency(self, state: EaState | None = None) -> str:
        """Return live Exness account currency from state snapshot (cached in runtime)."""
        st = state if state is not None else self.bridge.read_ea_state()
        if st is not None and st.currency:
            self.runtime.account_currency = st.currency.strip().upper()
        return self.runtime.account_currency

    @staticmethod
    def _meter_bar(ratio: float, width: int = 8) -> str:
        clamped = max(0.0, min(1.0, ratio))
        filled = int(round(clamped * width))
        return ("▰" * filled) + ("▱" * (width - filled))

    @staticmethod
    def _bias_badge(bias: str) -> str:
        if bias == "BULLISH":
            return "▲ BULLISH"
        if bias == "BEARISH":
            return "▼ BEARISH"
        return "◇ NEUTRAL"

    def _format_status(self, state: EaState | None) -> str:
        session = evaluate_session()
        kz_tag = "▪ KILLZONE" if session.in_killzone else f"▫ {html.escape(session.session_name)}"
        filter_tag = "KillZone" if self.settings.require_killzone else "24/5"
        mode_icon = {"auto": "■ AUTO", "confirm": "◇ CONFIRM", "pause": "▫ PAUSE"}.get(
            self.runtime.mode, self.runtime.mode.upper()
        )
        if state is None:
            return (
                "<b>■ EA-LEARNINGX STATUS</b>\n"
                "──────────────────\n"
                f"<b>MODE</b>     <code>{mode_icon}</code> · <code>{filter_tag}</code>\n"
                f"<b>SESSION</b>  <code>{kz_tag}</code>\n\n"
                "<i>MT5 offline. Open Exness MT5 &amp; attach EA_LearningX_Gold.</i>"
            )

        age = int(time.time()) - state.updated_at_unix
        flags = []
        if state.paused:
            flags.append("▫ PAUSED")
        if state.daily_halt:
            flags.append("× DAILY_HALT")
        if state.hmr_blackout:
            flags.append("× HMR_BLACKOUT")
        flag_str = " · ".join(flags) if flags else "■ ACTIVE"
        stale_warn = "  <i>(MT5 stale)</i>" if age > 60 else ""
        pnl_dir = "▲" if state.daily_pnl_usd >= 0 else "▼"
        ccy = self._resolve_account_currency(state)

        lines = [
            f"<b>■ EA-LEARNINGX</b>  <code>{html.escape(state.symbol)}</code>",
            "──────────────────",
            f"<b>MODE</b>     <code>{mode_icon}</code>  |  <b>EA</b> <code>{flag_str}</code>",
            f"<b>SYNC</b>     <code>{age}s ago</code>{stale_warn}",
            f"<b>SESSION</b>  <code>{kz_tag}</code> <i>({filter_tag})</i>",
            "",
            f"<b>PRICE</b>    <code>{state.bid:.2f} / {state.ask:.2f}</code>  <i>(sp ${state.spread_usd:.2f})</i>",
            f"<b>BALANCE</b>  <code>{self._format_money(state.balance, ccy)}</code>  |  <b>EQ</b> <code>{self._format_money(state.equity, ccy)}</code>",
            f"<b>DAY PNL</b>  {pnl_dir} <code>{self._format_money(state.daily_pnl_usd, ccy, signed=True)}</code>  |  <b>FREE</b> <code>{self._format_money(state.free_margin, ccy)}</code>",
            "",
            f"<b>POSITIONS ({len(state.positions)})</b>",
        ]
        if not state.positions:
            lines.append("   <i>None</i>")
        for p in state.positions:
            be_tag = " [BE]" if p.be_applied else ""
            hold_tag = " [KEPT]" if p.held else " [ML]"
            p_dir = "▲" if p.profit >= 0 else "▼"
            lines.append(
                f"   {p_dir} <code>#{p.ticket} {p.type} {p.volume:.2f} @ {p.open_price:.2f}</code>{hold_tag}\n"
                f"      SL <code>{p.sl:.2f}</code> · TP <code>{p.tp:.2f}</code> · <b>{self._format_money(p.profit, ccy, signed=True)}</b>{be_tag}"
            )

        lines.append(f"\n<b>PENDING ({len(state.pending_orders)})</b>")
        if not state.pending_orders:
            lines.append("   <i>None</i>")
        for o in state.pending_orders:
            lines.append(
                f"   ▪ <code>#{o.ticket} {html.escape(o.type)} {o.volume:.2f} @ {o.price:.2f}</code>\n"
                f"      SL <code>{o.sl:.2f}</code> · TP <code>{o.tp:.2f}</code>"
            )
        if self.runtime.last_report is not None:
            r = self.runtime.last_report
            lines.append(
                f"\n<b>LAST AI</b>  {self._bias_badge(r.decision.bias)} · <code>{html.escape(r.decision.order_intent)}</code> "
                f"({r.confluence_score}/5) — <i>{html.escape(r.validation_note[:75])}</i>"
            )
        return "\n".join(lines)

    def _format_analysis_caption(self, report: AnalysisReport, badge: str) -> str:
        """Monochrome compact HTML card (fits Telegram 1024-char caption cleanly)."""
        d = report.decision
        s = report.signal
        sess_txt = html.escape(report.session_status.session_name if report.session_status else "UNKNOWN")
        conf_bar = self._meter_bar(d.confidence, 8)
        score_bar = self._meter_bar(report.confluence_score / 5.0, 5)
        bias_badge = self._bias_badge(d.bias)
        exec_badge = "✓ ALLOWED" if s.trade_allowed else "× WAIT"

        lines = [
            f"<b>{html.escape(badge)}</b>",
            "──────────────────",
            f"{bias_badge}  ·  <code>{html.escape(d.order_intent)}</code>  (<code>{exec_badge}</code>)",
            f"<b>CONF</b>     <code>{conf_bar}</code> <b>{d.confidence:.0%}</b>  |  <b>SCORE</b> <code>{score_bar}</code> <b>{report.confluence_score}/5</b>",
            f"<b>SESSION</b>  <code>{sess_txt}</code>  |  <b>SENT</b> <code>{d.news_sentiment_score:+.2f}</code>",
        ]
        if d.order_intent != "NONE":
            entry_txt = f"{d.limit_price:.2f}" if d.limit_price > 0 else "MARKET"
            lines.append(
                f"<b>SETUP</b>    <code>{entry_txt}</code>  |  <b>SL</b> <code>{d.sl_price:.2f}</code>  |  <b>TP</b> <code>{d.tp_price:.2f}</code>"
            )
        if report.smc_snapshot is not None:
            smc = report.smc_snapshot
            lines.append(
                f"<b>SMC</b>      H4 <code>{smc.h4.trend}</code> · H1 <code>{smc.h1.trend}</code> · M15 <code>{smc.m15.trend}</code> · P/D <code>{smc.h1.premium_discount_ratio:.2f}</code>"
            )
        if report.hmr_blackout_active:
            lines.append(f"<b>HMR</b>      × <code>{html.escape(report.hmr_event_title or 'Active')}</code>")

        lines.append(f"\n<b>NOTE</b>  <i>{html.escape(report.validation_note[:120])}</i>")
        body_no_quote = "\n".join(lines)
        remaining = 960 - len(body_no_quote)
        if remaining > 40:
            raw_snippet = d.technical_summary.strip()
            if len(raw_snippet) > remaining - 30:
                raw_snippet = raw_snippet[: max(10, remaining - 33)] + "..."
            lines.append(
                f"<blockquote expandable><b>READ:</b> {html.escape(raw_snippet)}</blockquote>"
            )
        return "\n".join(lines)

    def _format_report(self, report: AnalysisReport, prefix: str = "") -> str:
        """Full monochrome HTML brief with expandable Macro/News/Technical sections."""
        d = report.decision
        s = report.signal
        sess_txt = html.escape(report.session_status.session_name if report.session_status else "UNKNOWN")
        conf_bar = self._meter_bar(d.confidence, 8)
        score_bar = self._meter_bar(report.confluence_score / 5.0, 5)
        exec_badge = "✓ ALLOWED" if s.trade_allowed else "× WAIT"

        lines = []
        if prefix:
            lines.append(f"<b>{html.escape(prefix)}</b>")
            lines.append("──────────────────")
        lines.extend([
            f"{self._bias_badge(d.bias)}  ·  <code>{html.escape(d.order_intent)}</code>  (<code>{exec_badge}</code>)",
            f"<b>CONFIDENCE</b>  <code>{conf_bar}</code> <b>{d.confidence:.0%}</b>",
            f"<b>CONFLUENCE</b>  <code>{score_bar}</code> <b>{report.confluence_score}/5</b>  |  <b>SENT</b> <code>{d.news_sentiment_score:+.2f}</code>",
            f"<b>SESSION</b>     <code>{sess_txt}</code>",
        ])
        if d.order_intent != "NONE":
            entry_txt = f"{d.limit_price:.2f}" if d.limit_price > 0 else "MARKET"
            lines.append(
                f"<b>ENTRY</b> <code>{entry_txt}</code>  |  <b>SL</b> <code>{d.sl_price:.2f}</code>  |  <b>TP</b> <code>{d.tp_price:.2f}</code>"
            )
        if report.smc_snapshot is not None:
            smc = report.smc_snapshot
            lines.append(
                f"<b>SMC</b>   H4 <code>{smc.h4.trend}({smc.h4.last_event})</code> · "
                f"H1 <code>{smc.h1.trend}({smc.h1.last_event})</code> · "
                f"M15 <code>{smc.m15.trend}</code> · P/D <code>{smc.h1.premium_discount_ratio:.2f}</code>"
            )
        if report.hmr_blackout_active:
            lines.append(f"<b>HMR GUARD</b>  × BLACKOUT ({html.escape(report.hmr_event_title or '')})")

        lines.append(f"\n<b>Validation:</b> <i>{html.escape(report.validation_note)}</i>")
        lines.append(f"<blockquote expandable><b>Macro:</b> {html.escape(d.macro_summary)}</blockquote>")
        lines.append(f"<blockquote expandable><b>News Sentiment:</b> {html.escape(d.news_sentiment_summary)}</blockquote>")
        lines.append(f"<blockquote expandable><b>Technical:</b> {html.escape(d.technical_summary)}</blockquote>")
        if report.confluence_notes:
            notes_txt = "\n".join(f"▪ {html.escape(n)}" for n in report.confluence_notes)
            lines.append(f"<blockquote expandable><b>Checklist:</b>\n{notes_txt}</blockquote>")
        if report.grounded_sources:
            lines.append(f"<b>Sources:</b> <i>{html.escape(', '.join(report.grounded_sources[:3]))}</i>")
        return "\n".join(lines)

    def _analysis_keyboard(self, pending_signal_id: str | None = None) -> InlineKeyboardMarkup:
        rows: list[list[InlineKeyboardButton]] = []
        if pending_signal_id:
            rows.append([
                InlineKeyboardButton("✓ Approve & Execute", callback_data=f"approve:{pending_signal_id}"),
                InlineKeyboardButton("× Reject", callback_data=f"reject:{pending_signal_id}"),
            ])
        rows.append([
            InlineKeyboardButton("▪ Full Brief", callback_data="menu:brief"),
            InlineKeyboardButton("▪ SMC Levels", callback_data="menu:smc_levels"),
        ])
        rows.append([
            InlineKeyboardButton("↻ Re-Analyze", callback_data="menu:analyze"),
            InlineKeyboardButton("▪ Status", callback_data="menu:status"),
            InlineKeyboardButton("■ Panel", callback_data="menu:panel"),
        ])
        return InlineKeyboardMarkup(rows)

    def _format_menu_text(self) -> str:
        mode_map = {"auto": "■ AUTO", "confirm": "◇ CONFIRM", "pause": "▫ PAUSED"}
        mode_badge = mode_map.get(self.runtime.mode, self.runtime.mode.upper())
        strat_label = (
            "M15 Precision [B2]" if self.runtime.strategy_mode == "M15_PRECISION" else "M5 Intraday [B2]"
        )
        if self.ml_runner is not None:
            ver_tag = (
                "✓ Verified"
                if self.ml_runner.is_mode_verified(self.runtime.strategy_mode)
                else "× Unverified"
            )
        else:
            ver_tag = "▫ ML Offline"
        hours = "▪ KillZone Only" if self.settings.require_killzone else "◇ 24/5 Flexible"
        sess = evaluate_session()
        state = self.bridge.read_ea_state()
        if state is None:
            mt5_txt = "▫ Offline"
        else:
            age = int(time.time()) - state.updated_at_unix
            mt5_txt = "■ Live" if age <= 60 and not state.paused else ("▫ Paused" if age <= 60 else "× MT5 Closed")
        sym_txt = html.escape(state.symbol if state is not None else self.settings.mt5_symbol)
        return (
            f"<b>■ EA-LEARNINGX PANEL</b>  <code>{sym_txt} · B2</code>\n"
            "──────────────────\n"
            f"<b>MODE</b>      <code>{mode_badge}</code>\n"
            f"<b>STRATEGY</b>  <code>{strat_label}</code>  <i>({ver_tag})</i>\n"
            f"<b>HOURS</b>     <code>{hours}</code>\n"
            f"<b>MARKET</b>    <code>{html.escape(sess.session_name)}</code>  ·  <b>MT5</b> <code>{mt5_txt}</code>"
        )

    def _build_menu_keyboard(self) -> InlineKeyboardMarkup:
        def _mode_label(mode_val: ExecutionMode, label: str) -> str:
            return f"■ {label}" if self.runtime.mode == mode_val else label

        def _strat_label(strat_val: StrategyMode, label: str) -> str:
            return f"■ {label}" if self.runtime.strategy_mode == strat_val else label

        kz_label = "Hours: KillZone" if self.settings.require_killzone else "Hours: 24/5"
        alerts_label = "Alerts: ON" if self.runtime.trade_alerts_enabled else "Alerts: OFF"
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton(_mode_label("auto", "Auto"), callback_data="menu:mode:auto"),
                InlineKeyboardButton(_mode_label("confirm", "Confirm"), callback_data="menu:mode:confirm"),
                InlineKeyboardButton(_mode_label("pause", "Stop"), callback_data="menu:mode:pause"),
            ],
            [
                InlineKeyboardButton(_strat_label("M15_PRECISION", "M15 Precision"), callback_data="strategy:M15_PRECISION"),
                InlineKeyboardButton(_strat_label("M5_INTRADAY", "M5 Intraday"), callback_data="strategy:M5_INTRADAY"),
            ],
            [
                InlineKeyboardButton(kz_label, callback_data="menu:toggle_kz"),
                InlineKeyboardButton("Analyze Now", callback_data="menu:analyze"),
            ],
            [
                InlineKeyboardButton("ML Models", callback_data="menu:ml_models"),
                InlineKeyboardButton("Retrain ML", callback_data="ml:retrain"),
            ],
            [
                InlineKeyboardButton("Status", callback_data="menu:status"),
                InlineKeyboardButton("SMC Chart", callback_data="menu:smc"),
                InlineKeyboardButton("Stats", callback_data="menu:stats"),
            ],
            [
                InlineKeyboardButton("Move BE", callback_data="menu:be"),
                InlineKeyboardButton(alerts_label, callback_data="menu:toggle_alerts"),
                InlineKeyboardButton("Close All", callback_data="menu:close_all"),
            ],
        ])

    def _format_ml_models_html(self) -> str:
        lines = [
            "<b>■ EA-LEARNINGX ML BUNDLES</b>",
            "──────────────────",
            f"<b>ACTIVE</b>  <code>{self.runtime.strategy_mode}</code>",
        ]
        for mode_name in ("M15_PRECISION", "M5_INTRADAY"):
            bundle = (
                self.ml_runner.get_bundle(mode_name)  # type: ignore[arg-type]
                if self.ml_runner is not None
                else load_bundle(mode_name, self.settings.ml_models_dir)  # type: ignore[arg-type]
            )
            tag = "▪" if mode_name == "M15_PRECISION" else "▫"
            if bundle is None:
                lines.append(f"\n{tag} <b>{mode_name}:</b> <i>Not trained</i>")
                continue
            m = bundle.metrics
            ver_badge = "✓ VERIFIED" if m.verified_for_live else "× UNVERIFIED"
            lines.append(
                f"\n{tag} <b>{mode_name}</b>  <code>{ver_badge}</code>\n"
                f"   Trades: <code>{m.oos_trades}</code> · WR: <code>{m.oos_win_rate * 100:.1f}%</code> · PF: <code>{m.oos_profit_factor:.2f}</code>\n"
                f"   Net: <code>${m.oos_net_pnl_usd:+.2f} (@0.02 split)</code> · Th: <code>{m.chosen_threshold:.2f}</code>"
            )
            if m.fail_reason:
                lines.append(f"   Note: <i>{html.escape(m.fail_reason)}</i>")
        if self.runtime.last_ml_prediction is not None:
            p = self.runtime.last_ml_prediction
            lines.append(
                f"\n<b>LAST BAR</b>  <code>{p.timeframe} · HMM {p.regime} · {p.order_intent}</code> "
                f"(L={p.prob_long:.2f}, S={p.prob_short:.2f})"
            )
        return "\n".join(lines)

    def _ml_models_keyboard(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("M15 Precision", callback_data="strategy:M15_PRECISION"),
                InlineKeyboardButton("M5 Intraday", callback_data="strategy:M5_INTRADAY"),
            ],
            [
                InlineKeyboardButton("↻ Retrain Active ML", callback_data="ml:retrain"),
                InlineKeyboardButton("■ Panel", callback_data="menu:panel"),
            ],
        ])

    async def _edit_menu_safely(self, query: CallbackQuery, unchanged_notice: str) -> None:
        """Edit the inline control panel, ignoring no-op edits with clear user feedback."""
        try:
            await query.edit_message_text(
                self._format_menu_text(),
                reply_markup=self._build_menu_keyboard(),
                parse_mode=ParseMode.HTML,
            )
        except BadRequest as exc:
            if "message is not modified" in str(exc).lower():
                if query.message is not None:
                    await query.message.reply_text(unchanged_notice, parse_mode=ParseMode.HTML)
                return
            raise

    def _format_stats_html(self) -> str:
        st = self.journal.get_stats()
        ccy = self._resolve_account_currency()
        wr_bar = self._meter_bar(st.win_rate_pct / 100.0, 10)
        pnl_dir = "▲" if st.net_realized_pnl >= 0 else "▼"
        pf_txt = "∞" if st.profit_factor >= 99.0 else f"{st.profit_factor:.2f}"
        return (
            "<b>■ PERFORMANCE JOURNAL</b>\n"
            "──────────────────\n"
            f"<b>ANALYSES</b>  <code>{st.total_analyses}</code>  ·  <b>SETUPS</b> <code>{st.actionable_signals}</code>\n"
            f"<b>TRADES</b>    <code>{st.trades_opened}</code> open · <code>{st.trades_closed}</code> closed\n"
            f"<b>BE LOCKS</b>  <code>{st.be_triggered_count}</code>\n"
            f"<b>WIN RATE</b>  <code>{wr_bar}</code> <b>{st.win_rate_pct:.1f}%</b> ({st.wins}W / {st.losses}L)\n"
            f"<b>PF</b>        <code>{pf_txt}</code>  |  <b>REALIZED</b> {pnl_dir} <code>{self._format_money(st.net_realized_pnl, ccy, signed=True)}</code>"
        )

    def _status_keyboard(self, state: EaState | None = None) -> InlineKeyboardMarkup:
        rows: list[list[InlineKeyboardButton]] = []
        if state is not None and state.positions:
            for p in state.positions[:4]:
                if p.held:
                    rows.append([
                        InlineKeyboardButton(
                            f"□ Release #{p.ticket}",
                            callback_data=f"unhold:{p.ticket}",
                        )
                    ])
                else:
                    rows.append([
                        InlineKeyboardButton(
                            f"■ Keep #{p.ticket} (Hold Safe)",
                            callback_data=f"hold:{p.ticket}",
                        )
                    ])
        rows.extend([
            [
                InlineKeyboardButton("↻ Refresh", callback_data="menu:status"),
                InlineKeyboardButton("▪ Analyze Now", callback_data="menu:analyze"),
            ],
            [
                InlineKeyboardButton("▪ Move BE", callback_data="menu:be"),
                InlineKeyboardButton("■ Panel", callback_data="menu:panel"),
            ],
        ])
        return InlineKeyboardMarkup(rows)

    def _smc_keyboard(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("↻ Refresh SMC", callback_data="menu:smc"),
                InlineKeyboardButton("▪ Analyze Now", callback_data="menu:analyze"),
            ],
            [
                InlineKeyboardButton("▪ Status", callback_data="menu:status"),
                InlineKeyboardButton("■ Panel", callback_data="menu:panel"),
            ],
        ])

    def _stats_keyboard(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("↻ Refresh", callback_data="menu:stats"),
                InlineKeyboardButton("▪ Analyze Now", callback_data="menu:analyze"),
                InlineKeyboardButton("■ Panel", callback_data="menu:panel"),
            ]
        ])

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return
        assert update.message is not None
        await update.message.reply_text(
            self._format_menu_text(),
            reply_markup=self._build_menu_keyboard(),
            parse_mode=ParseMode.HTML,
        )

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return
        state = self.bridge.read_ea_state()
        assert update.message is not None
        await update.message.reply_text(
            self._format_status(state),
            reply_markup=self._status_keyboard(state),
            parse_mode=ParseMode.HTML,
        )

    async def _send_smc_card(self, chat_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
        session = evaluate_session()
        market = self.bridge.read_market_context()
        if market is None or not market.h1_bars:
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    f"🧱 <b>SMC Structure</b>\n"
                    f"🧭 <b>Session:</b> <code>{html.escape(session.session_name)}</code> ({html.escape(session.reason)})\n\n"
                    "⚠️ <i>No market_context.json bars exported yet by MT5 EA.</i>"
                ),
                reply_markup=self._smc_keyboard(),
                parse_mode=ParseMode.HTML,
            )
            return

        smc = build_smc_snapshot(market)
        chart_png = await asyncio.to_thread(render_analysis_chart, market, None, smc, None)
        smc_summary = html.escape(format_smc_summary(smc))
        if len(smc_summary) > 780:
            smc_summary = smc_summary[:777] + "..."
        caption = (
            f"<b>■ XAUUSD SMC MAP</b>  ·  <code>{html.escape(session.session_name)}</code>\n"
            f"──────────────────\n"
            f"H4 <b>{smc.h4.trend}</b> · H1 <b>{smc.h1.trend}</b> · M15 <b>{smc.m15.trend}</b> · P/D <code>{smc.h1.premium_discount_ratio:.2f}</code>\n"
            f"<blockquote expandable>{smc_summary}</blockquote>"
        )
        if chart_png is not None:
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=io.BytesIO(chart_png),
                caption=caption,
                reply_markup=self._smc_keyboard(),
                parse_mode=ParseMode.HTML,
            )
            return
        await context.bot.send_message(
            chat_id=chat_id,
            text=caption,
            reply_markup=self._smc_keyboard(),
            parse_mode=ParseMode.HTML,
        )

    async def cmd_smc(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return
        assert update.message is not None
        await self._send_smc_card(update.message.chat_id, context)

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return
        assert update.message is not None
        await update.message.reply_text(
            self._format_stats_html(),
            reply_markup=self._stats_keyboard(),
            parse_mode=ParseMode.HTML,
        )

    async def cmd_session(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return
        assert update.message is not None
        args = context.args or []
        if not args or args[0].lower() not in {"on", "off"}:
            await update.message.reply_text(
                f"Kill-Zone filter is currently: {'ON' if self.settings.require_killzone else 'OFF'}\n"
                "Usage: /session on | /session off"
            )
            return
        self.settings.require_killzone = args[0].lower() == "on"
        await update.message.reply_text(
            f"Kill-Zone filter set to: {'ON (London/NY only)' if self.settings.require_killzone else 'OFF (24/5 allowed)'}"
        )

    async def cmd_mode(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return
        assert update.message is not None
        args = context.args or []
        if not args or args[0].lower() not in {"auto", "confirm", "pause"}:
            await update.message.reply_text(
                f"Current mode: {self.runtime.mode}\nUsage: /mode auto | /mode confirm | /mode pause"
            )
            return

        new_mode: ExecutionMode = args[0].lower()  # type: ignore[assignment]
        if (
            new_mode == "auto"
            and self.ml_runner is not None
            and not self.ml_runner.is_mode_verified(self.runtime.strategy_mode)
        ):
            self.runtime.mode = "confirm"
            self.bridge.send_command("RESUME")
            await update.message.reply_text(
                f"⚠️ AUTO blocked: {self.runtime.strategy_mode} ML bundle is unverified (PF < 1.05 or trades < 20). Staying in CONFIRM."
            )
            return

        self.runtime.mode = new_mode
        if new_mode == "pause":
            self.bridge.send_command("PAUSE")
        else:
            self.bridge.send_command("RESUME")

        await update.message.reply_text(f"Execution mode updated to: {new_mode.upper()}")

    async def cmd_be(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return
        cmd = self.bridge.send_command("MOVE_BE_NOW")
        assert update.message is not None
        await update.message.reply_text(f"Sent MOVE_BE_NOW command (id={cmd.command_id}).")

    async def cmd_cancel_pending(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return
        cmd = self.bridge.send_command("CANCEL_PENDING")
        assert update.message is not None
        await update.message.reply_text(f"Sent CANCEL_PENDING command (id={cmd.command_id}).")

    async def cmd_close_all(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("Confirm CLOSE ALL", callback_data="confirm_close_all"),
                InlineKeyboardButton("Cancel", callback_data="cancel_close_all"),
            ]
        ])
        assert update.message is not None
        await update.message.reply_text(
            "Close all open EA-LearningX positions now?",
            reply_markup=keyboard,
        )

    async def _send_analysis_card(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        report: AnalysisReport,
        market: MarketContext | None,
        header: str,
        pending_signal_id: str | None = None,
        ml_prediction: MlPrediction | None = None,
    ) -> None:
        kb = self._analysis_keyboard(pending_signal_id=pending_signal_id)
        if market is not None and len(market.h1_bars) >= 5:
            chart_png = await asyncio.to_thread(
                render_analysis_chart,
                market,
                report,
                None,
                ml_prediction,
            )
            if chart_png is not None:
                caption = self._format_analysis_caption(report, header)
                sent_photo = await self._broadcast_photo(
                    context,
                    photo_bytes=chart_png,
                    caption=caption,
                    reply_markup=kb,
                )
                if sent_photo:
                    return

        await self._broadcast(context, self._format_report(report, header), reply_markup=kb)

    async def _execute_analysis_cycle(self, context: ContextTypes.DEFAULT_TYPE, manual: bool) -> None:
        """Read-only Gemini macro & SMC briefing cycle; never writes execution signals."""
        session = evaluate_session()
        if not manual:
            if not session.market_open or session.session_name in {"DAILY_ROLLOVER", "FRIDAY_LATE_GUARD"}:
                logger.debug("Skipping scheduled analysis during market/rollover guard: %s", session.session_name)
                return
            if self.settings.require_killzone and not session.in_killzone:
                logger.debug("Skipping scheduled analysis outside killzone: %s", session.session_name)
                return

        market = self.bridge.read_market_context()
        ea_state = self.bridge.read_ea_state()

        try:
            report = await self.analyst.analyze(
                market=market,
                ea_state=ea_state,
                publish_allowed=False,
            )
            self.runtime.quota_alert_sent = False
        except Exception as exc:  # noqa: BLE001
            logger.exception("Analysis failed")
            if manual:
                await self._broadcast(context, f"⚠️ <b>Briefing failed:</b> <code>{html.escape(str(exc)[:220])}</code>")
            return

        self.runtime.last_report = report
        self.journal.record_analysis(report)
        ml_pred = self.runtime.last_ml_prediction
        await self._send_analysis_card(
            context,
            report,
            market,
            "■ GEMINI MACRO BRIEF · READ-ONLY",
            ml_prediction=ml_pred,
        )

    async def _run_manual_analysis(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            await self._execute_analysis_cycle(context, manual=True)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Manual analysis failed: %s", exc)
            await self._broadcast(context, f"⚠️ <b>Analysis failed:</b> <code>{html.escape(str(exc))}</code>")

    async def cmd_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return
        assert update.message is not None
        progress_msg = await update.message.reply_text(
            "<i>▪ Analyzing XAUUSD (Gemini + SMC)...</i>",
            parse_mode=ParseMode.HTML,
        )
        await self._run_manual_analysis(context)
        try:
            await progress_msg.delete()
        except Exception:  # noqa: BLE001
            pass

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return

        query = update.callback_query
        assert query is not None
        data = query.data or ""

        if data.startswith("menu:mode:"):
            new_mode: ExecutionMode = data.split(":", 2)[2]  # type: ignore[assignment]
            if (
                new_mode == "auto"
                and self.ml_runner is not None
                and not self.ml_runner.is_mode_verified(self.runtime.strategy_mode)
            ):
                if self.runtime.mode == "auto":
                    self.runtime.mode = "confirm"
                if self.runtime.mode != "pause":
                    self.bridge.send_command("RESUME")
                await self._safe_answer(query, "AUTO blocked: ML model unverified", show_alert=True)
                if query.message is not None:
                    await query.message.reply_text(
                        f"× <b>Cannot enable AUTO:</b> <code>{self.runtime.strategy_mode}</code> ML bundle is unverified "
                        "(PF &lt; 1.05 or OOS trades &lt; 20). Staying in <b>CONFIRM/PAUSE</b> — tap <b>Retrain ML</b> first.",
                        parse_mode=ParseMode.HTML,
                    )
                return

            already_selected = self.runtime.mode == new_mode
            self.runtime.mode = new_mode
            ea_cmd = "PAUSE" if new_mode == "pause" else "RESUME"
            self.bridge.send_command(ea_cmd)
            toast = (
                f"Already in {new_mode.upper()} mode ({ea_cmd} sent)"
                if already_selected
                else f"Mode updated to {new_mode.upper()}"
            )
            await self._safe_answer(query, toast)
            await self._edit_menu_safely(
                query,
                unchanged_notice=f"■ Mode is already <b>{new_mode.upper()}</b> (<code>{ea_cmd}</code> sent to MT5).",
            )
            return

        if data.startswith("strategy:"):
            raw_strat = data.split(":", 1)[1]
            if raw_strat not in {"M15_PRECISION", "M5_INTRADAY"}:
                await self._safe_answer(query, "Unknown strategy mode")
                return
            new_strat: StrategyMode = raw_strat  # type: ignore[assignment]
            already_selected = self.runtime.strategy_mode == new_strat
            self.runtime.strategy_mode = new_strat
            downgraded = False
            if (
                self.runtime.mode == "auto"
                and self.ml_runner is not None
                and not self.ml_runner.is_mode_verified(new_strat)
            ):
                self.runtime.mode = "confirm"
                downgraded = True
            toast = (
                f"Already on {new_strat}"
                if already_selected
                else f"Strategy set to {new_strat}"
            )
            await self._safe_answer(query, toast)
            notice = f"■ Strategy switched to <b>{new_strat}</b>."
            if downgraded:
                notice += " × <i>Unverified bundle — downgraded execution to CONFIRM.</i>"
            await self._edit_menu_safely(query, unchanged_notice=notice)
            if downgraded and query.message is not None:
                await query.message.reply_text(notice, parse_mode=ParseMode.HTML)
            return

        if data == "menu:toggle_kz":
            self.settings.require_killzone = not self.settings.require_killzone
            hours = "KillZone Only" if self.settings.require_killzone else "Flexible (24/5)"
            await self._safe_answer(query, f"Hours: {hours}")
            await self._edit_menu_safely(query, unchanged_notice=f"▪ Hours: <b>{hours}</b>")
            return

        if data == "menu:toggle_alerts":
            self.runtime.trade_alerts_enabled = not self.runtime.trade_alerts_enabled
            state_lbl = "ON" if self.runtime.trade_alerts_enabled else "OFF"
            await self._safe_answer(query, f"Trade alerts: {state_lbl}")
            await self._edit_menu_safely(query, unchanged_notice=f"▪ Trade alerts: <b>{state_lbl}</b>")
            return

        await self._safe_answer(query)

        if data == "confirm_close_all":
            cmd = self.bridge.send_command("CLOSE_ALL")
            await query.edit_message_text(
                f"× <b>CLOSE_ALL</b> sent to MT5 EA (<code>{cmd.command_id}</code>).",
                parse_mode=ParseMode.HTML,
            )
            return
        if data == "cancel_close_all":
            await query.edit_message_text("▫ <b>CLOSE_ALL</b> cancelled.", parse_mode=ParseMode.HTML)
            return

        if data.startswith("menu:"):
            action = data.split(":", 1)[1]
            assert query.message is not None
            if action == "panel":
                await query.message.reply_text(
                    self._format_menu_text(),
                    reply_markup=self._build_menu_keyboard(),
                    parse_mode=ParseMode.HTML,
                )
                return
            if action == "status":
                st_live = self.bridge.read_ea_state()
                await query.message.reply_text(
                    self._format_status(st_live),
                    reply_markup=self._status_keyboard(st_live),
                    parse_mode=ParseMode.HTML,
                )
                return
            if action == "smc":
                await self._send_smc_card(query.message.chat_id, context)
                return
            if action == "smc_levels":
                sess = evaluate_session()
                mkt = self.bridge.read_market_context()
                if mkt is None or not mkt.h1_bars:
                    await query.message.reply_text(
                        f"▪ <b>{html.escape(sess.session_name)}</b> — No MT5 bars exported yet.",
                        parse_mode=ParseMode.HTML,
                    )
                else:
                    smc = build_smc_snapshot(mkt)
                    await query.message.reply_text(
                        f"<b>■ SMC LEVELS</b>  ·  <code>{html.escape(sess.session_name)}</code>\n"
                        f"──────────────────\n"
                        f"<blockquote expandable>{html.escape(format_smc_summary(smc))}</blockquote>",
                        reply_markup=self._smc_keyboard(),
                        parse_mode=ParseMode.HTML,
                    )
                return
            if action == "brief":
                if self.runtime.last_report is None:
                    await query.message.reply_text("▫ No analysis run cached yet. Tap <b>Analyze Now</b>.", parse_mode=ParseMode.HTML)
                else:
                    await query.message.reply_text(
                        self._format_report(self.runtime.last_report, "■ FULL AI & SMC BRIEF"),
                        reply_markup=self._analysis_keyboard(),
                        parse_mode=ParseMode.HTML,
                    )
                return
            if action == "stats":
                await query.message.reply_text(
                    self._format_stats_html(),
                    reply_markup=self._stats_keyboard(),
                    parse_mode=ParseMode.HTML,
                )
                return
            if action == "ml_models":
                await query.message.reply_text(
                    self._format_ml_models_html(),
                    reply_markup=self._ml_models_keyboard(),
                    parse_mode=ParseMode.HTML,
                )
                return
            if action == "analyze":
                progress_msg = None
                try:
                    progress_msg = await query.message.reply_text(
                        "<i>▪ Analyzing XAUUSD (Gemini + SMC)...</i>",
                        parse_mode=ParseMode.HTML,
                    )
                except (TimedOut, NetworkError) as exc:
                    logger.warning("Progress message timed out; continuing analysis: %s", exc)
                await self._run_manual_analysis(context)
                if progress_msg is not None:
                    try:
                        await progress_msg.delete()
                    except Exception:  # noqa: BLE001
                        pass
                return
            if action == "be":
                cmd = self.bridge.send_command("MOVE_BE_NOW")
                await query.message.reply_text(
                    f"■ Sent <b>MOVE_BE_NOW</b> (<code>{cmd.command_id}</code>).",
                    parse_mode=ParseMode.HTML,
                )
                return
            if action == "close_all":
                kb = InlineKeyboardMarkup([[
                    InlineKeyboardButton("× Confirm CLOSE ALL", callback_data="confirm_close_all"),
                    InlineKeyboardButton("Cancel", callback_data="cancel_close_all"),
                ]])
                await query.message.reply_text(
                    "<b>Close all open EA-LearningX positions now?</b>",
                    reply_markup=kb,
                    parse_mode=ParseMode.HTML,
                )
                return

        if data.startswith("approve:"):
            sig_id = data.split(":", 1)[1]
            sig = self.runtime.pending_signals.pop(sig_id, None)
            if sig is None:
                await query.edit_message_reply_markup(reply_markup=self._analysis_keyboard())
                await query.message.reply_text(
                    f"▫ Signal <code>{html.escape(sig_id)}</code> already handled or expired.",
                    parse_mode=ParseMode.HTML,
                )
                return
            if int(time.time()) > sig.expires_at_unix:
                await query.edit_message_reply_markup(reply_markup=self._analysis_keyboard())
                await query.message.reply_text(
                    f"▫ Signal <code>{html.escape(sig_id)}</code> expired; not sent.",
                    parse_mode=ParseMode.HTML,
                )
                return
            approved_sig = sig.model_copy(
                update={
                    "signal_id": f"ml_app_{uuid.uuid4().hex[:8]}",
                    "generated_at_unix": int(time.time()),
                    "trade_allowed": True,
                }
            )
            self.bridge.write_signal(approved_sig)
            await query.edit_message_reply_markup(reply_markup=self._analysis_keyboard())
            await query.message.reply_text(
                f"✓ <b>Approved</b> <code>{html.escape(approved_sig.order_intent)}</code> (<code>{html.escape(approved_sig.signal_id)}</code>) → sent to MT5 EA.",
                parse_mode=ParseMode.HTML,
            )
            return

        if data.startswith("hold:") or data.startswith("unhold:"):
            is_hold = data.startswith("hold:")
            ticket_val = int(data.split(":", 1)[1])
            cmd = self.bridge.send_command("HOLD_TICKET" if is_hold else "UNHOLD_TICKET", ticket=ticket_val)
            assert query.message is not None
            msg = (
                f"■ <b>Kept #{ticket_val}</b> (<code>{cmd.command_id}</code>) — pinned safe."
                if is_hold
                else f"□ <b>Released #{ticket_val}</b> (<code>{cmd.command_id}</code>) — returned to EA."
            )
            await query.message.reply_text(msg, reply_markup=self._status_keyboard(self.bridge.read_ea_state()), parse_mode=ParseMode.HTML)
            return

        if data == "ml:retrain":
            assert query.message is not None
            if self.ml_runner is None:
                await query.message.reply_text("× ML runner is not configured.")
                return
            strat = self.runtime.strategy_mode
            await query.message.reply_text(
                f"<i>↻ Retraining &amp; walk-forward validating <b>{strat}</b>...</i>",
                parse_mode=ParseMode.HTML,
            )
            try:
                bundle = await self.ml_runner.retrain_mode(strat)
                m = bundle.metrics
                status_line = (
                    "✓ <b>PASSED OOS GATE — Live model updated</b>"
                    if m.verified_for_live
                    else f"× <b>FAILED OOS GATE — Auto blocked / Prior bundle kept</b> (<i>{html.escape(m.fail_reason)}</i>)"
                )
                await self._broadcast(
                    context,
                    f"<b>■ ML RETRAIN COMPLETE ({strat})</b>\n"
                    "──────────────────\n"
                    f"{status_line}\n"
                    f"Trades: <code>{m.oos_trades}</code> · WR: <code>{m.oos_win_rate * 100:.1f}%</code> · PF: <code>{m.oos_profit_factor:.2f}</code>\n"
                    f"Net: <code>${m.oos_net_pnl_usd:+.2f}</code> · Th: <code>{m.chosen_threshold:.2f}</code>",
                    reply_markup=self._ml_models_keyboard(),
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Manual ML retrain failed")
                await self._broadcast(
                    context,
                    f"× <b>ML Retrain failed ({strat}):</b> <code>{html.escape(str(exc)[:220])}</code>",
                )
            return

        if data.startswith("reject:"):
            sig_id = data.split(":", 1)[1]
            self.runtime.pending_signals.pop(sig_id, None)
            await query.edit_message_reply_markup(reply_markup=self._analysis_keyboard())
            await query.message.reply_text(
                f"× Rejected signal <code>{html.escape(sig_id)}</code>.",
                parse_mode=ParseMode.HTML,
            )

    async def scheduled_ml_cycle_job(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Poll closed M15/M5 bars every 15s and execute/queue ML signals on bar close."""
        if self.ml_runner is None or self.runtime.mode == "pause":
            return
        try:
            pred, sig, is_new_bar = await self.ml_runner.step_live_cycle(
                self.runtime.strategy_mode,
                self.runtime.mode,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ML live cycle skipped/error: %s", exc)
            return

        if pred is not None:
            self.runtime.last_ml_prediction = pred

        if not is_new_bar or sig is None or pred is None:
            return

        self.journal.record_ml_signal(pred, sig, evaluate_session().session_name)
        market = self.bridge.read_market_context()
        if sig.trade_allowed:
            side_dot = "▲ BUY" if sig.order_intent == "MARKET_BUY" else "▼ SELL"
            sl_d = abs(pred.entry_price - sig.sl_price) if sig.sl_price > 0 else 0.0
            tp1_d = abs(sig.tp1_price - pred.entry_price) if sig.tp1_price > 0 else 0.0
            tp2_d = abs(sig.tp_price - pred.entry_price) if sig.tp_price > 0 else 0.0
            rr2 = (tp2_d / sl_d) if sl_d > 0 else 0.0
            lines = [
                f"<b>{side_dot} · {pred.mode}</b>  <code>{pred.timeframe}</code>",
                "──────────────────",
                f"<b>ENTRY</b>   <code>{pred.entry_price:.2f}</code>",
                f"<b>SL</b>      <code>{sig.sl_price:.2f}</code>  <i>(-${sl_d:.2f})</i>",
            ]
            if sig.tp1_price > 0:
                lines.append(f"<b>TP1</b>     <code>{sig.tp1_price:.2f}</code>  <i>(+${tp1_d:.2f} · 50% + BE)</i>")
            lines.append(f"<b>TP2</b>     <code>{sig.tp_price:.2f}</code>  <i>(+${tp2_d:.2f} · 1:{rr2:.1f}R)</i>")
            if sig.trail_atr_usd > 0:
                lines.append(f"<b>TRAIL</b>   <code>${sig.trail_atr_usd:.2f}</code>")
            lines.extend([
                "──────────────────",
                f"<code>HMM {pred.regime}</code> · L <code>{pred.prob_long:.0%}</code> / S <code>{pred.prob_short:.0%}</code> <i>(th {pred.threshold:.0%})</i>",
            ])
            msg = "\n".join(lines)
            chart_png = (
                await asyncio.to_thread(render_analysis_chart, market, None, None, pred)
                if market is not None
                else None
            )
            if chart_png is not None:
                await self._broadcast_photo(context, chart_png, msg, reply_markup=self._status_keyboard())
            else:
                await self._broadcast(context, msg, reply_markup=self._status_keyboard())
        elif self.runtime.mode == "confirm" and pred.order_intent != "NONE":
            confirmed_sig = sig.model_copy(
                update={
                    "trade_allowed": True,
                    "max_hold_until_unix": pred.max_hold_until_unix,
                }
            )
            self.runtime.pending_signals[sig.signal_id] = confirmed_sig
            kb = self._analysis_keyboard(pending_signal_id=sig.signal_id)
            tp1_str = f" · <b>TP1</b> <code>{pred.tp1_price:.2f}</code>" if pred.tp1_price > 0 else ""
            msg = (
                f"<b>◇ APPROVAL REQUIRED ({pred.mode})</b> · <code>{sig.signal_id}</code>\n"
                "──────────────────\n"
                f"<code>{pred.timeframe}</code> · <code>HMM {pred.regime}</code> · <b>{pred.order_intent}</b>\n"
                f"<b>ENTRY</b> <code>{pred.entry_price:.2f}</code> · <b>SL</b> <code>{pred.sl_price:.2f}</code>{tp1_str} · <b>TP2</b> <code>{pred.tp_price:.2f}</code>\n"
                f"Prob L=<code>{pred.prob_long:.2f}</code> / S=<code>{pred.prob_short:.2f}</code>"
            )
            chart_png = (
                await asyncio.to_thread(render_analysis_chart, market, None, None, pred)
                if market is not None
                else None
            )
            if chart_png is not None:
                await self._broadcast_photo(context, chart_png, msg, reply_markup=kb)
            else:
                await self._broadcast(context, msg, reply_markup=kb)

    async def scheduled_ml_retrain_job(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Daily walk-forward retrain across both strategy modes at rollover hour."""
        if self.ml_runner is None:
            return
        for strat in ("M15_PRECISION", "M5_INTRADAY"):
            try:
                bundle = await self.ml_runner.retrain_mode(strat)  # type: ignore[arg-type]
                m = bundle.metrics
                if not m.verified_for_live:
                    if self.runtime.strategy_mode == strat and self.runtime.mode == "auto" and not self.ml_runner.is_mode_verified(strat):  # type: ignore[arg-type]
                        self.runtime.mode = "confirm"
                    await self._broadcast(
                        context,
                        f"⚠️ <b>Nightly Retrain Failed ({strat}):</b> {html.escape(m.fail_reason)} "
                        f"(PF={m.oos_profit_factor:.2f}, trades={m.oos_trades}).",
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Nightly retrain error for %s: %s", strat, exc)

    async def watch_ea_state_job(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Detect trade opens, Break-Even activations, and closes from ea_state.json."""
        state = self.bridge.read_ea_state()
        if state is None:
            return

        self._resolve_account_currency(state)
        pos_map = {p.ticket: p for p in state.positions}
        ord_map = {o.ticket: o for o in state.pending_orders}
        be_set = {p.ticket for p in state.positions if p.be_applied}

        for p in state.positions:
            self.journal.upsert_open_position(state.symbol, p)
        if state.closed_deals:
            self.journal.sync_closed_deals(state.symbol, state.closed_deals)

        if not self.runtime.state_initialized:
            self.runtime.known_position_tickets = set(pos_map.keys())
            self.runtime.known_be_tickets = be_set
            self.runtime.known_pending_tickets = set(ord_map.keys())
            self.runtime.state_initialized = True
            return

        for ticket in sorted(set(pos_map.keys()) - self.runtime.known_position_tickets):
            p = pos_map[ticket]
            if self.runtime.trade_alerts_enabled:
                p_dir = "▲" if p.type == "BUY" else "▼"
                await self._broadcast(
                    context,
                    f"<b>{p_dir} MT5 OPENED</b>  <code>#{p.ticket} {p.type} {p.volume:.2f} {html.escape(state.symbol)}</code>\n"
                    f"<b>IN</b> <code>{p.open_price:.2f}</code> · <b>SL</b> <code>{p.sl:.2f}</code> · <b>TP</b> <code>{p.tp:.2f}</code>",
                    reply_markup=self._status_keyboard(state),
                )

        for ticket in sorted(be_set - self.runtime.known_be_tickets):
            if ticket in pos_map and self.runtime.trade_alerts_enabled:
                p = pos_map[ticket]
                await self._broadcast(
                    context,
                    f"<b>■ MT5 BREAK-EVEN</b>  <code>#{p.ticket} {p.type}</code>\n"
                    f"SL locked at <code>{p.sl:.2f}</code> (entry <code>{p.open_price:.2f}</code>).",
                )

        for ticket in sorted(self.runtime.known_position_tickets - set(pos_map.keys())):
            self.journal.mark_position_closed(ticket)
            if self.runtime.trade_alerts_enabled:
                ccy = self._resolve_account_currency(state)
                pnl_dir = "▲" if state.daily_pnl_usd >= 0 else "▼"
                await self._broadcast(
                    context,
                    f"<b>■ MT5 CLOSED</b>  <code>#{ticket}</code>\n"
                    f"<b>EQ</b> <code>{self._format_money(state.equity, ccy)}</code> · <b>DAY</b> {pnl_dir} <code>{self._format_money(state.daily_pnl_usd, ccy, signed=True)}</code>",
                    reply_markup=self._stats_keyboard(),
                )

        # Pending stops ratchet rapidly in M1 SAR mode; keep chat clean unless pending limit placed.
        for ticket in sorted(set(ord_map.keys()) - self.runtime.known_pending_tickets):
            o = ord_map[ticket]
            if self.runtime.trade_alerts_enabled and "LIMIT" in o.type.upper():
                await self._broadcast(
                    context,
                    f"<b>▪ MT5 PENDING</b>  <code>#{o.ticket} {html.escape(o.type)} {o.volume:.2f}</code>\n"
                    f"@ <code>{o.price:.2f}</code> · <b>SL</b> <code>{o.sl:.2f}</code> · <b>TP</b> <code>{o.tp:.2f}</code>",
                )

        self.runtime.known_position_tickets = set(pos_map.keys())
        self.runtime.known_be_tickets = be_set
        self.runtime.known_pending_tickets = set(ord_map.keys())

    async def _error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.error("Unhandled Telegram update error (update=%s): %s", update, context.error, exc_info=context.error)

    def build_application(self) -> Application:
        """Create and wire the python-telegram-bot Application."""
        if not self.settings.telegram_bot_token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured in .env")

        app = (
            Application.builder()
            .token(self.settings.telegram_bot_token)
            .concurrent_updates(True)
            .connect_timeout(15.0)
            .read_timeout(25.0)
            .write_timeout(25.0)
            .pool_timeout(15.0)
            .build()
        )
        app.add_error_handler(self._error_handler)
        app.add_handler(CommandHandler(["start", "help", "menu"], self.cmd_start))
        app.add_handler(CommandHandler("status", self.cmd_status))
        app.add_handler(CommandHandler("smc", self.cmd_smc))
        app.add_handler(CommandHandler("stats", self.cmd_stats))
        app.add_handler(CommandHandler("analyze", self.cmd_analyze))
        app.add_handler(CommandHandler("mode", self.cmd_mode))
        app.add_handler(CommandHandler("session", self.cmd_session))
        app.add_handler(CommandHandler("be", self.cmd_be))
        app.add_handler(CommandHandler("cancel_pending", self.cmd_cancel_pending))
        app.add_handler(CommandHandler("close_all", self.cmd_close_all))
        app.add_handler(CallbackQueryHandler(self.handle_callback))

        if app.job_queue is not None:
            app.job_queue.run_repeating(
                self.scheduled_ml_cycle_job,
                interval=self.settings.ml_poll_interval_seconds,
                first=5,
            )
            app.job_queue.run_repeating(
                self.watch_ea_state_job,
                interval=10,
                first=5,
            )
            app.job_queue.run_daily(
                self.scheduled_ml_retrain_job,
                time=dt_time(hour=self.settings.ml_retrain_hour_utc, minute=0, tzinfo=timezone.utc),
            )
        return app
