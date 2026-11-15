"""Telegram bot remote controller & real-time alert notifier for Android."""

from __future__ import annotations

import asyncio
import html
import io
import logging
import time
from dataclasses import dataclass, field

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
from .schemas import AnalysisReport, EaSignalPayload, EaState, ExecutionMode, MarketContext
from .sessions import evaluate_session
from .smc import build_smc_snapshot, format_smc_summary

logger = logging.getLogger(__name__)


@dataclass
class BotRuntimeState:
    """Mutable runtime state shared across Telegram handlers and scheduled jobs."""

    mode: ExecutionMode
    pending_signals: dict[str, EaSignalPayload] = field(default_factory=dict)
    known_position_tickets: set[int] = field(default_factory=set)
    known_be_tickets: set[int] = field(default_factory=set)
    known_pending_tickets: set[int] = field(default_factory=set)
    state_initialized: bool = False
    last_report: AnalysisReport | None = None


class TelegramController:
    """Manages Telegram commands, inline approvals, scheduled analysis, and alerts."""

    def __init__(
        self,
        settings: Settings,
        bridge: Mt5Bridge,
        analyst: GoldAnalyst,
        journal: TradeJournal,
    ) -> None:
        self.settings = settings
        self.bridge = bridge
        self.analyst = analyst
        self.journal = journal
        self.runtime = BotRuntimeState(mode=settings.default_execution_mode)

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
                for attempt in range(2):
                    try:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=chunk,
                            reply_markup=markup,
                            parse_mode=parse_mode,
                        )
                        break
                    except (TimedOut, NetworkError) as exc:
                        if attempt == 0:
                            logger.warning(
                                "Transient Telegram send error to %s (retrying): %s",
                                chat_id,
                                exc,
                            )
                            await asyncio.sleep(1.5)
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
            for attempt in range(2):
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
                    if attempt == 0:
                        logger.warning("Transient Telegram photo send error to %s (retrying): %s", chat_id, exc)
                        await asyncio.sleep(1.5)
                        continue
                    logger.warning("Failed sending Telegram photo to %s: %s", chat_id, exc)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed sending Telegram photo to %s: %s", chat_id, exc)
                    break
        return sent_any

    @staticmethod
    def _meter_bar(ratio: float, width: int = 8) -> str:
        clamped = max(0.0, min(1.0, ratio))
        filled = int(round(clamped * width))
        return ("█" * filled) + ("░" * (width - filled))

    @staticmethod
    def _bias_badge(bias: str) -> str:
        if bias == "BULLISH":
            return "🟢 BULLISH"
        if bias == "BEARISH":
            return "🔴 BEARISH"
        return "⚪ NEUTRAL"

    def _format_status(self, state: EaState | None) -> str:
        session = evaluate_session()
        kz_tag = "🎯 KillZone ON" if session.in_killzone else f"🕒 {html.escape(session.session_name)}"
        filter_tag = "KillZone Only" if self.settings.require_killzone else "24/5 Flexible"
        mode_icon = {"auto": "🟢 AUTO", "confirm": "🟡 CONFIRM", "pause": "⏸ STOP"}.get(
            self.runtime.mode, self.runtime.mode.upper()
        )
        if state is None:
            return (
                "📡 <b>EA-LearningX Live Status</b>\n"
                "━━━━━━━━━━━━━━━━━━\n"
                f"🎛 <b>Mode:</b> {mode_icon}  ·  <b>Hours:</b> {filter_tag}\n"
                f"🧭 <b>Session:</b> {kz_tag}\n\n"
                "⚠️ <i>MT5 EA State not connected yet. Open Exness MT5 &amp; attach EA_LearningX_Gold to XAUUSD.</i>"
            )

        age = int(time.time()) - state.updated_at_unix
        flags = []
        if state.paused:
            flags.append("⏸ PAUSED")
        if state.daily_halt:
            flags.append("🛑 DAILY_HALT")
        if state.hmr_blackout:
            flags.append("📰 HMR_BLACKOUT")
        flag_str = " · ".join(flags) if flags else "🟢 ACTIVE"
        stale_warn = "  ⚠️ <i>(MT5 closed/stale)</i>" if age > 60 else ""
        pnl_icon = "🟢" if state.daily_pnl_usd >= 0 else "🔴"

        lines = [
            f"📡 <b>EA-LearningX</b>  <code>{html.escape(state.symbol)}</code>",
            "━━━━━━━━━━━━━━━━━━",
            f"🎛 <b>Mode:</b> {mode_icon}  |  <b>EA:</b> {flag_str}",
            f"⏱ <b>Heartbeat:</b> <code>{age}s ago</code>{stale_warn}",
            f"🧭 <b>Session:</b> {kz_tag} <i>({filter_tag})</i>",
            "",
            f"💱 <b>Bid / Ask:</b> <code>{state.bid:.2f}</code> / <code>{state.ask:.2f}</code>  <i>(Spread ${state.spread_usd:.2f})</i>",
            f"💰 <b>Balance:</b> <code>${state.balance:,.2f}</code>  |  <b>Equity:</b> <code>${state.equity:,.2f}</code>",
            f"{pnl_icon} <b>Daily PnL:</b> <code>${state.daily_pnl_usd:+.2f}</code>  |  <b>Free:</b> <code>${state.free_margin:,.2f}</code>",
            "",
            f"📌 <b>Open Positions ({len(state.positions)}):</b>",
        ]
        if not state.positions:
            lines.append("   <i>None</i>")
        for p in state.positions:
            be_tag = " 🛡[BE]" if p.be_applied else ""
            p_icon = "🟢" if p.profit >= 0 else "🔴"
            lines.append(
                f"   {p_icon} <code>#{p.ticket} {p.type} {p.volume:.2f} @ {p.open_price:.2f}</code>\n"
                f"      SL <code>{p.sl:.2f}</code> · TP <code>{p.tp:.2f}</code> · <b>${p.profit:+.2f}</b>{be_tag}"
            )

        lines.append(f"\n⏳ <b>Pending Orders ({len(state.pending_orders)}):</b>")
        if not state.pending_orders:
            lines.append("   <i>None</i>")
        for o in state.pending_orders:
            lines.append(
                f"   ⚡ <code>#{o.ticket} {html.escape(o.type)} {o.volume:.2f} @ {o.price:.2f}</code>\n"
                f"      SL <code>{o.sl:.2f}</code> · TP <code>{o.tp:.2f}</code>"
            )
        return "\n".join(lines)

    def _format_analysis_caption(self, report: AnalysisReport, badge: str) -> str:
        """Compact rich-HTML card (fits Telegram 1024-char photo caption limit cleanly)."""
        d = report.decision
        s = report.signal
        sess_txt = html.escape(report.session_status.session_name if report.session_status else "UNKNOWN")
        conf_bar = self._meter_bar(d.confidence, 8)
        score_bar = self._meter_bar(report.confluence_score / 5.0, 5)
        bias_badge = self._bias_badge(d.bias)
        exec_badge = "✅ ALLOWED" if s.trade_allowed else "⛔ NO TRADE"

        lines = [
            f"<b>{html.escape(badge)}</b>",
            "━━━━━━━━━━━━━━━━━━",
            f"{bias_badge}  ·  <code>{html.escape(d.order_intent)}</code>  ({exec_badge})",
            f"📊 <b>Conf:</b> <code>{conf_bar}</code> <b>{d.confidence:.0%}</b>  |  <b>Score:</b> <code>{score_bar}</code> <b>{report.confluence_score}/5</b>",
            f"🧭 <b>Session:</b> <code>{sess_txt}</code>  |  <b>Sent:</b> <code>{d.news_sentiment_score:+.2f}</code>",
        ]
        if d.order_intent != "NONE":
            entry_txt = f"{d.limit_price:.2f}" if d.limit_price > 0 else "MARKET"
            lines.append(
                f"🎯 <b>Setup:</b> <code>{entry_txt}</code>  |  🛑 <b>SL</b> <code>{d.sl_price:.2f}</code>  |  🏆 <b>TP</b> <code>{d.tp_price:.2f}</code>"
            )
        if report.smc_snapshot is not None:
            smc = report.smc_snapshot
            lines.append(
                f"🧱 <b>SMC:</b> H4 <code>{smc.h4.trend}</code> · H1 <code>{smc.h1.trend}</code> · M15 <code>{smc.m15.trend}</code> · P/D <code>{smc.h1.premium_discount_ratio:.2f}</code>"
            )
        if report.hmr_blackout_active:
            lines.append(f"🚨 <b>HMR Guard:</b> {html.escape(report.hmr_event_title or 'Active')}")

        lines.append(f"\n📝 <b>Gate:</b> <i>{html.escape(report.validation_note[:120])}</i>")
        body_no_quote = "\n".join(lines)
        remaining = 960 - len(body_no_quote)
        if remaining > 40:
            raw_snippet = d.technical_summary.strip()
            if len(raw_snippet) > remaining - 30:
                raw_snippet = raw_snippet[: max(10, remaining - 33)] + "..."
            lines.append(
                f"<blockquote expandable>📐 <b>Quick Read:</b> {html.escape(raw_snippet)}</blockquote>"
            )
        return "\n".join(lines)

    def _format_report(self, report: AnalysisReport, prefix: str = "") -> str:
        """Full HTML card with expandable Macro/News/Technical sections."""
        d = report.decision
        s = report.signal
        sess_txt = html.escape(report.session_status.session_name if report.session_status else "UNKNOWN")
        conf_bar = self._meter_bar(d.confidence, 8)
        score_bar = self._meter_bar(report.confluence_score / 5.0, 5)
        exec_badge = "✅ ALLOWED" if s.trade_allowed else "⛔ NO TRADE"

        lines = []
        if prefix:
            lines.append(f"<b>{html.escape(prefix)}</b>")
            lines.append("━━━━━━━━━━━━━━━━━━")
        lines.extend([
            f"{self._bias_badge(d.bias)}  ·  <code>{html.escape(d.order_intent)}</code>  ({exec_badge})",
            f"📊 <b>Confidence:</b> <code>{conf_bar}</code> <b>{d.confidence:.0%}</b>",
            f"🧩 <b>Confluence:</b> <code>{score_bar}</code> <b>{report.confluence_score}/5</b>  |  <b>Sent:</b> <code>{d.news_sentiment_score:+.2f}</code>",
            f"🧭 <b>Session:</b> <code>{sess_txt}</code>",
        ])
        if d.order_intent != "NONE":
            entry_txt = f"{d.limit_price:.2f}" if d.limit_price > 0 else "MARKET"
            lines.append(
                f"🎯 <b>Entry:</b> <code>{entry_txt}</code>  |  🛑 <b>SL:</b> <code>{d.sl_price:.2f}</code>  |  🏆 <b>TP:</b> <code>{d.tp_price:.2f}</code>"
            )
        if report.smc_snapshot is not None:
            smc = report.smc_snapshot
            lines.append(
                f"🧱 <b>SMC:</b> H4 <code>{smc.h4.trend}({smc.h4.last_event})</code> · "
                f"H1 <code>{smc.h1.trend}({smc.h1.last_event})</code> · "
                f"M15 <code>{smc.m15.trend}</code> · P/D <code>{smc.h1.premium_discount_ratio:.2f}</code>"
            )
        if report.hmr_blackout_active:
            lines.append(f"🚨 <b>HMR Guard:</b> BLACKOUT ({html.escape(report.hmr_event_title or '')})")

        lines.append(f"\n📝 <b>Validation:</b> <i>{html.escape(report.validation_note)}</i>")
        lines.append(f"<blockquote expandable>🌐 <b>Macro:</b> {html.escape(d.macro_summary)}</blockquote>")
        lines.append(f"<blockquote expandable>📰 <b>News Sentiment:</b> {html.escape(d.news_sentiment_summary)}</blockquote>")
        lines.append(f"<blockquote expandable>📐 <b>Technical:</b> {html.escape(d.technical_summary)}</blockquote>")
        if report.confluence_notes:
            notes_txt = "\n".join(f"• {html.escape(n)}" for n in report.confluence_notes)
            lines.append(f"<blockquote expandable>✅ <b>Checklist:</b>\n{notes_txt}</blockquote>")
        if report.grounded_sources:
            lines.append(f"🔗 <b>Sources:</b> <i>{html.escape(', '.join(report.grounded_sources[:3]))}</i>")
        return "\n".join(lines)

    def _analysis_keyboard(self, pending_signal_id: str | None = None) -> InlineKeyboardMarkup:
        rows: list[list[InlineKeyboardButton]] = []
        if pending_signal_id:
            rows.append([
                InlineKeyboardButton("✅ Approve & Execute", callback_data=f"approve:{pending_signal_id}"),
                InlineKeyboardButton("❌ Reject", callback_data=f"reject:{pending_signal_id}"),
            ])
        rows.append([
            InlineKeyboardButton("📰 Full AI Brief", callback_data="menu:brief"),
            InlineKeyboardButton("🧱 SMC Levels", callback_data="menu:smc_levels"),
        ])
        rows.append([
            InlineKeyboardButton("🔄 Re-Analyze", callback_data="menu:analyze"),
            InlineKeyboardButton("📡 Status", callback_data="menu:status"),
            InlineKeyboardButton("🎛 Panel", callback_data="menu:panel"),
        ])
        return InlineKeyboardMarkup(rows)

    def _format_menu_text(self) -> str:
        mode_map = {"auto": "🟢 AUTO", "confirm": "🟡 CONFIRM", "pause": "⏸ PAUSED"}
        mode_badge = mode_map.get(self.runtime.mode, self.runtime.mode.upper())
        hours = "🎯 KillZone Only" if self.settings.require_killzone else "🌐 Flexible (24/5)"
        sess = evaluate_session()
        state = self.bridge.read_ea_state()
        if state is None:
            mt5_txt = "⚪ Offline"
        else:
            age = int(time.time()) - state.updated_at_unix
            mt5_txt = "🟢 Live" if age <= 60 and not state.paused else ("⏸ Paused" if age <= 60 else "⚠️ MT5 Closed")
        return (
            "🎛 <b>EA-LearningX Control Panel</b>  <i>(Exness XAUUSD)</i>\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"⚙️ <b>Mode:</b> {mode_badge}\n"
            f"🕒 <b>Hours:</b> {hours}\n"
            f"🧭 <b>Market:</b> <code>{html.escape(sess.session_name)}</code>  ·  <b>MT5:</b> {mt5_txt}"
        )

    def _build_menu_keyboard(self) -> InlineKeyboardMarkup:
        def _mode_label(mode_val: ExecutionMode, label: str) -> str:
            return f"✅ {label}" if self.runtime.mode == mode_val else label

        kz_label = "🎯 KillZone Only" if self.settings.require_killzone else "🌐 24/5 Flexible"
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton(_mode_label("auto", "Auto"), callback_data="menu:mode:auto"),
                InlineKeyboardButton(_mode_label("confirm", "Confirm"), callback_data="menu:mode:confirm"),
                InlineKeyboardButton(_mode_label("pause", "Stop"), callback_data="menu:mode:pause"),
            ],
            [
                InlineKeyboardButton(kz_label, callback_data="menu:toggle_kz"),
                InlineKeyboardButton("⚡ Analyze Now", callback_data="menu:analyze"),
            ],
            [
                InlineKeyboardButton("📡 Status", callback_data="menu:status"),
                InlineKeyboardButton("🧱 SMC Chart", callback_data="menu:smc"),
                InlineKeyboardButton("📊 Stats", callback_data="menu:stats"),
            ],
            [
                InlineKeyboardButton("🛡 Move BE", callback_data="menu:be"),
                InlineKeyboardButton("🚨 Close All", callback_data="menu:close_all"),
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
        wr_bar = self._meter_bar(st.win_rate_pct / 100.0, 10)
        pnl_icon = "🟢" if st.net_realized_pnl >= 0 else "🔴"
        return (
            "📊 <b>EA-LearningX Performance Journal</b>\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"🧪 <b>Analyses Logged:</b> <code>{st.total_analyses}</code>  ·  <b>Setups:</b> <code>{st.actionable_signals}</code>\n"
            f"📥 <b>Trades Opened:</b> <code>{st.trades_opened}</code>  ·  <b>Closed:</b> <code>{st.trades_closed}</code>\n"
            f"🛡 <b>Break-Even Triggered:</b> <code>{st.be_triggered_count}</code>\n"
            f"🎯 <b>Win Rate:</b> <code>{wr_bar}</code> <b>{st.win_rate_pct:.1f}%</b> ({st.wins}W / {st.losses}L)\n"
            f"{pnl_icon} <b>Net Closed PnL:</b> <code>${st.net_realized_pnl:+.2f}</code>"
        )

    def _status_keyboard(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("🔄 Refresh", callback_data="menu:status"),
                InlineKeyboardButton("⚡ Analyze Now", callback_data="menu:analyze"),
            ],
            [
                InlineKeyboardButton("🛡 Move BE", callback_data="menu:be"),
                InlineKeyboardButton("🎛 Panel", callback_data="menu:panel"),
            ],
        ])

    def _smc_keyboard(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("🔄 Refresh SMC", callback_data="menu:smc"),
                InlineKeyboardButton("⚡ Analyze Now", callback_data="menu:analyze"),
            ],
            [
                InlineKeyboardButton("📡 Status", callback_data="menu:status"),
                InlineKeyboardButton("🎛 Panel", callback_data="menu:panel"),
            ],
        ])

    def _stats_keyboard(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("🔄 Refresh", callback_data="menu:stats"),
                InlineKeyboardButton("⚡ Analyze Now", callback_data="menu:analyze"),
                InlineKeyboardButton("🎛 Panel", callback_data="menu:panel"),
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
            reply_markup=self._status_keyboard(),
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
        chart_png = render_analysis_chart(market, smc=smc)
        smc_summary = html.escape(format_smc_summary(smc))
        if len(smc_summary) > 780:
            smc_summary = smc_summary[:777] + "..."
        caption = (
            f"🧱 <b>XAUUSD SMC Map</b>  ·  <code>{html.escape(session.session_name)}</code>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
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
    ) -> None:
        kb = self._analysis_keyboard(pending_signal_id=pending_signal_id)
        if market is not None and len(market.h1_bars) >= 5:
            chart_png = render_analysis_chart(market=market, report=report)
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
        is_auto = self.runtime.mode == "auto"

        try:
            report = await self.analyst.analyze(
                market=market,
                ea_state=ea_state,
                publish_allowed=is_auto,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Analysis failed")
            if manual:
                await self._broadcast(context, f"⚠️ <b>Analysis failed:</b> <code>{html.escape(str(exc))}</code>")
            return

        self.runtime.last_report = report
        self.journal.record_analysis(report)
        sig = report.signal

        if self.runtime.mode == "pause":
            sig_paused = sig.model_copy(update={"trade_allowed": False})
            self.bridge.write_signal(sig_paused)
            if manual:
                await self._send_analysis_card(context, report, market, "⏸ PAUSED MODE · NO EXECUTION")
            return

        if is_auto:
            self.bridge.write_signal(sig)
            if manual or sig.trade_allowed:
                header = "⚡ AUTO SIGNAL SENT TO MT5" if sig.trade_allowed else "🤖 AUTO ANALYSIS · NO TRADE"
                await self._send_analysis_card(context, report, market, header)
            return

        if sig.order_intent != "NONE" and report.validation_note.startswith("Validated setup"):
            self.bridge.write_signal(sig.model_copy(update={"trade_allowed": False}))
            self.runtime.pending_signals[sig.signal_id] = sig.model_copy(update={"trade_allowed": True})
            await self._send_analysis_card(
                context,
                report,
                market,
                f"🟡 APPROVAL REQUIRED · id={sig.signal_id}",
                pending_signal_id=sig.signal_id,
            )
        else:
            self.bridge.write_signal(sig)
            if manual:
                await self._send_analysis_card(context, report, market, "🟡 CONFIRM MODE · NO TRADE")

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
            "⏳ <i>Analyzing XAUUSD (Gemini 4 Hydrogen + SMC)...</i>",
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
            already_selected = self.runtime.mode == new_mode
            self.runtime.mode = new_mode
            ea_cmd = "PAUSE" if new_mode == "pause" else "RESUME"
            self.bridge.send_command(ea_cmd)
            toast = (
                f"Already in {new_mode.upper()} mode ({ea_cmd} sent)"
                if already_selected
                else f"Mode updated to {new_mode.upper()}"
            )
            await query.answer(toast)
            await self._edit_menu_safely(
                query,
                unchanged_notice=f"✅ Mode is already <b>{new_mode.upper()}</b> (<code>{ea_cmd}</code> sent to MT5).",
            )
            return

        if data == "menu:toggle_kz":
            self.settings.require_killzone = not self.settings.require_killzone
            hours = "KillZone Only" if self.settings.require_killzone else "Flexible (24/5)"
            await query.answer(f"Hours: {hours}")
            await self._edit_menu_safely(query, unchanged_notice=f"🕒 Hours: <b>{hours}</b>")
            return

        await query.answer()

        if data == "confirm_close_all":
            cmd = self.bridge.send_command("CLOSE_ALL")
            await query.edit_message_text(
                f"🚨 <b>CLOSE_ALL</b> sent to MT5 EA (<code>{cmd.command_id}</code>).",
                parse_mode=ParseMode.HTML,
            )
            return
        if data == "cancel_close_all":
            await query.edit_message_text("❎ <b>CLOSE_ALL</b> cancelled.", parse_mode=ParseMode.HTML)
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
                await query.message.reply_text(
                    self._format_status(self.bridge.read_ea_state()),
                    reply_markup=self._status_keyboard(),
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
                        f"🧭 <b>{html.escape(sess.session_name)}</b> — No MT5 bars exported yet.",
                        parse_mode=ParseMode.HTML,
                    )
                else:
                    smc = build_smc_snapshot(mkt)
                    await query.message.reply_text(
                        f"🧱 <b>SMC Levels</b>  ·  <code>{html.escape(sess.session_name)}</code>\n"
                        f"<blockquote expandable>{html.escape(format_smc_summary(smc))}</blockquote>",
                        reply_markup=self._smc_keyboard(),
                        parse_mode=ParseMode.HTML,
                    )
                return
            if action == "brief":
                if self.runtime.last_report is None:
                    await query.message.reply_text("ℹ️ No analysis run cached yet. Tap ⚡ <b>Analyze Now</b>.", parse_mode=ParseMode.HTML)
                else:
                    await query.message.reply_text(
                        self._format_report(self.runtime.last_report, "📰 FULL AI & SMC BRIEF"),
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
            if action == "analyze":
                progress_msg = await query.message.reply_text(
                    "⏳ <i>Analyzing XAUUSD (Gemini 4 Hydrogen + SMC)...</i>",
                    parse_mode=ParseMode.HTML,
                )
                await self._run_manual_analysis(context)
                try:
                    await progress_msg.delete()
                except Exception:  # noqa: BLE001
                    pass
                return
            if action == "be":
                cmd = self.bridge.send_command("MOVE_BE_NOW")
                await query.message.reply_text(
                    f"🛡 Sent <b>MOVE_BE_NOW</b> (<code>{cmd.command_id}</code>).",
                    parse_mode=ParseMode.HTML,
                )
                return
            if action == "close_all":
                kb = InlineKeyboardMarkup([[
                    InlineKeyboardButton("🚨 Confirm CLOSE ALL", callback_data="confirm_close_all"),
                    InlineKeyboardButton("Cancel", callback_data="cancel_close_all"),
                ]])
                await query.message.reply_text(
                    "⚠️ <b>Close all open EA-LearningX positions now?</b>",
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
                    f"ℹ️ Signal <code>{html.escape(sig_id)}</code> already handled or expired.",
                    parse_mode=ParseMode.HTML,
                )
                return
            if int(time.time()) > sig.expires_at_unix:
                await query.edit_message_reply_markup(reply_markup=self._analysis_keyboard())
                await query.message.reply_text(
                    f"⌛ Signal <code>{html.escape(sig_id)}</code> expired; not sent.",
                    parse_mode=ParseMode.HTML,
                )
                return
            self.bridge.write_signal(sig)
            await query.edit_message_reply_markup(reply_markup=self._analysis_keyboard())
            await query.message.reply_text(
                f"✅ <b>Approved</b> <code>{html.escape(sig.order_intent)}</code> (<code>{html.escape(sig_id)}</code>) → sent to MT5 EA.",
                parse_mode=ParseMode.HTML,
            )
            return

        if data.startswith("reject:"):
            sig_id = data.split(":", 1)[1]
            self.runtime.pending_signals.pop(sig_id, None)
            await query.edit_message_reply_markup(reply_markup=self._analysis_keyboard())
            await query.message.reply_text(
                f"❌ Rejected signal <code>{html.escape(sig_id)}</code>.",
                parse_mode=ParseMode.HTML,
            )

    async def scheduled_analysis_job(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        if self.runtime.mode == "pause":
            return
        await self._execute_analysis_cycle(context, manual=False)

    async def watch_ea_state_job(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Detect trade opens, Break-Even activations, and closes from ea_state.json."""
        state = self.bridge.read_ea_state()
        if state is None:
            return

        pos_map = {p.ticket: p for p in state.positions}
        ord_map = {o.ticket: o for o in state.pending_orders}
        be_set = {p.ticket for p in state.positions if p.be_applied}

        for p in state.positions:
            self.journal.upsert_open_position(state.symbol, p)

        if not self.runtime.state_initialized:
            self.runtime.known_position_tickets = set(pos_map.keys())
            self.runtime.known_be_tickets = be_set
            self.runtime.known_pending_tickets = set(ord_map.keys())
            self.runtime.state_initialized = True
            return

        for ticket in sorted(set(pos_map.keys()) - self.runtime.known_position_tickets):
            p = pos_map[ticket]
            p_icon = "🟢" if p.type == "BUY" else "🔴"
            await self._broadcast(
                context,
                f"{p_icon} <b>MT5 OPENED</b>  <code>#{p.ticket} {p.type} {p.volume:.2f} {html.escape(state.symbol)}</code>\n"
                f"🎯 Entry <code>{p.open_price:.2f}</code> · 🛑 SL <code>{p.sl:.2f}</code> · 🏆 TP <code>{p.tp:.2f}</code>",
                reply_markup=self._status_keyboard(),
            )

        for ticket in sorted(be_set - self.runtime.known_be_tickets):
            if ticket in pos_map:
                p = pos_map[ticket]
                await self._broadcast(
                    context,
                    f"🛡 <b>MT5 BREAK-EVEN</b>  <code>#{p.ticket} {p.type}</code>\n"
                    f"SL locked at <code>{p.sl:.2f}</code> (entry <code>{p.open_price:.2f}</code>).",
                )

        for ticket in sorted(self.runtime.known_position_tickets - set(pos_map.keys())):
            self.journal.mark_position_closed(ticket)
            pnl_icon = "🟢" if state.daily_pnl_usd >= 0 else "🔴"
            await self._broadcast(
                context,
                f"🏁 <b>MT5 CLOSED</b>  <code>#{ticket}</code>\n"
                f"💰 Equity <code>${state.equity:,.2f}</code> · {pnl_icon} Daily PnL <code>${state.daily_pnl_usd:+.2f}</code>",
                reply_markup=self._stats_keyboard(),
            )

        for ticket in sorted(set(ord_map.keys()) - self.runtime.known_pending_tickets):
            o = ord_map[ticket]
            await self._broadcast(
                context,
                f"⏳ <b>MT5 PENDING PLACED</b>  <code>#{o.ticket} {html.escape(o.type)} {o.volume:.2f}</code>\n"
                f"🎯 @ <code>{o.price:.2f}</code> · 🛑 SL <code>{o.sl:.2f}</code> · 🏆 TP <code>{o.tp:.2f}</code>",
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
            .connect_timeout(20.0)
            .read_timeout(30.0)
            .write_timeout(30.0)
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
                self.scheduled_analysis_job,
                interval=self.settings.analysis_interval_minutes * 60,
                first=20,
            )
            app.job_queue.run_repeating(
                self.watch_ea_state_job,
                interval=10,
                first=5,
            )
        return app
