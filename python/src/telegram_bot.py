"""Telegram bot remote controller & real-time alert notifier for Android."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

from .analyst import GoldAnalyst
from .bridge import Mt5Bridge
from .config import Settings
from .journal import TradeJournal
from .schemas import AnalysisReport, EaSignalPayload, EaState, ExecutionMode
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
    ) -> None:
        for chat_id in self.settings.telegram_allowed_user_ids:
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    reply_markup=reply_markup,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed sending Telegram message to %s: %s", chat_id, exc)

    def _format_status(self, state: EaState | None) -> str:
        session = evaluate_session()
        kz_tag = "KILLZONE ON" if session.in_killzone else f"OFF-KZ ({session.session_name})"
        if state is None:
            return (
                f"Mode: {self.runtime.mode.upper()} | Session: {kz_tag} (Filter={'ON' if self.settings.require_killzone else 'OFF'})\n"
                "MT5 EA State: Not connected yet (attach EA_LearningX_Gold to XAUUSD chart in Exness MT5)."
            )

        age = int(time.time()) - state.updated_at_unix
        flags = []
        if state.paused:
            flags.append("PAUSED")
        if state.daily_halt:
            flags.append("DAILY_HALT")
        if state.hmr_blackout:
            flags.append("HMR_BLACKOUT")
        flag_str = ", ".join(flags) if flags else "ACTIVE"

        lines = [
            f"EA-LearningX ({state.symbol}) | Mode: {self.runtime.mode.upper()}",
            f"EA Status: {flag_str} (heartbeat {age}s ago)",
            f"Session: {session.session_name} ({'KillZone' if session.in_killzone else 'No KillZone'} | Filter={'ON' if self.settings.require_killzone else 'OFF'})",
            f"Bid/Ask: {state.bid:.2f} / {state.ask:.2f} (Spread ${state.spread_usd:.2f})",
            f"Balance: ${state.balance:.2f} | Equity: ${state.equity:.2f} | Daily PnL: ${state.daily_pnl_usd:+.2f}",
            f"Free Margin: ${state.free_margin:.2f}",
            "",
            f"Open Positions ({len(state.positions)}):",
        ]
        if not state.positions:
            lines.append("  None")
        for p in state.positions:
            be_tag = " [BE]" if p.be_applied else ""
            lines.append(
                f"  #{p.ticket} {p.type} {p.volume:.2f}@{p.open_price:.2f} | SL={p.sl:.2f} TP={p.tp:.2f} | ${p.profit:+.2f}{be_tag}"
            )

        lines.append(f"\nPending Orders ({len(state.pending_orders)}):")
        if not state.pending_orders:
            lines.append("  None")
        for o in state.pending_orders:
            lines.append(
                f"  #{o.ticket} {o.type} {o.volume:.2f}@{o.price:.2f} | SL={o.sl:.2f} TP={o.tp:.2f}"
            )
        return "\n".join(lines)

    def _format_report(self, report: AnalysisReport, prefix: str = "") -> str:
        d = report.decision
        s = report.signal
        lines = []
        if prefix:
            lines.append(prefix)
        sess_txt = report.session_status.session_name if report.session_status else "UNKNOWN"
        lines.extend([
            f"AI Bias: {d.bias} | Conf: {d.confidence:.0%} | Sent: {d.news_sentiment_score:+.2f} | Confluence: {report.confluence_score}/5",
            f"Session: {sess_txt} | Intent: {d.order_intent} (allowed={s.trade_allowed})",
        ])
        if d.order_intent != "NONE":
            entry_txt = f"@ {d.limit_price:.2f}" if d.limit_price > 0 else "@ MARKET"
            lines.append(f"Setup: {d.order_intent} {entry_txt} | SL={d.sl_price:.2f} | TP={d.tp_price:.2f}")
        if report.smc_snapshot is not None:
            smc = report.smc_snapshot
            lines.append(
                f"SMC: H4={smc.h4.trend}({smc.h4.last_event}) | H1={smc.h1.trend}({smc.h1.last_event}) | P/D={smc.h1.premium_discount_ratio:.2f}"
            )
        if report.hmr_blackout_active:
            lines.append(f"HMR Guard: BLACKOUT ACTIVE ({report.hmr_event_title})")
        lines.extend([
            "",
            f"Macro: {d.macro_summary}",
            f"News Sentiment: {d.news_sentiment_summary}",
            f"Technical: {d.technical_summary}",
            f"Validation: {report.validation_note}",
        ])
        if report.confluence_notes:
            lines.append("Checklist: " + " | ".join(report.confluence_notes))
        if report.grounded_sources:
            lines.append(f"Sources: {', '.join(report.grounded_sources[:3])}")
        return "\n".join(lines)

    def _build_menu_keyboard(self) -> InlineKeyboardMarkup:
        kz_label = "Hours: KillZone Only" if self.settings.require_killzone else "Hours: Flexible (24/5)"
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("ON (Auto)", callback_data="menu:mode:auto"),
                InlineKeyboardButton("ON (Confirm)", callback_data="menu:mode:confirm"),
                InlineKeyboardButton("STOP (Pause)", callback_data="menu:mode:pause"),
            ],
            [
                InlineKeyboardButton(kz_label, callback_data="menu:toggle_kz"),
                InlineKeyboardButton("Analyze Now", callback_data="menu:analyze"),
            ],
            [
                InlineKeyboardButton("Status", callback_data="menu:status"),
                InlineKeyboardButton("SMC", callback_data="menu:smc"),
                InlineKeyboardButton("Stats", callback_data="menu:stats"),
            ],
            [
                InlineKeyboardButton("Move BE Now", callback_data="menu:be"),
                InlineKeyboardButton("Close All", callback_data="menu:close_all"),
            ],
        ])

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return
        msg = (
            "EA-LearningX Remote Controller (Exness XAUUSD)\n"
            f"Mode: {self.runtime.mode.upper()} | Hours: {'KillZone Only' if self.settings.require_killzone else 'Flexible (24/5)'}\n\n"
            "Tap a button below anytime to turn trading ON/STOP or check status:"
        )
        assert update.message is not None
        await update.message.reply_text(msg, reply_markup=self._build_menu_keyboard())

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return
        state = self.bridge.read_ea_state()
        assert update.message is not None
        await update.message.reply_text(self._format_status(state))

    async def cmd_smc(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return
        assert update.message is not None
        session = evaluate_session()
        market = self.bridge.read_market_context()
        if market is None or not market.h1_bars:
            await update.message.reply_text(
                f"Session: {session.session_name} ({session.reason})\n"
                "No market_context.json bars exported yet by MT5 EA."
            )
            return
        smc = build_smc_snapshot(market)
        await update.message.reply_text(
            f"Session: {session.session_name} ({session.reason})\n\n{format_smc_summary(smc)}"
        )

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return
        st = self.journal.get_stats()
        assert update.message is not None
        await update.message.reply_text(
            "EA-LearningX Journal Stats\n"
            f"Analyses Logged: {st.total_analyses}\n"
            f"Actionable Setups: {st.actionable_signals}\n"
            f"Trades Opened: {st.trades_opened} | Closed: {st.trades_closed}\n"
            f"Break-Even Triggered: {st.be_triggered_count}\n"
            f"Wins / Losses: {st.wins}W / {st.losses}L ({st.win_rate_pct:.1f}%)\n"
            f"Net Closed PnL: ${st.net_realized_pnl:+.2f}"
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
                await self._broadcast(context, f"Analysis failed: {exc}")
            return

        self.runtime.last_report = report
        self.journal.record_analysis(report)
        sig = report.signal

        if self.runtime.mode == "pause":
            sig_paused = sig.model_copy(update={"trade_allowed": False})
            self.bridge.write_signal(sig_paused)
            if manual:
                await self._broadcast(context, self._format_report(report, "[PAUSED MODE - No Execution]"))
            return

        if is_auto:
            self.bridge.write_signal(sig)
            if manual or sig.trade_allowed:
                header = "[AUTO SIGNAL SENT TO MT5]" if sig.trade_allowed else "[AUTO ANALYSIS - NO TRADE]"
                await self._broadcast(context, self._format_report(report, header))
            return

        if sig.order_intent != "NONE" and report.validation_note.startswith("Validated setup"):
            self.bridge.write_signal(sig.model_copy(update={"trade_allowed": False}))
            self.runtime.pending_signals[sig.signal_id] = sig.model_copy(update={"trade_allowed": True})
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("Approve & Execute", callback_data=f"approve:{sig.signal_id}"),
                    InlineKeyboardButton("Reject", callback_data=f"reject:{sig.signal_id}"),
                ]
            ])
            await self._broadcast(
                context,
                self._format_report(report, f"[APPROVAL REQUIRED | id={sig.signal_id}]"),
                reply_markup=keyboard,
            )
        else:
            self.bridge.write_signal(sig)
            if manual:
                await self._broadcast(context, self._format_report(report, "[CONFIRM MODE - NO TRADE]"))

    async def cmd_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return
        assert update.message is not None
        await update.message.reply_text(
            "Running Gemini 4 Hydrogen macro, news, sentiment & deterministic SMC analysis..."
        )
        await self._execute_analysis_cycle(context, manual=True)

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return

        query = update.callback_query
        assert query is not None
        await query.answer()
        data = query.data or ""

        if data == "confirm_close_all":
            cmd = self.bridge.send_command("CLOSE_ALL")
            await query.edit_message_text(f"CLOSE_ALL sent to MT5 EA (id={cmd.command_id}).")
            return
        if data == "cancel_close_all":
            await query.edit_message_text("CLOSE_ALL cancelled.")
            return

        if data.startswith("menu:"):
            action = data.split(":", 1)[1]
            if action.startswith("mode:"):
                new_mode: ExecutionMode = action.split(":", 1)[1]  # type: ignore[assignment]
                self.runtime.mode = new_mode
                self.bridge.send_command("PAUSE" if new_mode == "pause" else "RESUME")
                await query.edit_message_text(
                    f"EA-LearningX Mode: {new_mode.upper()} | Hours: {'KillZone Only' if self.settings.require_killzone else 'Flexible (24/5)'}",
                    reply_markup=self._build_menu_keyboard(),
                )
                return
            if action == "toggle_kz":
                self.settings.require_killzone = not self.settings.require_killzone
                await query.edit_message_text(
                    f"EA-LearningX Mode: {self.runtime.mode.upper()} | Hours: {'KillZone Only' if self.settings.require_killzone else 'Flexible (24/5)'}",
                    reply_markup=self._build_menu_keyboard(),
                )
                return
            if action == "status":
                await query.message.reply_text(self._format_status(self.bridge.read_ea_state()))
                return
            if action == "smc":
                sess = evaluate_session()
                mkt = self.bridge.read_market_context()
                if mkt is None or not mkt.h1_bars:
                    await query.message.reply_text(f"Session: {sess.session_name}\nNo MT5 bars exported yet.")
                else:
                    await query.message.reply_text(f"Session: {sess.session_name}\n\n{format_smc_summary(build_smc_snapshot(mkt))}")
                return
            if action == "stats":
                st = self.journal.get_stats()
                await query.message.reply_text(
                    f"Stats: {st.wins}W/{st.losses}L ({st.win_rate_pct:.1f}%) | Closed PnL: ${st.net_realized_pnl:+.2f} | Analyses: {st.total_analyses}"
                )
                return
            if action == "analyze":
                await query.message.reply_text("Running Gemini 4 Hydrogen + SMC analysis...")
                await self._execute_analysis_cycle(context, manual=True)
                return
            if action == "be":
                cmd = self.bridge.send_command("MOVE_BE_NOW")
                await query.message.reply_text(f"Sent MOVE_BE_NOW (id={cmd.command_id}).")
                return
            if action == "close_all":
                kb = InlineKeyboardMarkup([[
                    InlineKeyboardButton("Confirm CLOSE ALL", callback_data="confirm_close_all"),
                    InlineKeyboardButton("Cancel", callback_data="cancel_close_all"),
                ]])
                await query.message.reply_text("Close all open EA-LearningX positions now?", reply_markup=kb)
                return

        if data.startswith("approve:"):
            sig_id = data.split(":", 1)[1]
            sig = self.runtime.pending_signals.pop(sig_id, None)
            if sig is None:
                await query.edit_message_reply_markup(reply_markup=None)
                await query.message.reply_text(f"Signal {sig_id} already handled or expired.")
                return
            if int(time.time()) > sig.expires_at_unix:
                await query.edit_message_reply_markup(reply_markup=None)
                await query.message.reply_text(f"Signal {sig_id} expired; not sent.")
                return
            self.bridge.write_signal(sig)
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text(
                f"Approved {sig.order_intent} (id={sig_id}) -> written to MT5 EA."
            )
            return

        if data.startswith("reject:"):
            sig_id = data.split(":", 1)[1]
            self.runtime.pending_signals.pop(sig_id, None)
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text(f"Rejected signal {sig_id}.")

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
            await self._broadcast(
                context,
                f"[MT5 OPENED] #{p.ticket} {p.type} {p.volume:.2f} {state.symbol} @ {p.open_price:.2f} | SL={p.sl:.2f} TP={p.tp:.2f}",
            )

        for ticket in sorted(be_set - self.runtime.known_be_tickets):
            if ticket in pos_map:
                p = pos_map[ticket]
                await self._broadcast(
                    context,
                    f"[MT5 BREAK-EVEN] #{p.ticket} {p.type} SL moved to {p.sl:.2f} (entry {p.open_price:.2f}).",
                )

        for ticket in sorted(self.runtime.known_position_tickets - set(pos_map.keys())):
            self.journal.mark_position_closed(ticket)
            await self._broadcast(
                context,
                f"[MT5 CLOSED] Position #{ticket} closed. Equity=${state.equity:.2f} | Daily PnL=${state.daily_pnl_usd:+.2f}",
            )

        for ticket in sorted(set(ord_map.keys()) - self.runtime.known_pending_tickets):
            o = ord_map[ticket]
            await self._broadcast(
                context,
                f"[MT5 PENDING PLACED] #{o.ticket} {o.type} {o.volume:.2f} @ {o.price:.2f} | SL={o.sl:.2f} TP={o.tp:.2f}",
            )

        self.runtime.known_position_tickets = set(pos_map.keys())
        self.runtime.known_be_tickets = be_set
        self.runtime.known_pending_tickets = set(ord_map.keys())

    def build_application(self) -> Application:
        """Create and wire the python-telegram-bot Application."""
        if not self.settings.telegram_bot_token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured in .env")

        app = Application.builder().token(self.settings.telegram_bot_token).build()
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
