"""SQLite signal & trade journal for tracking confluence quality and win rate."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .schemas import AnalysisReport, EaClosedDeal, EaPosition, EaSignalPayload, MlPrediction


@dataclass
class JournalStats:
    """Aggregated signal & trade performance metrics."""

    total_analyses: int
    actionable_signals: int
    trades_opened: int
    trades_closed: int
    be_triggered_count: int
    wins: int
    losses: int
    win_rate_pct: float
    net_realized_pnl: float
    profit_factor: float = 0.0


class TradeJournal:
    """Persists signals and MT5 ticket lifecycles in SQLite."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    signal_id TEXT PRIMARY KEY,
                    created_at INTEGER NOT NULL,
                    bias TEXT NOT NULL,
                    order_intent TEXT NOT NULL,
                    trade_allowed INTEGER NOT NULL,
                    confluence_score INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    sentiment_score REAL NOT NULL,
                    session_name TEXT NOT NULL,
                    limit_price REAL NOT NULL,
                    sl_price REAL NOT NULL,
                    tp_price REAL NOT NULL,
                    validation_note TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trades (
                    ticket INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    volume REAL NOT NULL,
                    open_price REAL NOT NULL,
                    sl REAL NOT NULL,
                    tp REAL NOT NULL,
                    opened_at INTEGER NOT NULL,
                    be_applied INTEGER NOT NULL DEFAULT 0,
                    last_profit REAL NOT NULL DEFAULT 0.0,
                    closed_at INTEGER,
                    status TEXT NOT NULL DEFAULT 'OPEN'
                );
                """
            )

    def record_analysis(self, report: AnalysisReport) -> None:
        """Log every analysis cycle and its confluence outcome."""
        d = report.decision
        s = report.signal
        session_name = report.session_status.session_name if report.session_status else "UNKNOWN"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO signals (
                    signal_id, created_at, bias, order_intent, trade_allowed,
                    confluence_score, confidence, sentiment_score, session_name,
                    limit_price, sl_price, tp_price, validation_note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    s.signal_id,
                    s.generated_at_unix,
                    d.bias,
                    d.order_intent,
                    1 if s.trade_allowed else 0,
                    report.confluence_score,
                    d.confidence,
                    d.news_sentiment_score,
                    session_name,
                    s.limit_price,
                    s.sl_price,
                    s.tp_price,
                    report.validation_note,
                ),
            )

    def record_ml_signal(
        self,
        pred: MlPrediction,
        sig: EaSignalPayload,
        session_name: str = "UNKNOWN",
    ) -> None:
        """Log every closed-bar ML signal evaluation into the signals table."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO signals (
                    signal_id, created_at, bias, order_intent, trade_allowed,
                    confluence_score, confidence, sentiment_score, session_name,
                    limit_price, sl_price, tp_price, validation_note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sig.signal_id,
                    sig.generated_at_unix,
                    sig.bias,
                    sig.order_intent,
                    1 if sig.trade_allowed else 0,
                    0,
                    sig.confidence,
                    0.0,
                    session_name,
                    sig.limit_price,
                    sig.sl_price,
                    sig.tp_price,
                    f"[{pred.mode}/{pred.regime}] {sig.reason}",
                ),
            )

    def upsert_open_position(self, symbol: str, pos: EaPosition) -> None:
        """Record or update an active MT5 position."""
        now = int(time.time())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trades (
                    ticket, symbol, side, volume, open_price, sl, tp, opened_at, be_applied, last_profit, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
                ON CONFLICT(ticket) DO UPDATE SET
                    sl = excluded.sl,
                    tp = excluded.tp,
                    be_applied = MAX(trades.be_applied, excluded.be_applied),
                    last_profit = excluded.last_profit,
                    status = 'OPEN'
                """,
                (
                    pos.ticket,
                    symbol,
                    pos.type,
                    pos.volume,
                    pos.open_price,
                    pos.sl,
                    pos.tp,
                    now,
                    1 if pos.be_applied else 0,
                    pos.profit,
                ),
            )

    def mark_position_closed(self, ticket: int) -> None:
        """Mark a previously open ticket as closed."""
        now = int(time.time())
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE trades
                SET status = 'CLOSED', closed_at = ?
                WHERE ticket = ? AND status = 'OPEN'
                """,
                (now, ticket),
            )

    def sync_closed_deals(self, symbol: str, deals: list[EaClosedDeal]) -> None:
        """Reconcile exact MT5 closed deal profit/swap/commission into SQLite."""
        if not deals:
            return
        with self._connect() as conn:
            for d in deals:
                updated = conn.execute(
                    """
                    UPDATE trades
                    SET last_profit = ?, closed_at = ?, status = 'CLOSED'
                    WHERE ticket = ?
                    """,
                    (d.net_pnl, d.closed_at, d.pos_id),
                ).rowcount
                if updated == 0:
                    conn.execute(
                        """
                        INSERT INTO trades (
                            ticket, symbol, side, volume, open_price, sl, tp,
                            opened_at, be_applied, last_profit, closed_at, status
                        ) VALUES (?, ?, ?, ?, ?, 0.0, 0.0, ?, 0, ?, ?, 'CLOSED')
                        """,
                        (
                            d.pos_id,
                            symbol,
                            d.side,
                            d.volume,
                            d.close_price,
                            d.closed_at,
                            d.net_pnl,
                            d.closed_at,
                        ),
                    )

    def get_stats(self) -> JournalStats:
        """Compute journal statistics for /stats command."""
        with self._connect() as conn:
            total_analyses = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
            actionable = conn.execute(
                "SELECT COUNT(*) FROM signals WHERE order_intent != 'NONE'"
            ).fetchone()[0]
            opened = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            closed = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE status = 'CLOSED'"
            ).fetchone()[0]
            be_count = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE be_applied = 1"
            ).fetchone()[0]
            wins = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE status = 'CLOSED' AND last_profit > 0"
            ).fetchone()[0]
            losses = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE status = 'CLOSED' AND last_profit <= 0"
            ).fetchone()[0]
            net_pnl = conn.execute(
                "SELECT COALESCE(SUM(last_profit), 0.0) FROM trades WHERE status = 'CLOSED'"
            ).fetchone()[0]
            gross_win = conn.execute(
                "SELECT COALESCE(SUM(last_profit), 0.0) FROM trades WHERE status = 'CLOSED' AND last_profit > 0"
            ).fetchone()[0]
            gross_loss = conn.execute(
                "SELECT COALESCE(ABS(SUM(last_profit)), 0.0) FROM trades WHERE status = 'CLOSED' AND last_profit <= 0"
            ).fetchone()[0]

        win_rate = (wins / closed * 100.0) if closed > 0 else 0.0
        pf = (float(gross_win) / float(gross_loss)) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)
        return JournalStats(
            total_analyses=total_analyses,
            actionable_signals=actionable,
            trades_opened=opened,
            trades_closed=closed,
            be_triggered_count=be_count,
            wins=wins,
            losses=losses,
            win_rate_pct=round(win_rate, 1),
            net_realized_pnl=round(float(net_pnl), 2),
            profit_factor=round(pf, 2),
        )

    def format_learning_context(self, limit_trades: int = 8, limit_signals: int = 5) -> str:
        """Build a compact self-learning feedback block from past closed trades and setups."""
        stats = self.get_stats()
        with self._connect() as conn:
            side_rows = conn.execute(
                """
                SELECT side,
                       COUNT(*) AS total,
                       SUM(CASE WHEN last_profit > 0 THEN 1 ELSE 0 END) AS wins,
                       COALESCE(SUM(last_profit), 0.0) AS pnl
                FROM trades
                WHERE status = 'CLOSED'
                GROUP BY side
                """
            ).fetchall()
            recent_trades = conn.execute(
                """
                SELECT ticket, side, volume, open_price, last_profit, be_applied
                FROM trades
                WHERE status = 'CLOSED'
                ORDER BY COALESCE(closed_at, opened_at) DESC
                LIMIT ?
                """,
                (limit_trades,),
            ).fetchall()
            recent_signals = conn.execute(
                """
                SELECT bias, order_intent, confluence_score, session_name, validation_note
                FROM signals
                WHERE order_intent != 'NONE'
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit_signals,),
            ).fetchall()

        if stats.trades_closed == 0 and not recent_signals:
            return "No closed trades or actionable history recorded yet (fresh journal baseline)."

        lines = [
            (
                f"Overall Closed Performance: {stats.wins}W/{stats.losses}L "
                f"({stats.win_rate_pct:.1f}%), BE triggered={stats.be_triggered_count}, "
                f"Net PnL=${stats.net_realized_pnl:+.2f}"
            )
        ]
        if side_rows:
            side_parts = [
                f"{side}: {wins}/{total} wins (${pnl:+.2f})"
                for side, total, wins, pnl in side_rows
            ]
            lines.append("By Side: " + " | ".join(side_parts))
        if recent_trades:
            lines.append("Recent Closed Trades (newest->oldest):")
            for ticket, side, vol, op, pnl, be in recent_trades:
                outcome = "WIN" if pnl > 0 else "LOSS"
                lines.append(
                    f"  - #{ticket} {side} {vol:.2f}@{op:.2f} -> {outcome} (${pnl:+.2f}, BE={'yes' if be else 'no'})"
                )
        if recent_signals:
            lines.append("Recent Actionable Setups (newest->oldest):")
            for bias, intent, score, sess, note in recent_signals:
                lines.append(
                    f"  - {intent} ({bias}) in {sess} | Score={score}/5 | {note}"
                )
        return "\n".join(lines)
