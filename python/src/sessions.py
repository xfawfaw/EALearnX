"""XAUUSD trading session & London/NY kill-zone filter (UTC)."""

from __future__ import annotations

from datetime import datetime, time, timezone

from .schemas import SessionStatus


def evaluate_session(now_utc: datetime | None = None) -> SessionStatus:
    """Evaluate current UTC time against XAUUSD liquidity kill-zones & rollover blocks.

    Prime XAUUSD windows (UTC):
    - London Kill Zone: 07:00 - 11:00 UTC (14:00 - 18:00 WIB)
    - NY / London-NY Overlap: 12:00 - 17:00 UTC (19:00 - 00:00 WIB)
    """
    dt = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    weekday = dt.weekday()  # Mon=0 .. Sun=6
    t = dt.time()

    # Weekend closure (Saturday all day, Sunday before 22:00 UTC, Friday after 21:55 UTC)
    if weekday == 5:
        return SessionStatus(
            session_name="WEEKEND_CLOSED",
            in_killzone=False,
            market_open=False,
            reason="Weekend market closed (Saturday).",
        )
    if weekday == 6 and t < time(22, 0):
        return SessionStatus(
            session_name="WEEKEND_CLOSED",
            in_killzone=False,
            market_open=False,
            reason="Weekend market closed (Sunday pre-open).",
        )
    if weekday == 4 and t >= time(21, 55):
        return SessionStatus(
            session_name="WEEKEND_CLOSED",
            in_killzone=False,
            market_open=False,
            reason="Weekend market closed (Friday close).",
        )

    # Friday late session guard (avoid holding fresh entries into weekend gap)
    if weekday == 4 and t >= time(19, 0):
        return SessionStatus(
            session_name="FRIDAY_LATE_GUARD",
            in_killzone=False,
            market_open=True,
            reason="Blocked: Friday >= 19:00 UTC weekend gap guard.",
        )

    # Exness daily rollover / low-liquidity spread spike window (21:00 - 23:00 UTC)
    if time(21, 0) <= t < time(23, 0):
        return SessionStatus(
            session_name="DAILY_ROLLOVER",
            in_killzone=False,
            market_open=True,
            reason="Blocked: Daily rollover spread-widening window (21:00-23:00 UTC).",
        )

    if time(7, 0) <= t < time(11, 0):
        return SessionStatus(
            session_name="LONDON_KILLZONE",
            in_killzone=True,
            market_open=True,
            reason="Active London Kill Zone (07:00-11:00 UTC).",
        )

    if time(12, 0) <= t < time(17, 0):
        return SessionStatus(
            session_name="NY_OVERLAP_KILLZONE",
            in_killzone=True,
            market_open=True,
            reason="Active NY / London-NY Overlap Kill Zone (12:00-17:00 UTC).",
        )

    if time(11, 0) <= t < time(12, 0):
        return SessionStatus(
            session_name="LONDON_LUNCH_LULL",
            in_killzone=False,
            market_open=True,
            reason="Outside kill zone: pre-NY lull (11:00-12:00 UTC).",
        )

    if time(17, 0) <= t < time(21, 0):
        return SessionStatus(
            session_name="NY_LATE_SESSION",
            in_killzone=False,
            market_open=True,
            reason="Outside kill zone: late NY session (17:00-21:00 UTC).",
        )

    return SessionStatus(
        session_name="ASIAN_SESSION",
        in_killzone=False,
        market_open=True,
        reason="Outside kill zone: low-volume Asian session (23:00-07:00 UTC).",
    )
