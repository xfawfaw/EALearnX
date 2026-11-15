"""Unit tests for economic calendar HMR blackout window detection."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python.src.data_feeds import compute_hmr_blackout, format_upcoming_calendar
from python.src.schemas import CalendarEvent


def test_hmr_blackout_window() -> None:
    nfp_time = datetime(2026, 11, 6, 13, 30, tzinfo=timezone.utc)
    events = [
        CalendarEvent(
            title="Non-Farm Employment Change",
            country="USD",
            impact="High",
            event_time_utc=nfp_time,
            forecast="180K",
            previous="175K",
        ),
        CalendarEvent(
            title="Medium Speech",
            country="USD",
            impact="Medium",
            event_time_utc=datetime(2026, 11, 6, 15, 0, tzinfo=timezone.utc),
        ),
    ]

    # 10 minutes before NFP -> active
    active, until_unix, title = compute_hmr_blackout(
        events,
        datetime(2026, 11, 6, 13, 20, tzinfo=timezone.utc),
        pre_minutes=15,
        post_minutes=10,
    )
    assert active is True
    assert title == "Non-Farm Employment Change"
    assert until_unix == int(datetime(2026, 11, 6, 13, 40, tzinfo=timezone.utc).timestamp())

    # 30 minutes before NFP -> inactive
    active2, until2, title2 = compute_hmr_blackout(
        events,
        datetime(2026, 11, 6, 13, 0, tzinfo=timezone.utc),
        pre_minutes=15,
        post_minutes=10,
    )
    assert active2 is False
    assert until2 == 0
    assert title2 is None


def test_format_upcoming_calendar() -> None:
    now = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
    events = [
        CalendarEvent(
            title="CPI y/y",
            country="USD",
            impact="High",
            event_time_utc=datetime(2026, 11, 6, 13, 30, tzinfo=timezone.utc),
            forecast="2.6%",
            previous="2.5%",
        )
    ]
    text = format_upcoming_calendar(events, now)
    assert "CPI y/y" in text
    assert "forecast=2.6%" in text
