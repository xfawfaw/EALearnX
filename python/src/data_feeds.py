"""Live macroeconomic calendar, news feeds, and Exness HMR blackout detection."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import feedparser
import httpx

from .schemas import CalendarEvent, NewsHeadline

logger = logging.getLogger(__name__)

FOREX_FACTORY_WEEK_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

DEFAULT_RSS_FEEDS: list[tuple[str, str]] = [
    ("FXStreet Gold", "https://www.fxstreet.com/rss/news?category=commodities"),
    ("FXStreet Macro", "https://www.fxstreet.com/rss/news"),
    ("Yahoo Finance Gold", "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC=F&region=US&lang=en-US"),
    ("Investing Gold", "https://www.investing.com/rss/news_14.rss"),
]


class MarketDataFeeds:
    """Fetches economic calendar events and gold/macro headlines with caching."""

    def __init__(self, finnhub_api_key: str = "") -> None:
        self.finnhub_api_key = finnhub_api_key
        self._calendar_cache: list[CalendarEvent] = []
        self._calendar_fetched_at: float = 0.0
        self._calendar_ttl_sec: float = 900.0  # 15 minutes

    async def fetch_economic_calendar(self, force: bool = False) -> list[CalendarEvent]:
        """Fetch this week's economic calendar (cached for 15 minutes)."""
        now_ts = time.monotonic()
        if not force and self._calendar_cache and (now_ts - self._calendar_fetched_at) < self._calendar_ttl_sec:
            return self._calendar_cache

        events: list[CalendarEvent] = []
        try:
            async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
                resp = await client.get(
                    FOREX_FACTORY_WEEK_URL,
                    headers={"User-Agent": "EA-LearningX/1.0"},
                )
                resp.raise_for_status()
                raw_items = resp.json()

            for item in raw_items:
                country = str(item.get("country", "")).upper()
                impact = str(item.get("impact", "")).capitalize()
                if country not in {"USD", "ALL"}:
                    continue
                if impact not in {"High", "Medium"}:
                    continue

                raw_date = str(item.get("date", "")).strip()
                if not raw_date:
                    continue
                try:
                    dt = datetime.fromisoformat(raw_date).astimezone(timezone.utc)
                except ValueError:
                    continue

                events.append(
                    CalendarEvent(
                        title=str(item.get("title", "Unknown Event")).strip(),
                        country=country,
                        impact=impact,
                        event_time_utc=dt,
                        forecast=str(item.get("forecast", "") or "").strip(),
                        previous=str(item.get("previous", "") or "").strip(),
                    )
                )

            events.sort(key=lambda e: e.event_time_utc)
            self._calendar_cache = events
            self._calendar_fetched_at = now_ts
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed fetching ForexFactory calendar: %s", exc)
            return self._calendar_cache

        return events

    async def fetch_gold_news(self, limit: int = 18) -> list[NewsHeadline]:
        """Fetch recent gold, USD, and macro headlines from RSS (+ optional Finnhub)."""
        headlines: list[NewsHeadline] = []
        seen_titles: set[str] = set()

        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            for source_name, url in DEFAULT_RSS_FEEDS:
                try:
                    resp = await client.get(url, headers={"User-Agent": "EA-LearningX/1.0"})
                    resp.raise_for_status()
                    parsed = feedparser.parse(resp.text)
                    for entry in parsed.entries[:10]:
                        title = str(entry.get("title", "")).strip()
                        if not title:
                            continue
                        norm = title.lower()
                        if norm in seen_titles:
                            continue
                        seen_titles.add(norm)
                        headlines.append(
                            NewsHeadline(
                                title=title,
                                source=source_name,
                                published=str(entry.get("published", "")).strip(),
                                link=str(entry.get("link", "")).strip(),
                            )
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("RSS feed %s unavailable: %s", source_name, exc)

            if self.finnhub_api_key:
                try:
                    resp = await client.get(
                        "https://finnhub.io/api/v1/news",
                        params={"category": "forex", "token": self.finnhub_api_key},
                    )
                    resp.raise_for_status()
                    for item in resp.json()[:10]:
                        title = str(item.get("headline", "")).strip()
                        if title and title.lower() not in seen_titles:
                            seen_titles.add(title.lower())
                            headlines.append(
                                NewsHeadline(
                                    title=title,
                                    source=f"Finnhub:{item.get('source', 'forex')}",
                                    published="",
                                    link=str(item.get("url", "")),
                                )
                            )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Finnhub news failed: %s", exc)

        return headlines[:limit]


def compute_hmr_blackout(
    events: list[CalendarEvent],
    now_utc: datetime,
    pre_minutes: int,
    post_minutes: int,
) -> tuple[bool, int, str | None]:
    """Check whether current UTC time falls inside an Exness HMR high-impact USD news window.

    Returns:
        (is_blackout_active, blackout_until_unix, active_event_title)
    """
    pre_delta = timedelta(minutes=pre_minutes)
    post_delta = timedelta(minutes=post_minutes)
    max_until: datetime | None = None
    active_title: str | None = None

    for ev in events:
        if ev.impact.lower() != "high" or ev.country.upper() not in {"USD", "ALL"}:
            continue
        start_win = ev.event_time_utc - pre_delta
        end_win = ev.event_time_utc + post_delta
        if start_win <= now_utc <= end_win:
            if max_until is None or end_win > max_until:
                max_until = end_win
                active_title = ev.title

    if max_until is None:
        return False, 0, None
    return True, int(max_until.timestamp()), active_title


def format_upcoming_calendar(
    events: list[CalendarEvent],
    now_utc: datetime,
    hours_ahead: int = 24,
    hours_behind: int = 4,
) -> str:
    """Format relevant recent and upcoming USD calendar events for LLM context."""
    window_start = now_utc - timedelta(hours=hours_behind)
    window_end = now_utc + timedelta(hours=hours_ahead)
    relevant = [e for e in events if window_start <= e.event_time_utc <= window_end]
    if not relevant:
        return "No High/Medium USD events in the -4h/+24h window."

    lines: list[str] = []
    for ev in relevant[:15]:
        ts = ev.event_time_utc.strftime("%Y-%m-%d %H:%M UTC")
        extra = []
        if ev.forecast:
            extra.append(f"forecast={ev.forecast}")
        if ev.previous:
            extra.append(f"prev={ev.previous}")
        suffix = f" ({', '.join(extra)})" if extra else ""
        lines.append(f"- [{ts}] [{ev.impact}] {ev.country} {ev.title}{suffix}")
    return "\n".join(lines)
