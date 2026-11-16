import asyncio
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from python.src.bridge import Mt5Bridge
from python.src.config import Settings
from python.src.ml_engine import MlSignalRunner
from python.src.ml_trainer import train_and_validate_mode
from python.src.schemas import CalendarEvent


def _bars(n: int = 700, step: int = 900, start: int = 1700000000) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    t = start + np.arange(n) * step
    c = 3850.0 + np.cumsum(np.sin(np.arange(n) / 18.0) * 0.4 + rng.normal(0.0, 0.5, n))
    o = np.roll(c, 1)
    o[0] = c[0]
    w = np.abs(rng.normal(0.8, 0.2, n)) + 0.1
    return pd.DataFrame(
        {
            "t": t,
            "o": o,
            "h": np.maximum(o, c) + w,
            "l": np.minimum(o, c) - w,
            "c": c,
            "v": np.full(n, 200.0),
        }
    )


class _FakeFeed:
    def __init__(self, m15_df: pd.DataFrame, m5_df: pd.DataFrame) -> None:
        self.m15_df = m15_df
        self.m5_df = m5_df

    def fetch_closed_bars(self, tf: str, count: int) -> pd.DataFrame:
        src = self.m5_df if tf == "M5" else self.m15_df
        return src.tail(count).reset_index(drop=True)

    def fetch_tick_spread(self) -> tuple[float, float, float]:
        return (3850.0, 3850.24, 0.24)


class _FakeDataFeeds:
    def __init__(self, events: list[CalendarEvent] | None = None) -> None:
        self.events = events or []

    async def fetch_economic_calendar(self, force: bool = False) -> list[CalendarEvent]:
        return self.events


def test_runner_deduplicates_same_closed_bar_and_blocks_unverified_auto(tmp_path):
    settings = Settings(
        _env_file=None,
        bridge_dir=tmp_path / "bridge",
        ml_models_dir=tmp_path / "models",
        require_killzone=False,
    )
    m15_df = _bars(700, 900)
    m5_df = _bars(1200, 300)

    train_and_validate_mode(
        "M15_PRECISION",
        m15_df,
        None,
        settings.ml_models_dir,
        min_pf=99.0,
        min_trades=20,
    )

    bridge = Mt5Bridge(settings.bridge_dir)
    runner = MlSignalRunner(settings, _FakeFeed(m15_df, m5_df), _FakeDataFeeds(), bridge)

    pred, sig, is_new = asyncio.run(
        runner.step_live_cycle(
            "M15_PRECISION",
            "auto",
            now_utc=datetime(2026, 11, 16, 13, 0, tzinfo=timezone.utc),
        )
    )
    assert is_new is True
    assert pred is not None
    assert sig is not None
    assert sig.trade_allowed is False
    assert "unverified" in sig.reason.lower()

    # Second call on exact same closed bar must not re-emit
    pred2, sig2, is_new2 = asyncio.run(
        runner.step_live_cycle(
            "M15_PRECISION",
            "auto",
            now_utc=datetime(2026, 11, 16, 13, 1, tzinfo=timezone.utc),
        )
    )
    assert is_new2 is False
    assert sig2 is None


def test_runner_verified_auto_emits_signal_with_max_hold_and_respects_hmr_blackout(tmp_path):
    settings = Settings(
        _env_file=None,
        bridge_dir=tmp_path / "bridge",
        ml_models_dir=tmp_path / "models",
        require_killzone=True,
    )
    m15_df = _bars(700, 900)
    m5_df = _bars(1400, 300)

    bundle = train_and_validate_mode(
        "M15_PRECISION",
        m15_df,
        None,
        settings.ml_models_dir,
        min_pf=0.20,
        min_trades=1,
    )
    assert bundle.metrics.verified_for_live is True

    # Force low threshold + mock high prob prediction or verify Confirm mode writes signal
    bridge = Mt5Bridge(settings.bridge_dir)
    now_kill = datetime(2026, 11, 16, 13, 30, tzinfo=timezone.utc)  # NY overlap killzone

    # 1) Active HMR blackout must block execution
    blackout_event = CalendarEvent(
        title="US CPI",
        country="USD",
        impact="High",
        event_time_utc=now_kill,
    )
    runner_blk = MlSignalRunner(
        settings,
        _FakeFeed(m15_df, m5_df),
        _FakeDataFeeds([blackout_event]),
        bridge,
    )
    assert runner_blk.is_mode_verified("M15_PRECISION") is True
    _, sig_blk, is_new = asyncio.run(
        runner_blk.step_live_cycle("M15_PRECISION", "auto", now_utc=now_kill)
    )
    assert is_new is True
    assert sig_blk is not None
    assert sig_blk.trade_allowed is False
    assert sig_blk.blackout_until_unix > int(now_kill.timestamp())

    # 2) In confirm mode, signal payload is written to bridge with trade_allowed=False
    runner_cfg = MlSignalRunner(settings, _FakeFeed(m15_df, m5_df), _FakeDataFeeds(), bridge)
    pred, sig_cfg, is_new_cfg = asyncio.run(
        runner_cfg.step_live_cycle("M15_PRECISION", "confirm", now_utc=now_kill)
    )
    assert is_new_cfg is True
    assert pred is not None
    assert sig_cfg is not None
    assert sig_cfg.trade_allowed is False
    assert bridge.signal_path.exists()

    # 3) Manual confirmation emits trade_allowed=True and positive max_hold_until_unix
    pred_forced = pred.model_copy(
        update={
            "order_intent": "MARKET_BUY",
            "entry_price": 3850.24,
            "sl_price": 3846.0,
            "tp_price": 3857.0,
            "max_hold_until_unix": int(now_kill.timestamp()) + 16 * 900,
        }
    )
    sig_conf = runner_cfg.confirm_prediction(pred_forced, now_utc=now_kill)
    assert sig_conf is not None
    assert sig_conf.trade_allowed is True
    assert sig_conf.max_hold_until_unix == int(now_kill.timestamp()) + 16 * 900
    assert sig_conf.strategy_mode == "M15_PRECISION"


def test_runner_retrain_mode_updates_cache_and_runs_m5(tmp_path):
    settings = Settings(
        _env_file=None,
        bridge_dir=tmp_path / "bridge",
        ml_models_dir=tmp_path / "models",
        require_killzone=False,
        ml_m15_bars=650,
        ml_m5_bars=1000,
    )
    m15_df = _bars(700, 900)
    m5_df = _bars(1200, 300)
    bridge = Mt5Bridge(settings.bridge_dir)
    runner = MlSignalRunner(settings, _FakeFeed(m15_df, m5_df), _FakeDataFeeds(), bridge)

    assert runner.get_bundle("M5_INTRADAY") is None
    bundle = asyncio.run(runner.retrain_mode("M5_INTRADAY", min_pf=0.1, min_trades=1))
    assert bundle.mode == "M5_INTRADAY"
    assert runner.is_mode_verified("M5_INTRADAY") is True

    pred, _ = runner.predict_mode("M5_INTRADAY")
    assert pred.timeframe == "M5"
    assert pred.regime in {"TRENDING", "RANGING", "VOLATILE"}
