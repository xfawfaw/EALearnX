"""Live Dual-Mode ML Inference & Exclusive Signal Runner for EA-LearningX."""

from __future__ import annotations

import asyncio
import logging
import uuid
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"hmmlearn.*")

import numpy as np
import pandas as pd

from .bridge import Mt5Bridge
from .config import Settings
from .data_feeds import MarketDataFeeds, compute_hmr_blackout
from .ml_features import FEATURE_COLUMNS, REGIME_COLUMNS, compute_features
from .ml_trainer import MODE_SPECS, TrainedBundle, load_bundle, train_and_validate_mode
from .mt5_feed import Mt5Feed
from .schemas import (
    Bias,
    EaSignalPayload,
    ExecutionMode,
    MlPrediction,
    OrderIntent,
    StrategyMode,
)
from .sessions import evaluate_session

logger = logging.getLogger(__name__)

_INFERENCE_BARS = 350


class MlSignalRunner:
    """Runs causal GaussianHMM + calibrated XGBoost inference and owns ai_signal.json."""

    def __init__(
        self,
        settings: Settings,
        mt5_feed: Mt5Feed,
        feeds: MarketDataFeeds,
        bridge: Mt5Bridge,
    ) -> None:
        self.settings = settings
        self.mt5_feed = mt5_feed
        self.feeds = feeds
        self.bridge = bridge
        self._bundles: dict[StrategyMode, TrainedBundle | None] = {}
        self._last_emitted_bar_ts: dict[StrategyMode, int] = {}

    def get_bundle(self, mode: StrategyMode) -> TrainedBundle | None:
        """Return cached model bundle for mode, loading from disk if not yet loaded."""
        if mode not in self._bundles or self._bundles[mode] is None:
            self._bundles[mode] = load_bundle(mode, self.settings.ml_models_dir)
        return self._bundles[mode]

    def is_mode_verified(self, mode: StrategyMode) -> bool:
        """Return True iff mode has a trained bundle passing OOS PF/trade count gates."""
        bundle = self.get_bundle(mode)
        return bundle is not None and bool(bundle.metrics.verified_for_live)

    def predict_mode(
        self,
        mode: StrategyMode,
        now_utc: datetime | None = None,
    ) -> tuple[MlPrediction, pd.DataFrame]:
        """Score latest closed bar for mode without mutating emitted bar dedup state."""
        bundle = self.get_bundle(mode)
        if bundle is None:
            raise RuntimeError(f"No trained ML bundle found for {mode}")

        spec = MODE_SPECS[mode]
        if mode == "M5_INTRADAY":
            htf_df = self.mt5_feed.fetch_closed_bars("M15", _INFERENCE_BARS)
            df = self.mt5_feed.fetch_closed_bars("M5", _INFERENCE_BARS)
        else:
            htf_df = None
            df = self.mt5_feed.fetch_closed_bars("M15", _INFERENCE_BARS)

        features_df = compute_features(df, htf_df=htf_df)
        if features_df.empty:
            raise RuntimeError(f"Insufficient closed bars for {mode} feature computation")

        X_regime = features_df[list(REGIME_COLUMNS)].to_numpy(dtype=float)
        regime_probs, regime_labels = bundle.regime_model.predict_proba_causal(X_regime)
        latest_regime = regime_labels[-1]
        latest_rprobs = {
            "TRENDING": round(float(regime_probs[-1, 0]), 4),
            "RANGING": round(float(regime_probs[-1, 1]), 4),
            "VOLATILE": round(float(regime_probs[-1, 2]), 4),
        }

        X_row = np.column_stack(
            [
                features_df[list(FEATURE_COLUMNS)].to_numpy(dtype=float)[-1:],
                regime_probs[-1:],
            ]
        )

        prob_long = float(bundle.xgb_long.predict_proba(X_row)[0, 1])
        prob_short = float(bundle.xgb_short.predict_proba(X_row)[0, 1])
        threshold = float(bundle.threshold)

        long_ok = prob_long >= threshold and prob_long > prob_short
        short_ok = prob_short >= threshold and prob_short > prob_long

        alignment_note = ""
        if mode == "M5_INTRADAY" and "htf_ema_21_50_diff" in features_df.columns:
            htf_diff = float(features_df["htf_ema_21_50_diff"].iloc[-1])
            if long_ok and htf_diff < 0.0:
                long_ok = False
                alignment_note = f"BUY vetoed by M15 HTF trend ({htf_diff:+.2f})"
            elif short_ok and htf_diff > 0.0:
                short_ok = False
                alignment_note = f"SELL vetoed by M15 HTF trend ({htf_diff:+.2f})"

        order_intent: OrderIntent = "NONE"
        if long_ok:
            order_intent = "MARKET_BUY"
        elif short_ok:
            order_intent = "MARKET_SELL"

        bid, ask, _ = self.mt5_feed.fetch_tick_spread()
        close_px = float(features_df["close"].iloc[-1])
        atr = max(float(features_df["atr_14"].iloc[-1]), 0.5)

        if order_intent == "MARKET_BUY":
            entry_price = ask if ask > 0.0 else close_px
            sl_price = round(entry_price - spec.sl_atr * atr, 2)
            tp1_price = round(entry_price + spec.tp1_atr * atr, 2)
            tp_price = round(entry_price + spec.tp2_atr * atr, 2)
            trail_atr_usd = round(spec.trail_atr * atr, 2)
        elif order_intent == "MARKET_SELL":
            entry_price = bid if bid > 0.0 else close_px
            sl_price = round(entry_price + spec.sl_atr * atr, 2)
            tp1_price = round(entry_price - spec.tp1_atr * atr, 2)
            tp_price = round(entry_price - spec.tp2_atr * atr, 2)
            trail_atr_usd = round(spec.trail_atr * atr, 2)
        else:
            entry_price = round((bid + ask) / 2.0, 2) if (bid > 0.0 and ask > 0.0) else round(close_px, 2)
            sl_price = 0.0
            tp1_price = 0.0
            tp_price = 0.0
            trail_atr_usd = 0.0

        now_dt = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        now_unix = int(now_dt.timestamp())
        max_hold_until = (
            (now_unix + spec.runner_horizon_bars * spec.bar_seconds)
            if order_intent != "NONE"
            else 0
        )
        bar_time_unix = int(features_df["t"].iloc[-1])
        confidence = int(round(max(prob_long, prob_short) * 100))

        reason_parts = [
            f"{mode} {latest_regime}",
            f"P(L)={prob_long:.2f}/P(S)={prob_short:.2f} (th={threshold:.2f})",
        ]
        if alignment_note:
            reason_parts.append(alignment_note)

        pred = MlPrediction(
            mode=mode,
            bar_time_unix=bar_time_unix,
            timeframe=spec.timeframe,
            regime=latest_regime,
            regime_probs=latest_rprobs,
            prob_long=round(prob_long, 4),
            prob_short=round(prob_short, 4),
            threshold=threshold,
            order_intent=order_intent,
            confidence=confidence,
            entry_price=round(entry_price, 2),
            sl_price=sl_price,
            tp_price=tp_price,
            tp1_price=tp1_price,
            trail_atr_usd=trail_atr_usd,
            max_hold_until_unix=max_hold_until,
            reason=" | ".join(reason_parts),
            verified_for_live=bool(bundle.metrics.verified_for_live),
        )
        return pred, df

    async def step_live_cycle(
        self,
        mode: StrategyMode,
        execution_mode: ExecutionMode,
        now_utc: datetime | None = None,
    ) -> tuple[MlPrediction | None, EaSignalPayload | None, bool]:
        """Evaluate latest closed bar once per bar close and emit ai_signal.json if eligible."""
        if self.get_bundle(mode) is None:
            return (None, None, False)

        now_dt = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        pred, _ = await asyncio.to_thread(self.predict_mode, mode, now_dt)

        if self._last_emitted_bar_ts.get(mode) == pred.bar_time_unix:
            return (pred, None, False)

        self._last_emitted_bar_ts[mode] = pred.bar_time_unix
        spec = MODE_SPECS[mode]
        now_unix = int(now_dt.timestamp())

        session = evaluate_session(now_dt)
        events = await self.feeds.fetch_economic_calendar()
        is_blackout, blackout_until, blackout_reason = compute_hmr_blackout(
            events,
            now_utc=now_dt,
            pre_minutes=self.settings.hmr_pre_news_minutes,
            post_minutes=self.settings.hmr_post_news_minutes,
        )

        blockers: list[str] = []
        if not pred.verified_for_live:
            blockers.append("Unverified ML bundle (PF/OOS gate failed)")
        if execution_mode == "pause":
            blockers.append("Execution paused")
        elif execution_mode == "confirm":
            blockers.append("Awaiting manual confirmation")
        if not session.market_open:
            blockers.append(session.reason)
        elif self.settings.require_killzone and not session.in_killzone:
            blockers.append(session.reason)
        if is_blackout:
            blockers.append(blackout_reason)

        ea_state = self.bridge.read_ea_state()
        if ea_state is not None:
            if ea_state.daily_halt:
                blockers.append("MT5 EA daily loss halt active")
            if ea_state.paused:
                blockers.append("MT5 EA paused")
            unheld = [p for p in ea_state.positions if not p.held]
            at_risk = [p for p in unheld if not p.be_applied]
            if at_risk:
                blockers.append(
                    f"Active unheld position #{at_risk[0].ticket} still at risk (pre-BE)"
                )
            elif len(unheld) >= 2:
                blockers.append(f"Max 2 unheld positions running ({len(unheld)} active)")

        _, _, spread_usd = self.mt5_feed.fetch_tick_spread()
        if spread_usd > 0.60:
            blockers.append(f"Spread ${spread_usd:.2f} > $0.60 limit")

        if pred.order_intent != "NONE" and pred.sl_price > 0.0 and pred.entry_price > 0.0:
            sl_dist = abs(pred.entry_price - pred.sl_price)
            if sl_dist > self.settings.ml_max_sl_usd:
                blockers.append(
                    f"SL distance ${sl_dist:.2f} > ${self.settings.ml_max_sl_usd:.2f} volatility cap"
                )

        trade_allowed = (
            pred.order_intent != "NONE"
            and execution_mode == "auto"
            and pred.verified_for_live
            and not blockers
        )

        if pred.order_intent == "MARKET_BUY":
            bias: Bias = "BULLISH"
        elif pred.order_intent == "MARKET_SELL":
            bias = "BEARISH"
        elif pred.prob_long > pred.prob_short + 0.05:
            bias = "BULLISH"
        elif pred.prob_short > pred.prob_long + 0.05:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        reason = pred.reason if not blockers else f"{'; '.join(blockers)} | {pred.reason}"
        payload = EaSignalPayload(
            signal_id=f"ml_{mode.lower()}_{pred.bar_time_unix}_{uuid.uuid4().hex[:6]}",
            generated_at_unix=now_unix,
            expires_at_unix=now_unix + spec.bar_seconds,
            bias=bias,
            order_intent=pred.order_intent,
            trade_allowed=trade_allowed,
            limit_price=pred.entry_price if pred.order_intent != "NONE" else 0.0,
            sl_price=pred.sl_price,
            tp_price=pred.tp_price,
            tp1_price=pred.tp1_price,
            trail_atr_usd=pred.trail_atr_usd,
            confidence=round(max(pred.prob_long, pred.prob_short), 3),
            blackout_until_unix=blackout_until,
            max_hold_until_unix=pred.max_hold_until_unix if trade_allowed else 0,
            strategy_mode=mode,
            regime=pred.regime,
            reason=reason[:220],
        )

        if execution_mode in {"auto", "confirm"}:
            self.bridge.write_signal(payload)

        return (pred, payload, True)

    def confirm_prediction(
        self,
        pred: MlPrediction,
        now_utc: datetime | None = None,
    ) -> EaSignalPayload | None:
        """Manually confirm and dispatch an actionable ML prediction to ai_signal.json."""
        if pred.order_intent == "NONE":
            return None
        spec = MODE_SPECS[pred.mode]
        now_dt = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        now_unix = int(now_dt.timestamp())
        bias: Bias = "BULLISH" if pred.order_intent == "MARKET_BUY" else "BEARISH"
        max_hold = pred.max_hold_until_unix or (now_unix + spec.runner_horizon_bars * spec.bar_seconds)

        payload = EaSignalPayload(
            signal_id=f"ml_cfg_{pred.mode.lower()}_{uuid.uuid4().hex[:8]}",
            generated_at_unix=now_unix,
            expires_at_unix=now_unix + spec.bar_seconds,
            bias=bias,
            order_intent=pred.order_intent,
            trade_allowed=True,
            limit_price=pred.entry_price,
            sl_price=pred.sl_price,
            tp_price=pred.tp_price,
            tp1_price=pred.tp1_price,
            trail_atr_usd=pred.trail_atr_usd,
            confidence=round(max(pred.prob_long, pred.prob_short), 3),
            blackout_until_unix=0,
            max_hold_until_unix=max_hold,
            strategy_mode=pred.mode,
            regime=pred.regime,
            reason=f"Manual Confirm | {pred.reason}"[:220],
        )
        self.bridge.write_signal(payload)
        return payload

    async def retrain_mode(
        self,
        mode: StrategyMode,
        min_pf: float | None = None,
        min_trades: int | None = None,
    ) -> TrainedBundle:
        """Retrain & walk-forward validate a strategy mode in a background thread."""
        pf_floor = self.settings.ml_min_profit_factor if min_pf is None else min_pf
        trades_floor = self.settings.ml_min_oos_trades if min_trades is None else min_trades

        def _worker() -> TrainedBundle:
            m15_df = self.mt5_feed.fetch_closed_bars("M15", self.settings.ml_m15_bars)
            if mode == "M5_INTRADAY":
                m5_df = self.mt5_feed.fetch_closed_bars("M5", self.settings.ml_m5_bars)
                return train_and_validate_mode(
                    mode="M5_INTRADAY",
                    df=m5_df,
                    htf_df=m15_df,
                    models_dir=self.settings.ml_models_dir,
                    spread_cost_usd=self.settings.ml_spread_cost_usd,
                    min_pf=pf_floor,
                    min_trades=trades_floor,
                    require_killzone=self.settings.require_killzone,
                    max_sl_usd=self.settings.ml_max_sl_usd,
                )
            return train_and_validate_mode(
                mode="M15_PRECISION",
                df=m15_df,
                htf_df=None,
                models_dir=self.settings.ml_models_dir,
                spread_cost_usd=self.settings.ml_spread_cost_usd,
                min_pf=pf_floor,
                min_trades=trades_floor,
                require_killzone=self.settings.require_killzone,
                max_sl_usd=self.settings.ml_max_sl_usd,
            )

        candidate = await asyncio.to_thread(_worker)
        # Keep active cache synchronized with disk (preserves prior verified bundle on rejection)
        self._bundles[mode] = load_bundle(mode, self.settings.ml_models_dir) or candidate
        return candidate
