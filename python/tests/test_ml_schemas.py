from python.src.config import Settings
from python.src.schemas import EaSignalPayload, MlMetricsSummary, MlPrediction


def test_ea_signal_payload_has_max_hold_default_and_override():
    default_sig = EaSignalPayload(
        signal_id="sig1",
        generated_at_unix=1700000000,
        expires_at_unix=1700000900,
    )
    assert default_sig.max_hold_until_unix == 0
    assert default_sig.tp1_price == 0.0
    assert default_sig.trail_atr_usd == 0.0

    ml_sig = EaSignalPayload(
        signal_id="sig2",
        generated_at_unix=1700000000,
        expires_at_unix=1700000900,
        order_intent="MARKET_BUY",
        trade_allowed=True,
        sl_price=3840.0,
        tp_price=3870.0,
        tp1_price=3856.0,
        trail_atr_usd=6.0,
        max_hold_until_unix=1700028800,
        strategy_mode="M15_PRECISION",
        regime="TRENDING",
    )
    assert ml_sig.max_hold_until_unix == 1700028800
    assert ml_sig.tp1_price == 3856.0
    assert ml_sig.trail_atr_usd == 6.0
    assert ml_sig.strategy_mode == "M15_PRECISION"
    assert ml_sig.regime == "TRENDING"


def test_ml_models_and_settings_defaults(tmp_path):
    summary = MlMetricsSummary(
        mode="M5_INTRADAY",
        trained_at_unix=1700000000,
        oos_trades=45,
        oos_win_rate=0.58,
        oos_profit_factor=1.38,
        oos_net_pnl_usd=18.5,
        chosen_threshold=0.55,
        verified_for_live=True,
    )
    pred = MlPrediction(
        mode="M5_INTRADAY",
        bar_time_unix=1700000000,
        timeframe="M5",
        regime="TRENDING",
        regime_probs={"TRENDING": 0.82, "RANGING": 0.12, "VOLATILE": 0.06},
        prob_long=0.68,
        prob_short=0.22,
        threshold=0.55,
        order_intent="MARKET_BUY",
        confidence=68,
        entry_price=3850.0,
        sl_price=3845.5,
        tp_price=3856.75,
        max_hold_until_unix=1700007200,
        reason="M15 bull trend + M5 breakout",
        verified_for_live=True,
    )
    s = Settings(_env_file=None, ml_models_dir=tmp_path / "models")
    assert summary.verified_for_live is True
    assert summary.fail_reason == ""
    assert pred.order_intent == "MARKET_BUY"
    assert s.default_strategy_mode == "M15_PRECISION"
    assert s.mt5_symbol == "XAUUSDm"
    assert s.require_killzone is True
    assert s.ml_spread_cost_usd == 0.30
    assert s.ml_min_profit_factor == 1.05
    assert s.ml_min_oos_trades == 20
    assert s.ml_m15_bars == 6000
    assert s.ml_m5_bars == 12000
    assert s.ml_poll_interval_seconds == 15
    assert s.ml_retrain_hour_utc == 22
    assert s.ml_max_sl_usd == 15.0
    assert s.ml_cent_lot_min == 0.55
    assert s.ml_cent_lot_max == 0.90
