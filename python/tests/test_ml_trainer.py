import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

from python.src.ml_trainer import (
    MODE_SPECS,
    CalibratedModelWrapper,
    compute_split_tp_outcomes,
    compute_triple_barrier_outcomes,
    load_bundle,
    search_best_threshold,
    simulate_oos_backtest,
    train_and_validate_mode,
)


def _trend_bars(n: int = 1000, step: int = 900, drift: float = 0.45, noise: float = 0.6) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    t = 1700000000 + np.arange(n) * step
    closes = [3800.0]
    for i in range(1, n):
        # Alternating smooth macro waves so both long & short labels occur naturally
        wave = drift * np.sin(i / 25.0)
        closes.append(closes[-1] + wave + rng.normal(0.0, noise))
    c = np.array(closes)
    o = np.roll(c, 1)
    o[0] = c[0]
    wick = np.abs(rng.normal(0.9, 0.25, n))
    return pd.DataFrame({
        "t": t,
        "o": o,
        "h": np.maximum(o, c) + wick,
        "l": np.minimum(o, c) - wick,
        "c": c,
        "v": rng.integers(100, 400, n).astype(float),
    })


def test_mode_specs_constants():
    assert "M15_PRECISION" in MODE_SPECS
    assert "M5_INTRADAY" in MODE_SPECS

    m15 = MODE_SPECS["M15_PRECISION"]
    assert m15.sl_atr == 1.5
    assert m15.tp_atr == 2.4
    assert m15.horizon_bars == 16
    assert m15.bar_seconds == 900
    assert m15.timeframe == "M15"
    assert m15.tp1_atr == 1.5
    assert m15.layer_tp1_atr == 1.25
    assert m15.tp2_atr == 5.0
    assert m15.trail_atr == 1.5
    assert m15.be_buffer_usd == 0.30
    assert m15.runner_horizon_bars == 32

    m5 = MODE_SPECS["M5_INTRADAY"]
    assert m5.sl_atr == 1.5
    assert m5.tp_atr == 2.25
    assert m5.horizon_bars == 24
    assert m5.bar_seconds == 300
    assert m5.timeframe == "M5"
    assert m5.tp1_atr == 1.5
    assert m5.layer_tp1_atr == 1.25
    assert m5.tp2_atr == 4.0
    assert m5.trail_atr == 1.5
    assert m5.be_buffer_usd == 0.30
    assert m5.runner_horizon_bars == 32


def test_compute_split_tp_outcomes_tp1_be_and_runner_trail():
    # Entry bar 0: c=100, atr=2.0 -> SL=97.0, TP1=103.0, TP2=110.0, trail_dist=3.0
    # Bar 1: h=103.5, l=99.0 -> hits TP1 (leg1=0.01 closes at +3.0 - 0.30 = +$2.70); SL snaps to 100.30 (BE+$0.30); trail SL=100.50
    # Bar 2: h=108.0, l=102.0 -> runner surges, best_px=108.0 -> trail SL ratchets to 105.0
    # Bar 3: h=106.0, l=104.5 -> runner stops out at 105.0 (leg2=0.01 closes at +5.0 - 0.30 = +$4.70)
    # Total Long PnL = $2.70 + $4.70 = $7.40
    df_trail = pd.DataFrame({
        "t": [0, 900, 1800, 2700],
        "o": [100.0, 100.0, 103.0, 107.0],
        "h": [100.5, 103.5, 108.0, 106.0],
        "l": [99.5, 99.0, 102.0, 104.5],
        "c": [100.0, 103.0, 107.0, 105.0],
        "atr_14": [2.0, 2.0, 2.0, 2.0],
    })
    pnl_l, pnl_s, hold_l, hold_s = compute_split_tp_outcomes(
        df_trail,
        sl_atr=1.5,
        tp1_atr=1.5,
        tp2_atr=5.0,
        trail_atr=1.5,
        be_buffer_usd=0.30,
        runner_horizon_bars=10,
        spread_cost_usd=0.30,
        lot_total=0.02,
        return_durations=True,
    )
    assert round(pnl_l[0], 2) == 7.40
    assert hold_l[0] == 3  # Bar 0 enters -> bar 3 hits trailing SL
    # Short on bar 1 hits SL (103.0) with full 0.02 lot -> (-3.0 - 0.30) * 2 = -$6.60
    assert round(pnl_s[0], 2) == -6.60
    assert hold_s[0] == 1  # Bar 0 enters -> bar 1 stops out immediately

    # Conservative SL-first tie-break: bar 1 touches both SL (97.0) and TP1 (103.0) -> full 0.02 lot stops at SL
    df_tie = pd.DataFrame({
        "t": [0, 900],
        "o": [100.0, 100.0],
        "h": [100.5, 104.0],
        "l": [99.5, 96.0],
        "c": [100.0, 100.0],
        "atr_14": [2.0, 2.0],
    })
    pnl_l_tie, pnl_s_tie = compute_split_tp_outcomes(
        df_tie,
        sl_atr=1.5,
        tp1_atr=1.5,
        tp2_atr=5.0,
        trail_atr=1.5,
        be_buffer_usd=0.30,
        runner_horizon_bars=10,
        spread_cost_usd=0.30,
        lot_total=0.02,
    )
    assert round(pnl_l_tie[0], 2) == -6.60
    assert round(pnl_s_tie[0], 2) == -6.60

    # TP1 hit then immediate pullback to BE+$0.30: leg1=+$2.70, leg2=(+0.30-0.30)=$0.00 -> total=+$2.70
    df_be = pd.DataFrame({
        "t": [0, 900, 1800],
        "o": [100.0, 100.0, 102.0],
        "h": [100.5, 103.0, 102.0],
        "l": [99.5, 99.0, 100.0],
        "c": [100.0, 102.0, 100.0],
        "atr_14": [2.0, 2.0, 2.0],
    })
    pnl_l_be, _ = compute_split_tp_outcomes(
        df_be,
        sl_atr=1.5,
        tp1_atr=1.5,
        tp2_atr=5.0,
        trail_atr=1.5,
        be_buffer_usd=0.30,
        runner_horizon_bars=10,
        spread_cost_usd=0.30,
        lot_total=0.02,
    )
    assert round(pnl_l_be[0], 2) == 2.70


def test_triple_barrier_applies_spread_and_sl_first_tiebreak():
    df = pd.DataFrame({
        "t": [0, 900, 1800, 2700],
        "o": [100.0, 100.0, 100.0, 100.0],
        "h": [101.0, 105.0, 101.0, 101.0],
        "l": [99.0, 95.0, 99.0, 99.0],
        "c": [100.0, 100.0, 100.0, 100.0],
        "atr_14": [2.0, 2.0, 2.0, 2.0],
    })
    y_l, y_s, pnl_l, pnl_s = compute_triple_barrier_outcomes(
        df, sl_atr=1.5, tp_atr=2.4, horizon_bars=2, spread_cost_usd=0.30
    )
    # Bar 1 touches both SL (97.0) and TP (104.8) for long -> conservative SL wins (-3.0 - 0.30 = -3.30)
    assert y_l[0] == 0
    assert pnl_l[0] == -3.30
    assert y_s[0] == 0
    assert pnl_s[0] == -3.30


def test_triple_barrier_tp_hit_and_timeout():
    # Bar 0: c=100, atr=2.0 -> long tp=104.8, sl=97.0; short tp=95.2, sl=103.0
    # Bar 1: h=106.0, l=98.0 -> long TP hits (104.8), short SL hits (103.0)
    df_tp = pd.DataFrame({
        "t": [0, 900, 1800],
        "o": [100.0, 100.0, 100.0],
        "h": [101.0, 106.0, 101.0],
        "l": [99.0, 98.0, 99.0],
        "c": [100.0, 104.0, 100.0],
        "atr_14": [2.0, 2.0, 2.0],
    })
    y_l, y_s, pnl_l, pnl_s = compute_triple_barrier_outcomes(
        df_tp, sl_atr=1.5, tp_atr=2.4, horizon_bars=2, spread_cost_usd=0.30
    )
    # Long TP: +4.8 gross - 0.30 = +4.50 net USD -> y_l = 1
    assert y_l[0] == 1
    assert round(pnl_l[0], 2) == 4.50
    # Short SL: -3.0 gross - 0.30 = -3.30 net USD -> y_s = 0
    assert y_s[0] == 0
    assert round(pnl_s[0], 2) == -3.30

    # Test timeout: neither SL nor TP touched, exits at close after horizon
    df_timeout = pd.DataFrame({
        "t": [0, 900, 1800],
        "o": [100.0, 100.0, 101.0],
        "h": [100.5, 101.0, 101.5],
        "l": [99.5, 99.0, 99.5],
        "c": [100.0, 100.5, 101.0],
        "atr_14": [2.0, 2.0, 2.0],
    })
    y_l_t, y_s_t, pnl_l_t, pnl_s_t = compute_triple_barrier_outcomes(
        df_timeout, sl_atr=1.5, tp_atr=2.4, horizon_bars=2, spread_cost_usd=0.30
    )
    # Long exits at c[2]=101.0: +1.0 gross - 0.30 = +0.70 net USD -> y_l = 1
    assert y_l_t[0] == 1
    assert round(pnl_l_t[0], 2) == 0.70
    # Short exits at c[2]=101.0: -1.0 gross - 0.30 = -1.30 net USD -> y_s = 0
    assert y_s_t[0] == 0
    assert round(pnl_s_t[0], 2) == -1.30


def test_simulate_oos_backtest_sequential_stepping():
    probs_long = np.array([0.65, 0.70, 0.60, 0.40, 0.80])
    probs_short = np.array([0.30, 0.20, 0.25, 0.30, 0.10])
    pnl_long = np.array([2.50, 1.00, -1.00, 0.00, 3.00])
    pnl_short = np.array([-2.50, -1.00, 1.00, 0.00, -3.00])
    horizon_bars = 2
    threshold = 0.55

    trades, win_rate, pf, net_pnl = simulate_oos_backtest(
        probs_long=probs_long,
        probs_short=probs_short,
        pnl_long=pnl_long,
        pnl_short=pnl_short,
        horizon_bars=horizon_bars,
        threshold=threshold,
    )
    # Bar 0: prob_l=0.65 >= 0.55 -> enters long, pnl=2.50. next_free_idx = 0 + 2 = 2.
    # Bar 1: skipped (1 < 2).
    # Bar 2: prob_l=0.60 >= 0.55 -> enters long, pnl=-1.00. next_free_idx = 2 + 2 = 4.
    # Bar 3: skipped (3 < 4).
    # Bar 4: prob_l=0.80 >= 0.55 -> enters long, pnl=3.00. next_free_idx = 4 + 2 = 6.
    # Total trades: 3 (pnls: 2.50, -1.00, 3.00)
    # Wins: 2 (2.50, 3.00 = 5.50), Loss: 1 (-1.00 = 1.00)
    # Win rate: 2/3 = 0.6667
    # Profit factor: 5.50 / 1.00 = 5.50
    # Net pnl: 2.50 - 1.00 + 3.00 = 4.50
    assert trades == 3
    assert round(win_rate, 4) == round(2 / 3, 4)
    assert round(pf, 2) == 5.50
    assert round(net_pnl, 2) == 4.50


def test_simulate_oos_backtest_with_directional_eligible_mask():
    probs_long = np.array([0.70, 0.70])
    probs_short = np.array([0.20, 0.20])
    pnl_long = np.array([2.00, 2.00])
    pnl_short = np.array([-2.00, -2.00])

    # 2D mask: bar 0 long not allowed, bar 1 long allowed
    mask_2d = np.array([[False, True], [True, False]])
    trades, win_rate, pf, net_pnl = simulate_oos_backtest(
        probs_long=probs_long,
        probs_short=probs_short,
        pnl_long=pnl_long,
        pnl_short=pnl_short,
        horizon_bars=1,
        threshold=0.50,
        eligible_mask=mask_2d,
    )
    # Bar 0: long not allowed, short prob 0.20 < 0.50 -> no trade
    # Bar 1: long allowed, long prob 0.70 >= 0.50 -> enters long, pnl=2.00
    assert trades == 1
    assert round(net_pnl, 2) == 2.00

    # 1D float mask (e.g. htf_ema_diff): negative means short allowed, positive means long allowed
    mask_1d_float = np.array([-1.5, 2.0])
    trades_f, _, _, net_pnl_f = simulate_oos_backtest(
        probs_long=probs_long,
        probs_short=probs_short,
        pnl_long=pnl_long,
        pnl_short=pnl_short,
        horizon_bars=1,
        threshold=0.50,
        eligible_mask=mask_1d_float,
    )
    # Bar 0: diff < 0 -> long not allowed -> no trade
    # Bar 1: diff >= 0 -> long allowed -> 1 trade
    assert trades_f == 1
    assert round(net_pnl_f, 2) == 2.00


def test_search_best_threshold():
    probs_long = np.array([0.52, 0.62, 0.72])
    probs_short = np.array([0.10, 0.10, 0.10])
    pnl_long = np.array([-2.0, 3.0, 5.0])
    pnl_short = np.array([0.0, 0.0, 0.0])

    # At th=0.50: bar 0 trades (-2.0), next_free_idx=1 -> bar 1 trades (3.0), bar 2 trades (5.0) -> net 6.0
    # At th=0.60: bar 0 skipped, bar 1 trades (3.0), bar 2 trades (5.0) -> net 8.0!
    th = search_best_threshold(
        probs_long=probs_long,
        probs_short=probs_short,
        pnl_long=pnl_long,
        pnl_short=pnl_short,
        horizon_bars=1,
    )
    assert th in [0.55, 0.60, 0.65, 0.70]

    # When validation PnL is slightly negative everywhere except 0-trade th=0.70, min_trades prevents dead 0-trade selection
    th_active = search_best_threshold(
        probs_long=np.array([0.45, 0.45, 0.45]),
        probs_short=np.array([0.10, 0.10, 0.10]),
        pnl_long=np.array([-1.0, 2.0, -2.0]),
        pnl_short=np.array([0.0, 0.0, 0.0]),
        horizon_bars=1,
        min_trades=2,
    )
    assert th_active <= 0.44


def test_failed_retrain_preserves_prior_verified_bundle_or_marks_unverified(tmp_path):
    models_dir = tmp_path / "models"
    df = _trend_bars(900)

    # First run with impossible PF=99.0 gate and no prior bundle -> unverified
    bundle_fail = train_and_validate_mode(
        mode="M15_PRECISION",
        df=df,
        htf_df=None,
        models_dir=models_dir,
        min_pf=99.0,
        min_trades=20,
    )
    assert bundle_fail.metrics.verified_for_live is False
    assert "profit factor" in bundle_fail.metrics.fail_reason.lower() or "trade count" in bundle_fail.metrics.fail_reason.lower()

    # Run with relaxed gate -> verified bundle saved on disk
    bundle_ok = train_and_validate_mode(
        mode="M15_PRECISION",
        df=df,
        htf_df=None,
        models_dir=models_dir,
        min_pf=0.50,
        min_trades=5,
    )
    assert bundle_ok.metrics.verified_for_live is True
    saved_ts = bundle_ok.metrics.trained_at_unix

    # Next retrain fails -> prior verified bundle on disk must be preserved!
    bundle_fail2 = train_and_validate_mode(
        mode="M15_PRECISION",
        df=df,
        htf_df=None,
        models_dir=models_dir,
        min_pf=99.0,
        min_trades=20,
    )
    assert bundle_fail2.metrics.verified_for_live is False

    reloaded = load_bundle("M15_PRECISION", models_dir)
    assert reloaded is not None
    assert reloaded.metrics.verified_for_live is True
    assert reloaded.metrics.trained_at_unix == saved_ts


def test_train_and_validate_mode_m5_with_htf(tmp_path):
    models_dir = tmp_path / "models"
    df_m5 = _trend_bars(1200, step=300)
    df_m15 = _trend_bars(400, step=900)

    bundle = train_and_validate_mode(
        mode="M5_INTRADAY",
        df=df_m5,
        htf_df=df_m15,
        models_dir=models_dir,
        min_pf=0.10,
        min_trades=1,
    )
    assert bundle.mode == "M5_INTRADAY"
    assert len(bundle.feature_cols) == 44
    assert bundle.metrics.mode == "M5_INTRADAY"

    reloaded = load_bundle("M5_INTRADAY", models_dir)
    assert reloaded is not None
    assert reloaded.mode == "M5_INTRADAY"


def test_load_bundle_missing_or_corrupted(tmp_path):
    assert load_bundle("M15_PRECISION", tmp_path / "nonexistent") is None

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir(parents=True)
    (bad_dir / "m15_precision_bundle.joblib").write_text("corrupted bytes", encoding="utf-8")
    assert load_bundle("M15_PRECISION", bad_dir) is None


def test_single_class_fallback_wrapper():
    from sklearn.dummy import DummyClassifier
    dummy = DummyClassifier(strategy="prior")
    X = np.random.randn(20, 5)
    y = np.zeros(20, dtype=int)
    dummy.fit(X, y)

    wrapper = CalibratedModelWrapper(dummy, default_class=0)
    probs = wrapper.predict_proba(X)
    assert probs.shape == (20, 2)
    assert np.all(probs[:, 0] == 1.0)
    assert np.all(probs[:, 1] == 0.0)

    preds = wrapper.predict(X)
    assert len(preds) == 20
    assert np.all(preds == 0)

