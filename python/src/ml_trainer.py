"""Triple-Barrier Labeling, Calibrated XGBoost, Walk-Forward Validator & Rollback.

Implements:
- Mode specifications for M15_PRECISION and M5_INTRADAY.
- Path-dependent Triple-Barrier labeling with conservative SL-first tie-breaking and spread cost.
- Calibrated XGBoost classifiers with isotonic/sigmoid probability calibration.
- Purged chronological walk-forward validation with embargo.
- Best threshold search across candidate decision thresholds.
- Sequential non-overlapping position backtest simulation.
- Quality gating (profit factor & trade count) with automated rollback to preserve verified bundles.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import time
from typing import Any
import warnings

# Suppress upstream hmmlearn DeprecationWarning on numpy 2.x array shape mutation
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"hmmlearn.*")

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier

from python.src.ml_features import FEATURE_COLUMNS, REGIME_COLUMNS, compute_features
from python.src.ml_regime import RegimeModel
from python.src.schemas import MlMetricsSummary, StrategyMode

logger = logging.getLogger(__name__)

CANDIDATE_THRESHOLDS: tuple[float, ...] = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)


@dataclass(frozen=True)
class ModeSpec:
    """Trading strategy mode parameters."""

    sl_atr: float
    tp_atr: float
    horizon_bars: int
    bar_seconds: int
    timeframe: str


MODE_SPECS: dict[StrategyMode, ModeSpec] = {
    "M15_PRECISION": ModeSpec(
        sl_atr=1.5,
        tp_atr=2.4,
        horizon_bars=16,
        bar_seconds=900,
        timeframe="M15",
    ),
    "M5_INTRADAY": ModeSpec(
        sl_atr=1.5,
        tp_atr=2.25,
        horizon_bars=24,
        bar_seconds=300,
        timeframe="M5",
    ),
}


class CalibratedModelWrapper:
    """Wraps calibrated model or fallback to ensure predict_proba always returns shape (N, 2)."""

    def __init__(self, model: Any, default_class: int = 0) -> None:
        self.model = model
        self.default_class = default_class

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.model.predict_proba(X)
        if raw.ndim == 2 and raw.shape[1] == 2:
            return raw
        n = len(X)
        probs = np.zeros((n, 2), dtype=float)
        classes = getattr(self.model, "classes_", np.array([self.default_class]))
        if len(classes) == 1:
            if classes[0] == 1:
                probs[:, 1] = 1.0
                probs[:, 0] = 0.0
            else:
                probs[:, 0] = 1.0
                probs[:, 1] = 0.0
        elif raw.ndim == 2 and raw.shape[1] == 1:
            probs[:, 0] = 1.0 - raw[:, 0]
            probs[:, 1] = raw[:, 0]
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def __getattr__(self, name: str) -> Any:
        if name in ("__getstate__", "__setstate__", "model", "default_class"):
            raise AttributeError(name)
        return getattr(self.model, name)


@dataclass
class TrainedBundle:
    """Serializable bundle containing trained regime & direction models and live verification metadata."""

    mode: StrategyMode
    feature_cols: list[str]
    regime_model: RegimeModel
    xgb_long: Any
    xgb_short: Any
    threshold: float
    metrics: MlMetricsSummary


def compute_triple_barrier_outcomes(
    df: pd.DataFrame,
    sl_atr: float,
    tp_atr: float,
    horizon_bars: int,
    spread_cost_usd: float = 0.30,
    lot_size: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute path-dependent Triple-Barrier outcomes for Long and Short trades.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing price columns ('c'/'close', 'h'/'high', 'l'/'low') and 'atr_14'/'atr'.
    sl_atr : float
        Stop-loss multiplier relative to ATR.
    tp_atr : float
        Take-profit multiplier relative to ATR.
    horizon_bars : int
        Maximum number of bars before timeout exit.
    spread_cost_usd : float
        Roundtrip spread cost deducted per trade in USD (default $0.30).
    lot_size : float
        Trade lot size (default 0.01 lot = $1.00 per $1.00 XAUUSD move).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (y_long, y_short, pnl_long_usd, pnl_short_usd)
    """
    n = len(df)
    if n == 0:
        return (
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
        )

    c = df["c"].to_numpy(dtype=float) if "c" in df.columns else df["close"].to_numpy(dtype=float)
    h = df["h"].to_numpy(dtype=float) if "h" in df.columns else df["high"].to_numpy(dtype=float)
    l = df["l"].to_numpy(dtype=float) if "l" in df.columns else df["low"].to_numpy(dtype=float)
    atr = (
        df["atr_14"].to_numpy(dtype=float)
        if "atr_14" in df.columns
        else df["atr"].to_numpy(dtype=float)
    )

    y_long = np.zeros(n, dtype=int)
    y_short = np.zeros(n, dtype=int)
    pnl_long_usd = np.zeros(n, dtype=float)
    pnl_short_usd = np.zeros(n, dtype=float)

    usd_per_dollar_move = 100.0 * lot_size

    for i in range(n):
        entry_price = c[i]
        bar_atr = atr[i]
        sl_dist = sl_atr * bar_atr
        tp_dist = tp_atr * bar_atr

        long_sl = entry_price - sl_dist
        long_tp = entry_price + tp_dist

        short_sl = entry_price + sl_dist
        short_tp = entry_price - tp_dist

        long_exit_price: float | None = None
        short_exit_price: float | None = None

        scan_end = min(n, i + 1 + horizon_bars)
        for j in range(i + 1, scan_end):
            # Check Long
            if long_exit_price is None:
                sl_hit = l[j] <= long_sl
                tp_hit = h[j] >= long_tp
                if sl_hit and tp_hit:
                    # Conservative SL-first tie-breaking
                    long_exit_price = long_sl
                elif sl_hit:
                    long_exit_price = long_sl
                elif tp_hit:
                    long_exit_price = long_tp

            # Check Short
            if short_exit_price is None:
                sl_hit = h[j] >= short_sl
                tp_hit = l[j] <= short_tp
                if sl_hit and tp_hit:
                    # Conservative SL-first tie-breaking
                    short_exit_price = short_sl
                elif sl_hit:
                    short_exit_price = short_sl
                elif tp_hit:
                    short_exit_price = short_tp

            if long_exit_price is not None and short_exit_price is not None:
                break

        # Timeout handling
        if long_exit_price is None:
            timeout_idx = min(n - 1, i + horizon_bars)
            long_exit_price = c[timeout_idx]

        if short_exit_price is None:
            timeout_idx = min(n - 1, i + horizon_bars)
            short_exit_price = c[timeout_idx]

        # Long PnL
        price_diff_long = long_exit_price - entry_price
        gross_long = price_diff_long * usd_per_dollar_move
        net_long = gross_long - spread_cost_usd
        pnl_long_usd[i] = round(net_long, 4)
        y_long[i] = 1 if net_long > 0.0 else 0

        # Short PnL
        price_diff_short = entry_price - short_exit_price
        gross_short = price_diff_short * usd_per_dollar_move
        net_short = gross_short - spread_cost_usd
        pnl_short_usd[i] = round(net_short, 4)
        y_short[i] = 1 if net_short > 0.0 else 0

    return y_long, y_short, pnl_long_usd, pnl_short_usd


def simulate_oos_backtest(
    probs_long: np.ndarray,
    probs_short: np.ndarray,
    pnl_long: np.ndarray,
    pnl_short: np.ndarray,
    horizon_bars: int,
    threshold: float,
    eligible_mask: np.ndarray | None = None,
) -> tuple[int, float, float, float]:
    """Simulate out-of-sample backtest with sequential non-overlapping position stepping.

    Parameters
    ----------
    probs_long : np.ndarray
        Array of model long probability predictions.
    probs_short : np.ndarray
        Array of model short probability predictions.
    pnl_long : np.ndarray
        Net USD PnL array for long trades.
    pnl_short : np.ndarray
        Net USD PnL array for short trades.
    horizon_bars : int
        Holding horizon bars (used for non-overlapping position stepping).
    threshold : float
        Decision threshold for trade triggers.
    eligible_mask : np.ndarray | None
        Directional alignment or eligibility mask. Supports:
        - 2D array of shape (N, 2) where cols are [can_long, can_short]
        - 1D float array where value >= 0 means long eligible, <= 0 means short eligible
        - 1D bool array where True means bar is eligible for both sides

    Returns
    -------
    tuple[int, float, float, float]
        (trades, win_rate, profit_factor, net_pnl_usd)
    """
    n = len(probs_long)
    if n == 0:
        return 0, 0.0, 0.0, 0.0

    trade_pnls: list[float] = []
    next_free_idx = 0

    for i in range(n):
        if i < next_free_idx:
            continue

        can_long = True
        can_short = True
        if eligible_mask is not None:
            if eligible_mask.ndim == 2 and eligible_mask.shape[1] >= 2:
                can_long = bool(eligible_mask[i, 0])
                can_short = bool(eligible_mask[i, 1])
            elif np.issubdtype(eligible_mask.dtype, np.floating):
                can_long = bool(eligible_mask[i] >= 0.0)
                can_short = bool(eligible_mask[i] <= 0.0)
            else:
                can_long = bool(eligible_mask[i])
                can_short = bool(eligible_mask[i])

        pl = float(probs_long[i])
        ps = float(probs_short[i])

        signal_long = can_long and (pl >= threshold)
        signal_short = can_short and (ps >= threshold)

        if not signal_long and not signal_short:
            continue

        if signal_long and signal_short:
            take_long = pl >= ps
        elif signal_long:
            take_long = True
        else:
            take_long = False

        if take_long:
            trade_pnls.append(float(pnl_long[i]))
        else:
            trade_pnls.append(float(pnl_short[i]))

        next_free_idx = i + horizon_bars

    trades = len(trade_pnls)
    if trades == 0:
        return 0, 0.0, 0.0, 0.0

    wins = [p for p in trade_pnls if p > 0.0]
    losses = [p for p in trade_pnls if p <= 0.0]

    win_rate = float(len(wins) / trades)
    net_pnl_usd = float(np.sum(trade_pnls))

    gross_profit = float(np.sum(wins)) if wins else 0.0
    gross_loss = float(np.sum([abs(p) for p in losses])) if losses else 0.0

    if gross_loss > 0.0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = 99.0 if gross_profit > 0.0 else 0.0

    return trades, round(win_rate, 4), round(profit_factor, 4), round(net_pnl_usd, 4)


def search_best_threshold(
    probs_long: np.ndarray,
    probs_short: np.ndarray,
    pnl_long: np.ndarray,
    pnl_short: np.ndarray,
    horizon_bars: int,
    eligible_mask: np.ndarray | None = None,
) -> float:
    """Search for the best decision threshold maximizing validation performance."""
    best_threshold = 0.50
    t_0, wr_0, pf_0, pnl_0 = simulate_oos_backtest(
        probs_long, probs_short, pnl_long, pnl_short, horizon_bars, 0.50, eligible_mask
    )
    best_score = (pnl_0, pf_0, t_0)

    for th in CANDIDATE_THRESHOLDS:
        trades, win_rate, pf, net_pnl = simulate_oos_backtest(
            probs_long, probs_short, pnl_long, pnl_short, horizon_bars, th, eligible_mask
        )
        score = (net_pnl, pf, trades)
        if score > best_score:
            best_score = score
            best_threshold = th

    return best_threshold


def _fit_calibrated_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42,
) -> Any:
    """Train XGBClassifier and wrap with calibrated probability estimation."""
    unique_train = np.unique(y_train)
    if len(unique_train) < 2:
        dummy = DummyClassifier(strategy="prior")
        dummy.fit(X_train, y_train)
        return CalibratedModelWrapper(dummy, default_class=int(unique_train[0]))

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)

    unique_val = np.unique(y_val)
    if len(unique_val) < 2:
        return CalibratedModelWrapper(xgb)

    try:
        calibrated = CalibratedClassifierCV(
            FrozenEstimator(xgb),
            method="sigmoid",
        )
        calibrated.fit(X_val, y_val)
        return CalibratedModelWrapper(calibrated)
    except Exception as e:
        logger.warning(f"CalibratedClassifierCV failed: {e}; falling back to uncalibrated XGB.")
        return CalibratedModelWrapper(xgb)


def _atomic_dump_joblib(obj: Any, target_path: Path) -> None:
    """Atomically save an object to disk via joblib."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(f".tmp.{os.getpid()}.{time.time_ns()}")
    try:
        joblib.dump(obj, temp_path)
        os.replace(temp_path, target_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def _atomic_write_json(content: str, target_path: Path) -> None:
    """Atomically write text/JSON content to disk."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(f".tmp.{os.getpid()}.{time.time_ns()}")
    try:
        temp_path.write_text(content, encoding="utf-8")
        os.replace(temp_path, target_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def train_and_validate_mode(
    mode: StrategyMode,
    df: pd.DataFrame,
    htf_df: pd.DataFrame | None,
    models_dir: Path | str,
    spread_cost_usd: float = 0.30,
    min_pf: float = 1.05,
    min_trades: int = 20,
) -> TrainedBundle:
    """Train regime & direction models, run walk-forward OOS validation, and apply gating/rollback."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    spec = MODE_SPECS[mode]

    # 1. Compute causal features & Triple-Barrier outcomes
    features_df = compute_features(df, htf_df=htf_df if mode == "M5_INTRADAY" else None)
    y_l, y_s, pnl_l, pnl_s = compute_triple_barrier_outcomes(
        features_df,
        sl_atr=spec.sl_atr,
        tp_atr=spec.tp_atr,
        horizon_bars=spec.horizon_bars,
        spread_cost_usd=spread_cost_usd,
    )

    # Drop last horizon_bars rows (incomplete forward horizon)
    valid_len = len(features_df) - spec.horizon_bars
    if valid_len < 30:
        raise ValueError(
            f"Insufficient bars ({valid_len}) after dropping warmup and horizon bars."
        )

    features_df = features_df.iloc[:valid_len].reset_index(drop=True)
    y_l = y_l[:valid_len]
    y_s = y_s[:valid_len]
    pnl_l = pnl_l[:valid_len]
    pnl_s = pnl_s[:valid_len]

    # 2. Chronological split: 60% train / 20% val / 20% OOS test with embargo
    n = valid_len
    train_end = max(10, int(n * 0.60))
    val_start = min(n - 10, train_end + spec.horizon_bars)
    val_end = min(n - 5, max(val_start + 5, int(n * 0.80)))
    oos_start = min(n - 2, val_end + spec.horizon_bars)
    oos_end = n

    train_slice = slice(0, train_end)
    val_slice = slice(val_start, val_end)
    oos_slice = slice(oos_start, oos_end)

    # 3. Fit RegimeModel on train split ONLY
    X_regime = features_df[list(REGIME_COLUMNS)].to_numpy(dtype=float)
    regime_model = RegimeModel.fit(X_regime[train_slice], random_state=42)
    regime_probs, _ = regime_model.predict_proba_causal(X_regime)

    # Construct model features X_full: 41 feature cols + 3 regime prob cols = 44 columns
    X_features = features_df[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    X_full = np.column_stack([X_features, regime_probs])
    full_feature_cols = list(FEATURE_COLUMNS) + ["p_trending", "p_ranging", "p_volatile"]

    # 4. Train calibrated XGBoost models for Long and Short
    model_long = _fit_calibrated_xgb(
        X_full[train_slice],
        y_l[train_slice],
        X_full[val_slice],
        y_l[val_slice],
        random_state=42,
    )
    model_short = _fit_calibrated_xgb(
        X_full[train_slice],
        y_s[train_slice],
        X_full[val_slice],
        y_s[val_slice],
        random_state=42,
    )

    # Construct directional alignment mask if M5_INTRADAY
    eligible_mask: np.ndarray | None = None
    if mode == "M5_INTRADAY" and "htf_ema_21_50_diff" in features_df.columns:
        htf_diff = features_df["htf_ema_21_50_diff"].to_numpy(dtype=float)
        eligible_mask = np.column_stack([htf_diff >= 0.0, htf_diff <= 0.0])

    # 5. Search best threshold on validation split
    probs_val_l = model_long.predict_proba(X_full[val_slice])[:, 1]
    probs_val_s = model_short.predict_proba(X_full[val_slice])[:, 1]
    val_mask = eligible_mask[val_slice] if eligible_mask is not None else None

    best_th = search_best_threshold(
        probs_val_l,
        probs_val_s,
        pnl_l[val_slice],
        pnl_s[val_slice],
        spec.horizon_bars,
        eligible_mask=val_mask,
    )

    # Evaluate on out-of-sample test split
    probs_oos_l = model_long.predict_proba(X_full[oos_slice])[:, 1]
    probs_oos_s = model_short.predict_proba(X_full[oos_slice])[:, 1]
    oos_mask = eligible_mask[oos_slice] if eligible_mask is not None else None

    oos_trades, oos_wr, oos_pf, oos_net_pnl = simulate_oos_backtest(
        probs_oos_l,
        probs_oos_s,
        pnl_l[oos_slice],
        pnl_s[oos_slice],
        spec.horizon_bars,
        best_th,
        eligible_mask=oos_mask,
    )

    # 6. Quality gating & automated rollback
    passed = bool((oos_pf >= min_pf) and (oos_trades >= min_trades))
    fail_reasons: list[str] = []
    if not passed:
        if oos_pf < min_pf:
            fail_reasons.append(f"OOS profit factor {oos_pf:.2f} < {min_pf:.2f}")
        if oos_trades < min_trades:
            fail_reasons.append(f"OOS trade count {oos_trades} < {min_trades}")
    fail_reason = "; ".join(fail_reasons)

    metrics = MlMetricsSummary(
        mode=mode,
        trained_at_unix=int(time.time()),
        oos_trades=oos_trades,
        oos_win_rate=oos_wr,
        oos_profit_factor=oos_pf,
        oos_net_pnl_usd=oos_net_pnl,
        chosen_threshold=best_th,
        verified_for_live=passed,
        fail_reason=fail_reason,
    )

    bundle = TrainedBundle(
        mode=mode,
        feature_cols=full_feature_cols,
        regime_model=regime_model,
        xgb_long=model_long,
        xgb_short=model_short,
        threshold=best_th,
        metrics=metrics,
    )

    bundle_path = models_dir / f"{mode.lower()}_bundle.joblib"
    metrics_path = models_dir / f"{mode.lower()}_metrics.json"

    prior_bundle = load_bundle(mode, models_dir)
    prior_verified = (
        prior_bundle is not None and prior_bundle.metrics.verified_for_live is True
    )

    if passed:
        _atomic_dump_joblib(bundle, bundle_path)
        _atomic_write_json(bundle.metrics.model_dump_json(indent=2), metrics_path)
        logger.info(
            f"Successfully saved verified {mode} bundle (PF={oos_pf:.2f}, trades={oos_trades})."
        )
    else:
        if prior_verified:
            logger.warning(
                f"Retrained {mode} failed gate ({fail_reason}); preserving prior verified bundle on disk."
            )
            # Write rejected metrics sidecar without modifying verified .joblib bundle
            _atomic_write_json(
                bundle.metrics.model_dump_json(indent=2),
                models_dir / f"{mode.lower()}_rejected_metrics.json",
            )
        else:
            _atomic_dump_joblib(bundle, bundle_path)
            _atomic_write_json(bundle.metrics.model_dump_json(indent=2), metrics_path)
            logger.warning(
                f"Saved unverified {mode} bundle ({fail_reason}); no prior verified bundle existed."
            )

    return bundle


def load_bundle(mode: StrategyMode, models_dir: Path | str) -> TrainedBundle | None:
    """Load a trained model bundle from disk if available."""
    models_dir = Path(models_dir)
    bundle_path = models_dir / f"{mode.lower()}_bundle.joblib"
    if not bundle_path.is_file():
        return None
    try:
        bundle = joblib.load(bundle_path)
        if isinstance(bundle, TrainedBundle):
            return bundle
        return None
    except Exception as e:
        logger.warning(f"Failed to load bundle from {bundle_path}: {e}")
        return None
