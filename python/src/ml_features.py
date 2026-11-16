"""Causal feature engineering pipeline for ML regime detection and XGBoost models.

Computes 37 stationary base features + 4 causal higher-timeframe (HTF) context
columns across returns, trend, momentum, volatility, directional movement,
candle geometry, session cyclicals, and SMC structure proxies.
Zero future lookahead is strictly enforced via causal shifting and backward merge_asof.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from python.src.schemas import OhlcvBar

EPS: float = 1e-9
WARMUP_BARS: int = 60

FEATURE_COLUMNS: tuple[str, ...] = (
    # 1. Returns & Momentum (6)
    "log_ret_1",
    "log_ret_3",
    "log_ret_6",
    "log_ret_12",
    "roc_10",
    "mom_norm_20",
    # 2. Moving Averages & Trend (7)
    "ema_9_dist",
    "ema_21_dist",
    "ema_50_dist",
    "ema_200_dist",
    "ema_9_21_diff",
    "ema_21_50_diff",
    "ema_21_slope_5",
    # 3. Oscillators & Volatility (9)
    "rsi_14",
    "macd_line_pct",
    "macd_hist_pct",
    "stoch_k_14",
    "stoch_d_3",
    "bb_width_20",
    "bb_pct_b_20",
    "atr_pct_14",
    "atr_ratio_14_50",
    # 4. Directional Movement (3)
    "adx_14",
    "plus_di_14",
    "minus_di_14",
    # 5. Candle Geometry & Volume (5)
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "range_atr_ratio",
    "vol_zscore_20",
    # 6. Session Cyclicals (4)
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    # 7. Causal SMC Proxies (3)
    "pd_ratio_20",
    "causal_fvg_dir",
    "causal_bos_dir",
    # 8. HTF M15 Context Columns (4)
    "htf_ema_21_50_diff",
    "htf_rsi_14",
    "htf_atr_pct_14",
    "htf_bos_dir",
)

REGIME_COLUMNS: tuple[str, ...] = (
    "log_ret_1",
    "atr_pct_14",
    "adx_14",
)


def bars_to_dataframe(bars: Sequence[OhlcvBar | dict]) -> pd.DataFrame:
    """Convert a sequence of OhlcvBar objects (or dicts) to a standardized DataFrame."""
    cols = ["t", "o", "h", "l", "c", "v"]
    if not bars:
        return pd.DataFrame(columns=cols)

    records: list[dict[str, float | int]] = []
    for b in bars:
        if isinstance(b, dict):
            records.append({
                "t": int(b["t"]),
                "o": float(b["o"]),
                "h": float(b["h"]),
                "l": float(b["l"]),
                "c": float(b["c"]),
                "v": float(b.get("v", 0.0)),
            })
        else:
            records.append({
                "t": int(b.t),
                "o": float(b.o),
                "h": float(b.h),
                "l": float(b.l),
                "c": float(getattr(b, "c", 0.0)),
                "v": float(getattr(b, "v", 0.0)),
            })
    return pd.DataFrame.from_records(records, columns=cols)


def _detect_bar_step(ts_series: pd.Series, default_step: int = 300) -> int:
    """Infer the bar duration in seconds from timestamps."""
    if len(ts_series) > 1:
        diffs = ts_series.diff().dropna()
        pos_diffs = diffs[diffs > 0]
        if not pos_diffs.empty:
            med = int(pos_diffs.median())
            if med > 0:
                return med
    return default_step


def _compute_raw_base(df: pd.DataFrame) -> pd.DataFrame:
    """Compute raw base indicators and features across the whole DataFrame without dropping warmup."""
    col_map = {
        "open": "o",
        "high": "h",
        "low": "l",
        "close": "c",
        "volume": "v",
        "tick_volume": "v",
        "time": "t",
    }
    work = df.rename(columns=lambda x: col_map.get(str(x).lower(), str(x))).copy()
    work = work.sort_values("t").reset_index(drop=True)

    t = work["t"].astype(int)
    o = work["o"].astype(float)
    h = work["h"].astype(float)
    l = work["l"].astype(float)
    c = work["c"].astype(float)
    v = work["v"].astype(float) if "v" in work.columns else pd.Series(0.0, index=work.index)

    # True Range & ATR (Wilder's smoothing: alpha = 1 / period)
    c_prev = c.shift(1).fillna(c)
    tr1 = (h - l).abs()
    tr2 = (h - c_prev).abs()
    tr3 = (l - c_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_14 = tr.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    atr_50 = tr.ewm(alpha=1.0 / 50.0, adjust=False).mean()

    # 1. Returns & Momentum (6)
    c_prev_1 = c.shift(1).fillna(c)
    c_prev_3 = c.shift(3).fillna(c)
    c_prev_6 = c.shift(6).fillna(c)
    c_prev_12 = c.shift(12).fillna(c)
    c_prev_10 = c.shift(10).fillna(c)
    c_prev_20 = c.shift(20).fillna(c)

    log_ret_1 = np.log(np.maximum(c / np.maximum(c_prev_1, EPS), EPS))
    log_ret_3 = np.log(np.maximum(c / np.maximum(c_prev_3, EPS), EPS))
    log_ret_6 = np.log(np.maximum(c / np.maximum(c_prev_6, EPS), EPS))
    log_ret_12 = np.log(np.maximum(c / np.maximum(c_prev_12, EPS), EPS))
    roc_10 = (c - c_prev_10) / (c_prev_10 + EPS)
    mom_norm_20 = (c - c_prev_20) / (atr_14 + EPS)

    # 2. Moving Averages & Trend (7)
    ema_9 = c.ewm(span=9, adjust=False).mean()
    ema_21 = c.ewm(span=21, adjust=False).mean()
    ema_50 = c.ewm(span=50, adjust=False).mean()
    ema_200 = c.ewm(span=200, adjust=False).mean()

    ema_9_dist = (c - ema_9) / (atr_14 + EPS)
    ema_21_dist = (c - ema_21) / (atr_14 + EPS)
    ema_50_dist = (c - ema_50) / (atr_14 + EPS)
    ema_200_dist = (c - ema_200) / (atr_14 + EPS)
    ema_9_21_diff = (ema_9 - ema_21) / (atr_14 + EPS)
    ema_21_50_diff = (ema_21 - ema_50) / (atr_14 + EPS)

    ema_21_prev_5 = ema_21.shift(5).fillna(ema_21)
    ema_21_slope_5 = (ema_21 - ema_21_prev_5) / (5.0 * atr_14 + EPS)

    # 3. Oscillators & Volatility (9)
    delta = c.diff().fillna(0.0)
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    rs = avg_gain / (avg_loss + EPS)
    rsi_14 = 100.0 - (100.0 / (1.0 + rs))

    ema_12 = c.ewm(span=12, adjust=False).mean()
    ema_26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    macd_line_pct = macd_line / (c + EPS)
    macd_hist_pct = macd_hist / (c + EPS)

    low_14 = l.rolling(14, min_periods=1).min()
    high_14 = h.rolling(14, min_periods=1).max()
    stoch_range = high_14 - low_14
    stoch_k_14 = 100.0 * (c - low_14) / (stoch_range + EPS)
    stoch_d_3 = stoch_k_14.rolling(3, min_periods=1).mean()

    sma_20 = c.rolling(20, min_periods=1).mean()
    std_20 = c.rolling(20, min_periods=1).std(ddof=0).fillna(0.0)
    bb_upper = sma_20 + 2.0 * std_20
    bb_lower = sma_20 - 2.0 * std_20
    bb_width_20 = (bb_upper - bb_lower) / (sma_20 + EPS)
    bb_pct_b_20 = (c - bb_lower) / ((bb_upper - bb_lower) + EPS)

    atr_pct_14 = atr_14 / (c + EPS)
    atr_ratio_14_50 = atr_14 / (atr_50 + EPS)

    # 4. Directional Movement (3)
    up_move = h.diff().fillna(0.0)
    down_move = (-l.diff()).fillna(0.0)
    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)

    smooth_plus_dm = pd.Series(plus_dm, index=work.index).ewm(alpha=1.0 / 14.0, adjust=False).mean()
    smooth_minus_dm = pd.Series(minus_dm, index=work.index).ewm(alpha=1.0 / 14.0, adjust=False).mean()
    plus_di_14 = 100.0 * smooth_plus_dm / (atr_14 + EPS)
    minus_di_14 = 100.0 * smooth_minus_dm / (atr_14 + EPS)

    di_sum = plus_di_14 + minus_di_14
    di_diff = (plus_di_14 - minus_di_14).abs()
    dx = 100.0 * di_diff / (di_sum + EPS)
    adx_14 = dx.ewm(alpha=1.0 / 14.0, adjust=False).mean()

    # 5. Candle Geometry & Volume (5)
    candle_range = (h - l).abs()
    body = (c - o).abs()
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l

    body_ratio = body / (candle_range + EPS)
    upper_wick_ratio = upper_wick / (candle_range + EPS)
    lower_wick_ratio = lower_wick / (candle_range + EPS)
    range_atr_ratio = candle_range / (atr_14 + EPS)

    vol_mean_20 = v.rolling(20, min_periods=1).mean()
    vol_std_20 = v.rolling(20, min_periods=1).std(ddof=0).fillna(0.0)
    vol_zscore_20 = (v - vol_mean_20) / (vol_std_20 + EPS)

    # 6. Session Cyclicals (4)
    dt = pd.to_datetime(t, unit="s", utc=True)
    hour_frac = dt.dt.hour + dt.dt.minute / 60.0 + dt.dt.second / 3600.0
    hour_sin = np.sin(2.0 * np.pi * hour_frac / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hour_frac / 24.0)

    dow_series = dt.dt.day_of_week
    dow_frac = dow_series + hour_frac / 24.0
    dow_sin = np.sin(2.0 * np.pi * dow_frac / 7.0)
    dow_cos = np.cos(2.0 * np.pi * dow_frac / 7.0)

    # 7. Causal SMC Proxies (3)
    high_20 = h.rolling(20, min_periods=1).max()
    low_20 = l.rolling(20, min_periods=1).min()
    pd_ratio_20 = (c - low_20) / ((high_20 - low_20) + EPS)

    h_shift2 = h.shift(2).fillna(h)
    l_shift2 = l.shift(2).fillna(l)
    fvg_bull = l > h_shift2
    fvg_bear = h < l_shift2
    causal_fvg_dir = np.where(fvg_bull, 1.0, np.where(fvg_bear, -1.0, 0.0))

    prior_high_20 = h.shift(1).fillna(h).rolling(20, min_periods=1).max()
    prior_low_20 = l.shift(1).fillna(l).rolling(20, min_periods=1).min()
    bos_bull = c > prior_high_20
    bos_bear = c < prior_low_20
    causal_bos_dir = np.where(bos_bull, 1.0, np.where(bos_bear, -1.0, 0.0))

    out = pd.DataFrame(
        {
            "t": t,
            "o": o,
            "h": h,
            "l": l,
            "c": c,
            "v": v,
            "close": c,
            "atr_14": atr_14,
            # Base 37 features
            "log_ret_1": log_ret_1,
            "log_ret_3": log_ret_3,
            "log_ret_6": log_ret_6,
            "log_ret_12": log_ret_12,
            "roc_10": roc_10,
            "mom_norm_20": mom_norm_20,
            "ema_9_dist": ema_9_dist,
            "ema_21_dist": ema_21_dist,
            "ema_50_dist": ema_50_dist,
            "ema_200_dist": ema_200_dist,
            "ema_9_21_diff": ema_9_21_diff,
            "ema_21_50_diff": ema_21_50_diff,
            "ema_21_slope_5": ema_21_slope_5,
            "rsi_14": rsi_14,
            "macd_line_pct": macd_line_pct,
            "macd_hist_pct": macd_hist_pct,
            "stoch_k_14": stoch_k_14,
            "stoch_d_3": stoch_d_3,
            "bb_width_20": bb_width_20,
            "bb_pct_b_20": bb_pct_b_20,
            "atr_pct_14": atr_pct_14,
            "atr_ratio_14_50": atr_ratio_14_50,
            "adx_14": adx_14,
            "plus_di_14": plus_di_14,
            "minus_di_14": minus_di_14,
            "body_ratio": body_ratio,
            "upper_wick_ratio": upper_wick_ratio,
            "lower_wick_ratio": lower_wick_ratio,
            "range_atr_ratio": range_atr_ratio,
            "vol_zscore_20": vol_zscore_20,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "pd_ratio_20": pd_ratio_20,
            "causal_fvg_dir": causal_fvg_dir,
            "causal_bos_dir": causal_bos_dir,
        },
        index=work.index,
    )
    return out


def compute_features(
    df: pd.DataFrame, htf_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Compute causal 37-feature + 4 HTF context pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Base timeframe OHLCV bar DataFrame (must have columns t, o, h, l, c, v).
    htf_df : pd.DataFrame | None
        Optional higher-timeframe bar DataFrame (e.g. M15 when df is M5).
        Merged causally backward on closed bar timestamp (htf.t + htf_step <= df.t + df_step).
        If None, all 4 HTF context features are filled with 0.0.

    Returns
    -------
    pd.DataFrame
        DataFrame with raw OHLCV columns (t, o, h, l, c, v), close, atr_14,
        and all 41 FEATURE_COLUMNS, with the first 60 warmup rows dropped.
        Returns an empty DataFrame if len(df) < 60.
    """
    output_cols = ["t", "o", "h", "l", "c", "v", "close", "atr_14", *FEATURE_COLUMNS]

    if df is None or len(df) < WARMUP_BARS:
        return pd.DataFrame(columns=output_cols)

    raw_base = _compute_raw_base(df)

    htf_cols = ("htf_ema_21_50_diff", "htf_rsi_14", "htf_atr_pct_14", "htf_bos_dir")

    if htf_df is not None and len(htf_df) > 0:
        htf_raw = _compute_raw_base(htf_df)
        htf_step = _detect_bar_step(htf_raw["t"], default_step=900)
        df_step = _detect_bar_step(raw_base["t"], default_step=300)

        htf_context = pd.DataFrame(
            {
                "closed_at": htf_raw["t"] + htf_step,
                "htf_ema_21_50_diff": htf_raw["ema_21_50_diff"],
                "htf_rsi_14": htf_raw["rsi_14"],
                "htf_atr_pct_14": htf_raw["atr_pct_14"],
                "htf_bos_dir": htf_raw["causal_bos_dir"],
            }
        ).sort_values("closed_at").reset_index(drop=True)

        df_merge = pd.DataFrame(
            {
                "closed_at": raw_base["t"] + df_step,
                "_orig_idx": np.arange(len(raw_base)),
            }
        ).sort_values("closed_at").reset_index(drop=True)

        merged = pd.merge_asof(
            df_merge,
            htf_context,
            on="closed_at",
            direction="backward",
        )
        merged = merged.sort_values("_orig_idx").reset_index(drop=True)

        for c_name in htf_cols:
            raw_base[c_name] = merged[c_name].fillna(0.0).to_numpy()
    else:
        for c_name in htf_cols:
            raw_base[c_name] = 0.0

    # Drop warmup bars (< 60)
    res = raw_base.iloc[WARMUP_BARS:].reset_index(drop=True)
    return res[output_cols]
