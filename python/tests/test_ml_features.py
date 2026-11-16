import numpy as np
import pandas as pd
import pytest

from python.src.ml_features import (
    FEATURE_COLUMNS,
    REGIME_COLUMNS,
    bars_to_dataframe,
    compute_features,
)
from python.src.schemas import OhlcvBar


def _make_ohlcv(
    n: int = 320, start_ts: int = 1700000000, step: int = 900, flat: bool = False
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    t = start_ts + np.arange(n) * step
    if flat:
        c = np.full(n, 3850.0)
        return pd.DataFrame({"t": t, "o": c, "h": c, "l": c, "c": c, "v": np.zeros(n)})
    noise = rng.normal(0.0, 1.5, size=n)
    c = 3800.0 + np.cumsum(noise) + np.linspace(0, 15.0, n)
    o = np.roll(c, 1)
    o[0] = c[0]
    spread = np.abs(rng.normal(1.2, 0.4, size=n))
    h = np.maximum(o, c) + spread
    l = np.minimum(o, c) - spread
    v = rng.integers(50, 500, size=n).astype(float)
    return pd.DataFrame({"t": t, "o": o, "h": h, "l": l, "c": c, "v": v})


def test_feature_columns_count_and_finiteness():
    assert len(FEATURE_COLUMNS) == 41
    assert set(REGIME_COLUMNS).issubset(FEATURE_COLUMNS)
    assert REGIME_COLUMNS == ("log_ret_1", "atr_pct_14", "adx_14")

    df = _make_ohlcv(320)
    feats = compute_features(df)
    assert len(feats) > 100
    assert list(feats[list(FEATURE_COLUMNS)].isna().sum()) == [0] * len(FEATURE_COLUMNS)
    assert np.isfinite(feats[list(FEATURE_COLUMNS)].to_numpy()).all()

    # Pre-flight ruling: raw t, o, h, l, c, v and raw atr_14 must be present
    for col in ("t", "o", "h", "l", "c", "v", "atr_14"):
        assert col in feats.columns
        assert np.isfinite(feats[col].to_numpy()).all()


def test_features_strictly_causal_under_tail_truncation():
    full = _make_ohlcv(320)
    truncated = full.iloc[:-40].copy()
    f_full = compute_features(full)
    f_trunc = compute_features(truncated)

    common_t = f_trunc["t"].to_numpy()
    sub_full = f_full[f_full["t"].isin(common_t)].reset_index(drop=True)
    np.testing.assert_allclose(
        sub_full[list(FEATURE_COLUMNS)].to_numpy(),
        f_trunc[list(FEATURE_COLUMNS)].reset_index(drop=True).to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )


def test_flat_bars_and_short_input_do_not_nan_or_crash():
    flat_feats = compute_features(_make_ohlcv(120, flat=True))
    assert np.isfinite(flat_feats[list(FEATURE_COLUMNS)].to_numpy()).all()
    short_feats = compute_features(_make_ohlcv(30))
    assert short_feats.empty


def test_bars_to_dataframe():
    bars = [
        OhlcvBar(t=1700000000 + i * 300, o=2000.0 + i, h=2005.0 + i, l=1995.0 + i, c=2002.0 + i, v=100.0)
        for i in range(5)
    ]
    df = bars_to_dataframe(bars)
    assert len(df) == 5
    assert list(df.columns) == ["t", "o", "h", "l", "c", "v"]
    assert df["t"].iloc[0] == 1700000000
    assert df["c"].iloc[4] == 2006.0

    empty_df = bars_to_dataframe([])
    assert empty_df.empty
    assert list(empty_df.columns) == ["t", "o", "h", "l", "c", "v"]


def test_htf_m15_causality_and_truncation():
    # M5 base (step=300s), M15 HTF (step=900s)
    start_ts = 1700000000
    m5_df = _make_ohlcv(n=360, start_ts=start_ts, step=300)
    m15_df = _make_ohlcv(n=140, start_ts=start_ts, step=900)

    # 1. When htf_df is None, 4 HTF columns default to 0.0
    f_no_htf = compute_features(m5_df, htf_df=None)
    for col in ("htf_ema_21_50_diff", "htf_rsi_14", "htf_atr_pct_14", "htf_bos_dir"):
        assert col in f_no_htf.columns
        assert (f_no_htf[col] == 0.0).all()

    # 2. When htf_df is provided, HTF features are populated
    f_full = compute_features(m5_df, htf_df=m15_df)
    assert np.isfinite(f_full[list(FEATURE_COLUMNS)].to_numpy()).all()

    # 3. Tail truncation of HTF leaves earlier M5 rows bit-identical
    m15_trunc = m15_df.iloc[:-25].copy()
    f_trunc = compute_features(m5_df, htf_df=m15_trunc)

    # Max closed time in truncated M15
    m15_trunc_last_closed = int(m15_trunc["t"].iloc[-1] + 900)
    # M5 bars that closed at or before m15_trunc_last_closed
    mask_full = (f_full["t"] + 300) <= m15_trunc_last_closed
    mask_trunc = (f_trunc["t"] + 300) <= m15_trunc_last_closed

    sub_full = f_full[mask_full].reset_index(drop=True)
    sub_trunc = f_trunc[mask_trunc].reset_index(drop=True)

    np.testing.assert_allclose(
        sub_full[list(FEATURE_COLUMNS)].to_numpy(),
        sub_trunc[list(FEATURE_COLUMNS)].reset_index(drop=True).to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )

    # 4. Strict closed bar visibility check:
    # An M15 bar starting at t0 closes at t0 + 900.
    # An M5 bar starting at t0 closes at t0 + 300 (cannot see M15 bar at t0).
    # An M5 bar starting at t0 + 600 closes at t0 + 900 (can see M15 bar at t0).
    # Check that M5 bar at t0 + 300 does not match M15 bar at t0.
    sample_m15_idx = 70
    t0_m15 = int(m15_df["t"].iloc[sample_m15_idx])
    m15_features = compute_features(m15_df)
    m15_row = m15_features[m15_features["t"] == t0_m15]
    if not m15_row.empty:
        target_rsi = m15_row["rsi_14"].iloc[0]
        # M5 bar closing at t0 + 900 starts at t0 + 600
        m5_at_900 = f_full[f_full["t"] == t0_m15 + 600]
        if not m5_at_900.empty:
            assert np.isclose(m5_at_900["htf_rsi_14"].iloc[0], target_rsi, atol=1e-7)

        # M5 bar closing at t0 + 600 starts at t0 + 300, must NOT have target_rsi
        # (it sees the prior M15 bar)
        m5_at_600 = f_full[f_full["t"] == t0_m15 + 300]
        prior_m15_row = m15_features[m15_features["t"] == t0_m15 - 900]
        if not m5_at_600.empty and not prior_m15_row.empty:
            prior_rsi = prior_m15_row["rsi_14"].iloc[0]
            assert np.isclose(m5_at_600["htf_rsi_14"].iloc[0], prior_rsi, atol=1e-7)
