# Dual-Mode HMM + XGBoost XAUUSD ML Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and integrate a dual-mode (`M15_PRECISION` & `M5_INTRADAY`) `GaussianHMM` + calibrated `XGBoost` XAUUSD signal engine powered by live MetaTrader5 bars, matched Triple-Barrier exits, walk-forward OOS validation, Telegram mode controls, and read-only Gemini macro briefings.

**Architecture:** Python reads closed `XAUUSDm` `M15`/`M5` bars via `MetaTrader5`, computes 37 causal TA/SMC features, classifies 3-state volatility/trend regimes via forward-only `GaussianHMM`, scores Long/Short Triple-Barrier setups with calibrated `XGBClassifier` bundles, and writes `ai_signal.json` exclusively via `MlSignalRunner` (`GoldAnalyst` becomes read-only commentary + HMR blackout sync). `EA_LearningX_Gold.mq5` executes those setups with matched Triple-Barrier SL/TP and `max_hold_until_unix` timeout exits while preserving `🔒 Keep #<ticket>` and daily loss protection.

**Tech Stack:** Python 3.14 (`.venv`), `MetaTrader5>=5.0.6000`, `xgboost>=3.0.0`, `hmmlearn>=0.3.3`, `scikit-learn>=1.8.0`, `numpy>=2.2.0`, `pandas>=2.2.0`, `joblib>=1.4.0`, `pydantic>=2.9.0`, `python-telegram-bot>=22.0`, `Pillow>=10.0.0`, `pytest>=8.3.0`, MQL5 (`EA_LearningX_Gold.mq5`).

**Spec:** [docs/superpowers/specs/2026-11-16-ml-hmm-xgboost-design.md](file:///d:/MATERIAL/Antigravity_Project/EA-LearningX/docs/superpowers/specs/2026-11-16-ml-hmm-xgboost-design.md)

## Global Constraints

- **Python Runtime:** Windows `.venv` on Python 3.14 (`d:\MATERIAL\Antigravity_Project\EA-LearningX\.venv\Scripts\python.exe`).
- **Dual Strategy Modes:**
  - `M15_PRECISION`: Closed `M15` bars (~2-6 setups/day), `SL = 1.5 * ATR_M15`, `TP = 2.4 * ATR_M15` (`1:1.6`), `H = 16` bars (`4h` max hold, `bar_seconds = 900`).
  - `M5_INTRADAY`: `M15` trend alignment + closed `M5` trigger (~6-15 setups/day), `SL = 1.5 * ATR_M5`, `TP = 2.25 * ATR_M5` (`1:1.5`), `H = 24` bars (`2h` max hold, `bar_seconds = 300`).
- **Transaction Cost Parity:** Spread + slippage deduction of `$0.30` per round-trip trade in both Triple-Barrier labeling and walk-forward backtesting.
- **Validation Gate & Rollback:** Out-of-sample walk-forward requires `PF >= 1.05` and `trades >= 20`. If retrain fails, keep the previous verified bundle; if none exists, set `verified_for_live = False`, block `Auto` execution (`Confirm`-only fallback), and alert Telegram.
- **Exclusive Signal Ownership:** `MlSignalRunner` exclusively writes execution signals to `ai_signal.json`. `GoldAnalyst` (Gemini) never triggers trades and runs read-only on `⚡ Analyze Now` / `📰 Full AI Brief` (+ HMR blackout schedule).
- **Matched Exit Parity:** Default MQL5 inputs become `InpUseSarEngine = false`, `InpUseAutoBE = false`, `InpUseTrailing = false`; `EA_LearningX_Gold.mq5` enforces `max_hold_until_unix` timeout close on non-held positions while respecting `🔒 Keep #<ticket>`.

## Review Focus

1. **Lookahead Leakage (Features & HMM):** Forming (unclosed) bars, forward `bfill`/global scaling, or backward Viterbi/forward-backward HMM smoothing would leak future bars into training/live features -> pin causal prefix invariance tests on `ml_features.py` and forward-only filtering tests on `ml_regime.py`.
2. **Degenerate / Flat OHLCV Windows:** Short warmups (`< 250` bars) or zero-range candles (`high == low`, `volume == 0`) can produce `NaN`/`Inf` in RSI/ATR/ADX/BB/OB -> pin finite-value and graceful empty/insufficient-bar error tests in `ml_features.py` and `mt5_feed.py`.
3. **Failed Retrain & Unverified Live Bundle Guard:** A retrain yielding `PF < 1.05` or `trades < 20` (or missing `.joblib` files) must preserve the previous verified bundle or enforce `verified_for_live = False` and reject switching execution mode to `Auto` -> pin tests in `test_ml_trainer.py` and `test_telegram_callbacks.py`.
4. **Duplicate Bar Evaluation:** Polling `MlSignalRunner` every 15s between bar closes must evaluate each closed `M15`/`M5` bar at most once per mode -> pin dedup test in `test_ml_engine.py`.
5. **MQL5 Timeout vs Hold Isolation:** Closing on `max_hold_until_unix` must skip positions locked via `🔒 Keep #<ticket>` (`IsHeldPosition`) and never close positions whose stored `max_hold_until_unix == 0` -> pin MQL5 helper logic and schema tests (`test_ml_schemas.py` + MetaEditor `0 errors, 0 warnings` compile check).

---

### Task 1: Dependencies, ML Schemas & Config Extensions

**Files:**
- Modify: `requirements.txt:1-9`
- Modify: `python/src/schemas.py:1-145,160-271`
- Modify: `python/src/config.py:1-87`
- Test: `python/tests/test_ml_schemas.py`

**Interfaces:**
- Produces in `python/src/schemas.py`:
  - `StrategyMode = Literal["M15_PRECISION", "M5_INTRADAY"]`
  - `RegimeLabel = Literal["TRENDING", "RANGING", "VOLATILE"]`
  - `EaSignalPayload.max_hold_until_unix: int = 0` (plus optional metadata `strategy_mode: str = ""`, `regime: str = ""`)
  - `MlMetricsSummary` (Pydantic model: `mode`, `trained_at_unix`, `oos_trades`, `oos_win_rate`, `oos_profit_factor`, `oos_net_pnl_usd`, `chosen_threshold`, `verified_for_live`, `fail_reason`)
  - `MlPrediction` (Pydantic model: `mode`, `bar_time_unix`, `timeframe`, `regime`, `regime_probs`, `prob_long`, `prob_short`, `threshold`, `order_intent`, `confidence`, `entry_price`, `sl_price`, `tp_price`, `max_hold_until_unix`, `reason`, `verified_for_live`)
- Produces in `python/src/config.py`:
  - `Settings` fields: `default_strategy_mode: StrategyMode = "M15_PRECISION"`, `ml_models_dir: Path`, `ml_spread_cost_usd: float = 0.30`, `ml_min_profit_factor: float = 1.05`, `ml_min_oos_trades: int = 20`, `ml_m15_bars: int = 4000`, `ml_m5_bars: int = 12000`, `ml_poll_interval_seconds: int = 15`, `ml_retrain_hour_utc: int = 22`

- [ ] **Step 1: Add ML & MT5 packages to `requirements.txt` and install into `.venv`**

Append `MetaTrader5>=5.0.6000`, `xgboost>=3.0.0`, `hmmlearn>=0.3.3`, `scikit-learn>=1.8.0`, `numpy>=2.2.0`, `pandas>=2.2.0`, `joblib>=1.4.0` to `requirements.txt` and run:
```powershell
d:\MATERIAL\Antigravity_Project\EA-LearningX\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Expected: Exit code `0`, `import xgboost, hmmlearn, sklearn, MetaTrader5, pandas, numpy, joblib` succeeds in `.venv`.

- [ ] **Step 2: Write the failing tests in `python/tests/test_ml_schemas.py`**

```python
from python.src.config import Settings
from python.src.schemas import EaSignalPayload, MlMetricsSummary, MlPrediction


def test_ea_signal_payload_has_max_hold_default_and_override():
    default_sig = EaSignalPayload(
        signal_id="sig1",
        generated_at_unix=1700000000,
        expires_at_unix=1700000900,
    )
    assert default_sig.max_hold_until_unix == 0

    ml_sig = EaSignalPayload(
        signal_id="sig2",
        generated_at_unix=1700000000,
        expires_at_unix=1700000900,
        order_intent="MARKET_BUY",
        trade_allowed=True,
        sl_price=3840.0,
        tp_price=3856.0,
        max_hold_until_unix=1700014400,
        strategy_mode="M15_PRECISION",
        regime="TRENDING",
    )
    assert ml_sig.max_hold_until_unix == 1700014400
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
    assert pred.order_intent == "MARKET_BUY"
    assert s.default_strategy_mode == "M15_PRECISION"
    assert s.ml_spread_cost_usd == 0.30
    assert s.ml_min_profit_factor == 1.05
    assert s.ml_min_oos_trades == 20
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest python/tests/test_ml_schemas.py -v`
Expected: FAIL with `ImportError` / missing fields.

- [ ] **Step 4: Implement schema & config additions in `python/src/schemas.py` and `python/src/config.py`**

Add `StrategyMode`, `RegimeLabel`, `MlMetricsSummary`, `MlPrediction`, and add `max_hold_until_unix: int = Field(default=0, ge=0)`, `strategy_mode: str = ""`, `regime: str = ""` to `EaSignalPayload`. Add the pinned ML configuration fields with defaults to `Settings` (storing models under `PROJECT_ROOT / "data" / "models"` by default).

- [ ] **Step 5: Run tests to verify schemas, config, and existing suite still pass**

Run: `.venv\Scripts\python.exe -m pytest python/tests/test_ml_schemas.py python/tests/test_schemas_and_bridge.py -v`
Expected: PASS (`0 failed`).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt python/src/schemas.py python/src/config.py python/tests/test_ml_schemas.py
git commit -m "feat: add dual-mode ML dependencies, schemas, and config settings"
```

---

### Task 2: Causal 37-Feature Engineering Pipeline (`ml_features.py`)

**Files:**
- Create: `python/src/ml_features.py`
- Test: `python/tests/test_ml_features.py`

**Interfaces:**
- Consumes: `OhlcvBar` from `python/src/schemas.py` (or `pandas.DataFrame` with columns `t, o, h, l, c, v`)
- Produces in `python/src/ml_features.py`:
  - `FEATURE_COLUMNS: tuple[str, ...]` (exact 37 causal feature names for M15; for M5, same 37 base features plus 4 strictly causal higher-timeframe `M15` trend/slope/RSI/ATR context columns if `htf_df` provided, default-filled to `0.0` when `htf_df is None` so column schema stays fixed)
  - `REGIME_COLUMNS: tuple[str, ...]` (`("log_ret_1", "atr_pct_14", "adx_14")`)
  - `compute_features(df: pd.DataFrame, htf_df: pd.DataFrame | None = None) -> pd.DataFrame`
    - Drops rows in warmup (`< 60` bars) or returns full indexed DataFrame with `atr_14`, `close`, `t`, plus all `FEATURE_COLUMNS` (finite floats, no `NaN`/`Inf` in valid rows after `warmup=60`).
  - `bars_to_dataframe(bars: list[OhlcvBar]) -> pd.DataFrame`

Exact 37 base feature groups (causal `.shift`/`.rolling`/`.ewm` only, zero centered windows or negative shifts):
1. Returns & Momentum (6): `log_ret_1`, `log_ret_3`, `log_ret_6`, `log_ret_12`, `roc_10`, `mom_norm_20`
2. Moving Averages & Trend (7): `ema_9_dist`, `ema_21_dist`, `ema_50_dist`, `ema_200_dist`, `ema_9_21_diff`, `ema_21_50_diff`, `ema_21_slope_5`
3. Oscillators & Volatility (9): `rsi_14`, `macd_line_pct`, `macd_hist_pct`, `stoch_k_14`, `stoch_d_3`, `bb_width_20`, `bb_pct_b_20`, `atr_pct_14`, `atr_ratio_14_50`
4. Directional Movement (3): `adx_14`, `plus_di_14`, `minus_di_14`
5. Candle Geometry & Volume (5): `body_ratio`, `upper_wick_ratio`, `lower_wick_ratio`, `range_atr_ratio`, `vol_zscore_20`
6. Session Cyclicals (4): `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`
7. Causal SMC Proxies (3): `pd_ratio_20`, `causal_fvg_dir` (`+1/-1/0` 3-bar closed pattern), `causal_bos_dir` (`+1/-1/0` close break of prior 20-bar high/low)
+ 4 HTF M15 context columns (`htf_ema_21_50_diff`, `htf_rsi_14`, `htf_atr_pct_14`, `htf_bos_dir`) via backward `pd.merge_asof(..., direction="backward")` on closed M15 `t + 900 <= m5.t` (strictly causal — an M15 bar ending at `t + 900` is visible to an M5 bar closing at or after `t + 900`, i.e., using `m15_closed_at = m15.t + 900` matched against `m5_closed_at = m5.t + 300`).

- [ ] **Step 1: Write the failing tests in `python/tests/test_ml_features.py`**

```python
import numpy as np
import pandas as pd
import pytest

from python.src.ml_features import (
    FEATURE_COLUMNS,
    REGIME_COLUMNS,
    compute_features,
)


def _make_ohlcv(n: int = 320, start_ts: int = 1700000000, step: int = 900, flat: bool = False) -> pd.DataFrame:
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
    assert len(FEATURE_COLUMNS) >= 37
    assert set(REGIME_COLUMNS).issubset(FEATURE_COLUMNS)
    df = _make_ohlcv(320)
    feats = compute_features(df)
    assert len(feats) > 100
    assert list(feats[list(FEATURE_COLUMNS)].isna().sum()) == [0] * len(FEATURE_COLUMNS)
    assert np.isfinite(feats[list(FEATURE_COLUMNS)].to_numpy()).all()


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest python/tests/test_ml_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'python.src.ml_features'`.

- [ ] **Step 3: Implement `python/src/ml_features.py`**

Implement `FEATURE_COLUMNS` (37 base features + 4 causal HTF features), `REGIME_COLUMNS`, `bars_to_dataframe`, and `compute_features(df, htf_df=None)` using vectorized numpy/pandas calculations with safe zero-division guards (`eps = 1e-9`) and closed-bar backward `merge_asof` for HTF context. Verify `htf_df` causality test as well (add test in `test_ml_features.py` asserting appending future M15 bars never alters historical M5 HTF columns).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest python/tests/test_ml_features.py -v`
Expected: PASS (`0 failed`).

- [ ] **Step 5: Commit**

```bash
git add python/src/ml_features.py python/tests/test_ml_features.py
git commit -m "feat: implement causal 37-feature + HTF engineering pipeline"
```

---

### Task 3: 3-State Forward-Only GaussianHMM Regime Classifier (`ml_regime.py`)

**Files:**
- Create: `python/src/ml_regime.py`
- Test: `python/tests/test_ml_regime.py`

**Interfaces:**
- Consumes: `REGIME_COLUMNS` from `python/src/ml_features.py`, `RegimeLabel` from `python/src/schemas.py`
- Produces in `python/src/ml_regime.py`:
  - `class RegimeModel`:
    - `@classmethod def fit(cls, X_regime: np.ndarray, random_state: int = 42) -> RegimeModel`
    - `def predict_proba_causal(self, X_regime: np.ndarray) -> tuple[np.ndarray, list[RegimeLabel]]` (strictly forward HMM recursion using `model.startprob_`, `model.transmat_`, `model._compute_log_likelihood` — zero future backward smoothing!)
    - `def predict_latest(self, X_regime: np.ndarray) -> tuple[RegimeLabel, dict[str, float]]`

Semantic state mapping algorithm (deterministic):
- State with highest fitted mean `atr_pct_14` -> `"VOLATILE"`.
- Among the remaining 2 states, higher fitted mean `adx_14` -> `"TRENDING"`, remaining -> `"RANGING"`.

- [ ] **Step 1: Write the failing tests in `python/tests/test_ml_regime.py`**

```python
import numpy as np

from python.src.ml_regime import RegimeModel


def _synthetic_regime_matrix(n_per_regime: int = 120) -> np.ndarray:
    rng = np.random.default_rng(7)
    ranging = np.column_stack([
        rng.normal(0.0, 0.0005, n_per_regime),
        rng.normal(0.0010, 0.0002, n_per_regime),
        rng.normal(14.0, 2.0, n_per_regime),
    ])
    trending = np.column_stack([
        rng.normal(0.0012, 0.0007, n_per_regime),
        rng.normal(0.0020, 0.0003, n_per_regime),
        rng.normal(34.0, 3.0, n_per_regime),
    ])
    volatile = np.column_stack([
        rng.normal(0.0, 0.0035, n_per_regime),
        rng.normal(0.0065, 0.0008, n_per_regime),
        rng.normal(22.0, 4.0, n_per_regime),
    ])
    return np.vstack([ranging, trending, volatile])


def test_regime_fit_maps_states_semantically():
    X = _synthetic_regime_matrix()
    model = RegimeModel.fit(X)
    probs, labels = model.predict_proba_causal(X)
    assert probs.shape == (len(X), 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert labels[60] == "RANGING"
    assert labels[180] == "TRENDING"
    assert labels[300] == "VOLATILE"


def test_forward_only_filtering_has_no_future_leakage():
    X = _synthetic_regime_matrix()
    model = RegimeModel.fit(X[:240])
    probs_full, _ = model.predict_proba_causal(X)
    probs_prefix, _ = model.predict_proba_causal(X[:180])
    np.testing.assert_allclose(probs_full[:180], probs_prefix, rtol=1e-9, atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest python/tests/test_ml_regime.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'python.src.ml_regime'`.

- [ ] **Step 3: Implement `python/src/ml_regime.py`**

Fit `StandardScaler` + 3-state `GaussianHMM(covariance_type="full", n_iter=200, random_state=random_state)` with tiny variance jitter (`+1e-8 * noise`) fallback if an input column is near-constant, map HMM state indices deterministically to `(TRENDING, RANGING, VOLATILE)`, and implement log-space forward recursion (`predict_proba_causal`) so appending future bars never alters historical regime probabilities.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest python/tests/test_ml_regime.py -v`
Expected: PASS (`0 failed`).

- [ ] **Step 5: Commit**

```bash
git add python/src/ml_regime.py python/tests/test_ml_regime.py
git commit -m "feat: add forward-only 3-state GaussianHMM regime classifier"
```

---

### Task 4: Triple-Barrier Labeling, Calibrated XGBoost, Walk-Forward Validator & Rollback (`ml_trainer.py`)

**Files:**
- Create: `python/src/ml_trainer.py`
- Test: `python/tests/test_ml_trainer.py`

**Interfaces:**
- Consumes: `compute_features`, `FEATURE_COLUMNS`, `REGIME_COLUMNS` from `python/src/ml_features.py`; `RegimeModel` from `python/src/ml_regime.py`; `StrategyMode`, `MlMetricsSummary` from `python/src/schemas.py`
- Produces in `python/src/ml_trainer.py`:
  - `MODE_SPECS: dict[StrategyMode, ModeSpec]` (`M15_PRECISION`: `sl_atr=1.5`, `tp_atr=2.4`, `horizon_bars=16`, `bar_seconds=900`, `timeframe="M15"`; `M5_INTRADAY`: `sl_atr=1.5`, `tp_atr=2.25`, `horizon_bars=24`, `bar_seconds=300`, `timeframe="M5"`)
  - `compute_triple_barrier_outcomes(df: pd.DataFrame, sl_atr: float, tp_atr: float, horizon_bars: int, spread_cost_usd: float = 0.30, lot_size: float = 0.01) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]` (`y_long`, `y_short`, `pnl_long_usd`, `pnl_short_usd`)
    - Entry on bar `i` close `c[i]`; scans `j in range(i + 1, min(len, i + 1 + horizon_bars))`.
    - Conservative SL-first tie-breaking if bar `j` touches both SL and TP (`sl_hit` beats `tp_hit`).
    - On timeout at `i + horizon_bars`, exit at `c[i + horizon_bars]`.
    - Deduct `spread_cost_usd` per trade converted to USD P&L at `lot_size` (`(price_diff - spread_cost_usd) * 100.0 * lot_size`, since 0.01 lot = `$1.00 / $1.00 price move`? Verify: on XAUUSD 1 standard lot = 100 oz -> `$1 move = $100` -> `0.01 lot * 100 * $1 = $1.00`! Exact!).
    - Binary label `1` iff simulated net P&L after `$0.30` cost is strictly `> 0.0` (meaning TP won or timeout finished net positive after spread). Wait: verify for `y_long`/`y_short` cleanly: `pnl_long_usd > 0` (`price_diff > spread_cost_usd`).
  - `class TrainedBundle` (dataclass / joblib-serializable: `mode`, `feature_cols`, `regime_model`, `xgb_long`, `xgb_short`, `threshold`, `metrics: MlMetricsSummary`)
  - `def simulate_oos_backtest(probs_long: np.ndarray, probs_short: np.ndarray, pnl_long: np.ndarray, pnl_short: np.ndarray, horizon_bars: int, threshold: float, eligible_mask: np.ndarray | None = None) -> tuple[int, float, float, float]` (`trades, win_rate, profit_factor, net_pnl_usd`) — sequential non-overlapping position stepping (`next_free_idx = i + exit_bars` or `horizon_bars` cap) so concurrent signals don't double-count!
  - `def search_best_threshold(probs_long, probs_short, pnl_long, pnl_short, horizon_bars, eligible_mask=None) -> float` across `[0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]`
  - `def train_and_validate_mode(mode: StrategyMode, df: pd.DataFrame, htf_df: pd.DataFrame | None, models_dir: Path, spread_cost_usd: float = 0.30, min_pf: float = 1.05, min_trades: int = 20) -> TrainedBundle`
  - `def load_bundle(mode: StrategyMode, models_dir: Path) -> TrainedBundle | None`

Rollback & Gate Logic in `train_and_validate_mode`:
1. Compute causal features & Triple-Barrier outcomes on `df` (`htf_df` when `M5_INTRADAY`), drop last `horizon_bars` rows (no forward horizon).
2. Split chronologically: `60%` train / `20%` calibration+threshold validation / `20%` purged out-of-sample test (embargo `= horizon_bars` between splits).
3. Fit `RegimeModel` on train split ONLY; generate causal regime probs `[p_trend, p_range, p_volatile]` as 3 additional model features (`X_full = [features, regime_probs]`).
4. Train `XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42)` on train split for Long and Short, wrap with `CalibratedClassifierCV(FrozenEstimator(base_model), method="isotonic" or "sigmoid")` fit on validation split! (Note: check sklearn 1.9 `FrozenEstimator` API when writing!)
5. Search best threshold on validation split, then evaluate on OOS test split (`trades, win_rate, pf, net_pnl`).
6. Gate check: `passed = (oos_pf >= min_pf) and (oos_trades >= min_trades)`.
   - If `passed`: save verified bundle to `models_dir / f"{mode.lower()}_bundle.joblib"` and sidecar `f"{mode.lower()}_metrics.json"`.
   - If NOT `passed`: if an existing verified bundle (`verified_for_live=True`) already exists at `models_dir / f"{mode.lower()}_bundle.joblib"`, **keep it untouched** and write sidecar/return report reflecting retrain rejected + preserved prior bundle; if no prior verified bundle exists, save/update bundle with `verified_for_live=False`.

- [ ] **Step 1: Write the failing tests in `python/tests/test_ml_trainer.py`**

```python
import numpy as np
import pandas as pd

from python.src.ml_trainer import (
    compute_triple_barrier_outcomes,
    load_bundle,
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
    _ = train_and_validate_mode(
        mode="M15_PRECISION",
        df=df,
        htf_df=None,
        models_dir=models_dir,
        min_pf=99.0,
        min_trades=20,
    )
    reloaded = load_bundle("M15_PRECISION", models_dir)
    assert reloaded is not None
    assert reloaded.metrics.verified_for_live is True
    assert reloaded.metrics.trained_at_unix == saved_ts
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest python/tests/test_ml_trainer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'python.src.ml_trainer'`.

- [ ] **Step 3: Implement `python/src/ml_trainer.py`**

Implement `MODE_SPECS`, `compute_triple_barrier_outcomes`, `simulate_oos_backtest`, `search_best_threshold`, `TrainedBundle`, `train_and_validate_mode`, and `load_bundle`. Verify scikit-learn `CalibratedClassifierCV` signature in `.venv` (`FrozenEstimator` + `cv="prefit"` check via python `-c` before finalizing). Note: when one side lacks both classes in a small/synthetic split, handle cleanly without crashing `XGBClassifier` (`DummyClassifier(strategy="prior")` fallback).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest python/tests/test_ml_trainer.py -v`
Expected: PASS (`0 failed`).

- [ ] **Step 5: Commit**

```bash
git add python/src/ml_trainer.py python/tests/test_ml_trainer.py
git commit -m "feat: implement Triple-Barrier labeling, calibrated XGBoost, OOS gate, and rollback"
```

---

### Task 5: Live MetaTrader5 Historical & Closed-Bar Feed (`mt5_feed.py`)

**Files:**
- Create: `python/src/mt5_feed.py`
- Test: `python/tests/test_mt5_feed.py`

**Interfaces:**
- Consumes: `OhlcvBar` from `python/src/schemas.py`, `bars_to_dataframe` from `python/src/ml_features.py`
- Produces in `python/src/mt5_feed.py`:
  - `class Mt5Feed`:
    - `__init__(self, symbol: str = "XAUUSDm", mt5_module=None)` (accepts injected fake module for deterministic unit tests; lazily imports `MetaTrader5` in production)
    - `def fetch_closed_bars(self, timeframe: Literal["M5", "M15", "H1"], count: int) -> pd.DataFrame`
      - Calls `initialize()` if needed, ensures symbol selected (`symbol_select`), fetches `count + 2` bars via `copy_rates_from_pos`, drops the currently forming unclosed bar (last row where `bar_t + tf_seconds > latest_server_tick_time`, or always drops the trailing bar if `copy_rates_from_pos(..., 0, count + 1)` includes current bar `pos=0` by slicing `rates[:-1]` when `rates[-1]["time"] + tf_sec > tick_time` or simply requesting starting at `start_pos=1` (`copy_rates_from_pos(symbol, tf, 1, count)`!Wait: `start_pos=1` in MetaTrader5 strictly fetches CLOSED bars only, never `pos=0`! Double check: still assert `df["t"].is_monotonic_increasing` and verify `start_pos=1` skips forming candle!)
    - `def fetch_tick_spread(self) -> tuple[float, float, float]` (`bid, ask, spread_usd`; returns `(0.0, 0.0, 0.0)` cleanly on error)

- [ ] **Step 1: Write the failing tests in `python/tests/test_mt5_feed.py`**

```python
import numpy as np

from python.src.mt5_feed import Mt5Feed


class _FakeMt5:
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_H1 = 16385

    def __init__(self):
        self.calls = []

    def initialize(self):
        return True

    def symbol_select(self, symbol, enable):
        return True

    def symbol_info_tick(self, symbol):
        class Tick:
            bid = 3850.00
            ask = 3850.24
        return Tick()

    def copy_rates_from_pos(self, symbol, tf, start_pos, count):
        self.calls.append((symbol, tf, start_pos, count))
        dtype = [("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"), ("close", "f8"), ("tick_volume", "i8")]
        arr = np.zeros(count, dtype=dtype)
        for i in range(count):
            arr[i] = (1700000000 + i * 900, 3800.0 + i, 3802.0 + i, 3799.0 + i, 3801.0 + i, 120 + i)
        return arr


def test_mt5_feed_fetches_closed_bars_starting_at_pos_1():
    fake = _FakeMt5()
    feed = Mt5Feed(symbol="XAUUSDm", mt5_module=fake)
    df = feed.fetch_closed_bars("M15", count=50)
    assert len(df) == 50
    assert list(df.columns) == ["t", "o", "h", "l", "c", "v"]
    assert fake.calls[-1] == ("XAUUSDm", _FakeMt5.TIMEFRAME_M15, 1, 50)
    bid, ask, spread = feed.fetch_tick_spread()
    assert round(spread, 2) == 0.24
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest python/tests/test_mt5_feed.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'python.src.mt5_feed'`.

- [ ] **Step 3: Implement `python/src/mt5_feed.py`**

Implement `Mt5Feed` cleanly with `start_pos=1` closed-bar fetch, ascending timestamp sorting/deduplication, `mt5_module` DI for testing, `TIMEFRAME_MAP` (`M5`, `M15`, `H1`), and clean `RuntimeError` messages if MT5 terminal isn't running or returns `None`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest python/tests/test_mt5_feed.py -v`
Expected: PASS (`0 failed`).

- [ ] **Step 5: Commit**

```bash
git add python/src/mt5_feed.py python/tests/test_mt5_feed.py
git commit -m "feat: add MetaTrader5 closed-bar feed with test double injection"
```

---

### Task 6: Dual-Mode Live Inference & Exclusive Signal Runner (`ml_engine.py`)

**Files:**
- Create: `python/src/ml_engine.py`
- Test: `python/tests/test_ml_engine.py`

**Interfaces:**
- Consumes: `Settings` (`config.py`), `Mt5Feed` (`mt5_feed.py`), `MarketDataFeeds` + `compute_hmr_blackout` (`data_feeds.py`), `evaluate_session` (`sessions.py`), `compute_features` + `FEATURE_COLUMNS` + `REGIME_COLUMNS` (`ml_features.py`), `MODE_SPECS` + `TrainedBundle` + `load_bundle` + `train_and_validate_mode` (`ml_trainer.py`), `EaSignalPayload` + `MlPrediction` + `StrategyMode` (`schemas.py`), `Mt5Bridge` (`bridge.py`)
- Produces in `python/src/ml_engine.py`:
  - `class MlSignalRunner`:
    - `def __init__(self, settings: Settings, mt5_feed: Mt5Feed, feeds: MarketDataFeeds, bridge: Mt5Bridge)`
    - `def get_bundle(self, mode: StrategyMode) -> TrainedBundle | None`
    - `def is_mode_verified(self, mode: StrategyMode) -> bool`
    - `def predict_mode(self, mode: StrategyMode) -> tuple[MlPrediction, pd.DataFrame]` (computes live regime + Long/Short calibrated probabilities + ATR SL/TP + `max_hold_until_unix` on latest closed bar without mutating `last_emitted_bar_ts`)
    - `async def step_live_cycle(self, mode: StrategyMode, execution_mode: ExecutionMode) -> tuple[MlPrediction | None, EaSignalPayload | None, bool]`
      - Returns `(prediction, emitted_signal, is_new_bar)`
      - Deduplicates by `(mode, prediction.bar_time_unix)`: only emits once per closed bar!
      - Checks kill-zone (`settings.require_killzone`) + Exness HMR blackout (`compute_hmr_blackout`) + `verified_for_live` gate (`execution_mode == "auto"` blocked -> downgraded `trade_allowed=False` if unverified!)
      - Writes `EaSignalPayload` to `bridge.write_signal(...)` when `is_new_bar` is True and `execution_mode in {"auto", "confirm"}` (in `confirm` mode or failed verification, `trade_allowed=False` unless user manually approves via Confirm button, which sends a confirmed `EaSignalPayload`).
    - `async def retrain_mode(self, mode: StrategyMode) -> TrainedBundle` (runs `train_and_validate_mode` in worker thread via `asyncio.to_thread`, refreshes cached bundle in memory)

- [ ] **Step 1: Write the failing tests in `python/tests/test_ml_engine.py`**

```python
import asyncio
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from python.src.bridge import Mt5Bridge
from python.src.config import Settings
from python.src.ml_engine import MlSignalRunner
from python.src.ml_trainer import train_and_validate_mode


def _bars(n: int = 700, step: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    t = 1700000000 + np.arange(n) * step
    c = 3850.0 + np.cumsum(np.sin(np.arange(n) / 18.0) * 0.4 + rng.normal(0.0, 0.5, n))
    o = np.roll(c, 1)
    o[0] = c[0]
    w = np.abs(rng.normal(0.8, 0.2, n)) + 0.1
    return pd.DataFrame({"t": t, "o": o, "h": np.maximum(o, c) + w, "l": np.minimum(o, c) - w, "c": c, "v": np.full(n, 200.0)})


class _FakeFeed:
    def __init__(self, m15_df, m5_df):
        self.m15_df = m15_df
        self.m5_df = m5_df

    def fetch_closed_bars(self, tf, count):
        return (self.m5_df if tf == "M5" else self.m15_df).tail(count).reset_index(drop=True)

    def fetch_tick_spread(self):
        return (3850.0, 3850.24, 0.24)


class _FakeDataFeeds:
    async def fetch_economic_calendar(self, force=False):
        return []


def test_runner_deduplicates_same_closed_bar_and_blocks_unverified_auto(tmp_path):
    settings = Settings(_env_file=None, bridge_dir=tmp_path / "bridge", ml_models_dir=tmp_path / "models", require_killzone=False)
    m15_df = _bars(700, 900)
    m5_df = _bars(1200, 300)

    # Create unverified bundle first
    train_and_validate_mode("M15_PRECISION", m15_df, None, settings.ml_models_dir, min_pf=99.0, min_trades=20)

    bridge = Mt5Bridge(settings.bridge_dir)
    runner = MlSignalRunner(settings, _FakeFeed(m15_df, m5_df), _FakeDataFeeds(), bridge)

    pred, sig, is_new = asyncio.run(runner.step_live_cycle("M15_PRECISION", "auto"))
    assert is_new is True
    assert sig is not None
    assert sig.trade_allowed is False
    assert "unverified" in sig.reason.lower()

    # Second call on exact same bar must not re-emit
    pred2, sig2, is_new2 = asyncio.run(runner.step_live_cycle("M15_PRECISION", "auto"))
    assert is_new2 is False
    assert sig2 is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest python/tests/test_ml_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'python.src.ml_engine'`.

- [ ] **Step 3: Implement `python/src/ml_engine.py`**

Implement `MlSignalRunner` handling both `M15_PRECISION` and `M5_INTRADAY` (`M5_INTRADAY` passes `M15` bars as `htf_df`), trend gate for `M5_INTRADAY` (only emits BUY when `htf_ema_21_50_diff >= 0`, SELL when `<= 0`), session & HMR blackout gating, `max_hold_until_unix` calculation (`now_unix + spec.horizon_bars * spec.bar_seconds`), verification guard, and per-mode closed-bar deduplication.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest python/tests/test_ml_engine.py -v`
Expected: PASS (`0 failed`).

- [ ] **Step 5: Commit**

```bash
git add python/src/ml_engine.py python/tests/test_ml_engine.py
git commit -m "feat: add dual-mode MlSignalRunner with bar dedup and OOS verification guard"
```

---

### Task 7: Read-Only Gemini Briefings, ML Chart Card & Telegram 2-Mode Control (`analyst.py`, `chart_card.py`, `telegram_bot.py`, `main.py`)

**Files:**
- Modify: `python/src/analyst.py` (add read-only `analyze_brief(...)` / ensure commentary & HMR blackout sync cleanly separated from signal execution)
- Modify: `python/src/chart_card.py` (add `ml_prediction: MlPrediction | None = None` rendering `🎯 M15` / `⚡ M5` + HMM regime + Long/Short probability pills)
- Modify: `python/src/telegram_bot.py` (`BotRuntimeState.strategy_mode: StrategyMode`, ML 15s watcher job replacing Gemini trade-loop, `🎯 M15 Precision` / `⚡ M5 Intraday` mode switcher, `🤖 ML Models` status & `🔄 Retrain ML`, `Auto` lock if `!is_mode_verified`, Confirm callback for ML pending signals, `⚡ Analyze Now` & `📰 Full AI Brief` strictly read-only)
- Modify: `python/src/main.py` (wire `Mt5Feed` + `MlSignalRunner` into `TelegramController`)
- Test: `python/tests/test_telegram_callbacks.py`

**Interfaces:**
- Consumes: `MlSignalRunner` (`ml_engine.py`), `MlPrediction` & `StrategyMode` (`schemas.py`)
- Produces:
  - `render_chart_card(..., ml_prediction: MlPrediction | None = None) -> bytes`
  - `TelegramController(..., ml_runner: MlSignalRunner)`
  - Callback routes in `TelegramController.handle_callback`: `strategy:M15_PRECISION`, `strategy:M5_INTRADAY`, `menu:ml_models`, `ml:retrain`

- [ ] **Step 1: Write the failing tests in `python/tests/test_telegram_callbacks.py`**

Add tests pinning:
1. Switching `strategy:M15_PRECISION` ↔ `strategy:M5_INTRADAY` updates `runtime.strategy_mode` cleanly.
2. Attempting `mode:auto` when `ml_runner.is_mode_verified(runtime.strategy_mode)` is `False` keeps mode in `confirm`/`pause` and returns a clear warning alerting the user.
3. Calling `⚡ Analyze Now` runs `GoldAnalyst` + chart render in read-only mode and **never** calls `bridge.write_signal(...)`.
4. `render_chart_card(..., ml_prediction=...)` returns valid PNG bytes with non-zero length.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest python/tests/test_telegram_callbacks.py -v`
Expected: FAIL on missing `ml_runner` / `strategy_mode` / `ml_prediction`.

- [ ] **Step 3: Implement updates in `analyst.py`, `chart_card.py`, `telegram_bot.py`, and `main.py`**

Make `GoldAnalyst` commentary-only (add clean `run_brief` / keep `analyze` returning `AnalysisReport` without callers ever writing `ai_signal.json` from Gemini), update `chart_card.py` header pills for optional `MlPrediction`, update `telegram_bot.py` menus/callbacks/jobs (15s ML watcher + nightly retrain schedule + `🔄 Retrain ML` handler reporting OOS metrics), and wire `Mt5Feed` + `MlSignalRunner` in `main.py` (with graceful backward-compatible default in tests).

- [ ] **Step 4: Run test suite to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest python/tests/test_telegram_callbacks.py -v`
Expected: PASS (`0 failed`).

- [ ] **Step 5: Commit**

```bash
git add python/src/analyst.py python/src/chart_card.py python/src/telegram_bot.py python/src/main.py python/tests/test_telegram_callbacks.py
git commit -m "feat: integrate dual-mode ML Telegram controls, chart pills, and read-only Gemini brief"
```

---

### Task 8: MQL5 Triple-Barrier Timeout Exit & ML-Safe Defaults (`EA_LearningX_Gold.mq5`)

**Files:**
- Modify: `mql5/Experts/EA_LearningX_Gold.mq5:30-90,510-706,707-900`

**Interfaces:**
- Consumes from `ai_signal.json`: `max_hold_until_unix` (long, UTC unix seconds; `0` = disabled)

Specific edits in `EA_LearningX_Gold.mq5`:
1. Change defaults: `input bool InpUseSarEngine = false;`, `input bool InpUseAutoBE = false;`, `input bool InpUseTrailing = false;`.
2. Add global helper `MaxHoldKey(ulong ticket)` (`"LX_MAXH_" + IntegerToString(InpMagicNumber) + "_" + IntegerToString((long)ticket)`), `SetMaxHoldUntil(ulong ticket, long until_unix)`, `GetMaxHoldUntil(ulong ticket)` -> `0` if absent, `ClearMaxHoldUntil(ulong ticket)`.
3. In `ProcessAiSignal()`: parse `JsonGetLong(json, "max_hold_until_unix", max_hold_until, 0);`. When a market order executes (`g_trade.Buy`/`g_trade.Sell` returns true), find the opened position matching `comment` (or `PositionGetTicket` from open positions) and call `SetMaxHoldUntil(pos_ticket, max_hold_until)` if `max_hold_until > 0`.
4. Also handle pending limit fills: store `max_hold_until` on order ticket or scan `LX_` comment match? Even cleaner: store `LX_MHSIG_<sig_prefix>` or check if pending order converts to position with same magic/comment (`comment = "LX_" + StringSubstr(sig_id, 0, 12)`) — since ML emits `MARKET_BUY`/`MARKET_SELL` orders directly, `g_trade.Buy`/`Sell` followed by scanning open positions matching `comment` catches the exact opened position ticket! Still check existing `ManageOpenPositions()` to delete stale `LX_MAXH_*` / `LX_HOLD_*` only for tickets no longer open, AND close non-held open positions when `max_hold > 0 && (long)TimeGMT() >= max_hold`!

- [ ] **Step 1: Update `EA_LearningX_Gold.mq5` with `max_hold_until_unix` and clean ML defaults**

Apply the exact changes to inputs, `MaxHoldKey` helpers, `ProcessAiSignal`, and `ManageOpenPositions`.

- [ ] **Step 2: Compile `EA_LearningX_Gold.mq5` with MetaEditor (`scripts/compile_ea.ps1`)**

Run:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\compile_ea.ps1
```
Expected: `Result: 0 errors, 0 warnings` in `mql5/compile.log`.

- [ ] **Step 3: Commit**

```bash
git add mql5/Experts/EA_LearningX_Gold.mq5
git commit -m "feat(mql5): enforce matched triple-barrier timeout exits and ML-safe defaults"
```

---

### Task 9: Full Verification, Live MT5 Historical Training & OOS Benchmark + Git Push

**Files:**
- Create: `scripts/train_ml.py` (CLI helper to fetch live Exness `XAUUSDm` bars from running MT5 terminal, train & walk-forward validate both `M15_PRECISION` and `M5_INTRADAY`, and print full OOS metrics table)
- Modify: `README.md` (document dual-mode HMM + XGBoost ML architecture, setup & retrain CLI)

- [ ] **Step 1: Add `scripts/train_ml.py` and run full unit test suite**

Run:
```powershell
d:\MATERIAL\Antigravity_Project\EA-LearningX\.venv\Scripts\python.exe -m pytest python/tests -v
```
Expected: All tests PASS (`0 failed`, `0 skipped`).

- [ ] **Step 2: Run live Exness `XAUUSDm` training & walk-forward backtest benchmark (`scripts/train_ml.py`)**

Run:
```powershell
d:\MATERIAL\Antigravity_Project\EA-LearningX\.venv\Scripts\python.exe scripts\train_ml.py
```
Expected: Prints real Exness `XAUUSDm` sample size, OOS trades, OOS Win Rate, OOS Profit Factor, OOS Net PnL ($0.30/trade deducted), chosen confidence threshold, and `verified_for_live` status for both `M15_PRECISION` and `M5_INTRADAY`.

- [ ] **Step 3: Update `README.md` accurately and push verified source changes**

Verify `git status` keeps `.env`, `.venv/`, `data/`, `logs/`, `*.log`, and `*.ex5` untracked/ignored, commit, and run `git push -4 origin main`.

---

## Self-Review Check

- **Spec coverage:** All 9 sections of [2026-11-16-ml-hmm-xgboost-design.md](file:///d:/MATERIAL/Antigravity_Project/EA-LearningX/docs/superpowers/specs/2026-11-16-ml-hmm-xgboost-design.md) map directly to Tasks 1–9 (`37 features + HTF`, `3-state forward-only GaussianHMM`, `Triple-Barrier $0.30 + SL-first + validation threshold tuning`, `OOS PF>=1.05 & trades>=20 + rollback`, `MlSignalRunner exclusive ownership`, `read-only GoldAnalyst`, `Telegram M15/M5 toggle + chart card pills`, `MQL5 max_hold_until_unix + clean defaults`, `live OOS benchmark`).
- **Type & signature consistency:** `StrategyMode`, `RegimeLabel`, `EaSignalPayload.max_hold_until_unix`, `FEATURE_COLUMNS`, `REGIME_COLUMNS`, `MODE_SPECS`, `TrainedBundle`, `Mt5Feed.fetch_closed_bars`, `MlSignalRunner.step_live_cycle`, and `render_chart_card(..., ml_prediction=...)` match cleanly across producer and consumer tasks.
- **Proportion:** Plan focuses strictly on architecture, file boundaries, signatures, exact spec constants, and TDD assertions without bloated implementation transcripts.
