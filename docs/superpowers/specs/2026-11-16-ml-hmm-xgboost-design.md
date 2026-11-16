# EA-LearningX Dual-Mode ML Engine (`HMM + XGBoost + SMC`) Design Spec

**Date:** 2026-11-16  
**Approach Selected:** Approach A — Local Python ML (`MetaTrader5` + `HMM` + `XGBoost`) + Existing MQL5 Bridge (`EA_LearningX_Gold.mq5`)  
**Reference Prior Art:** `GifariKemal/xaubot-ai`

---

## 1. Intent & Why

Live testing on Exness `XAUUSDm` (`$0.24` spread) showed two failure modes in earlier versions:
1. **LLM-only 15m/30m entries** hit `gemini-4-hydrogen` free-tier limits (`20 req/day`) and had no statistical edge validation.
2. **Pure M1 tight-stop/SAR scalping** suffered severe spread tax (`$0.24` vs `$0.55–$1.10` SL) and 1-minute wick whipsaws (`-280k IDR` during 13:38 chop; `-107.9k IDR` at 14:45–14:55).

This upgrade adopts `xaubot-ai`'s proven quantitative structure on top of our existing MQL5 execution safety and Telegram UI:
- Pull thousands of real Exness `XAUUSDm` candles directly from the open MT5 terminal (`MetaTrader5` Python API).
- Detect market state mathematically with a **3-state Hidden Markov Model (`GaussianHMM`)**.
- Predict directional edge with **calibrated `XGBoost` models** trained with spread-aware **triple-barrier labels** across ~37 quantitative + SMC + session features.
- Require **out-of-sample walk-forward verification** (Net PnL & Profit Factor) *before* deploying trained models.
- Offer **2 switchable live strategy modes** on Telegram: **`🎯 M15 Precision`** and **`⚡ M5 Intraday`**.

---

## 2. Goals, Non-Goals & Anti-Goals

### Goals
- **Two Switchable Strategy Modes (`🎯 M15 Precision` / `⚡ M5 Intraday`):**
  - **`M15_PRECISION`:** Evaluates each closed `M15` candle (~2–6 setups/day), wider `M15` ATR-based SL (`1.5x ATR_14`) and TP (`2.4x ATR_14`, `1:1.6` R:R), highest confidence & regime filtering for clean Profit Factor.
  - **`M5_INTRADAY`:** Uses `M15` trend + HMM regime alignment while triggering on closed `M5` candles (~6–15 setups/day), `M5` ATR-based SL (`1.5x ATR_14`) and TP (`2.25x ATR_14`, `1:1.5` R:R), faster intraday feedback.
- **Zero-API-Quota Live Inference:** ML predictions + HMM regime run locally in `<100ms` on every new closed `M5`/`M15` bar without touching Gemini quotas. Gemini remains purely on-demand for `⚡ Analyze Now` chart cards and `📰 Full AI Brief`.
- **Keep Existing MQL5 Protections:** Retain `🔒 Keep #<ticket>` hold locks, real MT5 SL/TP on every trade, Break-Even (`0.7R`), trailing stops, Exness HMR USD news blackout, daily drawdown halt, and real closed-deal PnL sync into `journal.db`.
- **Automated Walk-Forward Training & Retraining:** Clean CLI (`python -m python.src.ml_trainer`) + Telegram `🔄 Retrain ML` button / scheduled weekly retrain.

### Non-Goals
- Replacing `EA_LearningX_Gold.mq5` with raw Python `mt5.order_send()` (Approach B).
- Building a Next.js web dashboard right now (Telegram mobile UI + PNG chart cards already fill that role).

### Anti-Goals
- Never trade unbacktested `M1` noise or place stops tight enough for `$0.24` spread + 1m wicks to auto-stop.
- Never claim backtest profits without subtracting Exness `$0.24+` spread/slippage in triple-barrier labeling & simulation.
- Never silently activate an uncalibrated model whose out-of-sample Profit Factor `< 1.05`.

---

## 3. Approaches Considered

| Approach | Summary | Trade-offs | Decision |
|---|---|---|---|
| **A. Python `HMM + XGBoost` -> Existing MQL5 Bridge** | Pull bars via `MetaTrader5` Python API, train/backtest locally, send ATR signals via `ai_signal.json` to `EA_LearningX_Gold.mq5` | Clean separation of ML math & millisecond MQL5 SL/BE/Keep safety; keeps all existing UI | **Selected** |
| **B. Full Python (`mt5.order_send`)** | Retire `.mq5` EA and trade directly from Python | Simpler 1-process execution, but loses tick-speed SL/BE/Keep & MT5 chart HUD | Rejected |
| **C. MQL5 Rule-Only M15/M5** | Indicator thresholds hardcoded in MQL5 without ML/HMM | No probability calibration, no regime fitting, repeats M1 rule-tuning failures | Rejected |

---

## 4. Architecture & Data Flow

```text
Open Exness MT5 (XAUUSDm)
   ├──► MetaTrader5 Python API (live M5/M15/H1 bars, no rate limits)
   │        └──► ml_features.py (37 features: TA + SMC + session + HTF)
   │                └──► ml_regime.py (3-state GaussianHMM: Trend/Range/Volatile)
   │                        └──► ml_engine.py (XGBoost M15 / M5 probability)
   │                                └──► Guards: HMR blackout, spread <= $0.60, session/cooldown, no duplicate
   │                                        └──► ai_signal.json (MARKET_BUY/SELL + ATR SL & TP)
   │                                                └──► EA_LearningX_Gold.mq5 (calculates lots, places order, BE/Trail, 🔒 Keep, exports closed_deals)
   └──► TelegramController (Switch 🎯 M15 / ⚡ M5, 🟢 Auto / Confirm / Pause, Chart Cards, 📊 Stats + PF)
```

---

## 5. Module-by-Module Specification

### 5.1 Dependencies (`requirements.txt`)
Add verified Python 3.14 `win_amd64` packages:
- `MetaTrader5>=5.0.0`, `xgboost>=3.0.0`, `hmmlearn>=0.3.3`, `scikit-learn>=1.6.0`, `numpy>=2.0.0`, `pandas>=2.2.0`, `joblib>=1.4.0`.

### 5.2 Historical Data Loader (`python/src/mt5_feed.py`)
- Uses `MetaTrader5.initialize()` and `mt5.copy_rates_from_pos("XAUUSDm", tf, 0, count)` (auto-resolving `XAUUSDm` / `XAUUSD` / configured `symbol`).
- Falls back gracefully in tests or if MT5 terminal is temporarily closed (exposing clean `is_connected` check; never crashes unit tests).

### 5.3 Quantitative + SMC Feature Pipeline (`python/src/ml_features.py`)
Computes **37 stationary features** per closed bar (zero future leakage — strictly shifted to `bar[i]` closed values):
- **Price & Returns (6):** log returns `r1`, `r3`, `r6`, `r12`; candle body-to-range ratio; upper/lower wick ratios.
- **Trend & Momentum (10):** `EMA(9)/EMA(21)` & `EMA(21)/EMA(50)` normalized spreads (`(ema_a - ema_b) / atr`), `ADX(14)`, `+DI - -DI`, `RSI(14)`, `RSI_slope(3)`, `MACD_hist / atr`, Stochastic `%K - %D`, Bollinger Band `%B` & width.
- **Volatility & Volume (6):** `ATR(14) / close`, `ATR_ratio = ATR(7) / ATR(21)`, rolling z-score of range, tick volume z-score `(v - ma20)/std20`, Parkinson volatility, spread-to-ATR ratio.
- **Deterministic SMC & Structure (9):** normalized distance to nearest H1/TF swing high & swing low in ATRs, Premium/Discount ratio `[0..1]`, recent BOS/CHoCH numeric flags (`+1/-1/0`), count & proximity (in ATR) to active Bullish/Bearish OB & FVG zones, BSL/SSL liquidity sweep flags.
- **HTF & Session Cyclical Context (6):** Higher-timeframe (`H1`) EMA trend sign & slope, `H1` RSI, cyclical `sin/cos(hour_utc)`, London/NY Kill-Zone active flag, Asian session flag.

### 5.4 Hidden Markov Regime Detector (`python/src/ml_regime.py`)
- Uses `hmmlearn.hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=200, random_state=42)` trained strictly on the **training split** of `[abs_log_return, normalized_atr_14, ema_trend_strength]` (avoiding look-ahead leakage!).
- Automatically labels the 3 latent states post-fit by cluster statistics:
  - **`TRENDING`**: highest directional `|ema_trend_strength|` & moderate ATR.
  - **`RANGING`**: lowest `normalized_atr` & lowest `|ema_trend_strength|`.
  - **`VOLATILE`**: highest `normalized_atr` (extreme spikes / news whipsaw).
- Outputs current `regime_id`, `regime_label`, and `regime_probs` (`p_trending`, `p_ranging`, `p_volatile`) as features AND as a trading gate (`RANGING` + low trend probability blocks weak entries; `VOLATILE` demands higher XGBoost probability threshold).

### 5.5 Triple-Barrier Labeling, Calibration & Walk-Forward Validation (`python/src/ml_trainer.py`)
- **Spread-Aware Matched Triple-Barrier Exits** (identical rules in both backtest & live MT5 execution — entering at `open[i+1]` with Exness cost buffer `spread_cost = $0.30`):
  - **Mode `M15_PRECISION`:** `SL = 1.5 * ATR_M15`, `TP = 2.4 * ATR_M15` (`1:1.6` R:R), max hold `H = 16` closed `M15` bars (`4h`).
  - **Mode `M5_INTRADAY`:** `SL = 1.5 * ATR_M5`, `TP = 2.25 * ATR_M5` (`1:1.5` R:R), max hold `H = 24` closed `M5` bars (`2h`), additionally gated by `M15` trend alignment.
  - Label `1 (BUY)` if Long hits TP before SL (with conservative tie-break: SL wins if both hit in same candle!); `2 (SELL)` if Short hits TP before SL; `0 (HOLD)` otherwise.
- **Model, Calibration & Validation-Tuned Thresholds *(Builder Call)*:**
  - `XGBClassifier` (`max_depth=4`, `n_estimators=250`, `learning_rate=0.04`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`) wrapped with isotonic/sigmoid probability calibration on balanced class weights across chronological splits (`70% train` / `15% validation` / `15% strict out-of-sample test`).
  - Instead of hardcoding arbitrary 3-class thresholds, the trainer searches candidate probability thresholds (`0.40..0.75`) on the **15% validation slice** to maximize validation Profit Factor while ensuring at least `20` out-of-sample trades (applying regime offsets: `+0.00` in `TRENDING`, `+0.06` in `VOLATILE`, `+0.10` in `RANGING`).
- **Realistic Backtester & Rollback Promotion Gate:**
  - Simulates the 15% out-of-sample slice chronologically with exact matched exits (`SL`, `TP`, and `H`-bar timeout close at market minus `$0.30` spread/slippage — zero `0.7R BE` or `$4.00` trailing mismatch).
  - Model bundle (`data/models/m15_bundle.joblib`, `m5_bundle.joblib`) passes ONLY if OOS `PF >= 1.05` and `trades >= 20`.
  - **On Failed Retrain:** if a previously verified bundle exists for that mode, keep the existing verified bundle (zero downgrade); if none exists, mark `verified_for_live = False`, block **`Auto`** live execution for that mode (**`Confirm`-only fallback**), and notify Telegram immediately.

### 5.6 Signal Ownership, Live ML Engine & Execution Bridge (`python/src/ml_engine.py`, `EA_LearningX_Gold.mq5`)
- **Exclusive Signal Ownership:**
  - **`MlSignalRunner` exclusively owns `ai_signal.json` trade execution** in both `AUTO` and `CONFIRM` modes (checking every 15s for newly closed `M15` or `M5` bar matching the active strategy mode).
  - **`GoldAnalyst` (Gemini) becomes strictly read-only commentary + HMR news guard:** `⚡ Analyze Now` and `📰 Full AI Brief` generate macro/news/SMC commentary & chart cards on Telegram without writing trade intents to `ai_signal.json` or `pending_signals`, preventing conflicting manual Gemini limit orders vs ML orders. (Scheduled background HMR news calendar sync still writes `blackout_until_unix` cleanly without overwriting active ML execution unless blackout is active.)
- **Live ML Decision Pipeline (`python/src/ml_engine.py`):**
  - On new closed bar (`M15` or `M5`): checks `MarketOpen`, Exness HMR blackout, spread `<= $0.60`, and zero unheld active position (`held` positions remain untouched).
  - Computes 37 features + HMM regime (`TRENDING`/`RANGING`/`VOLATILE`) + XGBoost probabilities `(p_hold, p_buy, p_sell)`.
  - Publishes `EaSignalPayload` (`MARKET_BUY`/`MARKET_SELL`, exact ATR-scaled `sl_price`, `tp_price`, `max_hold_until_unix = now + H * bar_sec`, `confidence`, `reason="ML [M15|M5] p=.. regime=.."`) when `prob >= tuned_threshold` and directional margin `>= 0.12`.
- **MQL5 Execution & Exit Parity (`EA_LearningX_Gold.mq5`):**
  - Defaults **`InpUseSarEngine = false`** (retires blind M1 churn).
  - Adds **`max_hold_until_unix`** support in `EaSignalPayload` / `EA_LearningX_Gold.mq5`: closes unheld ML positions automatically on timeout (`TimeGMT() >= max_hold_until_unix`) to match the `H`-bar triple-barrier simulator!
  - In ML mode (`InpUseAutoBE = false` & `InpUseTrailing = false` defaults, or skipped when `max_hold_until_unix > 0`), MT5 uses pure matched `SL` / `TP` / `H`-bar timeout (while keeping manual **`🛡 Move BE`** button and **`🔒 Keep #<ticket>`** fully functional).
  - Calculates `0.5%` equity risk lot size against exact ATR `sl_price` (`OrderCalcProfit` + margin cap `InpMaxLotSize`), and continues exporting real `closed_deals` every 2s.

### 5.7 Telegram Controls & Visual Cards (`python/src/telegram_bot.py`, `chart_card.py`)
- Replace KillZone-only focus on `/start` Control Panel with **2-Mode Strategy Selector**:
  - **`🎯 M15 Precision`** ↔ **`⚡ M5 Intraday`** (`menu:strategy`), plus **`🤖 ML Models`** (displays OOS Win Rate, PF, trade count, last trained date, and **`🔄 Retrain ML`** button).
- **Unified Chart Card Rendering (`chart_card.py`):** Renders the active mode timeframe (`M15` or `M5` candles) + SMC zones + **HMM Regime & XGBoost Probability pills** on both live ML trade alerts and on-demand `⚡ Analyze Now` / `🧱 SMC Chart` cards.

---

## 6. Verification & Rollout Plan
1. Unit tests for `ml_features.py` (37 features, zero NaNs after warmup, strict no-lookahead), `ml_regime.py`, `ml_trainer.py` (triple-barrier math, validation threshold search, rollback gate), `ml_engine.py`, `EaSignalPayload` `max_hold_until_unix`, and Telegram strategy mode switching.
2. Compile `EA_LearningX_Gold.mq5` (`0 errors, 0 warnings`).
3. Run live training (`python -m python.src.ml_trainer`) against actual Exness MT5 `XAUUSDm` history and verify OOS **Win Rate, Profit Factor (>= 1.05), and Max Drawdown** for both `M15` and `M5` before turning `Auto` ON.

---

## 7. Builder Calls Logged
1. **Validation-Tuned Threshold Search:** Instead of fixed `0.58/0.66/0.70` thresholds on calibrated 3-class probabilities, tune probability thresholds on the 15% validation slice maximizing PF with `>= 20` trades (`ml_trainer.py`).
2. **Matched Timeout Field (`max_hold_until_unix`):** Add `max_hold_until_unix: int = 0` to `EaSignalPayload` / `JsonLite.mqh` read so `EA_LearningX_Gold.mq5` enforces exact `H`-bar timeout parity automatically when `max_hold_until_unix > 0` without breaking manual `🛡 Move BE` or legacy modes.
3. **Read-Only Gemini Chart Integration:** Show `M15`/`M5` + HMM/XGBoost pills inside `⚡ Analyze Now` chart cards as well so user sees ML + LLM + SMC together in one visual card without signal collisions.
