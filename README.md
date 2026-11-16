# EA-LearningX (Exness XAUUSD Dual-Mode ML + HMM/XGBoost + Gemini Brief + Telegram)

Hybrid autonomous trading system for **XAUUSD** on **Exness MetaTrader 5** (Windows PC), monitorable live from **MT5 Android** and controllable via **Telegram**.

## Architecture

1. **MQL5 Execution & Trade Management Layer (`mql5/Experts/EA_LearningX_Gold.mq5`)**
   - Runs natively inside Exness MT5 Desktop (tick-by-tick + 2s timer).
   - Executes ML setups (`MARKET_BUY`, `MARKET_SELL`, `BUY_LIMIT`, `SELL_LIMIT`) with matched Triple-Barrier SL/TP and `max_hold_until_unix` (`H`-bar) timeout closes.
   - Preserves pinned positions via Telegram **`🔒 Keep #<ticket>`** (`HoldKey` lock) and enforces risk-% lot sizing, `MaxSpreadUSD`, `MaxDailyLossPct`, free margin buffer, and **HMR news blackout**.

2. **Dual-Mode Python ML Engine (`python/src/ml_*.py` + `MetaTrader5`)**
   - **`🎯 M15 Precision`**: Closed `M15` bars (~2–6 setups/day), `SL = 1.5 * ATR_M15`, `TP = 2.4 * ATR_M15` (`1:1.6`), `H = 16` bars (`4h` max hold).
   - **`⚡ M5 Intraday`**: `M15` trend alignment + closed `M5` trigger (~6–15 setups/day), `SL = 1.5 * ATR_M5`, `TP = 2.25 * ATR_M5` (`1:1.5`), `H = 24` bars (`2h` max hold).
   - Computes 37+4 causal TA/SMC/HTF features, classifies 3-state volatility/trend regimes via forward-only `GaussianHMM`, scores entries via calibrated `XGBoost` (`FrozenEstimator`), and gates live execution behind spread-adjusted (`$0.30`) walk-forward OOS validation (`PF >= 1.05`, `trades >= 20`) with automatic rollback.

3. **Read-Only Gemini Macro Analyst & Telegram Control (`python/src/analyst.py`, `telegram_bot.py`)**
   - **Gemini** provides read-only macro, ForexFactory/RSS news sentiment, and SMC chart briefings (`⚡ Analyze Now` / `📰 Full AI Brief`) — never overrides ML trade execution.
   - **Telegram Remote Control**: `/status`, `/analyze`, `/mode auto|confirm|pause`, inline **`🎯 M15 Precision` / `⚡ M5 Intraday`** strategy switcher, **`🤖 ML Models` / `🔄 Retrain ML`**, **`🔒 Keep #<ticket>`**, and real closed-deal P&L stats.

4. **Shared IPC Bridge (`%APPDATA%\MetaQuotes\Terminal\Common\Files\EA-LearningX\`)**
   - `market_context.json` & `ea_state.json` (EA -> Python)
   - `ai_signal.json` & `ea_command.json` (Python -> EA)

---

## Quick Setup (Exness Demo First!)

### 1. Configure `.env`
Copy `.env.example` to `.env` and fill in:
- `GEMINI_API_KEY` (Google AI Studio key for read-only macro/SMC briefs)
- `TELEGRAM_BOT_TOKEN` (from `@BotFather`)
- `TELEGRAM_ALLOWED_USER_IDS` (your numeric Telegram ID from `@userinfobot`)

### 2. Deploy & Compile the MQL5 EA
Run in PowerShell:
```powershell
.\scripts\compile_ea.ps1
```
Then in **Exness MT5 Desktop**:
1. Enable **Algo Trading** (`Tools -> Options -> Expert Advisors -> Allow algorithmic trading`).
2. Open an **Exness Demo (`Exness-MT5Trial`)** chart for `XAUUSD` / `XAUUSDm` / `XAUUSDc` (M15 or H1).
3. Refresh **Navigator -> Expert Advisors** and drag **`EA_LearningX_Gold`** onto the chart.

### 3. Train & Walk-Forward Validate Both ML Models
With Exness MT5 open:
```powershell
.\.venv\Scripts\python.exe .\scripts\train_ml.py
```

### 4. Start the Python ML & Telegram Bot
```powershell
.\scripts\run_bot.ps1
```
*(Starts in `DEFAULT_EXECUTION_MODE=confirm` and automatically blocks `Auto` mode unless the active strategy bundle passed OOS `PF >= 1.05` & `trades >= 20`.)*

### 4. Monitor from MT5 Android
Log into the same Exness MT5 account on **MT5 Android** to watch positions, `Buy Limit` orders, `SL`, and `BE` updates in real time, and use Telegram on your phone to trigger `/analyze`, `/status`, `/be`, or `/close_all`.

---

## Disclaimer
Algorithmic XAUUSD trading carries significant financial risk, especially around high-impact news and Exness HMR dynamic leverage changes. Always test thoroughly on `Exness-MT5Trial` demo before considering live capital.
