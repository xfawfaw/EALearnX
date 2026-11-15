# EA-LearningX (Exness XAUUSD Autonomous EA + Gemini 4 Hydrogen + Telegram)

Hybrid autonomous trading system for **XAUUSD** on **Exness MetaTrader 5** (Windows PC), monitorable live from **MT5 Android** and controllable via **Telegram**.

## Architecture

1. **MQL5 Execution & Trade Management Layer (`mql5/Experts/EA_LearningX_Gold.mq5`)**
   - Runs natively inside Exness MT5 Desktop (tick-by-tick + 2s timer).
   - Executes `MARKET_BUY`, `MARKET_SELL`, `BUY_LIMIT`, and `SELL_LIMIT`.
   - Manages **Stop Loss (`SL`)**, **Take Profit (`TP`)**, automatic **Break-Even (`BE`)** at `1.0R` (configurable), **Trailing Stop**, and remote **Close All / Cancel Pending**.
   - Enforces deterministic Exness guards: risk-% lot sizing, `MaxSpreadUSD`, `MaxDailyLossPct`, free margin buffer, and **HMR news blackout**.

2. **Python AI Analyst & Telegram Controller (`python/src/`)**
   - **Gemini 4 Hydrogen (`gemini-4-hydrogen`)** 2-stage pipeline:
     1. Live macro & news sentiment intelligence (Google Search grounding + ForexFactory USD calendar + Gold/Macro RSS feeds).
     2. Structured JSON trade decision using live Exness `H4`/`H1`/`M15` bars & `H1 ATR(14)`.
   - **Telegram Remote Control** (`/status`, `/analyze`, `/mode auto|confirm|pause`, `/be`, `/cancel_pending`, `/close_all`) plus inline **Approve & Execute / Reject** buttons and trade/BE push alerts for Android.

3. **Shared IPC Bridge (`%APPDATA%\MetaQuotes\Terminal\Common\Files\EA-LearningX\`)**
   - `market_context.json` & `ea_state.json` (EA -> Python)
   - `ai_signal.json` & `ea_command.json` (Python -> EA)

---

## Quick Setup (Exness Demo First!)

### 1. Configure `.env`
Copy `.env.example` to `.env` and fill in:
- `GEMINI_API_KEY` (Google AI Studio key)
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

### 3. Start the Python AI & Telegram Bot
```powershell
.\scripts\run_bot.ps1
```
*(Starts in `DEFAULT_EXECUTION_MODE=confirm` so you can verify signals on Telegram before switching to `/mode auto`.)*

### 4. Monitor from MT5 Android
Log into the same Exness MT5 account on **MT5 Android** to watch positions, `Buy Limit` orders, `SL`, and `BE` updates in real time, and use Telegram on your phone to trigger `/analyze`, `/status`, `/be`, or `/close_all`.

---

## Disclaimer
Algorithmic XAUUSD trading carries significant financial risk, especially around high-impact news and Exness HMR dynamic leverage changes. Always test thoroughly on `Exness-MT5Trial` demo before considering live capital.
