//+------------------------------------------------------------------+
//|                                          EA_LearningX_Gold.mq5   |
//|   Autonomous XAUUSD Execution & Management EA for Exness MT5     |
//|   Reads AI macro/news/sentiment signals from Python via          |
//|   FILE_COMMON/EA-LearningX and manages Market, Buy/Sell Limit,   |
//|   SL, TP, Break-Even (BE), Trailing Stop, and Exness HMR guards. |
//+------------------------------------------------------------------+
#property copyright "EA-LearningX"
#property version   "1.00"

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\OrderInfo.mqh>
#include "..\Include\JsonLite.mqh"

//--- Enums
enum ENUM_LOT_MODE
  {
   LOT_DYNAMIC_RISK = 0, // Dynamic Risk (% Equity / Effective Stop)
   LOT_FIXED        = 1  // Fixed Lot Size
  };

//--- Inputs
input group "=== General & Bridge ==="
input long   InpMagicNumber         = 20261114;      // Magic Number
input string InpBridgeFolder        = "EA-LearningX";// Subfolder inside Common\Files
input int    InpTimerSeconds        = 2;             // Bridge poll & state sync interval (sec)

input group "=== M1 Momentum & Tight Trailing Engine ==="
input bool          InpUseSarEngine       = true;             // Enable fast M1 Tight Trailing Engine
input bool          InpAllowInstantFlip   = false;            // False=Trend-only (no chop whip-flips), True=SAR flip
input ENUM_LOT_MODE InpLotMode            = LOT_DYNAMIC_RISK; // Lot Sizing Mode
input double        InpFixedLot           = 0.05;             // Starter Fixed Lot (used when LOT_FIXED)
input double        InpMinAdxM1           = 22.0;             // Min M1 ADX(14) trend strength to enter (0=off)
input double        InpMinAtrM1_USD       = 0.35;             // Min M1 ATR(14) volatility to avoid dead chop
input double        InpSAR_MinTrailUSD    = 0.65;             // Min Tight Trail / SL distance (USD)
input double        InpSAR_MaxTrailUSD    = 1.10;             // Max Tight Trail / SL distance (USD)
input double        InpSAR_TrailStepUSD   = 0.05;             // Trailing SL ratchet step (USD)
input double        InpSAR_SpreadMult     = 2.6;              // Min trail multiple of spread
input double        InpTightBE_TriggerUSD = 0.45;             // Profit (USD move) to trigger instant BE lock
input double        InpTightBE_LockUSD    = 0.15;             // Locked profit buffer at BE (USD)
input double        InpSAR_HardSL_USD     = 0.0;              // Extra Hard SL if InpAllowInstantFlip=true (0=use tight SL)
input int           InpSAR_FlipCooldownSec= 15;               // Post-close cooldown before next entry (sec)

input group "=== Risk & Exness Safety ==="
input double InpRiskPercent         = 0.5;           // Risk per trade (% of equity)
input double InpMaxLotSize          = 0.50;          // Hard max lot cap
input double InpMaxSpreadUSD        = 0.60;          // Max allowed spread in USD (e.g. 0.60 = 60 cents)
input double InpMaxDailyLossPct     = 20.0;          // Daily max loss halt (% of day start equity)
input int    InpMaxOpenPositions    = 2;             // Max concurrent positions (legacy AI limit mode)
input int    InpMaxPendingOrders    = 2;             // Max concurrent pending orders (legacy AI limit mode)
input bool   InpCloseOnOppositeBias = false;         // Close open trade if AI reverses strongly

input group "=== Break-Even & Trailing (USD) (Legacy Mode) ==="
input bool   InpUseAutoBE           = true;          // Enable automatic Break-Even
input double InpBE_TriggerR         = 0.7;           // Move to BE when profit >= N * initial SL risk
input double InpBE_BufferUSD        = 0.25;          // Lock-in profit buffer at BE (USD)
input bool   InpUseTrailing         = true;          // Trail SL after BE is active
input double InpTrailingDistUSD     = 4.00;          // Trailing distance behind price (USD)
input double InpTrailingStepUSD     = 1.00;          // Min step before modifying SL (USD)

input group "=== Fallback SL/TP (if AI omits) ==="
input double InpFallbackSL_USD      = 12.00;         // Fallback Stop Loss distance (USD)
input double InpFallbackTP_USD      = 24.00;         // Fallback Take Profit distance (USD)

//--- Globals
CTrade        g_trade;
CPositionInfo g_pos;
COrderInfo    g_ord;

string   g_signal_path;
string   g_command_path;
string   g_state_path;
string   g_context_path;

string   g_last_signal_id   = "";
string   g_last_command_id  = "";
string   g_ai_bias          = "NEUTRAL";
string   g_ai_reason        = "M1 Trailing SAR Ready";
bool     g_paused_by_user   = false;
long     g_blackout_until   = 0;

int      g_last_day_of_year = -1;
double   g_daily_start_eq   = 0.0;
bool     g_daily_halt       = false;
datetime g_last_context_bar = 0;

int      g_atr_m1_handle    = INVALID_HANDLE;
int      g_adx_m1_handle    = INVALID_HANDLE;
int      g_ema_fast_handle  = INVALID_HANDLE;
int      g_ema_slow_handle  = INVALID_HANDLE;
int      g_ema_mid_handle   = INVALID_HANDLE;
datetime g_last_sar_action  = 0;
double   g_last_sar_dist    = 0.80;

//+------------------------------------------------------------------+
//| Helper: Normalize price to symbol digits & tick size             |
//+------------------------------------------------------------------+
double NormPrice(const double price)
  {
   double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   if(tick_size > 0.0)
      return NormalizeDouble(MathRound(price / tick_size) * tick_size, digits);
   return NormalizeDouble(price, digits);
  }

//+------------------------------------------------------------------+
//| Helper: Select supported Exness filling mode                     |
//+------------------------------------------------------------------+
void ConfigureFillingMode()
  {
   long filling = SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
   if((filling & SYMBOL_FILLING_IOC) == SYMBOL_FILLING_IOC)
      g_trade.SetTypeFilling(ORDER_FILLING_IOC);
   else if((filling & SYMBOL_FILLING_FOK) == SYMBOL_FILLING_FOK)
      g_trade.SetTypeFilling(ORDER_FILLING_FOK);
   else
      g_trade.SetTypeFilling(ORDER_FILLING_RETURN);
  }

//+------------------------------------------------------------------+
//| Helper: Check/reset daily equity baseline                        |
//+------------------------------------------------------------------+
void UpdateDailyRiskState()
  {
   MqlDateTime dt;
   TimeToStruct(TimeGMT(), dt);
   if(g_last_day_of_year != dt.day_of_year)
     {
      g_last_day_of_year = dt.day_of_year;
      g_daily_start_eq   = AccountInfoDouble(ACCOUNT_BALANCE);
      g_daily_halt       = false;
     }

   if(g_daily_start_eq > 0.0)
     {
      double eq = AccountInfoDouble(ACCOUNT_EQUITY);
      double pnl_pct = ((eq - g_daily_start_eq) / g_daily_start_eq) * 100.0;
      if(pnl_pct <= -InpMaxDailyLossPct)
         g_daily_halt = true;
     }
  }

//+------------------------------------------------------------------+
//| Helper: Count EA open positions and pending orders               |
//+------------------------------------------------------------------+
void CountMyOrders(int &open_pos, int &pending_ord)
  {
   open_pos    = 0;
   pending_ord = 0;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(g_pos.SelectByIndex(i) && g_pos.Symbol() == _Symbol && g_pos.Magic() == InpMagicNumber)
         open_pos++;
     }

   for(int i = OrdersTotal() - 1; i >= 0; i--)
     {
      if(g_ord.SelectByIndex(i) && g_ord.Symbol() == _Symbol && g_ord.Magic() == InpMagicNumber)
         pending_ord++;
     }
  }

//+------------------------------------------------------------------+
//| Helper: Check margin & clamp lot size cleanly                    |
//+------------------------------------------------------------------+
double ClampLotWithMargin(const double raw_lots, const double entry_price, const ENUM_ORDER_TYPE order_type)
  {
   double min_lot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_lot  = MathMin(SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX), InpMaxLotSize);
   double step_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(step_lot <= 0.0) step_lot = 0.01;

   double lots = MathFloor(raw_lots / step_lot) * step_lot;
   lots = MathMax(min_lot, MathMin(max_lot, lots));
   lots = NormalizeDouble(lots, 2);

   // OrderCalcMargin requires ORDER_TYPE_BUY / ORDER_TYPE_SELL (not pending types).
   ENUM_ORDER_TYPE margin_type = (order_type == ORDER_TYPE_SELL || order_type == ORDER_TYPE_SELL_LIMIT || order_type == ORDER_TYPE_SELL_STOP)
                                 ? ORDER_TYPE_SELL
                                 : ORDER_TYPE_BUY;
   double free_margin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   double req_margin  = 0.0;
   while(lots >= min_lot)
     {
      if(OrderCalcMargin(margin_type, _Symbol, lots, entry_price, req_margin))
        {
         if(req_margin <= free_margin * 0.85)
            return lots;
        }
      lots = NormalizeDouble(lots - step_lot, 2);
     }
   return 0.0;
  }

//+------------------------------------------------------------------+
//| Helper: Deterministic lot sizing from SL distance & equity risk  |
//+------------------------------------------------------------------+
double CalculateLotSize(const double entry_price, const double sl_price, const ENUM_ORDER_TYPE order_type)
  {
   if(InpLotMode == LOT_FIXED && InpFixedLot > 0.0)
      return ClampLotWithMargin(InpFixedLot, entry_price, order_type);

   double sl_dist = MathAbs(entry_price - sl_price);
   if(sl_dist <= _Point)
      return 0.0;

   ENUM_ORDER_TYPE calc_type = (order_type == ORDER_TYPE_SELL || order_type == ORDER_TYPE_SELL_LIMIT || order_type == ORDER_TYPE_SELL_STOP)
                               ? ORDER_TYPE_SELL
                               : ORDER_TYPE_BUY;
   double loss_per_lot = 0.0;
   double profit_1_lot = 0.0;
   if(OrderCalcProfit(calc_type, _Symbol, 1.0, entry_price, sl_price, profit_1_lot) && profit_1_lot != 0.0)
     {
      loss_per_lot = MathAbs(profit_1_lot);
     }
   else
     {
      double tick_size  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
      double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE_LOSS);
      if(tick_value <= 0.0)
         tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
      if(tick_size <= 0.0 || tick_value <= 0.0)
         return 0.0;
      loss_per_lot = (sl_dist / tick_size) * tick_value;
     }

   if(loss_per_lot <= 0.0)
      return 0.0;

   double equity     = AccountInfoDouble(ACCOUNT_EQUITY);
   double risk_money = equity * (InpRiskPercent / 100.0);
   double raw_lots   = risk_money / loss_per_lot;
   return ClampLotWithMargin(raw_lots, entry_price, order_type);
  }

//+------------------------------------------------------------------+
//| Action: Close all EA positions                                   |
//+------------------------------------------------------------------+
void CloseAllPositions(const string reason)
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(g_pos.SelectByIndex(i) && g_pos.Symbol() == _Symbol && g_pos.Magic() == InpMagicNumber)
        {
         ulong ticket = g_pos.Ticket();
         if(!g_trade.PositionClose(ticket))
            PrintFormat("[EA-LearningX] Failed to close #%I64u (%s): %s", ticket, reason, g_trade.ResultRetcodeDescription());
         else
            PrintFormat("[EA-LearningX] Closed #%I64u (%s)", ticket, reason);
        }
     }
  }

//+------------------------------------------------------------------+
//| Action: Cancel all EA pending orders                             |
//+------------------------------------------------------------------+
void CancelAllPendingOrders(const string reason)
  {
   for(int i = OrdersTotal() - 1; i >= 0; i--)
     {
      if(g_ord.SelectByIndex(i) && g_ord.Symbol() == _Symbol && g_ord.Magic() == InpMagicNumber)
        {
         ulong ticket = g_ord.Ticket();
         if(!g_trade.OrderDelete(ticket))
            PrintFormat("[EA-LearningX] Failed to delete pending #%I64u (%s): %s", ticket, reason, g_trade.ResultRetcodeDescription());
         else
            PrintFormat("[EA-LearningX] Deleted pending #%I64u (%s)", ticket, reason);
        }
     }
  }

//+------------------------------------------------------------------+
//| Action: Force Break-Even on all profitable positions now         |
//+------------------------------------------------------------------+
void ForceBreakEvenNow()
  {
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double stops_dist = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(!g_pos.SelectByIndex(i) || g_pos.Symbol() != _Symbol || g_pos.Magic() != InpMagicNumber)
         continue;

      double open_p = g_pos.PriceOpen();
      double cur_sl = g_pos.StopLoss();
      double cur_tp = g_pos.TakeProfit();

      if(g_pos.PositionType() == POSITION_TYPE_BUY)
        {
         double target_sl = NormPrice(open_p + InpBE_BufferUSD);
         if(bid - target_sl > stops_dist && (cur_sl == 0.0 || target_sl > cur_sl))
            g_trade.PositionModify(g_pos.Ticket(), target_sl, cur_tp);
        }
      else if(g_pos.PositionType() == POSITION_TYPE_SELL)
        {
         double target_sl = NormPrice(open_p - InpBE_BufferUSD);
         if(target_sl - ask > stops_dist && (cur_sl == 0.0 || target_sl < cur_sl))
            g_trade.PositionModify(g_pos.Ticket(), target_sl, cur_tp);
        }
     }
  }

//+------------------------------------------------------------------+
//| Trade Management: Automatic Break-Even & Trailing Stop           |
//+------------------------------------------------------------------+
void ManageOpenPositions()
  {
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double stops_dist = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(!g_pos.SelectByIndex(i) || g_pos.Symbol() != _Symbol || g_pos.Magic() != InpMagicNumber)
         continue;

      ulong  ticket = g_pos.Ticket();
      double open_p = g_pos.PriceOpen();
      double cur_sl = g_pos.StopLoss();
      double cur_tp = g_pos.TakeProfit();

      if(g_pos.PositionType() == POSITION_TYPE_BUY)
        {
         bool be_applied = (cur_sl >= open_p && cur_sl > 0.0);

         // 1. Auto Break-Even
         if(InpUseAutoBE && !be_applied && cur_sl > 0.0 && cur_sl < open_p)
           {
            double init_risk = open_p - cur_sl;
            if(bid >= open_p + init_risk * InpBE_TriggerR)
              {
               double be_sl = NormPrice(open_p + InpBE_BufferUSD);
               if(bid - be_sl > stops_dist && be_sl > cur_sl)
                 {
                  if(g_trade.PositionModify(ticket, be_sl, cur_tp))
                    {
                     PrintFormat("[EA-LearningX] BE applied on BUY #%I64u -> SL=%.2f", ticket, be_sl);
                     cur_sl = be_sl;
                     be_applied = true;
                    }
                 }
              }
           }

         // 2. Trailing Stop after BE
         if(InpUseTrailing && be_applied)
           {
            double trail_sl = NormPrice(bid - InpTrailingDistUSD);
            if(trail_sl > cur_sl + InpTrailingStepUSD && (bid - trail_sl) > stops_dist)
               g_trade.PositionModify(ticket, trail_sl, cur_tp);
           }
        }
      else if(g_pos.PositionType() == POSITION_TYPE_SELL)
        {
         bool be_applied = (cur_sl > 0.0 && cur_sl <= open_p);

         // 1. Auto Break-Even
         if(InpUseAutoBE && !be_applied && cur_sl > open_p)
           {
            double init_risk = cur_sl - open_p;
            if(ask <= open_p - init_risk * InpBE_TriggerR)
              {
               double be_sl = NormPrice(open_p - InpBE_BufferUSD);
               if(be_sl - ask > stops_dist && (cur_sl == 0.0 || be_sl < cur_sl))
                 {
                  if(g_trade.PositionModify(ticket, be_sl, cur_tp))
                    {
                     PrintFormat("[EA-LearningX] BE applied on SELL #%I64u -> SL=%.2f", ticket, be_sl);
                     cur_sl = be_sl;
                     be_applied = true;
                    }
                 }
              }
           }

         // 2. Trailing Stop after BE
         if(InpUseTrailing && be_applied)
           {
            double trail_sl = NormPrice(ask + InpTrailingDistUSD);
            if((cur_sl == 0.0 || trail_sl < cur_sl - InpTrailingStepUSD) && (trail_sl - ask) > stops_dist)
               g_trade.PositionModify(ticket, trail_sl, cur_tp);
           }
        }
     }
  }

//+------------------------------------------------------------------+
//| Hold / Keep ticket helpers (exclude pinned positions from SAR)   |
//+------------------------------------------------------------------+
string HoldKey(const ulong ticket)
  {
   return "LX_HOLD_" + IntegerToString((long)ticket);
  }

bool IsHeldPosition(const ulong ticket, const string comment = "")
  {
   if(GlobalVariableCheck(HoldKey(ticket)))
      return true;
   if(comment != "" && StringFind(comment, "KEEP") >= 0)
      return true;
   return false;
  }

void SetHeldPosition(const ulong ticket, const bool hold)
  {
   if(ticket == 0)
      return;
   string key = HoldKey(ticket);
   if(hold)
     {
      GlobalVariableSet(key, 1.0);
      PrintFormat("[EA-LearningX] Locked position #%I64u as KEPT (ignored by SAR flips).", ticket);
     }
   else
     {
      GlobalVariableDel(key);
      PrintFormat("[EA-LearningX] Released position #%I64u back to SAR management.", ticket);
     }
  }

void CountSarOrders(int &sar_open, int &sar_pending, int &held_open)
  {
   sar_open    = 0;
   sar_pending = 0;
   held_open   = 0;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(!g_pos.SelectByIndex(i) || g_pos.Symbol() != _Symbol || g_pos.Magic() != InpMagicNumber)
         continue;
      if(IsHeldPosition(g_pos.Ticket(), g_pos.Comment()))
         held_open++;
      else
         sar_open++;
     }

   for(int i = OrdersTotal() - 1; i >= 0; i--)
     {
      if(g_ord.SelectByIndex(i) && g_ord.Symbol() == _Symbol && g_ord.Magic() == InpMagicNumber)
         sar_pending++;
     }
  }

//+------------------------------------------------------------------+
//| Process remote Telegram commands from ea_command.json            |
//+------------------------------------------------------------------+
void ProcessCommands()
  {
   string json;
   if(!ReadCommonUtf8File(g_command_path, json))
      return;

   string cmd_id, action;
   if(!JsonGetString(json, "command_id", cmd_id) || cmd_id == "" || cmd_id == g_last_command_id)
      return;

   JsonGetString(json, "action", action, "");
   long issued_at = 0;
   long cmd_ticket = 0;
   JsonGetLong(json, "issued_at_unix", issued_at, 0);
   JsonGetLong(json, "ticket", cmd_ticket, 0);

   // Ignore stale commands older than 5 minutes
   if(issued_at > 0 && (long)TimeGMT() - issued_at > 300)
     {
      g_last_command_id = cmd_id;
      return;
     }

   g_last_command_id = cmd_id;
   PrintFormat("[EA-LearningX] Executing remote command: %s (id=%s, ticket=%I64d)", action, cmd_id, cmd_ticket);

   if(action == "CLOSE_ALL")
      CloseAllPositions("Telegram CLOSE_ALL");
   else if(action == "CANCEL_PENDING")
      CancelAllPendingOrders("Telegram CANCEL_PENDING");
   else if(action == "MOVE_BE_NOW")
      ForceBreakEvenNow();
   else if(action == "PAUSE")
      g_paused_by_user = true;
   else if(action == "RESUME")
     {
      g_paused_by_user = false;
      if(g_daily_halt)
        {
         g_daily_halt = false;
         g_daily_start_eq = AccountInfoDouble(ACCOUNT_EQUITY);
        }
     }
   else if(action == "HOLD_TICKET" && cmd_ticket > 0)
      SetHeldPosition((ulong)cmd_ticket, true);
   else if(action == "UNHOLD_TICKET" && cmd_ticket > 0)
      SetHeldPosition((ulong)cmd_ticket, false);
  }

//+------------------------------------------------------------------+
//| Process AI signal from ai_signal.json                            |
//+------------------------------------------------------------------+
void ProcessAiSignal()
  {
   string json;
   if(!ReadCommonUtf8File(g_signal_path, json))
      return;

   string sig_id;
   if(!JsonGetString(json, "signal_id", sig_id) || sig_id == "")
      return;

   // Always refresh display metadata & blackout window
   JsonGetString(json, "bias", g_ai_bias, "NEUTRAL");
   JsonGetString(json, "reason", g_ai_reason, "");
   JsonGetLong(json, "blackout_until_unix", g_blackout_until, 0);

   if(sig_id == g_last_signal_id)
      return;

   string intent;
   bool trade_allowed = false;
   long expires_at = 0;
   double limit_price = 0.0, sl_price = 0.0, tp_price = 0.0;

   JsonGetString(json, "order_intent", intent, "NONE");
   JsonGetBool(json, "trade_allowed", trade_allowed, false);
   JsonGetLong(json, "expires_at_unix", expires_at, 0);
   JsonGetDouble(json, "limit_price", limit_price, 0.0);
   JsonGetDouble(json, "sl_price", sl_price, 0.0);
   JsonGetDouble(json, "tp_price", tp_price, 0.0);

   // Mark consumed immediately so a rejected signal is never retried endlessly
   g_last_signal_id = sig_id;

   // Optional opposite-bias close
   if(InpCloseOnOppositeBias && (g_ai_bias == "BULLISH" || g_ai_bias == "BEARISH"))
     {
      for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
         if(!g_pos.SelectByIndex(i) || g_pos.Symbol() != _Symbol || g_pos.Magic() != InpMagicNumber)
            continue;
         if((g_pos.PositionType() == POSITION_TYPE_BUY && g_ai_bias == "BEARISH") ||
            (g_pos.PositionType() == POSITION_TYPE_SELL && g_ai_bias == "BULLISH"))
           {
            g_trade.PositionClose(g_pos.Ticket());
           }
        }
     }

   if(InpUseSarEngine)
      return;

   if(!trade_allowed || intent == "NONE")
      return;

   long now_utc = (long)TimeGMT();
   if(expires_at > 0 && now_utc > expires_at)
     {
      PrintFormat("[EA-LearningX] Signal %s expired before execution.", sig_id);
      return;
     }

   if(g_paused_by_user || g_daily_halt)
     {
      PrintFormat("[EA-LearningX] Signal %s blocked (paused=%s, daily_halt=%s).",
                  sig_id, g_paused_by_user ? "true" : "false", g_daily_halt ? "true" : "false");
      return;
     }

   if(g_blackout_until > now_utc)
     {
      PrintFormat("[EA-LearningX] Signal %s blocked by Exness HMR/News blackout.", sig_id);
      return;
     }

   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) || !MQLInfoInteger(MQL_TRADE_ALLOWED))
     {
      Print("[EA-LearningX] Algo trading disabled in MT5 terminal.");
      return;
     }

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double spread = ask - bid;
   if(spread > InpMaxSpreadUSD)
     {
      PrintFormat("[EA-LearningX] Signal %s skipped: spread %.2f > MaxSpread %.2f", sig_id, spread, InpMaxSpreadUSD);
      return;
     }

   int open_pos = 0, pending_ord = 0;
   CountMyOrders(open_pos, pending_ord);

   double stops_dist = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;
   string comment = "LX_" + StringSubstr(sig_id, 0, 12);

   if(intent == "MARKET_BUY" || intent == "MARKET_SELL")
     {
      if(open_pos >= InpMaxOpenPositions)
        {
         Print("[EA-LearningX] Max open positions reached.");
         return;
        }

      bool is_buy = (intent == "MARKET_BUY");
      double entry = is_buy ? ask : bid;
      if(sl_price <= 0.0)
         sl_price = is_buy ? (entry - InpFallbackSL_USD) : (entry + InpFallbackSL_USD);
      if(tp_price <= 0.0)
         tp_price = is_buy ? (entry + InpFallbackTP_USD) : (entry - InpFallbackTP_USD);

      sl_price = NormPrice(sl_price);
      tp_price = NormPrice(tp_price);

      // Validate SL/TP side
      if(is_buy && (sl_price >= entry - stops_dist || tp_price <= entry + stops_dist))
        {
         PrintFormat("[EA-LearningX] Invalid BUY SL/TP (entry=%.2f sl=%.2f tp=%.2f)", entry, sl_price, tp_price);
         return;
        }
      if(!is_buy && (sl_price <= entry + stops_dist || tp_price >= entry - stops_dist))
        {
         PrintFormat("[EA-LearningX] Invalid SELL SL/TP (entry=%.2f sl=%.2f tp=%.2f)", entry, sl_price, tp_price);
         return;
        }

      ENUM_ORDER_TYPE otype = is_buy ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
      double lots = CalculateLotSize(entry, sl_price, otype);
      if(lots <= 0.0)
        {
         Print("[EA-LearningX] Lot calculation returned 0 (check margin/SL).");
         return;
        }

      bool ok = is_buy ? g_trade.Buy(lots, _Symbol, entry, sl_price, tp_price, comment)
                       : g_trade.Sell(lots, _Symbol, entry, sl_price, tp_price, comment);
      if(!ok)
         PrintFormat("[EA-LearningX] %s failed: %s", intent, g_trade.ResultRetcodeDescription());
      else
         PrintFormat("[EA-LearningX] %s executed: lots=%.2f sl=%.2f tp=%.2f", intent, lots, sl_price, tp_price);
     }
   else if(intent == "BUY_LIMIT" || intent == "SELL_LIMIT")
     {
      if(pending_ord >= InpMaxPendingOrders || (open_pos + pending_ord) >= (InpMaxOpenPositions + InpMaxPendingOrders))
        {
         Print("[EA-LearningX] Max pending orders reached.");
         return;
        }

      bool is_buy_limit = (intent == "BUY_LIMIT");
      limit_price = NormPrice(limit_price);

      if(is_buy_limit && (limit_price <= 0.0 || limit_price >= ask - stops_dist))
        {
         PrintFormat("[EA-LearningX] Invalid BUY_LIMIT price %.2f (ask=%.2f)", limit_price, ask);
         return;
        }
      if(!is_buy_limit && (limit_price <= 0.0 || limit_price <= bid + stops_dist))
        {
         PrintFormat("[EA-LearningX] Invalid SELL_LIMIT price %.2f (bid=%.2f)", limit_price, bid);
         return;
        }

      if(sl_price <= 0.0)
         sl_price = is_buy_limit ? (limit_price - InpFallbackSL_USD) : (limit_price + InpFallbackSL_USD);
      if(tp_price <= 0.0)
         tp_price = is_buy_limit ? (limit_price + InpFallbackTP_USD) : (limit_price - InpFallbackTP_USD);

      sl_price = NormPrice(sl_price);
      tp_price = NormPrice(tp_price);

      if(is_buy_limit && (sl_price >= limit_price - stops_dist || tp_price <= limit_price + stops_dist))
        {
         PrintFormat("[EA-LearningX] Invalid BUY_LIMIT SL/TP (limit=%.2f sl=%.2f tp=%.2f)", limit_price, sl_price, tp_price);
         return;
        }
      if(!is_buy_limit && (sl_price <= limit_price + stops_dist || tp_price >= limit_price - stops_dist))
        {
         PrintFormat("[EA-LearningX] Invalid SELL_LIMIT SL/TP (limit=%.2f sl=%.2f tp=%.2f)", limit_price, sl_price, tp_price);
         return;
        }

      ENUM_ORDER_TYPE otype = is_buy_limit ? ORDER_TYPE_BUY_LIMIT : ORDER_TYPE_SELL_LIMIT;
      double lots = CalculateLotSize(limit_price, sl_price, otype);
      if(lots <= 0.0)
        {
         Print("[EA-LearningX] Lot calculation returned 0 for pending order.");
         return;
        }

      datetime exp_time = (expires_at > now_utc) ? (datetime)(TimeCurrent() + MathMax(60L, expires_at - now_utc)) : 0;
      ENUM_ORDER_TYPE_TIME exp_type = (exp_time > 0) ? ORDER_TIME_SPECIFIED : ORDER_TIME_GTC;

      bool ok = is_buy_limit
         ? g_trade.BuyLimit(lots, limit_price, _Symbol, sl_price, tp_price, exp_type, exp_time, comment)
         : g_trade.SellLimit(lots, limit_price, _Symbol, sl_price, tp_price, exp_type, exp_time, comment);

      if(!ok)
         PrintFormat("[EA-LearningX] %s failed: %s", intent, g_trade.ResultRetcodeDescription());
      else
         PrintFormat("[EA-LearningX] %s placed @ %.2f lots=%.2f sl=%.2f tp=%.2f", intent, limit_price, lots, sl_price, tp_price);
     }
  }

//+------------------------------------------------------------------+
//| Helper: Serialize last 20 closed EA deals for exact PnL sync     |
//+------------------------------------------------------------------+
string BuildClosedDealsJson()
  {
   datetime to_t   = TimeCurrent();
   datetime from_t = to_t - 86400;
   if(!HistorySelect(from_t, to_t))
      return "[]";

   string out = "[";
   int total = HistoryDealsTotal();
   int added = 0;
   for(int i = total - 1; i >= 0 && added < 20; i--)
     {
      ulong deal_t = HistoryDealGetTicket(i);
      if(deal_t == 0)
         continue;
      if(HistoryDealGetString(deal_t, DEAL_SYMBOL) != _Symbol)
         continue;
      if(HistoryDealGetInteger(deal_t, DEAL_MAGIC) != InpMagicNumber)
         continue;
      long entry_mode = HistoryDealGetInteger(deal_t, DEAL_ENTRY);
      if(entry_mode != DEAL_ENTRY_OUT && entry_mode != DEAL_ENTRY_INOUT)
         continue;

      long dtype = HistoryDealGetInteger(deal_t, DEAL_TYPE);
      string orig_side = (dtype == DEAL_TYPE_SELL) ? "BUY" : "SELL";
      long pos_id = HistoryDealGetInteger(deal_t, DEAL_POSITION_ID);
      double net_pnl = HistoryDealGetDouble(deal_t, DEAL_PROFIT)
                     + HistoryDealGetDouble(deal_t, DEAL_SWAP)
                     + HistoryDealGetDouble(deal_t, DEAL_COMMISSION);

      if(added > 0)
         out += ",";
      out += StringFormat(
         "{\"pos_id\":%I64d,\"side\":\"%s\",\"volume\":%.2f,\"close_price\":%.2f,\"net_pnl\":%.2f,\"closed_at\":%I64d}",
         pos_id,
         orig_side,
         HistoryDealGetDouble(deal_t, DEAL_VOLUME),
         HistoryDealGetDouble(deal_t, DEAL_PRICE),
         net_pnl,
         HistoryDealGetInteger(deal_t, DEAL_TIME)
      );
      added++;
     }
   out += "]";
   return out;
  }

//+------------------------------------------------------------------+
//| Export live EA & Account State to ea_state.json                  |
//+------------------------------------------------------------------+
void ExportEaState()
  {
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double eq  = AccountInfoDouble(ACCOUNT_EQUITY);
   double bal = AccountInfoDouble(ACCOUNT_BALANCE);
   double daily_pnl = (g_daily_start_eq > 0.0) ? (eq - g_daily_start_eq) : 0.0;

   string pos_json = "[";
   bool first_pos = true;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(!g_pos.SelectByIndex(i) || g_pos.Symbol() != _Symbol || g_pos.Magic() != InpMagicNumber)
         continue;
      if(!first_pos) pos_json += ",";
      first_pos = false;

      string ptype = (g_pos.PositionType() == POSITION_TYPE_BUY) ? "BUY" : "SELL";
      bool be_done = (g_pos.PositionType() == POSITION_TYPE_BUY)
                     ? (g_pos.StopLoss() >= g_pos.PriceOpen() && g_pos.StopLoss() > 0.0)
                     : (g_pos.StopLoss() > 0.0 && g_pos.StopLoss() <= g_pos.PriceOpen());

      bool is_held = IsHeldPosition(g_pos.Ticket(), g_pos.Comment());
      pos_json += StringFormat(
         "{\"ticket\":%I64u,\"type\":\"%s\",\"volume\":%.2f,\"open_price\":%.2f,\"sl\":%.2f,\"tp\":%.2f,\"profit\":%.2f,\"be_applied\":%s,\"held\":%s}",
         g_pos.Ticket(), ptype, g_pos.Volume(), g_pos.PriceOpen(),
         g_pos.StopLoss(), g_pos.TakeProfit(), g_pos.Profit() + g_pos.Swap(),
         be_done ? "true" : "false",
         is_held ? "true" : "false"
      );
     }
   pos_json += "]";

   string ord_json = "[";
   bool first_ord = true;
   for(int i = OrdersTotal() - 1; i >= 0; i--)
     {
      if(!g_ord.SelectByIndex(i) || g_ord.Symbol() != _Symbol || g_ord.Magic() != InpMagicNumber)
         continue;
      if(!first_ord) ord_json += ",";
      first_ord = false;

      ord_json += StringFormat(
         "{\"ticket\":%I64u,\"type\":\"%s\",\"volume\":%.2f,\"price\":%.2f,\"sl\":%.2f,\"tp\":%.2f}",
         g_ord.Ticket(), EnumToString(g_ord.OrderType()), g_ord.VolumeInitial(),
         g_ord.PriceOpen(), g_ord.StopLoss(), g_ord.TakeProfit()
      );
     }
   ord_json += "]";

   long now_utc = (long)TimeGMT();
   bool hmr_active = (g_blackout_until > now_utc);

   string json = StringFormat(
      "{"
      "\"updated_at_unix\":%I64d,"
      "\"symbol\":\"%s\","
      "\"digits\":%d,"
      "\"bid\":%.2f,"
      "\"ask\":%.2f,"
      "\"spread_usd\":%.2f,"
      "\"balance\":%.2f,"
      "\"equity\":%.2f,"
      "\"free_margin\":%.2f,"
      "\"daily_pnl_usd\":%.2f,"
      "\"paused\":%s,"
      "\"daily_halt\":%s,"
      "\"hmr_blackout\":%s,"
      "\"last_signal_id\":\"%s\","
      "\"last_command_id\":\"%s\","
      "\"positions\":%s,"
      "\"pending_orders\":%s,"
      "\"closed_deals\":%s"
      "}",
      now_utc, JsonEscape(_Symbol), (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS),
      bid, ask, ask - bid, bal, eq, AccountInfoDouble(ACCOUNT_MARGIN_FREE),
      daily_pnl,
      g_paused_by_user ? "true" : "false",
      g_daily_halt ? "true" : "false",
      hmr_active ? "true" : "false",
      JsonEscape(g_last_signal_id),
      JsonEscape(g_last_command_id),
      pos_json, ord_json, BuildClosedDealsJson()
   );

   WriteCommonUtf8File(g_state_path, json);
  }

//+------------------------------------------------------------------+
//| Helper: Serialize closed candles for a timeframe                 |
//+------------------------------------------------------------------+
string BuildBarsJson(const ENUM_TIMEFRAMES tf, const int count)
  {
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int copied = CopyRates(_Symbol, tf, 1, count, rates);
   if(copied <= 0)
      return "[]";

   string out = "[";
   // Output oldest -> newest
   for(int i = copied - 1; i >= 0; i--)
     {
      if(i < copied - 1) out += ",";
      out += StringFormat(
         "{\"t\":%I64d,\"o\":%.2f,\"h\":%.2f,\"l\":%.2f,\"c\":%.2f,\"v\":%I64d}",
         (long)rates[i].time, rates[i].open, rates[i].high, rates[i].low, rates[i].close, rates[i].tick_volume
      );
     }
   out += "]";
   return out;
  }

//+------------------------------------------------------------------+
//| Export H4/H1/M15 market context to market_context.json           |
//+------------------------------------------------------------------+
void ExportMarketContext(const bool force = false)
  {
   datetime m15_bar = iTime(_Symbol, PERIOD_M15, 0);
   if(!force && m15_bar == g_last_context_bar)
      return;
   g_last_context_bar = m15_bar;

   double atr_h1_val = 0.0;
   int atr_handle = iATR(_Symbol, PERIOD_H1, 14);
   if(atr_handle != INVALID_HANDLE)
     {
      double buf[];
      if(CopyBuffer(atr_handle, 0, 1, 1, buf) > 0)
         atr_h1_val = buf[0];
      IndicatorRelease(atr_handle);
     }

   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   string json = StringFormat(
      "{"
      "\"updated_at_unix\":%I64d,"
      "\"symbol\":\"%s\","
      "\"bid\":%.2f,"
      "\"ask\":%.2f,"
      "\"spread_usd\":%.2f,"
      "\"atr_h1_14\":%.2f,"
      "\"h4_bars\":%s,"
      "\"h1_bars\":%s,"
      "\"m15_bars\":%s"
      "}",
      (long)TimeGMT(), JsonEscape(_Symbol), bid, ask, ask - bid, atr_h1_val,
      BuildBarsJson(PERIOD_H4, 40),
      BuildBarsJson(PERIOD_H1, 60),
      BuildBarsJson(PERIOD_M15, 80)
   );

   WriteCommonUtf8File(g_context_path, json);
  }

//+------------------------------------------------------------------+
//| Compute smart spread-safe tight trail distance (USD)             |
//+------------------------------------------------------------------+
double GetSmartSarDistanceUSD()
  {
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double spread = MathMax(0.0, ask - bid);
   double stops_dist = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;

   double atr_m1 = 0.0;
   if(g_atr_m1_handle != INVALID_HANDLE)
     {
      double buf[];
      if(CopyBuffer(g_atr_m1_handle, 0, 1, 1, buf) > 0)
         atr_m1 = buf[0];
     }

   double base_dist = (atr_m1 > 0.0) ? (atr_m1 * 1.4) : InpSAR_MinTrailUSD;
   base_dist = MathMax(InpSAR_MinTrailUSD, MathMin(InpSAR_MaxTrailUSD, base_dist));
   double floor_dist = MathMax(spread * InpSAR_SpreadMult, stops_dist + 0.15);
   double eff = MathMax(base_dist, floor_dist);
   g_last_sar_dist = eff;
   return eff;
  }

//+------------------------------------------------------------------+
//| Anti-Chop M1 Expansion Direction (ADX + ATR + EMA 5/13/26)       |
//+------------------------------------------------------------------+
bool GetInitialSarDirection(ENUM_ORDER_TYPE &out_type)
  {
   double atr_buf[];
   if(g_atr_m1_handle != INVALID_HANDLE && CopyBuffer(g_atr_m1_handle, 0, 1, 1, atr_buf) > 0)
     {
      if(atr_buf[0] < InpMinAtrM1_USD)
         return false;
     }

   double adx_val = 25.0, pdi_val = 0.0, mdi_val = 0.0;
   if(InpMinAdxM1 > 0.0 && g_adx_m1_handle != INVALID_HANDLE)
     {
      double adx_buf[], pdi_buf[], mdi_buf[];
      if(CopyBuffer(g_adx_m1_handle, 0, 0, 1, adx_buf) <= 0 ||
         CopyBuffer(g_adx_m1_handle, 1, 0, 1, pdi_buf) <= 0 ||
         CopyBuffer(g_adx_m1_handle, 2, 0, 1, mdi_buf) <= 0)
         return false;
      adx_val = adx_buf[0];
      pdi_val = pdi_buf[0];
      mdi_val = mdi_buf[0];
      if(adx_val < InpMinAdxM1)
         return false;
     }

   double ema_fast[], ema_slow[], ema_mid[];
   if(g_ema_fast_handle != INVALID_HANDLE && g_ema_slow_handle != INVALID_HANDLE && g_ema_mid_handle != INVALID_HANDLE)
     {
      if(CopyBuffer(g_ema_fast_handle, 0, 0, 1, ema_fast) > 0 &&
         CopyBuffer(g_ema_slow_handle, 0, 0, 1, ema_slow) > 0 &&
         CopyBuffer(g_ema_mid_handle, 0, 0, 1, ema_mid) > 0)
        {
         double c0 = iClose(_Symbol, PERIOD_M1, 0);
         if(ema_fast[0] > ema_slow[0] && ema_slow[0] > ema_mid[0] && c0 > ema_fast[0] && (InpMinAdxM1 <= 0.0 || pdi_val > mdi_val))
           {
            out_type = ORDER_TYPE_BUY;
            return true;
           }
         if(ema_fast[0] < ema_slow[0] && ema_slow[0] < ema_mid[0] && c0 < ema_fast[0] && (InpMinAdxM1 <= 0.0 || mdi_val > pdi_val))
           {
            out_type = ORDER_TYPE_SELL;
            return true;
           }
         return false;
        }
     }
   return false;
  }

//+------------------------------------------------------------------+
//| Delete all EA pending orders except an optional ticket           |
//+------------------------------------------------------------------+
void DeleteStrayPendingExcept(const ulong keep_ticket)
  {
   for(int i = OrdersTotal() - 1; i >= 0; i--)
     {
      if(!g_ord.SelectByIndex(i) || g_ord.Symbol() != _Symbol || g_ord.Magic() != InpMagicNumber)
         continue;
      if(g_ord.Ticket() != keep_ticket)
         g_trade.OrderDelete(g_ord.Ticket());
     }
  }

//+------------------------------------------------------------------+
//| Tight Real MT5 SL + Fast BE Lock + Tick Trailing (Never 0 SL)    |
//+------------------------------------------------------------------+
void TightTrailOpenPositions(const double trail_dist)
  {
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(ask <= 0.0 || bid <= 0.0)
      return;
   double stops_dist = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;
   double min_gap    = MathMax(stops_dist + 0.03, 0.05);

   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(!g_pos.SelectByIndex(i) || g_pos.Symbol() != _Symbol || g_pos.Magic() != InpMagicNumber)
         continue;
      if(IsHeldPosition(g_pos.Ticket(), g_pos.Comment()))
         continue;

      ulong  ticket = g_pos.Ticket();
      double open_p = g_pos.PriceOpen();
      double cur_sl = g_pos.StopLoss();
      double cur_tp = g_pos.TakeProfit();

      if(g_pos.PositionType() == POSITION_TYPE_BUY)
        {
         double desired_sl = NormPrice(bid - trail_dist);
         if((bid - open_p) >= InpTightBE_TriggerUSD)
            desired_sl = MathMax(desired_sl, NormPrice(open_p + InpTightBE_LockUSD));

         if(desired_sl < bid - min_gap)
           {
            if(cur_sl == 0.0 || desired_sl >= cur_sl + InpSAR_TrailStepUSD)
               g_trade.PositionModify(ticket, desired_sl, cur_tp);
           }
        }
      else if(g_pos.PositionType() == POSITION_TYPE_SELL)
        {
         double desired_sl = NormPrice(ask + trail_dist);
         if((open_p - ask) >= InpTightBE_TriggerUSD)
            desired_sl = MathMin(desired_sl, NormPrice(open_p - InpTightBE_LockUSD));

         if(desired_sl > ask + min_gap)
           {
            if(cur_sl == 0.0 || desired_sl <= cur_sl - InpSAR_TrailStepUSD)
               g_trade.PositionModify(ticket, desired_sl, cur_tp);
           }
        }
     }
  }

//+------------------------------------------------------------------+
//| M1 Momentum + Tight Trailing Engine (Safe Real SL + Optional SAR)|
//+------------------------------------------------------------------+
void ManageSarEngine()
  {
   if(!InpUseSarEngine)
      return;

   double trail_dist = GetSmartSarDistanceUSD();
   // Always enforce & ratchet real SL on open positions even during Pause/Halt/Blackout
   TightTrailOpenPositions(trail_dist);

   long now_utc = (long)TimeGMT();
   bool trading_blocked = g_paused_by_user || g_daily_halt || (g_blackout_until > now_utc) ||
                          !TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) || !MQLInfoInteger(MQL_TRADE_ALLOWED);

   int open_pos = 0, pending_ord = 0, held_pos = 0;
   CountSarOrders(open_pos, pending_ord, held_pos);

   if(trading_blocked)
     {
      if(pending_ord > 0)
         CancelAllPendingOrders("Paused/Blackout Guard (Real SL preserved)");
      return;
     }

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(ask <= 0.0 || bid <= 0.0)
      return;
   double spread = ask - bid;
   double stops_dist = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;

   // State 2: Hedge flip cleanup when 2+ unheld positions exist -> keep newest, close older
   if(open_pos > 1)
     {
      ulong newest_ticket = 0;
      ulong newest_time   = 0;
      for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
         if(!g_pos.SelectByIndex(i) || g_pos.Symbol() != _Symbol || g_pos.Magic() != InpMagicNumber)
            continue;
         if(IsHeldPosition(g_pos.Ticket(), g_pos.Comment()))
            continue;
         ulong t_msc = g_pos.TimeMsc();
         if(t_msc > newest_time || (t_msc == newest_time && g_pos.Ticket() > newest_ticket))
           {
            newest_time   = t_msc;
            newest_ticket = g_pos.Ticket();
           }
        }

      for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
         if(!g_pos.SelectByIndex(i) || g_pos.Symbol() != _Symbol || g_pos.Magic() != InpMagicNumber)
            continue;
         if(IsHeldPosition(g_pos.Ticket(), g_pos.Comment()))
            continue;
         if(g_pos.Ticket() != newest_ticket)
           {
            if(!g_trade.PositionClose(g_pos.Ticket()))
               PrintFormat("[EA-LearningX] Flip close failed #%I64u: %s", g_pos.Ticket(), g_trade.ResultRetcodeDescription());
            else
               PrintFormat("[EA-LearningX] Flip closed old #%I64u, active #%I64u", g_pos.Ticket(), newest_ticket);
           }
        }

      if(pending_ord > 0)
         DeleteStrayPendingExcept(0);
      g_last_sar_action = TimeCurrent();
      return;
     }

   // State 3 & 4: 1 active unheld position
   if(open_pos == 1)
     {
      g_last_sar_action = TimeCurrent();

      if(!InpAllowInstantFlip)
        {
         // Trend-only tight trailing mode: no blind instant flip order needed; TightTrailOpenPositions manages SL.
         if(pending_ord > 0)
            DeleteStrayPendingExcept(0);
         return;
        }

      ulong  pos_ticket = 0;
      ENUM_POSITION_TYPE pos_type = POSITION_TYPE_BUY;
      double pos_vol = 0.0;
      for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
         if(g_pos.SelectByIndex(i) && g_pos.Symbol() == _Symbol && g_pos.Magic() == InpMagicNumber)
           {
            if(IsHeldPosition(g_pos.Ticket(), g_pos.Comment()))
               continue;
            pos_ticket = g_pos.Ticket();
            pos_type   = g_pos.PositionType();
            pos_vol    = g_pos.Volume();
            break;
           }
        }
      if(pos_ticket == 0)
         return;

      ENUM_ORDER_TYPE desired_stop_type = (pos_type == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL_STOP : ORDER_TYPE_BUY_STOP;

      ulong  sar_ord_ticket = 0;
      double sar_ord_price  = 0.0;
      for(int i = OrdersTotal() - 1; i >= 0; i--)
        {
         if(!g_ord.SelectByIndex(i) || g_ord.Symbol() != _Symbol || g_ord.Magic() != InpMagicNumber)
            continue;
         if(g_ord.OrderType() == desired_stop_type && sar_ord_ticket == 0)
           {
            sar_ord_ticket = g_ord.Ticket();
            sar_ord_price  = g_ord.PriceOpen();
           }
         else
           {
            g_trade.OrderDelete(g_ord.Ticket());
           }
        }

      double target_stop = (pos_type == POSITION_TYPE_BUY)
                           ? NormPrice(bid - trail_dist)
                           : NormPrice(ask + trail_dist);
      double min_gap = MathMax(stops_dist + 0.03, 0.05);
      double opp_sl  = (desired_stop_type == ORDER_TYPE_SELL_STOP)
                       ? NormPrice(target_stop + ((InpSAR_HardSL_USD > 0.0) ? InpSAR_HardSL_USD : trail_dist))
                       : NormPrice(target_stop - ((InpSAR_HardSL_USD > 0.0) ? InpSAR_HardSL_USD : trail_dist));

      if(sar_ord_ticket == 0)
        {
         if(desired_stop_type == ORDER_TYPE_SELL_STOP && target_stop < bid - min_gap)
            g_trade.SellStop(pos_vol, target_stop, _Symbol, opp_sl, 0.0, ORDER_TIME_GTC, 0, "LX_SAR");
         else if(desired_stop_type == ORDER_TYPE_BUY_STOP && target_stop > ask + min_gap)
            g_trade.BuyStop(pos_vol, target_stop, _Symbol, opp_sl, 0.0, ORDER_TIME_GTC, 0, "LX_SAR");
         return;
        }

      if(desired_stop_type == ORDER_TYPE_SELL_STOP)
        {
         if(target_stop >= sar_ord_price + InpSAR_TrailStepUSD && target_stop < bid - min_gap)
            g_trade.OrderModify(sar_ord_ticket, target_stop, opp_sl, 0.0, ORDER_TIME_GTC, 0);
        }
      else
        {
         if(target_stop <= sar_ord_price - InpSAR_TrailStepUSD && target_stop > ask + min_gap)
            g_trade.OrderModify(sar_ord_ticket, target_stop, opp_sl, 0.0, ORDER_TIME_GTC, 0);
        }
      return;
     }

   // State 1: Flat (0 unheld open) -> clean stale orders, enforce spread & cooldown, check M1 ADX/EMA trend
   if(pending_ord > 0)
     {
      DeleteStrayPendingExcept(0);
      return;
     }

   if(spread > InpMaxSpreadUSD)
      return;
   if(TimeCurrent() - g_last_sar_action < InpSAR_FlipCooldownSec)
      return;

   ENUM_ORDER_TYPE init_type = ORDER_TYPE_BUY;
   if(!GetInitialSarDirection(init_type))
      return;

   double entry = (init_type == ORDER_TYPE_BUY) ? ask : bid;
   // Real protective SL attached right on entry (offset slightly wider if instant-flip SAR stop also placed)
   double sl_offset = InpAllowInstantFlip ? (trail_dist + 0.15) : trail_dist;
   double real_sl   = NormPrice((init_type == ORDER_TYPE_BUY) ? (bid - sl_offset) : (ask + sl_offset));
   double eff_risk_sl = NormPrice((init_type == ORDER_TYPE_BUY)
                                  ? (entry - MathMax(trail_dist * 2.0, 2.0))
                                  : (entry + MathMax(trail_dist * 2.0, 2.0)));
   double lots = CalculateLotSize(entry, eff_risk_sl, init_type);
   if(lots <= 0.0)
     {
      Print("[EA-LearningX] Momentum lot calculation returned 0.");
      return;
     }

   bool ok = (init_type == ORDER_TYPE_BUY)
             ? g_trade.Buy(lots, _Symbol, entry, real_sl, 0.0, "LX_MOM")
             : g_trade.Sell(lots, _Symbol, entry, real_sl, 0.0, "LX_MOM");
   if(ok)
     {
      g_last_sar_action = TimeCurrent();
      PrintFormat("[EA-LearningX] Momentum %s @ %.2f lots=%.2f sl=%.2f (trail=$%.2f)",
                  (init_type == ORDER_TYPE_BUY ? "BUY" : "SELL"), entry, lots, real_sl, trail_dist);
     }
   else
     {
      PrintFormat("[EA-LearningX] Momentum entry failed: %s", g_trade.ResultRetcodeDescription());
     }
  }

//+------------------------------------------------------------------+
//| Render concise chart status                                      |
//+------------------------------------------------------------------+
void UpdateChartComment()
  {
   int sar_open = 0, sar_pending = 0, held_pos = 0;
   CountSarOrders(sar_open, sar_pending, held_pos);
   double eq  = AccountInfoDouble(ACCOUNT_EQUITY);
   double pnl = (g_daily_start_eq > 0.0) ? (eq - g_daily_start_eq) : 0.0;
   long now_utc = (long)TimeGMT();
   bool hmr = (g_blackout_until > now_utc);

   string status = g_daily_halt ? "HALTED (DAILY LOSS)"
                 : (g_paused_by_user ? "PAUSED"
                 : (hmr ? "HMR/NEWS BLACKOUT" : "ACTIVE (M1 SAR)"));

   Comment(StringFormat(
      "=== EA-LearningX (%s) ===\n"
      "Status: %s | SAR Trail: $%.2f | AI Bias: %s\n"
      "Equity: %.2f | Daily PnL: %+.2f | Spread: $%.2f\n"
      "SAR Open: %d | Kept (Hold): %d | SAR Stops: %d\n"
      "Note: %s",
      _Symbol, status, g_last_sar_dist, g_ai_bias, eq, pnl,
      SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID),
      sar_open, held_pos, sar_pending, g_ai_reason
   ));
  }

//+------------------------------------------------------------------+
//| Initialization                                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   FolderCreate(InpBridgeFolder, FILE_COMMON);
   g_signal_path  = InpBridgeFolder + "\\ai_signal.json";
   g_command_path = InpBridgeFolder + "\\ea_command.json";
   g_state_path   = InpBridgeFolder + "\\ea_state.json";
   g_context_path = InpBridgeFolder + "\\market_context.json";

   g_trade.SetExpertMagicNumber(InpMagicNumber);
   g_trade.SetDeviationInPoints(30);
   ConfigureFillingMode();

   g_atr_m1_handle   = iATR(_Symbol, PERIOD_M1, 14);
   g_adx_m1_handle   = iADX(_Symbol, PERIOD_M1, 14);
   g_ema_fast_handle = iMA(_Symbol, PERIOD_M1, 5, 0, MODE_EMA, PRICE_CLOSE);
   g_ema_slow_handle = iMA(_Symbol, PERIOD_M1, 13, 0, MODE_EMA, PRICE_CLOSE);
   g_ema_mid_handle  = iMA(_Symbol, PERIOD_M1, 26, 0, MODE_EMA, PRICE_CLOSE);

   UpdateDailyRiskState();
   ExportMarketContext(true);
   ExportEaState();
   UpdateChartComment();

   EventSetTimer(MathMax(1, InpTimerSeconds));
   PrintFormat("[EA-LearningX] Initialized on %s (Magic=%I64d, TightMom=%s, InstantFlip=%s)",
               _Symbol, InpMagicNumber, InpUseSarEngine ? "ON" : "OFF", InpAllowInstantFlip ? "ON" : "OFF");
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Deinitialization                                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
   if(g_atr_m1_handle != INVALID_HANDLE)   IndicatorRelease(g_atr_m1_handle);
   if(g_adx_m1_handle != INVALID_HANDLE)   IndicatorRelease(g_adx_m1_handle);
   if(g_ema_fast_handle != INVALID_HANDLE) IndicatorRelease(g_ema_fast_handle);
   if(g_ema_slow_handle != INVALID_HANDLE) IndicatorRelease(g_ema_slow_handle);
   if(g_ema_mid_handle != INVALID_HANDLE)  IndicatorRelease(g_ema_mid_handle);
   Comment("");
  }

//+------------------------------------------------------------------+
//| Tick handler: fast SAR trailing + immediate Telegram reaction    |
//+------------------------------------------------------------------+
void OnTick()
  {
   UpdateDailyRiskState();
   ProcessCommands();
   ProcessAiSignal();
   if(InpUseSarEngine)
      ManageSarEngine();
   else
      ManageOpenPositions();
   ExportMarketContext(false);
  }

//+------------------------------------------------------------------+
//| Timer handler: keeps bridge & chart responsive                   |
//+------------------------------------------------------------------+
void OnTimer()
  {
   UpdateDailyRiskState();
   ProcessCommands();
   ProcessAiSignal();
   if(InpUseSarEngine)
      ManageSarEngine();
   else
      ManageOpenPositions();
   ExportMarketContext(false);
   ExportEaState();
   UpdateChartComment();
  }
//+------------------------------------------------------------------+
