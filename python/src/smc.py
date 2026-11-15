"""Deterministic Smart Money Concepts (SMC) analyzer for XAUUSD."""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import (
    AnchoredSetup,
    Bias,
    FvgZone,
    MarketContext,
    OhlcvBar,
    OrderBlockZone,
    SmcSnapshot,
    TimeframeStructure,
)


@dataclass
class SwingPoint:
    """Detected local swing high or swing low."""

    index: int
    price: float
    kind: str  # "HIGH" or "LOW"


def find_swings(bars: list[OhlcvBar], lookback: int = 2) -> list[SwingPoint]:
    """Identify fractal swing highs and lows without future lookahead bias."""
    n = len(bars)
    if n < (2 * lookback + 1):
        return []

    swings: list[SwingPoint] = []
    for i in range(lookback, n - lookback):
        hi = bars[i].h
        lo = bars[i].l
        is_high = all(hi >= bars[j].h for j in range(i - lookback, i)) and all(
            hi > bars[j].h for j in range(i + 1, i + lookback + 1)
        )
        is_low = all(lo <= bars[j].l for j in range(i - lookback, i)) and all(
            lo < bars[j].l for j in range(i + 1, i + lookback + 1)
        )
        if is_high:
            swings.append(SwingPoint(index=i, price=hi, kind="HIGH"))
        if is_low:
            swings.append(SwingPoint(index=i, price=lo, kind="LOW"))

    swings.sort(key=lambda s: s.index)
    return swings


def analyze_timeframe_structure(
    bars: list[OhlcvBar],
    timeframe: str,
    lookback: int = 2,
) -> TimeframeStructure:
    """Compute BOS/CHoCH trend, dealing range, and premium/discount ratio."""
    if not bars:
        return TimeframeStructure(
            timeframe=timeframe,
            trend="RANGING",
            last_event="NONE",
            swing_high=0.0,
            swing_low=0.0,
            premium_discount_ratio=0.5,
        )

    swings = find_swings(bars, lookback=lookback)
    highs = [s for s in swings if s.kind == "HIGH"]
    lows = [s for s in swings if s.kind == "LOW"]

    swing_high = highs[-1].price if highs else max(b.h for b in bars[-20:])
    swing_low = lows[-1].price if lows else min(b.l for b in bars[-20:])
    # Ensure range encapsulates recent price meaningfully
    range_high = max(swing_high, max(b.h for b in bars[-12:]))
    range_low = min(swing_low, min(b.l for b in bars[-12:]))

    current_close = bars[-1].c
    span = range_high - range_low
    pd_ratio = (current_close - range_low) / span if span > 0 else 0.5
    pd_ratio = round(max(0.0, min(1.0, pd_ratio)), 3)

    trend: str = "RANGING"
    last_event = "NONE"

    if highs or lows:
        prev_trend = "RANGING"
        if len(highs) >= 2 and len(lows) >= 2:
            if highs[-1].price > highs[-2].price and lows[-1].price > lows[-2].price:
                prev_trend = "BULLISH"
            elif highs[-1].price < highs[-2].price and lows[-1].price < lows[-2].price:
                prev_trend = "BEARISH"

        broke_up = bool(highs) and any(
            bars[k].c > highs[-1].price for k in range(highs[-1].index + 1, len(bars))
        )
        broke_down = bool(lows) and any(
            bars[k].c < lows[-1].price for k in range(lows[-1].index + 1, len(bars))
        )

        if broke_up and not broke_down:
            last_event = "CHOCH_UP" if prev_trend == "BEARISH" else "BOS_UP"
            trend = "BULLISH"
        elif broke_down and not broke_up:
            last_event = "CHOCH_DOWN" if prev_trend == "BULLISH" else "BOS_DOWN"
            trend = "BEARISH"
        else:
            trend = prev_trend

    return TimeframeStructure(
        timeframe=timeframe,
        trend=trend,  # type: ignore[arg-type]
        last_event=last_event,
        swing_high=round(range_high, 2),
        swing_low=round(range_low, 2),
        premium_discount_ratio=pd_ratio,
    )


def find_unmitigated_fvgs(
    bars: list[OhlcvBar],
    timeframe: str,
    current_price: float,
    min_gap_usd: float = 0.50,
) -> tuple[list[FvgZone], list[FvgZone]]:
    """Detect unmitigated 3-candle Fair Value Gaps (FVG)."""
    bullish: list[FvgZone] = []
    bearish: list[FvgZone] = []
    n = len(bars)
    if n < 4:
        return bullish, bearish

    for i in range(2, n - 1):
        c0, c2 = bars[i - 2], bars[i]
        # Bullish FVG: candle[i-2].high < candle[i].low
        if c2.l - c0.h >= min_gap_usd:
            gap_low, gap_high = round(c0.h, 2), round(c2.l, 2)
            # Check if later candles fully mitigated the gap (touched below gap_low)
            mitigated = any(bars[k].l <= gap_low for k in range(i + 1, n))
            if not mitigated and gap_high <= current_price + 1.0:
                bullish.append(
                    FvgZone(
                        direction="BULLISH",
                        low=gap_low,
                        high=gap_high,
                        mid=round((gap_low + gap_high) / 2.0, 2),
                        timeframe=timeframe,
                    )
                )

        # Bearish FVG: candle[i-2].low > candle[i].high
        if c0.l - c2.h >= min_gap_usd:
            gap_low, gap_high = round(c2.h, 2), round(c0.l, 2)
            mitigated = any(bars[k].h >= gap_high for k in range(i + 1, n))
            if not mitigated and gap_low >= current_price - 1.0:
                bearish.append(
                    FvgZone(
                        direction="BEARISH",
                        low=gap_low,
                        high=gap_high,
                        mid=round((gap_low + gap_high) / 2.0, 2),
                        timeframe=timeframe,
                    )
                )

    # Sort closest to current price first
    bullish.sort(key=lambda z: abs(current_price - z.high))
    bearish.sort(key=lambda z: abs(z.low - current_price))
    return bullish[:4], bearish[:4]


def find_order_blocks(
    bars: list[OhlcvBar],
    timeframe: str,
    current_price: float,
    atr: float,
) -> tuple[list[OrderBlockZone], list[OrderBlockZone]]:
    """Detect unmitigated Bullish & Bearish Order Blocks preceding impulsive displacement."""
    bull_obs: list[OrderBlockZone] = []
    bear_obs: list[OrderBlockZone] = []
    n = len(bars)
    if n < 5:
        return bull_obs, bear_obs

    min_impulse = max(1.0, atr * 0.45)

    for i in range(1, n - 2):
        ob_bar = bars[i]
        imp1 = bars[i + 1]
        imp2 = bars[i + 2]

        # Bullish OB: bearish candle followed by strong bullish displacement closing above OB high
        if ob_bar.c < ob_bar.o:
            move_up = max(imp1.c, imp2.c) - ob_bar.h
            if move_up >= min_impulse and imp1.c > imp1.o:
                mitigated = any(bars[k].c < ob_bar.l for k in range(i + 2, n))
                if not mitigated and ob_bar.h < current_price:
                    bull_obs.append(
                        OrderBlockZone(
                            direction="BULLISH",
                            low=round(ob_bar.l, 2),
                            high=round(ob_bar.h, 2),
                            entry_optimal=round((ob_bar.h + ob_bar.l) * 0.5 + (ob_bar.h - ob_bar.l) * 0.25, 2),
                            timeframe=timeframe,
                        )
                    )

        # Bearish OB: bullish candle followed by strong bearish displacement closing below OB low
        if ob_bar.c > ob_bar.o:
            move_down = ob_bar.l - min(imp1.c, imp2.c)
            if move_down >= min_impulse and imp1.c < imp1.o:
                mitigated = any(bars[k].c > ob_bar.h for k in range(i + 2, n))
                if not mitigated and ob_bar.l > current_price:
                    bear_obs.append(
                        OrderBlockZone(
                            direction="BEARISH",
                            low=round(ob_bar.l, 2),
                            high=round(ob_bar.h, 2),
                            entry_optimal=round((ob_bar.h + ob_bar.l) * 0.5 - (ob_bar.h - ob_bar.l) * 0.25, 2),
                            timeframe=timeframe,
                        )
                    )

    bull_obs.sort(key=lambda z: abs(current_price - z.high))
    bear_obs.sort(key=lambda z: abs(z.low - current_price))
    return bull_obs[:4], bear_obs[:4]


def find_liquidity_pools(bars: list[OhlcvBar], current_price: float) -> tuple[list[float], list[float]]:
    """Return Buy-Side Liquidity (above price) and Sell-Side Liquidity (below price)."""
    swings = find_swings(bars, lookback=2)
    bsl = sorted({round(s.price, 2) for s in swings if s.kind == "HIGH" and s.price > current_price})
    ssl = sorted(
        {round(s.price, 2) for s in swings if s.kind == "LOW" and s.price < current_price},
        reverse=True,
    )
    return bsl[:4], ssl[:4]


def _build_anchored_limit(
    direction: str,
    current_price: float,
    atr: float,
    obs: list[OrderBlockZone],
    fvgs: list[FvgZone],
    targets: list[float],
    min_rr: float = 1.8,
) -> AnchoredSetup | None:
    """Compute a concrete BUY_LIMIT or SELL_LIMIT candidate anchored to nearest OB/FVG."""
    buffer_sl = max(1.5, round(atr * 0.35, 2))
    atr_sl_floor = max(4.0, round(atr * 0.8, 2))

    if direction == "BUY_LIMIT":
        entry = None
        sl = None
        label = ""
        if obs:
            z = obs[0]
            entry = min(z.entry_optimal, round(current_price - 0.5, 2))
            sl = round(min(z.low - buffer_sl, entry - atr_sl_floor), 2)
            label = f"{z.timeframe} Bullish OB [{z.low:.2f}-{z.high:.2f}]"
        elif fvgs:
            z = fvgs[0]
            entry = min(z.mid, round(current_price - 0.5, 2))
            sl = round(min(z.low - buffer_sl, entry - atr_sl_floor), 2)
            label = f"{z.timeframe} Bullish FVG [{z.low:.2f}-{z.high:.2f}]"
        if entry is None or sl is None or entry >= current_price:
            return None
        risk = entry - sl
        if risk <= 0:
            return None
        min_tp = round(entry + risk * min_rr, 2)
        tp = next((t for t in targets if t >= min_tp), min_tp)
        return AnchoredSetup(
            direction="BUY_LIMIT",
            entry=round(entry, 2),
            sl=sl,
            tp=round(tp, 2),
            rr=round((tp - entry) / risk, 2),
            zone_label=label,
        )

    # SELL_LIMIT
    entry = None
    sl = None
    label = ""
    if obs:
        z = obs[0]
        entry = max(z.entry_optimal, round(current_price + 0.5, 2))
        sl = round(max(z.high + buffer_sl, entry + atr_sl_floor), 2)
        label = f"{z.timeframe} Bearish OB [{z.low:.2f}-{z.high:.2f}]"
    elif fvgs:
        z = fvgs[0]
        entry = max(z.mid, round(current_price + 0.5, 2))
        sl = round(max(z.high + buffer_sl, entry + atr_sl_floor), 2)
        label = f"{z.timeframe} Bearish FVG [{z.low:.2f}-{z.high:.2f}]"
    if entry is None or sl is None or entry <= current_price:
        return None
    risk = sl - entry
    if risk <= 0:
        return None
    min_tp = round(entry - risk * min_rr, 2)
    tp = next((t for t in targets if t <= min_tp), min_tp)
    return AnchoredSetup(
        direction="SELL_LIMIT",
        entry=round(entry, 2),
        sl=sl,
        tp=round(tp, 2),
        rr=round((entry - tp) / risk, 2),
        zone_label=label,
    )


def build_smc_snapshot(market: MarketContext) -> SmcSnapshot:
    """Build full multi-timeframe deterministic SMC snapshot from Exness bars."""
    mid_price = round((market.bid + market.ask) / 2.0, 2)
    atr = market.atr_h1_14 if market.atr_h1_14 > 0 else 8.0

    h4 = analyze_timeframe_structure(market.h4_bars, "H4")
    h1 = analyze_timeframe_structure(market.h1_bars, "H1")
    m15 = analyze_timeframe_structure(market.m15_bars, "M15")

    if h4.trend == "BULLISH" and h1.trend in {"BULLISH", "RANGING"}:
        htf_bias: Bias = "BULLISH"
    elif h4.trend == "BEARISH" and h1.trend in {"BEARISH", "RANGING"}:
        htf_bias = "BEARISH"
    elif h4.trend == "RANGING" and h1.trend in {"BULLISH", "BEARISH"}:
        htf_bias = h1.trend  # type: ignore[assignment]
    else:
        htf_bias = "NEUTRAL"

    h1_bull_fvg, h1_bear_fvg = find_unmitigated_fvgs(market.h1_bars, "H1", mid_price)
    m15_bull_fvg, m15_bear_fvg = find_unmitigated_fvgs(market.m15_bars, "M15", mid_price)

    h1_bull_ob, h1_bear_ob = find_order_blocks(market.h1_bars, "H1", mid_price, atr)
    m15_bull_ob, m15_bear_ob = find_order_blocks(market.m15_bars, "M15", mid_price, atr * 0.6)

    bull_obs = sorted(h1_bull_ob + m15_bull_ob, key=lambda z: abs(mid_price - z.high))[:5]
    bear_obs = sorted(h1_bear_ob + m15_bear_ob, key=lambda z: abs(z.low - mid_price))[:5]
    bull_fvgs = sorted(h1_bull_fvg + m15_bull_fvg, key=lambda z: abs(mid_price - z.high))[:5]
    bear_fvgs = sorted(h1_bear_fvg + m15_bear_fvg, key=lambda z: abs(z.low - mid_price))[:5]

    bsl_h1, ssl_h1 = find_liquidity_pools(market.h1_bars, mid_price)
    bsl_h4, ssl_h4 = find_liquidity_pools(market.h4_bars, mid_price)
    bsl = sorted(set(bsl_h1 + bsl_h4))[:5]
    ssl = sorted(set(ssl_h1 + ssl_h4), reverse=True)[:5]

    buy_limit = _build_anchored_limit("BUY_LIMIT", mid_price, atr, bull_obs, bull_fvgs, bsl)
    sell_limit = _build_anchored_limit("SELL_LIMIT", mid_price, atr, bear_obs, bear_fvgs, ssl)

    return SmcSnapshot(
        symbol=market.symbol,
        current_price=mid_price,
        atr_h1_14=round(atr, 2),
        h4=h4,
        h1=h1,
        m15=m15,
        htf_aligned_bias=htf_bias,
        bullish_obs=bull_obs,
        bearish_obs=bear_obs,
        bullish_fvgs=bull_fvgs,
        bearish_fvgs=bear_fvgs,
        bsl_levels=bsl,
        ssl_levels=ssl,
        anchored_buy_limit=buy_limit,
        anchored_sell_limit=sell_limit,
    )


def format_smc_summary(smc: SmcSnapshot) -> str:
    """Format deterministic SMC levels concisely for LLM context & Telegram /smc."""
    lines = [
        f"SMC ({smc.symbol}) @ {smc.current_price:.2f} | H1 ATR=${smc.atr_h1_14:.2f} | HTF Bias={smc.htf_aligned_bias}",
        f"  H4: {smc.h4.trend} ({smc.h4.last_event}) | Range [{smc.h4.swing_low:.2f}-{smc.h4.swing_high:.2f}] | P/D={smc.h4.premium_discount_ratio:.2f}",
        f"  H1: {smc.h1.trend} ({smc.h1.last_event}) | Range [{smc.h1.swing_low:.2f}-{smc.h1.swing_high:.2f}] | P/D={smc.h1.premium_discount_ratio:.2f}",
        f"  M15: {smc.m15.trend} ({smc.m15.last_event}) | P/D={smc.m15.premium_discount_ratio:.2f}",
        f"  Bullish OBs: {', '.join(f'{z.timeframe}[{z.low:.2f}-{z.high:.2f}]' for z in smc.bullish_obs[:3]) or 'None'}",
        f"  Bearish OBs: {', '.join(f'{z.timeframe}[{z.low:.2f}-{z.high:.2f}]' for z in smc.bearish_obs[:3]) or 'None'}",
        f"  Bullish FVGs: {', '.join(f'{z.timeframe}[{z.low:.2f}-{z.high:.2f}]' for z in smc.bullish_fvgs[:3]) or 'None'}",
        f"  Bearish FVGs: {', '.join(f'{z.timeframe}[{z.low:.2f}-{z.high:.2f}]' for z in smc.bearish_fvgs[:3]) or 'None'}",
        f"  Liquidity BSL: {', '.join(f'{x:.2f}' for x in smc.bsl_levels[:3]) or 'None'} | SSL: {', '.join(f'{x:.2f}' for x in smc.ssl_levels[:3]) or 'None'}",
    ]
    if smc.anchored_buy_limit:
        a = smc.anchored_buy_limit
        lines.append(
            f"  Anchored BUY_LIMIT: {a.entry:.2f} | SL={a.sl:.2f} | TP={a.tp:.2f} (R:R 1:{a.rr:.2f}, {a.zone_label})"
        )
    if smc.anchored_sell_limit:
        a = smc.anchored_sell_limit
        lines.append(
            f"  Anchored SELL_LIMIT: {a.entry:.2f} | SL={a.sl:.2f} | TP={a.tp:.2f} (R:R 1:{a.rr:.2f}, {a.zone_label})"
        )
    return "\n".join(lines)
