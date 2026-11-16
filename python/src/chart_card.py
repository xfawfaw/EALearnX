"""Dark-mode XAUUSD candlestick + SMC zone chart renderer for Telegram cards."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .schemas import AnalysisReport, MarketContext, MlPrediction, OhlcvBar, SmcSnapshot

# Dark institutional palette
BG_CANVAS = (11, 15, 23, 255)
BG_PLOT = (17, 24, 36, 255)
BG_HEADER = (15, 21, 32, 255)
BORDER_SUBTLE = (35, 48, 68, 255)
GRID_COLOR = (28, 39, 56, 255)
TEXT_PRIMARY = (232, 238, 248, 255)
TEXT_MUTED = (139, 155, 180, 255)

CANDLE_BULL = (38, 166, 154, 255)
CANDLE_BEAR = (239, 83, 80, 255)

BULL_OB_FILL = (22, 163, 74, 48)
BULL_OB_EDGE = (34, 197, 94, 175)
BEAR_OB_FILL = (220, 38, 38, 48)
BEAR_OB_EDGE = (248, 113, 113, 175)

BULL_FVG_FILL = (6, 182, 212, 34)
BULL_FVG_EDGE = (34, 211, 238, 130)
BEAR_FVG_FILL = (245, 158, 11, 34)
BEAR_FVG_EDGE = (251, 191, 36, 130)

COLOR_BSL = (96, 165, 250, 170)
COLOR_SSL = (192, 132, 252, 170)
COLOR_EQ = (148, 163, 184, 110)

COLOR_ENTRY = (56, 189, 248, 235)
COLOR_SL = (248, 113, 113, 235)
COLOR_TP = (74, 222, 128, 235)
COLOR_PRICE = (250, 204, 21, 215)


def _load_font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    candidates: list[str]
    if mono:
        candidates = [
            "C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ]
    else:
        candidates = [
            "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _draw_dashed_hline(
    draw: ImageDraw.ImageDraw,
    x0: int,
    x1: int,
    y: int,
    color: tuple[int, int, int, int],
    dash: int = 8,
    gap: int = 6,
    width: int = 1,
) -> None:
    x = x0
    while x < x1:
        end_x = min(x + dash, x1)
        draw.line([(x, y), (end_x, y)], fill=color, width=width)
        x += dash + gap


def _draw_pill(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    bg: tuple[int, int, int, int],
    fg: tuple[int, int, int, int],
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    pad_x: int = 10,
    pad_y: int = 5,
) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    w = tw + pad_x * 2
    h = th + pad_y * 2 + 2
    draw.rounded_rectangle([(x, y), (x + w, y + h)], radius=8, fill=bg)
    draw.text((x + pad_x, y + pad_y - 1), text, font=font, fill=fg)
    return x + w + 8


def _draw_axis_tag(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    label: str,
    bg: tuple[int, int, int, int],
    fg: tuple[int, int, int, int],
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> None:
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    top = y - th // 2 - 4
    draw.rounded_rectangle([(x + 3, top), (x + tw + 15, top + th + 8)], radius=5, fill=bg)
    draw.text((x + 9, top + 3), label, font=font, fill=fg)


def _compute_price_bounds(
    bars: list[OhlcvBar],
    smc: SmcSnapshot | None,
    report: AnalysisReport | None,
    ml_prediction: MlPrediction | None = None,
) -> tuple[float, float]:
    prices: list[float] = []
    for b in bars:
        prices.extend([b.h, b.l])

    if smc is not None:
        prices.extend([smc.h1.swing_high, smc.h1.swing_low])
        for ob in smc.bullish_obs[:2] + smc.bearish_obs[:2]:
            prices.extend([ob.low, ob.high])
        for fvg in smc.bullish_fvgs[:2] + smc.bearish_fvgs[:2]:
            prices.extend([fvg.low, fvg.high])
        prices.extend(smc.bsl_levels[:2])
        prices.extend(smc.ssl_levels[:2])

    if report is not None and report.decision.order_intent != "NONE":
        d = report.decision
        for val in (d.limit_price, d.sl_price, d.tp_price):
            if val > 0:
                prices.append(val)
    elif ml_prediction is not None and ml_prediction.order_intent != "NONE":
        for val in (ml_prediction.entry_price, ml_prediction.sl_price, ml_prediction.tp_price):
            if val > 0:
                prices.append(val)

    valid = [p for p in prices if p > 0]
    lo = min(valid)
    hi = max(valid)
    span = hi - lo
    if span < 1.0:
        span = max(lo * 0.005, 5.0)
        lo -= span * 0.5
        hi += span * 0.5
    pad = span * 0.12
    return lo - pad, hi + pad


def render_analysis_chart(
    market: MarketContext,
    report: AnalysisReport | None = None,
    smc: SmcSnapshot | None = None,
    ml_prediction: MlPrediction | None = None,
) -> bytes | None:
    """Render an H1 candlestick + SMC/ML Setup PNG chart card. Returns None if bars are missing."""
    bars = market.h1_bars[-60:]
    if len(bars) < 5:
        return None

    smc_data = smc or (report.smc_snapshot if report is not None else None)

    width, height = 1040, 620
    img = Image.new("RGBA", (width, height), BG_CANVAS)
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    ov_draw = ImageDraw.Draw(overlay)

    f_title = _load_font(22, bold=True)
    f_sub = _load_font(15, bold=False)
    f_pill = _load_font(14, bold=True)
    f_mono = _load_font(13, bold=False, mono=True)
    f_mono_bold = _load_font(13, bold=True, mono=True)

    # Header bar
    draw.rectangle([(0, 0), (width, 66)], fill=BG_HEADER)
    draw.line([(0, 66), (width, 66)], fill=BORDER_SUBTLE, width=1)

    last_price = market.bid if market.bid > 0 else bars[-1].c
    draw.text((24, 14), f"{market.symbol}  •  H1", font=f_title, fill=TEXT_PRIMARY)
    draw.text((24, 42), f"Live/Last ${last_price:.2f}  •  Spread ${market.spread_usd:.2f}  •  ATR14 ${market.atr_h1_14:.2f}", font=f_sub, fill=TEXT_MUTED)

    # Right header status pills
    pill_x = 505
    if report is not None:
        bias = report.decision.bias
        bias_bg = (22, 101, 52, 255) if bias == "BULLISH" else (153, 27, 27, 255) if bias == "BEARISH" else (51, 65, 85, 255)
        pill_x = _draw_pill(draw, pill_x, 19, f"{bias}", bias_bg, TEXT_PRIMARY, f_pill)
        pill_x = _draw_pill(draw, pill_x, 19, f"CONF {report.decision.confidence:.0%}", (30, 41, 59, 255), TEXT_PRIMARY, f_pill)
        pill_x = _draw_pill(draw, pill_x, 19, f"SCORE {report.confluence_score}/5", (30, 58, 138, 255), TEXT_PRIMARY, f_pill)
        if report.session_status is not None:
            _draw_pill(draw, pill_x, 19, report.session_status.session_name.replace("_", " "), (39, 39, 58, 255), TEXT_MUTED, f_pill)
    elif ml_prediction is not None:
        bias_bg = (
            (22, 101, 52, 255)
            if ml_prediction.order_intent == "MARKET_BUY"
            else (153, 27, 27, 255)
            if ml_prediction.order_intent == "MARKET_SELL"
            else (51, 65, 85, 255)
        )
        pill_x = _draw_pill(draw, pill_x, 19, f"{ml_prediction.timeframe} ML", (30, 58, 138, 255), TEXT_PRIMARY, f_pill)
        pill_x = _draw_pill(draw, pill_x, 19, f"HMM {ml_prediction.regime}", (39, 39, 58, 255), TEXT_PRIMARY, f_pill)
        _draw_pill(draw, pill_x, 19, f"{ml_prediction.order_intent}", bias_bg, TEXT_PRIMARY, f_pill)
    elif smc_data is not None:
        bias = smc_data.htf_aligned_bias
        bias_bg = (22, 101, 52, 255) if bias == "BULLISH" else (153, 27, 27, 255) if bias == "BEARISH" else (51, 65, 85, 255)
        pill_x = _draw_pill(draw, pill_x, 19, f"HTF {bias}", bias_bg, TEXT_PRIMARY, f_pill)
        _draw_pill(draw, pill_x, 19, f"H1 {smc_data.h1.trend}", (30, 41, 59, 255), TEXT_PRIMARY, f_pill)

    # Plot geometry
    plot_l, plot_t = 24, 82
    plot_r, plot_b = width - 118, height - 88
    plot_w = plot_r - plot_l
    plot_h = plot_b - plot_t

    draw.rounded_rectangle([(plot_l, plot_t), (plot_r, plot_b)], radius=10, fill=BG_PLOT, outline=BORDER_SUBTLE, width=1)

    p_lo, p_hi = _compute_price_bounds(bars, smc_data, report, ml_prediction=ml_prediction)
    p_span = max(p_hi - p_lo, 1e-6)

    def y_of(price: float) -> int:
        ratio = (p_hi - price) / p_span
        y = int(round(plot_t + ratio * plot_h))
        return max(plot_t + 2, min(plot_b - 2, y))

    # Horizontal grid + right-axis prices
    grid_steps = 6
    for i in range(grid_steps + 1):
        frac = i / grid_steps
        gy = int(round(plot_t + frac * plot_h))
        price_val = p_hi - frac * p_span
        if 0 < i < grid_steps:
            draw.line([(plot_l + 1, gy), (plot_r - 1, gy)], fill=GRID_COLOR, width=1)
        draw.text((plot_r + 10, gy - 8), f"{price_val:.2f}", font=f_mono, fill=TEXT_MUTED)

    # Shaded SMC Order Blocks and FVGs
    if smc_data is not None:
        for fvg in smc_data.bullish_fvgs[:2]:
            y1, y2 = y_of(fvg.high), y_of(fvg.low)
            ov_draw.rectangle([(plot_l + 2, y1), (plot_r - 2, y2)], fill=BULL_FVG_FILL, outline=BULL_FVG_EDGE, width=1)
            ov_draw.text((plot_l + 10, max(plot_t + 4, y1 + 2)), f"Bull FVG {fvg.timeframe}", font=f_mono, fill=(103, 232, 249, 220))

        for fvg in smc_data.bearish_fvgs[:2]:
            y1, y2 = y_of(fvg.high), y_of(fvg.low)
            ov_draw.rectangle([(plot_l + 2, y1), (plot_r - 2, y2)], fill=BEAR_FVG_FILL, outline=BEAR_FVG_EDGE, width=1)
            ov_draw.text((plot_l + 10, max(plot_t + 4, y1 + 2)), f"Bear FVG {fvg.timeframe}", font=f_mono, fill=(253, 224, 71, 220))

        for ob in smc_data.bullish_obs[:2]:
            y1, y2 = y_of(ob.high), y_of(ob.low)
            ov_draw.rectangle([(plot_l + 2, y1), (plot_r - 2, y2)], fill=BULL_OB_FILL, outline=BULL_OB_EDGE, width=1)
            ov_draw.text((plot_l + 10, max(plot_t + 4, y1 + 2)), f"Bull OB ({ob.low:.1f}-{ob.high:.1f})", font=f_mono_bold, fill=(134, 239, 172, 240))

        for ob in smc_data.bearish_obs[:2]:
            y1, y2 = y_of(ob.high), y_of(ob.low)
            ov_draw.rectangle([(plot_l + 2, y1), (plot_r - 2, y2)], fill=BEAR_OB_FILL, outline=BEAR_OB_EDGE, width=1)
            ov_draw.text((plot_l + 10, max(plot_t + 4, y1 + 2)), f"Bear OB ({ob.low:.1f}-{ob.high:.1f})", font=f_mono_bold, fill=(252, 165, 165, 240))

        # Equilibrium line (H1 50%)
        eq_price = (smc_data.h1.swing_high + smc_data.h1.swing_low) / 2.0
        if p_lo <= eq_price <= p_hi:
            ey = y_of(eq_price)
            _draw_dashed_hline(ov_draw, plot_l + 2, plot_r - 2, ey, COLOR_EQ, dash=4, gap=6, width=1)
            ov_draw.text((plot_r - 125, max(plot_t + 4, ey - 16)), f"EQ 50% {eq_price:.1f}", font=f_mono, fill=(148, 163, 184, 200))

        # Liquidity levels BSL / SSL
        for lvl in smc_data.bsl_levels[:2]:
            if p_lo <= lvl <= p_hi:
                ly = y_of(lvl)
                _draw_dashed_hline(ov_draw, plot_l + 2, plot_r - 2, ly, COLOR_BSL, dash=5, gap=5, width=1)
                ov_draw.text((plot_r - 92, max(plot_t + 4, ly - 15)), f"BSL {lvl:.1f}", font=f_mono, fill=(147, 197, 253, 230))

        for lvl in smc_data.ssl_levels[:2]:
            if p_lo <= lvl <= p_hi:
                ly = y_of(lvl)
                _draw_dashed_hline(ov_draw, plot_l + 2, plot_r - 2, ly, COLOR_SSL, dash=5, gap=5, width=1)
                ov_draw.text((plot_r - 92, max(plot_t + 4, ly + 3)), f"SSL {lvl:.1f}", font=f_mono, fill=(216, 180, 254, 230))

    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)

    # Candlesticks
    n = len(bars)
    inner_l = plot_l + 14
    inner_r = plot_r - 14
    step = (inner_r - inner_l) / max(n, 1)
    candle_w = max(3, int(step * 0.62))

    for i, bar in enumerate(bars):
        cx = int(round(inner_l + (i + 0.5) * step))
        color = CANDLE_BULL if bar.c >= bar.o else CANDLE_BEAR
        yh = y_of(bar.h)
        yl = y_of(bar.l)
        yo = y_of(bar.o)
        yc = y_of(bar.c)
        draw.line([(cx, yh), (cx, yl)], fill=color, width=1)
        top_b = min(yo, yc)
        bot_b = max(yo, yc)
        if bot_b - top_b < 2:
            bot_b = top_b + 2
        half = candle_w // 2
        draw.rectangle([(cx - half, top_b), (cx + half, bot_b)], fill=color)

    # Current price line + tag
    py = y_of(last_price)
    _draw_dashed_hline(draw, plot_l + 2, plot_r - 2, py, COLOR_PRICE, dash=6, gap=4, width=1)
    _draw_axis_tag(draw, plot_r, py, f"{last_price:.2f}", (202, 138, 4, 255), (15, 23, 42, 255), f_mono_bold)

    # Trade Setup lines (active report decision or live ML prediction)
    setup_entry, setup_sl, setup_tp, setup_label = 0.0, 0.0, 0.0, ""
    if report is not None and report.decision.order_intent != "NONE":
        d = report.decision
        setup_entry = d.limit_price if d.limit_price > 0 else last_price
        setup_sl = d.sl_price
        setup_tp = d.tp_price
        setup_label = d.order_intent
    elif ml_prediction is not None and ml_prediction.order_intent != "NONE":
        setup_entry = ml_prediction.entry_price if ml_prediction.entry_price > 0 else last_price
        setup_sl = ml_prediction.sl_price
        setup_tp = ml_prediction.tp_price
        setup_label = ml_prediction.order_intent

    # Optional ML Regime + Probabilities HUD inside top-right plot corner
    if ml_prediction is not None:
        ml_badge = f"{ml_prediction.timeframe} | HMM {ml_prediction.regime} | L {ml_prediction.prob_long:.0%} / S {ml_prediction.prob_short:.0%}"
        _draw_pill(draw, plot_l + 12, plot_t + 10, ml_badge, (15, 23, 42, 230), (125, 211, 252, 255), f_mono_bold, pad_x=10, pad_y=4)

    if setup_tp > 0:
        ty = y_of(setup_tp)
        _draw_dashed_hline(draw, plot_l + 2, plot_r - 2, ty, COLOR_TP, dash=10, gap=4, width=2)
        _draw_axis_tag(draw, plot_r, ty, f"TP {setup_tp:.1f}", (21, 128, 61, 255), TEXT_PRIMARY, f_mono_bold)

    if setup_entry > 0:
        ey = y_of(setup_entry)
        _draw_dashed_hline(draw, plot_l + 2, plot_r - 2, ey, COLOR_ENTRY, dash=10, gap=4, width=2)
        _draw_axis_tag(draw, plot_r, ey, f"IN {setup_entry:.1f}", (2, 132, 199, 255), TEXT_PRIMARY, f_mono_bold)

    if setup_sl > 0:
        sy = y_of(setup_sl)
        _draw_dashed_hline(draw, plot_l + 2, plot_r - 2, sy, COLOR_SL, dash=10, gap=4, width=2)
        _draw_axis_tag(draw, plot_r, sy, f"SL {setup_sl:.1f}", (185, 28, 28, 255), TEXT_PRIMARY, f_mono_bold)

    # Footer card
    footer_y = plot_b + 14
    draw.line([(0, footer_y - 4), (width, footer_y - 4)], fill=BORDER_SUBTLE, width=1)

    if smc_data is not None:
        pd = smc_data.h1.premium_discount_ratio
        zone_txt = "DISCOUNT" if pd < 0.45 else "PREMIUM" if pd > 0.55 else "EQUILIBRIUM"
        struct_txt = (
            f"H4 {smc_data.h4.trend} ({smc_data.h4.last_event})   •   "
            f"H1 {smc_data.h1.trend} ({smc_data.h1.last_event})   •   "
            f"M15 {smc_data.m15.trend} ({smc_data.m15.last_event})   •   "
            f"P/D {pd:.2f} ({zone_txt})"
        )
        draw.text((24, footer_y + 4), struct_txt, font=f_sub, fill=TEXT_PRIMARY)

        # Mini P/D slider bar on bottom right
        bar_x, bar_y, bar_w = 780, footer_y + 11, 220
        draw.rounded_rectangle([(bar_x, bar_y), (bar_x + bar_w, bar_y + 8)], radius=4, fill=(30, 41, 59, 255))
        mid_x = bar_x + bar_w // 2
        draw.line([(mid_x, bar_y - 2), (mid_x, bar_y + 10)], fill=TEXT_MUTED, width=1)
        dot_x = bar_x + int(max(0.0, min(1.0, pd)) * bar_w)
        dot_col = CANDLE_BULL if pd < 0.5 else CANDLE_BEAR
        draw.ellipse([(dot_x - 6, bar_y - 2), (dot_x + 6, bar_y + 10)], fill=dot_col, outline=TEXT_PRIMARY, width=1)

    if report is not None:
        intent_txt = (
            f"Setup: {setup_label} @ {setup_entry:.2f} | SL {setup_sl:.2f} | TP {setup_tp:.2f}"
            if setup_label
            else f"Status: NO TRADE  —  {report.validation_note[:75]}"
        )
        draw.text((24, footer_y + 32), intent_txt, font=f_mono, fill=TEXT_MUTED)
    elif smc_data is not None:
        parts = []
        if smc_data.anchored_buy_limit:
            b = smc_data.anchored_buy_limit
            parts.append(f"Anchored BUY_LIMIT @ {b.entry:.2f} (RR {b.rr:.1f})")
        if smc_data.anchored_sell_limit:
            s = smc_data.anchored_sell_limit
            parts.append(f"Anchored SELL_LIMIT @ {s.entry:.2f} (RR {s.rr:.1f})")
        draw.text((24, footer_y + 32), "  |  ".join(parts) if parts else "EA-LearningX SMC Map", font=f_mono, fill=TEXT_MUTED)

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


render_chart_card = render_analysis_chart
