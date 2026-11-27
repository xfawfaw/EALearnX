"""Pure Black & White monochrome TradingView/LuxAlgo candlestick + SMC chart renderer."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .schemas import AnalysisReport, MarketContext, MlPrediction, OhlcvBar, SmcSnapshot

# Pure Black & White institutional palette (matching monochrome LuxAlgo reference)
BG_CANVAS = (0, 0, 0, 255)
BG_PLOT = (0, 0, 0, 255)
BG_HEADER = (6, 6, 6, 255)
BORDER_SUBTLE = (38, 38, 38, 255)
GRID_COLOR = (18, 18, 18, 255)
TEXT_PRIMARY = (245, 245, 245, 255)
TEXT_MUTED = (145, 145, 145, 255)
TEXT_DARK = (8, 8, 8, 255)

CANDLE_WHITE = (242, 242, 242, 255)
CANDLE_BLACK = (0, 0, 0, 255)

OB_FILL = (255, 255, 255, 26)
OB_EDGE = (140, 140, 140, 125)
FVG_FILL = (255, 255, 255, 16)
FVG_EDGE = (100, 100, 100, 95)

COLOR_BSL = (185, 185, 185, 145)
COLOR_SSL = (145, 145, 145, 145)
COLOR_EQ = (105, 105, 105, 120)

COLOR_ENTRY = (255, 255, 255, 235)
COLOR_SL = (135, 135, 135, 210)
COLOR_TP1 = (210, 210, 210, 225)
COLOR_TP = (245, 245, 245, 240)
COLOR_PRICE = (225, 225, 225, 180)
COLOR_EMA21 = (200, 200, 200, 150)
COLOR_EMA50 = (95, 95, 95, 145)

RISK_BOX_FILL = (255, 255, 255, 12)
TP1_BOX_FILL = (255, 255, 255, 28)
TP2_BOX_FILL = (255, 255, 255, 16)


def _load_font(
    size: int, bold: bool = False, mono: bool = False
) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    candidates: list[str]
    if mono:
        candidates = [
            "C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ]
    else:
        candidates = [
            "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
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
    outline: tuple[int, int, int, int] | None = None,
    pad_x: int = 10,
    pad_y: int = 5,
) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    w = tw + pad_x * 2
    h = th + pad_y * 2 + 2
    draw.rounded_rectangle([(x, y), (x + w, y + h)], radius=6, fill=bg, outline=outline, width=1)
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
    outline: tuple[int, int, int, int] | None = None,
) -> None:
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    top = y - th // 2 - 4
    draw.rounded_rectangle(
        [(x + 4, top), (x + tw + 16, top + th + 8)],
        radius=4,
        fill=bg,
        outline=outline,
        width=1,
    )
    draw.text((x + 10, top + 3), label, font=font, fill=fg)


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
        for val in (
            ml_prediction.entry_price,
            ml_prediction.sl_price,
            ml_prediction.tp1_price,
            ml_prediction.tp_price,
        ):
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
    pad = span * 0.10
    return lo - pad, hi + pad


def _ema_series(closes: list[float], period: int) -> list[float]:
    if not closes:
        return []
    k = 2.0 / (period + 1.0)
    out = [closes[0]]
    for c in closes[1:]:
        out.append(c * k + out[-1] * (1.0 - k))
    return out


def _draw_non_overlapping_axis_tags(
    draw: ImageDraw.ImageDraw,
    x: int,
    plot_t: int,
    plot_b: int,
    tags: list[
        tuple[int, str, tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int] | None]
    ],
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    min_gap: int = 23,
) -> None:
    """Draw right-axis price pills with vertical anti-collision so labels never overlap."""
    if not tags:
        return
    ordered = sorted(tags, key=lambda item: item[0])
    placed_ys: list[int] = []
    for raw_y, _, _, _, _ in ordered:
        y = max(plot_t + 12, min(plot_b - 12, raw_y))
        if placed_ys and y - placed_ys[-1] < min_gap:
            y = placed_ys[-1] + min_gap
        placed_ys.append(y)
    if placed_ys and placed_ys[-1] > plot_b - 12:
        overflow = placed_ys[-1] - (plot_b - 12)
        placed_ys = [max(plot_t + 12, y - overflow) for y in placed_ys]
    for y, (_, label, bg, fg, outline) in zip(placed_ys, ordered):
        _draw_axis_tag(draw, x, y, label, bg, fg, font, outline=outline)


def render_analysis_chart(
    market: MarketContext,
    report: AnalysisReport | None = None,
    smc: SmcSnapshot | None = None,
    ml_prediction: MlPrediction | None = None,
) -> bytes | None:
    """Render a monochrome black & white LuxAlgo-style candlestick + SMC/ML PNG chart card."""
    use_m15 = ml_prediction is not None and len(market.m15_bars) >= 10
    chart_tf = "M15" if use_m15 else "H1"
    bars = market.m15_bars[-55:] if use_m15 else market.h1_bars[-60:]
    if len(bars) < 5:
        return None

    smc_data = smc or (report.smc_snapshot if report is not None else None)

    width, height = 1080, 640
    img = Image.new("RGBA", (width, height), BG_CANVAS)
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    ov_draw = ImageDraw.Draw(overlay)

    f_title = _load_font(22, bold=True)
    f_sub = _load_font(14, bold=False)
    f_pill = _load_font(13, bold=True)
    f_mono = _load_font(13, bold=False, mono=True)
    f_mono_bold = _load_font(13, bold=True, mono=True)

    # Monochrome Header Bar
    draw.rectangle([(0, 0), (width, 64)], fill=BG_HEADER)
    draw.line([(0, 64), (width, 64)], fill=BORDER_SUBTLE, width=1)

    last_price = market.bid if market.bid > 0 else bars[-1].c
    draw.text((24, 12), f"{market.symbol}  ·  {chart_tf}", font=f_title, fill=TEXT_PRIMARY)
    draw.text(
        (24, 40),
        f"Price ${last_price:.2f}  ·  Spread ${market.spread_usd:.2f}  ·  EMA 21/50  ·  SMC",
        font=f_sub,
        fill=TEXT_MUTED,
    )

    # Monochrome Header Pills (inverted white primary, bordered dark secondary)
    pill_x = 530
    if report is not None:
        bias = report.decision.bias
        bias_bg = CANDLE_WHITE if bias == "BULLISH" else (34, 34, 34, 255) if bias == "BEARISH" else (18, 18, 18, 255)
        bias_fg = TEXT_DARK if bias == "BULLISH" else TEXT_PRIMARY
        pill_x = _draw_pill(draw, pill_x, 18, f"{bias}", bias_bg, bias_fg, f_pill, outline=OB_EDGE)
        pill_x = _draw_pill(draw, pill_x, 18, f"CONF {report.decision.confidence:.0%}", (18, 18, 18, 255), TEXT_PRIMARY, f_pill, outline=BORDER_SUBTLE)
        pill_x = _draw_pill(draw, pill_x, 18, f"SCORE {report.confluence_score}/5", (18, 18, 18, 255), TEXT_PRIMARY, f_pill, outline=BORDER_SUBTLE)
        if report.session_status is not None:
            _draw_pill(draw, pill_x, 18, report.session_status.session_name.replace("_", " "), (10, 10, 10, 255), TEXT_MUTED, f_pill, outline=BORDER_SUBTLE)
    elif ml_prediction is not None:
        is_action = ml_prediction.order_intent in {"MARKET_BUY", "MARKET_SELL"}
        intent_lbl = (
            "BUY" if ml_prediction.order_intent == "MARKET_BUY"
            else "SELL" if ml_prediction.order_intent == "MARKET_SELL"
            else "WAIT"
        )
        pill_x = _draw_pill(draw, pill_x, 18, f"{ml_prediction.timeframe} B2", (18, 18, 18, 255), TEXT_PRIMARY, f_pill, outline=BORDER_SUBTLE)
        pill_x = _draw_pill(draw, pill_x, 18, f"HMM {ml_prediction.regime}", (18, 18, 18, 255), TEXT_MUTED, f_pill, outline=BORDER_SUBTLE)
        pill_x = _draw_pill(
            draw,
            pill_x,
            18,
            intent_lbl,
            CANDLE_WHITE if is_action else (24, 24, 24, 255),
            TEXT_DARK if is_action else TEXT_MUTED,
            f_pill,
            outline=CANDLE_WHITE if is_action else BORDER_SUBTLE,
        )
        _draw_pill(draw, pill_x, 18, f"L {ml_prediction.prob_long:.0%} / S {ml_prediction.prob_short:.0%}", (10, 10, 10, 255), TEXT_PRIMARY, f_pill, outline=BORDER_SUBTLE)
    elif smc_data is not None:
        bias = smc_data.htf_aligned_bias
        is_bull = bias == "BULLISH"
        pill_x = _draw_pill(
            draw,
            pill_x,
            18,
            f"HTF {bias}",
            CANDLE_WHITE if is_bull else (24, 24, 24, 255),
            TEXT_DARK if is_bull else TEXT_PRIMARY,
            f_pill,
            outline=OB_EDGE,
        )
        _draw_pill(draw, pill_x, 18, f"H1 {smc_data.h1.trend}", (14, 14, 14, 255), TEXT_MUTED, f_pill, outline=BORDER_SUBTLE)

    # Plot Frame
    plot_l, plot_t = 24, 78
    plot_r, plot_b = width - 134, height - 74
    plot_h = plot_b - plot_t

    draw.rounded_rectangle([(plot_l, plot_t), (plot_r, plot_b)], radius=8, fill=BG_PLOT, outline=BORDER_SUBTLE, width=1)

    p_lo, p_hi = _compute_price_bounds(bars, smc_data, report, ml_prediction=ml_prediction)
    p_span = max(p_hi - p_lo, 1e-6)

    def y_of(price: float) -> int:
        ratio = (p_hi - price) / p_span
        y = int(round(plot_t + ratio * plot_h))
        return max(plot_t + 2, min(plot_b - 2, y))

    # Minimal Horizontal Grid + Right Axis Prices
    grid_steps = 6
    for i in range(grid_steps + 1):
        frac = i / grid_steps
        gy = int(round(plot_t + frac * plot_h))
        price_val = p_hi - frac * p_span
        if 0 < i < grid_steps:
            draw.line([(plot_l + 1, gy), (plot_r - 1, gy)], fill=GRID_COLOR, width=1)
        draw.text((plot_r + 10, gy - 8), f"{price_val:.2f}", font=f_mono, fill=TEXT_MUTED)

    # Trade Setup levels
    setup_entry, setup_sl, setup_tp1, setup_tp, setup_label = 0.0, 0.0, 0.0, 0.0, ""
    if report is not None and report.decision.order_intent != "NONE":
        d = report.decision
        setup_entry = d.limit_price if d.limit_price > 0 else last_price
        setup_sl = d.sl_price
        setup_tp = d.tp_price
        setup_label = d.order_intent
    elif ml_prediction is not None and ml_prediction.order_intent != "NONE":
        setup_entry = ml_prediction.entry_price if ml_prediction.entry_price > 0 else last_price
        setup_sl = ml_prediction.sl_price
        setup_tp1 = ml_prediction.tp1_price
        setup_tp = ml_prediction.tp_price
        setup_label = ml_prediction.order_intent

    # Candlesticks Geometry
    n = len(bars)
    proj_slots = 6 if setup_entry > 0 else 2
    inner_l = plot_l + 14
    inner_r = plot_r - 10
    step = (inner_r - inner_l) / max(n + proj_slots, 1)
    candle_w = max(4, int(step * 0.64))
    last_cx = int(round(inner_l + (n - 0.5) * step))

    # Monochrome Risk / Reward Zones on Right Edge
    if setup_entry > 0 and setup_sl > 0:
        box_l = max(plot_l + 4, last_cx - int(step * 2))
        box_r = plot_r - 2
        ye = y_of(setup_entry)
        ys = y_of(setup_sl)
        ov_draw.rectangle([(box_l, min(ye, ys)), (box_r, max(ye, ys))], fill=RISK_BOX_FILL)
        if setup_tp1 > 0:
            yt1 = y_of(setup_tp1)
            ov_draw.rectangle([(box_l, min(ye, yt1)), (box_r, max(ye, yt1))], fill=TP1_BOX_FILL)
            if setup_tp > 0:
                yt2 = y_of(setup_tp)
                ov_draw.rectangle([(box_l, min(yt1, yt2)), (box_r, max(yt1, yt2))], fill=TP2_BOX_FILL)
        elif setup_tp > 0:
            yt2 = y_of(setup_tp)
            ov_draw.rectangle([(box_l, min(ye, yt2)), (box_r, max(ye, yt2))], fill=TP1_BOX_FILL)

    # Monochrome LuxAlgo SMC Blocks, FVGs, Equilibrium & Liquidity Levels
    if smc_data is not None:
        for fvg in smc_data.bullish_fvgs[:2] + smc_data.bearish_fvgs[:2]:
            y1, y2 = y_of(fvg.high), y_of(fvg.low)
            ov_draw.rectangle([(plot_l + 2, y1), (plot_r - 2, y2)], fill=FVG_FILL, outline=FVG_EDGE, width=1)
            ov_draw.text((plot_l + 10, max(plot_t + 3, y1 + 2)), f"FVG {fvg.timeframe}", font=f_mono, fill=(130, 130, 130, 185))

        for ob in smc_data.bullish_obs[:2]:
            y1, y2 = y_of(ob.high), y_of(ob.low)
            ov_draw.rectangle([(plot_l + 2, y1), (plot_r - 2, y2)], fill=OB_FILL, outline=OB_EDGE, width=1)
            ov_draw.text((plot_l + 10, max(plot_t + 3, y1 + 2)), f"+OB {ob.low:.1f}-{ob.high:.1f}", font=f_mono_bold, fill=(215, 215, 215, 220))

        for ob in smc_data.bearish_obs[:2]:
            y1, y2 = y_of(ob.high), y_of(ob.low)
            ov_draw.rectangle([(plot_l + 2, y1), (plot_r - 2, y2)], fill=OB_FILL, outline=OB_EDGE, width=1)
            ov_draw.text((plot_l + 10, max(plot_t + 3, y1 + 2)), f"-OB {ob.low:.1f}-{ob.high:.1f}", font=f_mono_bold, fill=(215, 215, 215, 220))

        eq_price = (smc_data.h1.swing_high + smc_data.h1.swing_low) / 2.0
        if p_lo <= eq_price <= p_hi:
            ey = y_of(eq_price)
            _draw_dashed_hline(ov_draw, plot_l + 2, plot_r - 2, ey, COLOR_EQ, dash=4, gap=6, width=1)
            ov_draw.text((plot_r - 125, max(plot_t + 3, ey - 15)), f"EQ 50% {eq_price:.1f}", font=f_mono, fill=(135, 135, 135, 190))

        for lvl in smc_data.bsl_levels[:2]:
            if p_lo <= lvl <= p_hi:
                ly = y_of(lvl)
                _draw_dashed_hline(ov_draw, plot_l + 2, plot_r - 2, ly, COLOR_BSL, dash=5, gap=5, width=1)
                ov_draw.text((plot_r - 92, max(plot_t + 3, ly - 15)), f"BSL {lvl:.1f}", font=f_mono, fill=(195, 195, 195, 215))

        for lvl in smc_data.ssl_levels[:2]:
            if p_lo <= lvl <= p_hi:
                ly = y_of(lvl)
                _draw_dashed_hline(ov_draw, plot_l + 2, plot_r - 2, ly, COLOR_SSL, dash=5, gap=5, width=1)
                ov_draw.text((plot_r - 92, max(plot_t + 3, ly + 3)), f"SSL {lvl:.1f}", font=f_mono, fill=(165, 165, 165, 215))

    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)

    # Monochrome EMA 21 / 50 Lines
    closes = [b.c for b in bars]
    ema21 = _ema_series(closes, 21)
    ema50 = _ema_series(closes, 50)
    pts21 = [(int(round(inner_l + (i + 0.5) * step)), y_of(v)) for i, v in enumerate(ema21)]
    pts50 = [(int(round(inner_l + (i + 0.5) * step)), y_of(v)) for i, v in enumerate(ema50)]
    if len(pts50) >= 2:
        draw.line(pts50, fill=COLOR_EMA50, width=1)
    if len(pts21) >= 2:
        draw.line(pts21, fill=COLOR_EMA21, width=2)

    # Monochrome LuxAlgo Candlesticks: Bull = Hollow Black + White Outline | Bear = Solid Crisp White
    for i, bar in enumerate(bars):
        cx = int(round(inner_l + (i + 0.5) * step))
        is_bull = bar.c >= bar.o
        yh = y_of(bar.h)
        yl = y_of(bar.l)
        yo = y_of(bar.o)
        yc = y_of(bar.c)
        top_b = min(yo, yc)
        bot_b = max(yo, yc)
        if bot_b - top_b < 2:
            bot_b = top_b + 2
        half = candle_w // 2

        draw.line([(cx, yh), (cx, top_b)], fill=CANDLE_WHITE, width=1)
        draw.line([(cx, bot_b), (cx, yl)], fill=CANDLE_WHITE, width=1)
        if is_bull:
            draw.rectangle([(cx - half, top_b), (cx + half, bot_b)], fill=CANDLE_BLACK, outline=CANDLE_WHITE, width=1)
        else:
            draw.rectangle([(cx - half, top_b), (cx + half, bot_b)], fill=CANDLE_WHITE)

    # Setup Lines & Crisp Non-Overlapping Monochrome Price Tags
    axis_tags: list[
        tuple[int, str, tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int] | None]
    ] = []

    py = y_of(last_price)
    if setup_entry <= 0:
        _draw_dashed_hline(draw, plot_l + 2, plot_r - 2, py, COLOR_PRICE, dash=5, gap=4, width=1)
        axis_tags.append((py, f"{last_price:.2f}", CANDLE_WHITE, TEXT_DARK, None))

    if setup_tp > 0:
        ty = y_of(setup_tp)
        _draw_dashed_hline(draw, plot_l + 2, plot_r - 2, ty, COLOR_TP, dash=10, gap=4, width=2)
        tp_prefix = "TP2" if setup_tp1 > 0 else "TP"
        axis_tags.append((ty, f"{tp_prefix} {setup_tp:.1f}", CANDLE_WHITE, TEXT_DARK, None))

    if setup_tp1 > 0:
        t1y = y_of(setup_tp1)
        _draw_dashed_hline(draw, plot_l + 2, plot_r - 2, t1y, COLOR_TP1, dash=6, gap=4, width=2)
        axis_tags.append((t1y, f"TP1 {setup_tp1:.1f}", (195, 195, 195, 255), TEXT_DARK, None))

    if setup_entry > 0:
        ey = y_of(setup_entry)
        _draw_dashed_hline(draw, plot_l + 2, plot_r - 2, ey, COLOR_ENTRY, dash=10, gap=4, width=2)
        axis_tags.append((ey, f"IN  {setup_entry:.1f}", CANDLE_WHITE, TEXT_DARK, None))

    if setup_sl > 0:
        sy = y_of(setup_sl)
        _draw_dashed_hline(draw, plot_l + 2, plot_r - 2, sy, COLOR_SL, dash=10, gap=4, width=2)
        axis_tags.append((sy, f"SL  {setup_sl:.1f}", (34, 34, 34, 255), TEXT_PRIMARY, (170, 170, 170, 255)))

    _draw_non_overlapping_axis_tags(draw, plot_r, plot_t, plot_b, axis_tags, f_mono_bold)

    # Monochrome Footer Strip
    footer_y = plot_b + 12
    draw.line([(0, footer_y - 4), (width, footer_y - 4)], fill=BORDER_SUBTLE, width=1)

    if ml_prediction is not None and report is None:
        if setup_entry > 0 and setup_sl > 0:
            sl_d = abs(setup_entry - setup_sl)
            tp1_d = abs(setup_tp1 - setup_entry) if setup_tp1 > 0 else 0.0
            tp2_d = abs(setup_tp - setup_entry) if setup_tp > 0 else 0.0
            rr2 = (tp2_d / sl_d) if sl_d > 0 else 0.0
            foot_1 = (
                f"RISK -${sl_d:.2f}   ·   TP1 +${tp1_d:.2f} (50% + BE)   ·   "
                f"TP2 +${tp2_d:.2f} (1:{rr2:.1f}R)   ·   TRAIL ${ml_prediction.trail_atr_usd:.2f}"
            )
        else:
            foot_1 = f"WAIT  ·  {ml_prediction.mode} ({ml_prediction.regime})  ·  TH {ml_prediction.threshold:.0%}"
        draw.text((24, footer_y + 8), foot_1, font=f_sub, fill=TEXT_PRIMARY)
        draw.text(
            (24, footer_y + 32),
            f"EA-LEARNINGX B2 MONOCHROME  ·  PROB L {ml_prediction.prob_long:.1%} / S {ml_prediction.prob_short:.1%}  (TH {ml_prediction.threshold:.0%})",
            font=f_mono,
            fill=TEXT_MUTED,
        )
    elif smc_data is not None:
        pd = smc_data.h1.premium_discount_ratio
        zone_txt = "DISCOUNT" if pd < 0.45 else "PREMIUM" if pd > 0.55 else "EQUILIBRIUM"
        struct_txt = (
            f"H4 {smc_data.h4.trend} ({smc_data.h4.last_event})   ·   "
            f"H1 {smc_data.h1.trend} ({smc_data.h1.last_event})   ·   "
            f"M15 {smc_data.m15.trend} ({smc_data.m15.last_event})   ·   "
            f"P/D {pd:.2f} ({zone_txt})"
        )
        draw.text((24, footer_y + 4), struct_txt, font=f_sub, fill=TEXT_PRIMARY)

        bar_x, bar_y, bar_w = 820, footer_y + 11, 220
        draw.rounded_rectangle([(bar_x, bar_y), (bar_x + bar_w, bar_y + 8)], radius=4, fill=(26, 26, 26, 255), outline=BORDER_SUBTLE, width=1)
        mid_x = bar_x + bar_w // 2
        draw.line([(mid_x, bar_y - 2), (mid_x, bar_y + 10)], fill=TEXT_MUTED, width=1)
        dot_x = bar_x + int(max(0.0, min(1.0, pd)) * bar_w)
        dot_fill = CANDLE_WHITE if pd < 0.5 else (28, 28, 28, 255)
        draw.ellipse([(dot_x - 6, bar_y - 2), (dot_x + 6, bar_y + 10)], fill=dot_fill, outline=CANDLE_WHITE, width=2)

    if report is not None:
        intent_txt = (
            f"SETUP {setup_label} @ {setup_entry:.2f}  |  SL {setup_sl:.2f}  |  TP {setup_tp:.2f}"
            if setup_label
            else f"STATUS: NO TRADE  —  {report.validation_note[:75]}"
        )
        draw.text((24, footer_y + 32), intent_txt, font=f_mono, fill=TEXT_MUTED)
    elif smc_data is not None and ml_prediction is None:
        parts = []
        if smc_data.anchored_buy_limit:
            b = smc_data.anchored_buy_limit
            parts.append(f"BUY_LIMIT @ {b.entry:.2f} (RR {b.rr:.1f})")
        if smc_data.anchored_sell_limit:
            s = smc_data.anchored_sell_limit
            parts.append(f"SELL_LIMIT @ {s.entry:.2f} (RR {s.rr:.1f})")
        draw.text((24, footer_y + 32), "  |  ".join(parts) if parts else "EA-LEARNINGX MONOCHROME SMC MAP", font=f_mono, fill=TEXT_MUTED)

    buf = io.BytesIO()
    # compress_level=1 cuts encode latency ~5x vs optimize=True for crisp Telegram responsiveness
    img.convert("RGB").save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


render_chart_card = render_analysis_chart
