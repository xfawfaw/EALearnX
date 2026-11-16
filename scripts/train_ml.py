"""CLI to fetch live Exness XAUUSDm history from MT5, train & walk-forward validate both ML modes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python.src.config import Settings
from python.src.ml_trainer import train_and_validate_mode
from python.src.mt5_feed import Mt5Feed


def main() -> int:
    parser = argparse.ArgumentParser(description="Train & OOS-validate EA-LearningX dual ML bundles.")
    parser.add_argument("--symbol", default="XAUUSDm", help="MT5 symbol name (default: XAUUSDm)")
    parser.add_argument("--models-dir", type=Path, default=None, help="Output directory for .joblib bundles")
    args = parser.parse_args()

    settings = Settings()
    models_dir = args.models_dir or settings.ml_models_dir
    feed = Mt5Feed(symbol=args.symbol)

    try:
        m15_df = feed.fetch_closed_bars("M15", settings.ml_m15_bars)
        m5_df = feed.fetch_closed_bars("M5", settings.ml_m5_bars)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR fetching MT5 history: {exc}", file=sys.stderr)
        return 1
    finally:
        feed.shutdown()

    print(f"Fetched closed bars: M15={len(m15_df)} | M5={len(m5_df)}")
    print(f"Models output dir : {models_dir}")
    print("-" * 78)

    b_m15 = train_and_validate_mode(
        mode="M15_PRECISION",
        df=m15_df,
        htf_df=None,
        models_dir=models_dir,
        spread_cost_usd=settings.ml_spread_cost_usd,
        min_pf=settings.ml_min_profit_factor,
        min_trades=settings.ml_min_oos_trades,
    )
    b_m5 = train_and_validate_mode(
        mode="M5_INTRADAY",
        df=m5_df,
        htf_df=m15_df,
        models_dir=models_dir,
        spread_cost_usd=settings.ml_spread_cost_usd,
        min_pf=settings.ml_min_profit_factor,
        min_trades=settings.ml_min_oos_trades,
    )

    for b in (b_m15, b_m5):
        m = b.metrics
        status = "VERIFIED" if m.verified_for_live else f"UNVERIFIED ({m.fail_reason})"
        print(
            f"{m.mode:<15} | OOS Trades={m.oos_trades:<4d} | WR={m.oos_win_rate * 100:5.1f}% | "
            f"PF={m.oos_profit_factor:5.2f} | Net=${m.oos_net_pnl_usd:+7.2f} | "
            f"Th={m.chosen_threshold:.2f} | {status}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
