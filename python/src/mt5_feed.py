"""Live MetaTrader 5 historical and closed-bar feed.

Provides deterministic closed-bar fetching starting at MT5 shift pos=1,
ensuring currently-forming bars are never leaked to causal inference pipelines.
Supports test-double injection for dependency injection and deterministic testing.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import pandas as pd

from python.src.ml_features import bars_to_dataframe
from python.src.schemas import OhlcvBar

logger = logging.getLogger("mt5_feed")

TimeframeStr = Literal["M5", "M15", "H1"]


class Mt5Feed:
    """Historical and closed-bar feed communicating with MetaTrader5 terminal."""

    TIMEFRAME_MAP: dict[str, int] = {
        "M5": 5,
        "M15": 15,
        "H1": 16385,
    }

    def __init__(self, symbol: str = "XAUUSDm", mt5_module: Any | None = None) -> None:
        self.symbol = symbol
        self._mt5 = mt5_module
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure MetaTrader5 connection is initialized and valid."""
        if self._mt5 is None:
            try:
                import MetaTrader5 as mt5_real

                if mt5_real is None:
                    raise ImportError("MetaTrader5 module is None")
                self._mt5 = mt5_real
            except (ImportError, Exception) as exc:
                raise RuntimeError(
                    f"MetaTrader5 package is not available or failed to import: {exc}"
                ) from exc

        if not self._initialized:
            init_ok = self._mt5.initialize()
            if not init_ok:
                err = getattr(self._mt5, "last_error", lambda: "unknown")()
                raise RuntimeError(f"MetaTrader5 initialize() failed: {err}")
            self._initialized = True

    def shutdown(self) -> None:
        """Shutdown MT5 connection if active."""
        if self._initialized and self._mt5 is not None:
            if hasattr(self._mt5, "shutdown"):
                try:
                    self._mt5.shutdown()
                except Exception as exc:
                    logger.warning("Error during MT5 shutdown: %s", exc)
            self._initialized = False

    def _resolve_timeframe(self, timeframe: TimeframeStr) -> int:
        """Resolve timeframe string ('M5', 'M15', 'H1') to MT5 timeframe integer constant."""
        tf_attr = f"TIMEFRAME_{timeframe}"
        if self._mt5 is not None and hasattr(self._mt5, tf_attr):
            return int(getattr(self._mt5, tf_attr))
        if timeframe in self.TIMEFRAME_MAP:
            return self.TIMEFRAME_MAP[timeframe]
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. Allowed timeframes: {list(self.TIMEFRAME_MAP.keys())}"
        )

    def fetch_closed_bars(self, timeframe: TimeframeStr, count: int) -> pd.DataFrame:
        """Fetch count closed bars starting strictly at start_pos=1.

        MT5 shift pos=0 is the currently forming unclosed candle.
        Requesting start_pos=1 guarantees only closed candles are retrieved.

        Returns DataFrame with schema: ['t', 'o', 'h', 'l', 'c', 'v']
        sorted ascending by 't' with duplicate timestamps dropped.
        """
        if count <= 0:
            raise ValueError(f"count must be a positive integer, got {count}")

        self._ensure_initialized()

        tf_val = self._resolve_timeframe(timeframe)

        if not self._mt5.symbol_select(self.symbol, True):
            err = getattr(self._mt5, "last_error", lambda: "unknown")()
            raise RuntimeError(
                f"Failed to select symbol '{self.symbol}' in MetaTrader5: {err}"
            )

        rates = self._mt5.copy_rates_from_pos(self.symbol, tf_val, 1, count)
        if rates is None or len(rates) == 0:
            err = getattr(self._mt5, "last_error", lambda: "unknown")()
            raise RuntimeError(
                f"copy_rates_from_pos('{self.symbol}', {timeframe}, 1, {count}) returned None or empty: {err}"
            )

        bars = self._parse_rates_to_ohlcv(rates)
        df = bars_to_dataframe(bars)
        if df.empty:
            raise RuntimeError(
                f"No valid bars parsed for '{self.symbol}' ({timeframe})"
            )

        # Drop duplicate timestamps, keeping the last occurrence, and sort ascending by 't'
        df = df.drop_duplicates(subset=["t"], keep="last").sort_values("t").reset_index(drop=True)

        if not df["t"].is_monotonic_increasing:
            raise RuntimeError(
                f"Bar timestamps for '{self.symbol}' ({timeframe}) are not monotonic increasing after sorting"
            )

        return df

    def fetch_tick_spread(self) -> tuple[float, float, float]:
        """Fetch current (bid, ask, spread_usd) from MT5.

        Returns (0.0, 0.0, 0.0) safely on any error, missing tick, or disconnected terminal.
        """
        try:
            self._ensure_initialized()
            if not self._mt5.symbol_select(self.symbol, True):
                return (0.0, 0.0, 0.0)

            tick = self._mt5.symbol_info_tick(self.symbol)
            if not tick:
                return (0.0, 0.0, 0.0)

            bid = float(getattr(tick, "bid", 0.0))
            ask = float(getattr(tick, "ask", 0.0))
            if bid <= 0.0 or ask <= 0.0:
                return (0.0, 0.0, 0.0)

            spread = ask - bid
            return (bid, ask, spread)
        except Exception as exc:
            logger.debug("Safe tick/spread fetch failure: %s", exc)
            return (0.0, 0.0, 0.0)

    @staticmethod
    def _parse_rates_to_ohlcv(rates: Any) -> list[OhlcvBar]:
        """Convert various rate formats (numpy structured array, records, dicts) to list[OhlcvBar]."""
        bars: list[OhlcvBar] = []

        if isinstance(rates, pd.DataFrame):
            records = rates.to_dict(orient="records")
            return Mt5Feed._parse_rates_to_ohlcv(records)

        for row in rates:
            if isinstance(row, OhlcvBar):
                bars.append(row)
            elif isinstance(row, dict):
                bars.append(
                    OhlcvBar(
                        t=int(row.get("t", row.get("time", 0))),
                        o=float(row.get("o", row.get("open", 0.0))),
                        h=float(row.get("h", row.get("high", 0.0))),
                        l=float(row.get("l", row.get("low", 0.0))),
                        c=float(row.get("c", row.get("close", 0.0))),
                        v=int(row.get("v", row.get("tick_volume", row.get("volume", 0)))),
                    )
                )
            elif hasattr(row, "dtype") and getattr(row.dtype, "names", None):
                names = row.dtype.names
                t = int(row["time"]) if "time" in names else int(row["t"])
                o = float(row["open"]) if "open" in names else float(row["o"])
                h = float(row["high"]) if "high" in names else float(row["h"])
                l = float(row["low"]) if "low" in names else float(row["l"])
                c = float(row["close"]) if "close" in names else float(row["c"])
                v = int(
                    row["tick_volume"]
                    if "tick_volume" in names
                    else (row["volume"] if "volume" in names else row["v"])
                )
                bars.append(OhlcvBar(t=t, o=o, h=h, l=l, c=c, v=v))
            else:
                t = getattr(row, "t", getattr(row, "time", None))
                o = getattr(row, "o", getattr(row, "open", None))
                h = getattr(row, "h", getattr(row, "high", None))
                l = getattr(row, "l", getattr(row, "low", None))
                c = getattr(row, "c", getattr(row, "close", None))
                v = getattr(row, "v", getattr(row, "tick_volume", getattr(row, "volume", 0)))
                if t is not None and o is not None:
                    bars.append(
                        OhlcvBar(t=int(t), o=float(o), h=float(h), l=float(l), c=float(c), v=int(v))
                    )
                elif hasattr(row, "__getitem__"):
                    bars.append(
                        OhlcvBar(
                            t=int(row[0]),
                            o=float(row[1]),
                            h=float(row[2]),
                            l=float(row[3]),
                            c=float(row[4]),
                            v=int(row[5]),
                        )
                    )
                else:
                    raise ValueError(f"Unsupported rate record format: {type(row)}")

        return bars
