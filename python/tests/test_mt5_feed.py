import numpy as np
import pytest

from python.src.mt5_feed import Mt5Feed


_UNSET = object()


class _FakeMt5:
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_H1 = 16385

    def __init__(
        self,
        init_ok: bool = True,
        select_ok: bool = True,
        rates_return=_UNSET,
        tick_return=_UNSET,
    ):
        self.init_ok = init_ok
        self.select_ok = select_ok
        self.rates_return = rates_return
        self.tick_return = tick_return
        self.calls = []
        self.shutdown_called = False

    def initialize(self):
        return self.init_ok

    def shutdown(self):
        self.shutdown_called = True

    def last_error(self):
        return (1, "Fake error description")

    def symbol_select(self, symbol, enable):
        return self.select_ok

    def symbol_info_tick(self, symbol):
        if self.tick_return is not _UNSET:
            return self.tick_return

        class Tick:
            bid = 3850.00
            ask = 3850.24

        return Tick()

    def copy_rates_from_pos(self, symbol, tf, start_pos, count):
        self.calls.append((symbol, tf, start_pos, count))
        if self.rates_return is not _UNSET:
            return self.rates_return

        dtype = [
            ("time", "i8"),
            ("open", "f8"),
            ("high", "f8"),
            ("low", "f8"),
            ("close", "f8"),
            ("tick_volume", "i8"),
        ]
        arr = np.zeros(count, dtype=dtype)
        for i in range(count):
            arr[i] = (
                1700000000 + i * 900,
                3800.0 + i,
                3802.0 + i,
                3799.0 + i,
                3801.0 + i,
                120 + i,
            )
        return arr


def test_mt5_feed_fetches_closed_bars_starting_at_pos_1():
    fake = _FakeMt5()
    feed = Mt5Feed(symbol="XAUUSDm", mt5_module=fake)
    df = feed.fetch_closed_bars("M15", count=50)
    assert len(df) == 50
    assert list(df.columns) == ["t", "o", "h", "l", "c", "v"]
    assert fake.calls[-1] == ("XAUUSDm", _FakeMt5.TIMEFRAME_M15, 1, 50)
    assert df["t"].is_monotonic_increasing
    bid, ask, spread = feed.fetch_tick_spread()
    assert round(spread, 2) == 0.24


def test_mt5_feed_timeframe_mapping():
    fake = _FakeMt5()
    feed = Mt5Feed(symbol="XAUUSDm", mt5_module=fake)

    feed.fetch_closed_bars("M5", count=10)
    assert fake.calls[-1] == ("XAUUSDm", _FakeMt5.TIMEFRAME_M5, 1, 10)

    feed.fetch_closed_bars("H1", count=10)
    assert fake.calls[-1] == ("XAUUSDm", _FakeMt5.TIMEFRAME_H1, 1, 10)

    with pytest.raises(ValueError, match="Unsupported timeframe"):
        feed.fetch_closed_bars("M1", count=10)


def test_mt5_feed_initialization_failure_raises():
    fake = _FakeMt5(init_ok=False)
    feed = Mt5Feed(symbol="XAUUSDm", mt5_module=fake)

    with pytest.raises(RuntimeError, match="MetaTrader5 initialize\\(\\) failed"):
        feed.fetch_closed_bars("M15", count=50)


def test_mt5_feed_symbol_selection_failure_raises():
    fake = _FakeMt5(select_ok=False)
    feed = Mt5Feed(symbol="XAUUSDm", mt5_module=fake)

    with pytest.raises(RuntimeError, match="Failed to select symbol 'XAUUSDm'"):
        feed.fetch_closed_bars("M15", count=50)


def test_mt5_feed_copy_rates_none_or_empty_raises():
    fake_none = _FakeMt5(rates_return=None)
    feed_none = Mt5Feed(symbol="XAUUSDm", mt5_module=fake_none)
    with pytest.raises(RuntimeError, match="copy_rates_from_pos.*returned None or empty"):
        feed_none.fetch_closed_bars("M15", count=50)

    fake_empty = _FakeMt5(rates_return=np.array([]))
    feed_empty = Mt5Feed(symbol="XAUUSDm", mt5_module=fake_empty)
    with pytest.raises(RuntimeError, match="copy_rates_from_pos.*returned None or empty"):
        feed_empty.fetch_closed_bars("M15", count=50)


def test_mt5_feed_invalid_count_raises():
    fake = _FakeMt5()
    feed = Mt5Feed(symbol="XAUUSDm", mt5_module=fake)
    with pytest.raises(ValueError, match="count must be a positive integer"):
        feed.fetch_closed_bars("M15", count=0)


def test_mt5_feed_deduplicates_and_sorts_ascending():
    dtype = [
        ("time", "i8"),
        ("open", "f8"),
        ("high", "f8"),
        ("low", "f8"),
        ("close", "f8"),
        ("tick_volume", "i8"),
    ]
    raw_data = np.array(
        [
            (1700000900, 3801.0, 3803.0, 3800.0, 3802.0, 150),
            (1700000000, 3800.0, 3802.0, 3799.0, 3801.0, 120),
            (1700000900, 3801.0, 3803.0, 3800.0, 3802.5, 155),  # duplicate timestamp
        ],
        dtype=dtype,
    )
    fake = _FakeMt5(rates_return=raw_data)
    feed = Mt5Feed(symbol="XAUUSDm", mt5_module=fake)

    df = feed.fetch_closed_bars("M15", count=2)
    assert len(df) == 2
    assert df["t"].tolist() == [1700000000, 1700000900]
    assert df["t"].is_monotonic_increasing
    # Should keep the latest duplicate
    assert df.iloc[-1]["c"] == 3802.5


def test_mt5_feed_fetch_tick_spread_safe_defaults():
    # 1. Init failure
    fake_init_fail = _FakeMt5(init_ok=False)
    feed_init_fail = Mt5Feed(symbol="XAUUSDm", mt5_module=fake_init_fail)
    assert feed_init_fail.fetch_tick_spread() == (0.0, 0.0, 0.0)

    # 2. Select failure
    fake_sel_fail = _FakeMt5(select_ok=False)
    feed_sel_fail = Mt5Feed(symbol="XAUUSDm", mt5_module=fake_sel_fail)
    assert feed_sel_fail.fetch_tick_spread() == (0.0, 0.0, 0.0)

    # 3. None tick
    fake_none_tick = _FakeMt5(tick_return=None)
    feed_none_tick = Mt5Feed(symbol="XAUUSDm", mt5_module=fake_none_tick)
    assert feed_none_tick.fetch_tick_spread() == (0.0, 0.0, 0.0)

    # 4. Tick with zero or invalid prices
    class BadTick:
        bid = 0.0
        ask = 0.0

    fake_bad_tick = _FakeMt5(tick_return=BadTick())
    feed_bad_tick = Mt5Feed(symbol="XAUUSDm", mt5_module=fake_bad_tick)
    assert feed_bad_tick.fetch_tick_spread() == (0.0, 0.0, 0.0)


def test_mt5_feed_shutdown():
    fake = _FakeMt5()
    feed = Mt5Feed(symbol="XAUUSDm", mt5_module=fake)
    feed.fetch_closed_bars("M15", count=5)
    feed.shutdown()
    assert fake.shutdown_called is True


def test_mt5_feed_lazy_import_error_when_no_module(monkeypatch):
    import sys

    # Ensure MetaTrader5 cannot be imported
    monkeypatch.setitem(sys.modules, "MetaTrader5", None)
    feed = Mt5Feed(symbol="XAUUSDm", mt5_module=None)

    with pytest.raises(RuntimeError, match="MetaTrader5 package is not available"):
        feed.fetch_closed_bars("M15", count=5)

    assert feed.fetch_tick_spread() == (0.0, 0.0, 0.0)


def test_mt5_feed_auto_reconnects_after_ipc_disconnect():
    class _FlakyIpcMt5(_FakeMt5):
        def __init__(self):
            super().__init__()
            self.init_calls = 0
            self.select_calls = 0

        def initialize(self):
            self.init_calls += 1
            return True

        def symbol_select(self, symbol, enable):
            self.select_calls += 1
            # Fail once simulating (-10001, 'IPC send failed') after MT5 restart
            return self.select_calls >= 2

        def last_error(self):
            return (-10001, "IPC send failed")

    flaky = _FlakyIpcMt5()
    feed = Mt5Feed(symbol="XAUUSDm", mt5_module=flaky)
    df = feed.fetch_closed_bars("M15", count=10)
    assert len(df) == 10
    assert flaky.shutdown_called is True
    assert flaky.init_calls == 2
    assert flaky.select_calls == 2

