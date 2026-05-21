"""
Tests for ORBStrategy.
Run with:  pytest tests/test_orb.py -v
"""
import sys
from pathlib import Path
from datetime import datetime, time

import pandas as pd
import pytz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.strategy.orb import ORBStrategy, _ORB_END

_ET = pytz.timezone("America/New_York")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bar(t: time, close: float, date_str: str = "2024-01-15", volume: float = 100_000) -> dict:
    """Create a single bar dict with an ET-aware timestamp."""
    dt = _ET.localize(datetime.strptime(f"{date_str} {t.hour:02d}:{t.minute:02d}", "%Y-%m-%d %H:%M"))
    return {"timestamp": dt, "open": close, "high": close, "low": close, "close": close, "volume": volume}


def _make_day(bars: list) -> pd.DataFrame:
    return pd.DataFrame(bars)


def _orb_bars(or_high: float = 100.0, or_low: float = 99.0, date_str: str = "2024-01-15") -> list:
    """15 opening-range bars from 9:30 to 9:44."""
    bars = []
    for minute in range(30, 45):   # 9:30 .. 9:44
        price = or_high if minute == 30 else (or_low if minute == 31 else (or_high + or_low) / 2)
        bars.append(_make_bar(time(9, minute), price, date_str))
    return bars


def _default_config() -> dict:
    return {"strategy": {"min_breakout_pct": 0.1, "volume_multiplier": 1.5}}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def strategy():
    return ORBStrategy(_default_config())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_signal_during_opening_range(strategy):
    """Bars 9:30–9:44 must never produce a signal."""
    bars = _orb_bars()
    df = _make_day(bars)
    strategy.reset_session()
    for i in range(len(df)):
        assert strategy.generate_signal(df, i) is None, f"Unexpected signal at bar {i}"


def test_buy_signal_on_breakout(strategy):
    """A close above opening_high + min_breakout_pct with sufficient volume should trigger BUY."""
    or_high = 100.0
    bars = _orb_bars(or_high=or_high)
    # OR bars each have volume=100_000 → avg=100_000; need breakout vol ≥ 150_000
    bars.append(_make_bar(time(9, 45), or_high * 1.002, volume=160_000))   # +0.2% > 0.1% threshold
    df = _make_day(bars)

    strategy.reset_session()
    signal = None
    for i in range(len(df)):
        s = strategy.generate_signal(df, i)
        if s:
            signal = s
            break

    assert signal is not None, "Expected a BUY signal after breakout bar"
    assert signal["signal"] == "BUY"
    assert signal["entry_price"] > signal["opening_high"]


def test_no_signal_below_threshold(strategy):
    """A close just above opening_high but below the min_breakout_pct should NOT trigger."""
    or_high = 100.0
    bars = _orb_bars(or_high=or_high)
    bars.append(_make_bar(time(9, 45), or_high * 1.0005))  # +0.05% < 0.1% threshold
    df = _make_day(bars)

    strategy.reset_session()
    for i in range(len(df)):
        assert strategy.generate_signal(df, i) is None


def test_only_one_signal_per_session(strategy):
    """Even if multiple breakout bars occur, only the first should fire."""
    or_high = 100.0
    bars = _orb_bars(or_high=or_high)
    for minute in range(45, 60):
        bars.append(_make_bar(time(9, minute), or_high * 1.005, volume=160_000))  # all break out
    df = _make_day(bars)

    strategy.reset_session()
    signals = []
    for i in range(len(df)):
        s = strategy.generate_signal(df, i)
        if s:
            signals.append(s)

    assert len(signals) == 1, f"Expected exactly 1 signal, got {len(signals)}"


def test_no_entry_signal_in_pm_window(strategy):
    """Bars in 15:30–16:00 must not trigger entry signals."""
    bars = _orb_bars()   # opening range establishes OR high/low
    # Add a PM bar that would otherwise break out
    or_high = max(b["close"] for b in bars)
    bars.append(_make_bar(time(15, 31), or_high * 1.005))
    df = _make_day(bars)

    strategy.reset_session()
    for i in range(len(df)):
        assert strategy.generate_signal(df, i) is None, "Should not signal in PM window"


def test_run_day_returns_signal(strategy):
    """run_day convenience method should return the first signal."""
    or_high = 100.0
    bars = _orb_bars(or_high=or_high)
    bars.append(_make_bar(time(9, 45), or_high * 1.002, volume=160_000))
    df = _make_day(bars)

    sig = strategy.run_day(df, "SPY")
    assert sig is not None
    assert sig["symbol"] == "SPY"
    assert sig["signal"] == "BUY"


def test_no_signal_at_or_after_10am(strategy):
    """Bars at 10:00 ET and later must never trigger a BUY entry signal."""
    or_high = 100.0
    bars = _orb_bars(or_high=or_high)
    # Add bars at 10:00, 10:01, and 10:30 — all would otherwise break out
    for t in [time(10, 0), time(10, 1), time(10, 30)]:
        bars.append(_make_bar(t, or_high * 1.005))
    df = _make_day(bars)

    strategy.reset_session()
    for i in range(len(df)):
        assert strategy.generate_signal(df, i) is None, \
            f"Unexpected signal at bar {i} ({df.iloc[i]['timestamp']})"


def test_reset_session_clears_state(strategy):
    """After reset_session, strategy should fire again on a new day."""
    or_high = 100.0
    bars = _orb_bars(or_high=or_high)
    bars.append(_make_bar(time(9, 45), or_high * 1.002, volume=160_000))
    df = _make_day(bars)

    sig1 = strategy.run_day(df, "SPY")
    sig2 = strategy.run_day(df, "SPY")   # run_day calls reset_session internally

    assert sig1 is not None
    assert sig2 is not None, "Strategy should fire again after reset_session"


def test_volume_confirmation_passes(strategy):
    """Breakout bar with volume ≥ 1.5× avg OR volume should produce a signal."""
    or_high = 100.0
    # OR bars: volume=100_000 each → avg=100_000; required = 150_000
    bars = _orb_bars(or_high=or_high)
    bars.append(_make_bar(time(9, 45), or_high * 1.002, volume=150_000))  # exactly at threshold
    df = _make_day(bars)

    sig = strategy.run_day(df, "SPY")
    assert sig is not None, "Volume exactly at threshold should pass"
    assert sig["volume_confirmed"] is True


def test_volume_confirmation_rejects(strategy):
    """Breakout bar with volume < 1.5× avg OR volume must NOT produce a signal."""
    or_high = 100.0
    # OR bars: volume=100_000 each → avg=100_000; required = 150_000
    bars = _orb_bars(or_high=or_high)
    bars.append(_make_bar(time(9, 45), or_high * 1.002, volume=149_999))  # just below threshold
    df = _make_day(bars)

    sig = strategy.run_day(df, "SPY")
    assert sig is None, "Volume below threshold should be rejected"


def test_regime_filter_blocks_signal_below_sma():
    """BUY signal must be suppressed when close < 20-day SMA and use_regime_filter=True."""
    cfg = {"strategy": {"min_breakout_pct": 0.1, "volume_multiplier": 1.5,
                        "use_regime_filter": True, "regime_sma_days": 20}}
    strat = ORBStrategy(cfg)
    or_high = 100.0
    bars = _orb_bars(or_high=or_high)
    breakout_price = or_high * 1.002   # above OR high + threshold
    bars.append(_make_bar(time(9, 45), breakout_price, volume=160_000))
    df = _make_day(bars)

    strat.reset_session()
    strat.set_regime_sma(breakout_price + 1.0)   # SMA above close → bearish regime

    sig = None
    for i in range(len(df)):
        s = strat.generate_signal(df, i)
        if s:
            sig = s
            break

    assert sig is None, "Regime filter should block signal when close < SMA"


def test_regime_filter_allows_signal_above_sma():
    """BUY signal must be emitted when close > 20-day SMA and use_regime_filter=True."""
    cfg = {"strategy": {"min_breakout_pct": 0.1, "volume_multiplier": 1.5,
                        "use_regime_filter": True, "regime_sma_days": 20}}
    strat = ORBStrategy(cfg)
    or_high = 100.0
    bars = _orb_bars(or_high=or_high)
    breakout_price = or_high * 1.002
    bars.append(_make_bar(time(9, 45), breakout_price, volume=160_000))
    df = _make_day(bars)

    strat.reset_session()
    strat.set_regime_sma(breakout_price - 1.0)   # SMA below close → bullish regime

    sig = None
    for i in range(len(df)):
        s = strat.generate_signal(df, i)
        if s:
            sig = s
            break

    assert sig is not None, "Regime filter should allow signal when close > SMA"
    assert sig["signal"] == "BUY"


def test_regime_filter_disabled_allows_signal():
    """When use_regime_filter=False, a valid breakout should fire even if close < SMA."""
    cfg = {"strategy": {"min_breakout_pct": 0.1, "volume_multiplier": 1.5,
                        "use_regime_filter": False}}
    strat = ORBStrategy(cfg)
    or_high = 100.0
    bars = _orb_bars(or_high=or_high)
    breakout_price = or_high * 1.002
    bars.append(_make_bar(time(9, 45), breakout_price, volume=160_000))
    df = _make_day(bars)

    strat.reset_session()
    strat.set_regime_sma(breakout_price + 5.0)   # SMA far above — would block if filter enabled

    sig = None
    for i in range(len(df)):
        s = strat.generate_signal(df, i)
        if s:
            sig = s
            break

    assert sig is not None, "Disabled regime filter should not block signal"
    assert sig["signal"] == "BUY"
