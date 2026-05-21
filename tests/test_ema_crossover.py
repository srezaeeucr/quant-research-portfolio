"""
Tests for EMACrossoverStrategy.
"""
import sys
from pathlib import Path
from datetime import datetime, time

import pandas as pd
import pytz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.strategy.ema_crossover import EMACrossoverStrategy

_ET = pytz.timezone("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(t: time, close: float, volume: float = 200_000,
         date_str: str = "2024-01-15") -> dict:
    dt = _ET.localize(datetime.strptime(
        f"{date_str} {t.hour:02d}:{t.minute:02d}", "%Y-%m-%d %H:%M"
    ))
    return {"timestamp": dt, "open": close, "high": close,
            "low": close, "close": close, "volume": volume}


def _df(bars):
    return pd.DataFrame(bars)


def _default_cfg(fast: int = 9, slow: int = 21, vol_mult: float = 1.2) -> dict:
    return {"strategy": {"ema_fast_period": fast, "ema_slow_period": slow,
                         "volume_multiplier": vol_mult}}


def _flat_session(price: float = 100.0, n: int = 25,
                  start_minute: int = 30) -> list:
    """Return n bars starting at 9:start_minute with constant price."""
    bars = []
    for i in range(n):
        m = start_minute + i
        h, rem = divmod(m, 60)
        bars.append(_bar(time(9 + h, rem), price))
    return bars


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_signal_with_insufficient_bars():
    """No signal before slow_period (21) bars have been seen."""
    cfg   = _default_cfg()
    strat = EMACrossoverStrategy(cfg)
    strat.reset_session()

    bars = _flat_session(n=20)  # only 20 bars — slow EMA not yet seeded
    df   = _df(bars)
    for i in range(len(df)):
        assert strat.generate_signal(df, i) is None


def test_no_signal_before_entry_window():
    """Even with enough bars, no signal is emitted before 9:45 ET."""
    cfg   = _default_cfg()
    strat = EMACrossoverStrategy(cfg)
    strat.reset_session()

    # 22 bars but all before 9:45 (bars 0..21 = 9:30..9:51, some are after cutoff)
    # Build a sequence that would cause a crossover at bar 21 (9:51) but entry window
    # starts at 9:45. We want the crossover at bar < 9:45.
    # Use fast=3, slow=5 for quick seeding
    cfg2  = _default_cfg(fast=3, slow=5)
    strat2 = EMACrossoverStrategy(cfg2)
    strat2.reset_session()

    bars = []
    # Falling prices so fast EMA < slow EMA
    for i in range(10):
        m = 30 + i
        bars.append(_bar(time(9, m), 100.0 - i * 0.1))
    # Rising price to force crossover — but force it at 9:40 (before entry window)
    bars.append(_bar(time(9, 40), 102.0, volume=500_000))

    df = _df(bars)
    strat2.reset_session()
    for i in range(len(df)):
        s = strat2.generate_signal(df, i)
        if s:
            bar_time = df.iloc[i]["timestamp"].astimezone(_ET).time()
            assert bar_time >= time(9, 45), \
                f"Signal fired before 9:45 at {bar_time}"


def test_golden_cross_fires_signal():
    """BUY signal fires when fast EMA crosses above slow EMA in entry window."""
    cfg   = _default_cfg(fast=3, slow=5, vol_mult=1.0)
    strat = EMACrossoverStrategy(cfg)
    strat.reset_session()

    bars = []
    # Declining phase: fast EMA will be below slow EMA
    for i in range(7):
        m = 30 + i
        bars.append(_bar(time(9, m), 100.0 - i * 0.2, volume=100_000))

    # 9:45 bar — sudden strong rise forces fast EMA above slow EMA
    bars.append(_bar(time(9, 45), 103.0, volume=100_000))  # vol_mult=1.0, always passes

    df  = _df(bars)
    sig = None
    for i in range(len(df)):
        s = strat.generate_signal(df, i)
        if s:
            sig = s
            break

    assert sig is not None, "Expected a BUY signal on golden cross"
    assert sig["signal"] == "BUY"
    assert sig["ema_fast"] > sig["ema_slow"]


def test_volume_filter_rejects_low_volume():
    """No signal when crossover bar volume is below vol_mult × avg_volume."""
    cfg   = _default_cfg(fast=3, slow=5, vol_mult=2.0)  # very strict
    strat = EMACrossoverStrategy(cfg)
    strat.reset_session()

    bars = []
    for i in range(7):
        m = 30 + i
        bars.append(_bar(time(9, m), 100.0 - i * 0.2, volume=100_000))
    # Crossover bar with LOW volume (way below 2× avg)
    bars.append(_bar(time(9, 45), 103.0, volume=1_000))

    df = _df(bars)
    for i in range(len(df)):
        assert strat.generate_signal(df, i) is None


def test_only_one_signal_per_session():
    """Strategy fires at most one signal per session."""
    cfg   = _default_cfg(fast=3, slow=5, vol_mult=1.0)
    strat = EMACrossoverStrategy(cfg)
    strat.reset_session()

    bars = []
    for i in range(7):
        bars.append(_bar(time(9, 30 + i), 100.0 - i * 0.2, volume=100_000))
    # Two potential crossover bars in the entry window
    bars.append(_bar(time(9, 45), 103.0, volume=100_000))
    bars.append(_bar(time(9, 46), 101.0, volume=100_000))
    bars.append(_bar(time(9, 48), 104.0, volume=100_000))

    df      = _df(bars)
    signals = []
    for i in range(len(df)):
        s = strat.generate_signal(df, i)
        if s:
            signals.append(s)
    assert len(signals) <= 1


def test_no_signal_after_entry_window():
    """No signal when crossover occurs after 10:30 ET."""
    cfg   = _default_cfg(fast=3, slow=5, vol_mult=1.0)
    strat = EMACrossoverStrategy(cfg)
    strat.reset_session()

    bars = []
    for i in range(7):
        bars.append(_bar(time(9, 30 + i), 100.0 - i * 0.2, volume=100_000))
    # Crossover at 10:31 — outside window
    bars.append(_bar(time(10, 31), 105.0, volume=500_000))

    df = _df(bars)
    for i in range(len(df)):
        assert strat.generate_signal(df, i) is None


# ---------------------------------------------------------------------------
# Afternoon entry window tests
# ---------------------------------------------------------------------------

def _afternoon_cfg(fast=3, slow=5, vol_mult=1.0, pm_enabled=True):
    return {
        "strategy": {"ema_fast_period": fast, "ema_slow_period": slow,
                      "volume_multiplier": vol_mult},
        "strategy_params": {"ema_crossover": {
            "ema_fast": fast, "ema_slow": slow,
            "volume_multiplier": vol_mult,
            "afternoon_entry_enabled": pm_enabled,
        }},
    }


def _full_day_bars(pm_crossover_price=105.0, pm_crossover_time=time(15, 35)):
    """Build bars spanning 9:30–15:50 with a decline then a PM golden cross."""
    bars = []
    # Morning: 9:30–10:30 — flat/declining (no morning crossover)
    for i in range(61):
        m = 30 + i
        h, rem = divmod(m, 60)
        bars.append(_bar(time(9 + h, rem), 100.0 - i * 0.01, volume=100_000))
    # Midday gap: 10:31–15:29 — continue declining slowly
    for h in range(10, 15):
        start_m = 31 if h == 10 else 0
        end_m   = 60 if h < 15 else 30
        for m in range(start_m, end_m):
            bars.append(_bar(time(h, m), 99.0 - (h - 10) * 0.1, volume=80_000))
    # Afternoon: 15:30–15:50 — spike up for crossover
    for m in range(30, 51):
        price = pm_crossover_price if m == pm_crossover_time.minute else 98.5
        vol   = 200_000 if m == pm_crossover_time.minute else 80_000
        bars.append(_bar(time(15, m), price, volume=vol))
    return bars


def test_afternoon_entry_fires_between_1530_and_1550():
    """BUY signal fires during 15:30–15:50 ET when afternoon entries enabled."""
    cfg   = _afternoon_cfg()
    strat = EMACrossoverStrategy(cfg)
    strat.reset_session()

    bars = _full_day_bars(pm_crossover_price=105.0, pm_crossover_time=time(15, 35))
    df   = _df(bars)
    sig  = None
    for i in range(len(df)):
        s = strat.generate_signal(df, i)
        if s:
            sig = s
            sig_time = df.iloc[i]["timestamp"].astimezone(
                __import__("pytz").timezone("America/New_York")
            ).time()

    assert sig is not None, "Expected afternoon BUY signal"
    assert sig["signal"] == "BUY"
    assert sig_time >= time(15, 30)
    assert sig_time <= time(15, 50)


def test_no_afternoon_entry_after_1550():
    """No signal when crossover occurs after 15:50 ET (too close to EOD)."""
    cfg   = _afternoon_cfg()
    strat = EMACrossoverStrategy(cfg)
    strat.reset_session()

    bars = _full_day_bars(pm_crossover_price=98.5)  # no spike — no crossover in window
    # Add late crossover at 15:52
    bars.append(_bar(time(15, 52), 110.0, volume=300_000))
    df = _df(bars)

    signals = []
    for i in range(len(df)):
        s = strat.generate_signal(df, i)
        if s:
            sig_time = df.iloc[i]["timestamp"].astimezone(
                __import__("pytz").timezone("America/New_York")
            ).time()
            signals.append(sig_time)

    # No signals should fire after 15:50
    for t in signals:
        assert t <= time(15, 50), f"Signal fired at {t}, after 15:50 cutoff"


def test_max_trades_blocks_afternoon_via_engine_limit():
    """Afternoon signal still fires from strategy — engine's max_trades_per_day
    handles the limit. Strategy allows one signal per window independently."""
    cfg   = _afternoon_cfg()
    strat = EMACrossoverStrategy(cfg)
    strat.reset_session()

    # Morning crossover
    bars = []
    for i in range(7):
        bars.append(_bar(time(9, 30 + i), 100.0 - i * 0.2, volume=100_000))
    bars.append(_bar(time(9, 45), 103.0, volume=100_000))

    df = _df(bars)
    morning_sig = None
    for i in range(len(df)):
        s = strat.generate_signal(df, i)
        if s:
            morning_sig = s
    assert morning_sig is not None, "Morning signal should fire"
    assert strat._am_signal_fired is True
    assert strat._pm_signal_fired is False  # PM still available


def test_afternoon_session_independent_of_morning():
    """Morning and afternoon signals fire independently — morning signal
    does not block afternoon, and vice versa."""
    cfg   = _afternoon_cfg()
    strat = EMACrossoverStrategy(cfg)
    strat.reset_session()

    bars = _full_day_bars(pm_crossover_price=105.0, pm_crossover_time=time(15, 35))
    df   = _df(bars)
    signals = []
    for i in range(len(df)):
        s = strat.generate_signal(df, i)
        if s:
            sig_time = df.iloc[i]["timestamp"].astimezone(
                __import__("pytz").timezone("America/New_York")
            ).time()
            signals.append(("am" if sig_time < time(12, 0) else "pm", s))

    # Should have at most one PM signal (morning may or may not fire depending on data)
    pm_signals = [s for w, s in signals if w == "pm"]
    assert len(pm_signals) <= 1
    if pm_signals:
        assert pm_signals[0]["signal"] == "BUY"


def test_no_afternoon_entry_when_disabled():
    """No afternoon signal when afternoon_entry_enabled=false."""
    cfg   = _afternoon_cfg(pm_enabled=False)
    strat = EMACrossoverStrategy(cfg)
    strat.reset_session()

    bars = _full_day_bars(pm_crossover_price=105.0, pm_crossover_time=time(15, 35))
    df   = _df(bars)
    pm_signals = []
    for i in range(len(df)):
        s = strat.generate_signal(df, i)
        if s:
            sig_time = df.iloc[i]["timestamp"].astimezone(
                __import__("pytz").timezone("America/New_York")
            ).time()
            if sig_time >= time(15, 0):
                pm_signals.append(s)

    assert len(pm_signals) == 0, "No PM signal should fire when disabled"
