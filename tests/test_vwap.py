"""
Tests for VWAPReversionStrategy (new implementation using close × volume VWAP).
"""
import sys
from pathlib import Path
from datetime import datetime, time

import pandas as pd
import pytz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.strategy.vwap_reversion import VWAPReversionStrategy

_ET = pytz.timezone("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(t: time, close: float, high: float = None, low: float = None,
         volume: float = 100_000, date_str: str = "2024-01-15") -> dict:
    dt = _ET.localize(datetime.strptime(
        f"{date_str} {t.hour:02d}:{t.minute:02d}", "%Y-%m-%d %H:%M"
    ))
    h = high if high is not None else close
    l = low  if low  is not None else close
    return {"timestamp": dt, "open": close, "high": h, "low": l,
            "close": close, "volume": volume}


def _df(bars):
    return pd.DataFrame(bars)


def _new_cfg(deviation: float = 0.002) -> dict:
    """Config using new strategy_params.vwap_reversion path."""
    return {"strategy_params": {"vwap_reversion": {"deviation_threshold": deviation}}}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_vwap_uses_close_times_volume():
    """VWAP should equal cumulative(close × volume) / cumulative(volume)."""
    cfg   = _new_cfg()
    strat = VWAPReversionStrategy(cfg)
    strat.reset_session()

    bars = [
        _bar(time(9, 30), close=100.0, volume=200_000),
        _bar(time(9, 31), close=102.0, volume=100_000),
    ]
    df = _df(bars)
    for i in range(len(df)):
        strat.generate_signal(df, i)

    # Expected VWAP after bar 1: (100*200k + 102*100k) / 300k = 100.667
    expected_vwap = (100.0 * 200_000 + 102.0 * 100_000) / 300_000
    computed_vwap = strat._cum_pv / strat._cum_vol
    assert abs(computed_vwap - expected_vwap) < 0.001, (
        f"Expected VWAP {expected_vwap:.4f}, got {computed_vwap:.4f}"
    )


def test_signal_fires_on_dip_and_bounce():
    """BUY signal fires when close dips >= deviation_threshold below VWAP and bounces."""
    cfg   = _new_cfg(deviation=0.002)   # 0.2% dip required
    strat = VWAPReversionStrategy(cfg)
    strat.reset_session()

    # 15 bars at 9:30-9:44: all close=100, VWAP=100
    bars = [_bar(time(9, 30 + m), 100.0, volume=200_000) for m in range(15)]
    # Dip bar 9:45: close=99.75 → deviation=(100-99.75)/100=0.25% ≥ 0.2%
    bars.append(_bar(time(9, 45), 99.75, volume=100_000))
    # Bounce bar 9:46: close=99.80 > 99.75
    bars.append(_bar(time(9, 46), 99.80, volume=100_000))

    df  = _df(bars)
    sig = None
    for i in range(len(df)):
        s = strat.generate_signal(df, i)
        if s:
            sig = s
            break

    assert sig is not None, "Expected a BUY signal on dip+bounce"
    assert sig["signal"] == "BUY"
    assert sig["entry_price"] == pytest.approx(99.80, abs=0.01)
    assert sig["deviation_pct"] > 0
    assert sig["vwap"] > sig["entry_price"]


def test_no_signal_without_bounce():
    """No signal when close is below VWAP but current close <= previous close."""
    cfg   = _new_cfg(deviation=0.002)
    strat = VWAPReversionStrategy(cfg)
    strat.reset_session()

    bars = [_bar(time(9, 30 + m), 100.0, volume=200_000) for m in range(15)]
    # Dip bar: valid dip
    bars.append(_bar(time(9, 45), 99.75, volume=100_000))
    # No bounce: close LOWER than dip bar
    bars.append(_bar(time(9, 46), 99.70, volume=100_000))

    df = _df(bars)
    strat.reset_session()
    for i in range(len(df)):
        assert strat.generate_signal(df, i) is None


def test_no_signal_outside_entry_window():
    """No signal when dip+bounce occurs before 9:45 ET."""
    cfg   = _new_cfg(deviation=0.002)
    strat = VWAPReversionStrategy(cfg)
    strat.reset_session()

    # All bars before entry window (9:30–9:44)
    bars = [_bar(time(9, 30 + m), 100.0, volume=200_000) for m in range(14)]
    # Dip bar at 9:44 (inside OR, outside entry window)
    bars.append(_bar(time(9, 44), 99.70, volume=100_000))

    df = _df(bars)
    strat.reset_session()
    for i in range(len(df)):
        assert strat.generate_signal(df, i) is None


def test_only_one_signal_per_session():
    """Second qualifying dip+bounce in same session must not produce a second signal."""
    cfg   = _new_cfg(deviation=0.002)
    strat = VWAPReversionStrategy(cfg)
    strat.reset_session()

    bars = [_bar(time(9, 30 + m), 100.0, volume=200_000) for m in range(15)]
    # First qualifying sequence
    bars.append(_bar(time(9, 45), 99.75, volume=100_000))
    bars.append(_bar(time(9, 46), 99.80, volume=100_000))
    # Second qualifying sequence
    bars.append(_bar(time(9, 50), 99.60, volume=100_000))
    bars.append(_bar(time(9, 51), 99.65, volume=100_000))

    df      = _df(bars)
    signals = []
    strat.reset_session()
    for i in range(len(df)):
        s = strat.generate_signal(df, i)
        if s:
            signals.append(s)

    assert len(signals) == 1, f"Expected 1 signal, got {len(signals)}"


def test_deviation_threshold_respected():
    """No signal when dip is below the configured deviation_threshold."""
    cfg   = _new_cfg(deviation=0.005)   # require 0.5% dip
    strat = VWAPReversionStrategy(cfg)
    strat.reset_session()

    bars = [_bar(time(9, 30 + m), 100.0, volume=200_000) for m in range(15)]
    # Only 0.2% dip — below 0.5% threshold
    bars.append(_bar(time(9, 45), 99.80, volume=100_000))
    bars.append(_bar(time(9, 46), 99.85, volume=100_000))

    df = _df(bars)
    strat.reset_session()
    for i in range(len(df)):
        assert strat.generate_signal(df, i) is None
