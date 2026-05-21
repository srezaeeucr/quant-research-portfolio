"""
Tests for MomentumStrategy.
"""
import sys
from pathlib import Path
from datetime import datetime, time

import pandas as pd
import pytz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.strategy.momentum import MomentumStrategy

_ET = pytz.timezone("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(t: time, close: float, open_: float = None,
         volume: float = 100_000, date_str: str = "2024-01-15") -> dict:
    dt = _ET.localize(datetime.strptime(
        f"{date_str} {t.hour:02d}:{t.minute:02d}", "%Y-%m-%d %H:%M"
    ))
    o = open_ if open_ is not None else close
    return {"timestamp": dt, "open": o, "high": close, "low": close,
            "close": close, "volume": volume}


def _df(bars):
    return pd.DataFrame(bars)


def _default_cfg(threshold: float = 0.3, vol_mult: float = 1.2) -> dict:
    return {"strategy": {"momentum_threshold_pct": threshold,
                         "volume_multiplier": vol_mult}}


def _or_session(open_price: float, final_close: float,
                vol_per_bar: float = 100_000) -> list:
    """Build 15 OR bars (9:30–9:44) with a linear move from open_price to final_close."""
    bars = []
    for i in range(15):
        progress = i / 14.0
        price    = open_price + (final_close - open_price) * progress
        t        = time(9, 30 + i)
        bars.append(_bar(t, close=price,
                         open_=open_price if i == 0 else price,
                         volume=vol_per_bar))
    return bars


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_signal_fires_at_threshold():
    """BUY signal fires at 9:45 when OR momentum >= threshold (0.3%)."""
    cfg   = _default_cfg(threshold=0.3)
    strat = MomentumStrategy(cfg)
    strat.reset_session()
    # No prev_or_volume set → skip volume check

    open_p = 100.0
    final  = 100.4  # 0.4% gain — above 0.3% threshold
    bars   = _or_session(open_p, final)
    bars.append(_bar(time(9, 45), final + 0.1))  # 9:45 bar

    df  = _df(bars)
    sig = None
    for i in range(len(df)):
        s = strat.generate_signal(df, i)
        if s:
            sig = s
            break

    assert sig is not None, "Expected a BUY signal at 9:45"
    assert sig["signal"] == "BUY"
    assert sig["momentum_pct"] >= 0.3
    assert sig["open_price"] == pytest.approx(open_p, abs=0.01)


def test_no_signal_below_threshold():
    """No signal when OR momentum is below threshold."""
    cfg   = _default_cfg(threshold=0.3)
    strat = MomentumStrategy(cfg)
    strat.reset_session()

    open_p = 100.0
    final  = 100.2  # only 0.2% — below 0.3% threshold
    bars   = _or_session(open_p, final)
    bars.append(_bar(time(9, 45), final))

    df = _df(bars)
    for i in range(len(df)):
        assert strat.generate_signal(df, i) is None


def test_momentum_pct_calculated_correctly():
    """momentum_pct = (or_close - open_price) / open_price × 100."""
    cfg   = _default_cfg(threshold=0.0)  # zero threshold so any move fires
    strat = MomentumStrategy(cfg)
    strat.reset_session()

    open_p = 200.0
    final  = 201.0  # exactly 0.5%
    bars   = _or_session(open_p, final)
    bars.append(_bar(time(9, 45), final))

    df  = _df(bars)
    sig = None
    for i in range(len(df)):
        s = strat.generate_signal(df, i)
        if s:
            sig = s
            break

    assert sig is not None
    assert abs(sig["momentum_pct"] - 0.5) < 0.01


def test_volume_filter_blocks_signal():
    """No signal when OR volume < vol_mult × prev_or_volume."""
    cfg   = _default_cfg(threshold=0.3, vol_mult=1.2)
    strat = MomentumStrategy(cfg)
    strat.reset_session()

    # prev_or_volume = 150_000; total OR volume = 15 bars × 10_000 = 150_000
    # required = 150_000 × 1.2 = 180_000 — actual 150_000 < 180_000 → reject
    strat.set_prev_or_volume(150_000.0)

    open_p = 100.0
    final  = 100.4  # above threshold
    bars   = _or_session(open_p, final, vol_per_bar=10_000)  # low volume
    bars.append(_bar(time(9, 45), final))

    df = _df(bars)
    for i in range(len(df)):
        assert strat.generate_signal(df, i) is None


def test_volume_filter_passes_high_volume():
    """Signal fires when OR volume >= vol_mult × prev_or_volume."""
    cfg   = _default_cfg(threshold=0.3, vol_mult=1.2)
    strat = MomentumStrategy(cfg)
    strat.reset_session()

    # prev_or_volume = 100_000; total OR volume = 15 × 10_000 = 150_000
    # required = 100_000 × 1.2 = 120_000; actual 150_000 >= 120_000 → pass
    strat.set_prev_or_volume(100_000.0)

    open_p = 100.0
    final  = 100.4
    bars   = _or_session(open_p, final, vol_per_bar=10_000)
    bars.append(_bar(time(9, 45), final))

    df  = _df(bars)
    sig = None
    for i in range(len(df)):
        s = strat.generate_signal(df, i)
        if s:
            sig = s
            break
    assert sig is not None


def test_no_signal_when_or_not_established():
    """No signal at 9:45 if the opening range bars were missing."""
    cfg   = _default_cfg(threshold=0.0)
    strat = MomentumStrategy(cfg)
    strat.reset_session()

    # Jump straight to 9:45 with no OR history
    bars = [_bar(time(9, 45), 100.5)]
    df   = _df(bars)
    assert strat.generate_signal(df, 0) is None
