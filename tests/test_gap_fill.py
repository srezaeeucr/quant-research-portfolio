"""
Tests for GapFillStrategy.
"""
import sys
from pathlib import Path
from datetime import datetime, time

import pandas as pd
import pytz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.strategy.gap_fill import GapFillStrategy

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


def _default_cfg(min_gap_pct: float = 0.3) -> dict:
    return {"strategy": {"min_gap_pct": min_gap_pct}}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_gap_down_signal_fires():
    """BUY signal fires when open gaps down by >= min_gap_pct from prev_close."""
    cfg   = _default_cfg(min_gap_pct=0.3)
    strat = GapFillStrategy(cfg)
    strat.reset_session()
    strat.set_prev_close(100.0)

    # Gap down: open = 99.60 → gap = -0.4% (< -0.3% threshold)
    bars = [_bar(time(9, 30), close=99.65, open_=99.60)]
    df   = _df(bars)

    strat.reset_session()
    strat.set_prev_close(100.0)
    sig = strat.generate_signal(df, 0)

    assert sig is not None, "Expected a BUY signal on gap-down"
    assert sig["signal"] == "BUY"
    assert sig["gap_pct"] < -0.3
    assert sig["prev_close"] == 100.0
    assert sig["gap_fill_target"] == 100.0


def test_no_signal_gap_too_small():
    """No signal when gap is smaller than min_gap_pct."""
    cfg   = _default_cfg(min_gap_pct=0.3)
    strat = GapFillStrategy(cfg)
    strat.reset_session()
    strat.set_prev_close(100.0)

    # Gap down only 0.1%: open = 99.90
    bars = [_bar(time(9, 30), close=99.92, open_=99.90)]
    df   = _df(bars)

    strat.reset_session()
    strat.set_prev_close(100.0)
    sig = strat.generate_signal(df, 0)
    assert sig is None, "Gap too small — should not signal"


def test_no_signal_gap_up():
    """No signal on gap-up (long-only strategy skips gap-up days)."""
    cfg   = _default_cfg(min_gap_pct=0.3)
    strat = GapFillStrategy(cfg)
    strat.reset_session()
    strat.set_prev_close(100.0)

    # Gap UP: open = 100.50
    bars = [_bar(time(9, 30), close=100.55, open_=100.50)]
    df   = _df(bars)

    strat.reset_session()
    strat.set_prev_close(100.0)
    sig = strat.generate_signal(df, 0)
    assert sig is None, "Gap-up should not generate a BUY signal"


def test_no_signal_without_prev_close():
    """No signal when prev_close has not been set (e.g. first trading day)."""
    cfg   = _default_cfg()
    strat = GapFillStrategy(cfg)
    strat.reset_session()
    # Deliberately do NOT call set_prev_close

    bars = [_bar(time(9, 30), close=99.60, open_=99.60)]
    df   = _df(bars)
    assert strat.generate_signal(df, 0) is None


def test_no_signal_after_cutoff():
    """No signal when bar is at or after 10:00 ET."""
    cfg   = _default_cfg()
    strat = GapFillStrategy(cfg)
    strat.reset_session()
    strat.set_prev_close(100.0)

    bars = [_bar(time(10, 0), close=99.60, open_=99.60)]
    df   = _df(bars)
    assert strat.generate_signal(df, 0) is None


def test_only_one_signal_per_session():
    """Only one BUY signal fires per day even if multiple qualifying bars exist."""
    cfg   = _default_cfg()
    strat = GapFillStrategy(cfg)
    strat.reset_session()
    strat.set_prev_close(100.0)

    bars = [
        _bar(time(9, 30), close=99.60, open_=99.60),
        _bar(time(9, 31), close=99.55, open_=99.55),
        _bar(time(9, 32), close=99.50, open_=99.50),
    ]
    df      = _df(bars)
    signals = []
    for i in range(len(df)):
        s = strat.generate_signal(df, i)
        if s:
            signals.append(s)

    assert len(signals) == 1
