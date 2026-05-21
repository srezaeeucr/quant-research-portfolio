"""
Tests for ATR helper + ATR-based dynamic stops in RiskManager.
Run with:  pytest tests/test_atr.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.risk.atr import calculate_atr
from src.risk.manager import RiskManager


# ---------------------------------------------------------------------------
# ATR helper unit tests
# ---------------------------------------------------------------------------

def test_atr_synthetic_constant_range():
    """ATR on bars with high-low=1.0 every bar should be ~1.0 (true range = max
    of high-low and abs differences with prev close — when consecutive closes
    are identical, true range = high - low = 1.0)."""
    bars = pd.DataFrame({
        "high":  [101.0] * 20,
        "low":   [100.0] * 20,
        "close": [100.5] * 20,
    })
    atr = calculate_atr(bars, period=14)
    assert atr is not None
    # Each bar's TR = max(1.0, |101 - 100.5|, |100 - 100.5|) = max(1.0, 0.5, 0.5) = 1.0
    assert atr == pytest.approx(1.0, abs=1e-6)


def test_atr_with_gaps():
    """When close gaps, true range includes the gap."""
    bars = pd.DataFrame({
        "high":  [10.0, 15.0, 12.0, 14.0, 13.0],
        "low":   [9.0,  14.0, 11.0, 13.0, 12.0],
        "close": [9.5,  14.5, 11.5, 13.5, 12.5],
    })
    atr = calculate_atr(bars, period=4)
    # TR[1] = max(1.0, |15-9.5|=5.5, |14-9.5|=4.5) = 5.5
    # TR[2] = max(1.0, |12-14.5|=2.5, |11-14.5|=3.5) = 3.5
    # TR[3] = max(1.0, |14-11.5|=2.5, |13-11.5|=1.5) = 2.5
    # TR[4] = max(1.0, |13-13.5|=0.5, |12-13.5|=1.5) = 1.5
    # ATR(4) at last row = (5.5 + 3.5 + 2.5 + 1.5) / 4 = 3.25
    assert atr == pytest.approx(3.25, abs=1e-6)


def test_atr_returns_none_when_too_few_bars():
    bars = pd.DataFrame({"high": [1.0]*5, "low": [0.5]*5, "close": [0.75]*5})
    assert calculate_atr(bars, period=14) is None


def test_atr_returns_none_when_columns_missing():
    bars = pd.DataFrame({"open": [1.0]*20, "close": [1.0]*20})
    assert calculate_atr(bars, period=14) is None


def test_atr_returns_none_when_df_is_none():
    assert calculate_atr(None, period=14) is None


# ---------------------------------------------------------------------------
# RiskManager ATR integration
# ---------------------------------------------------------------------------

def _bars_around(price: float, range_pct: float = 1.0, n: int = 20) -> pd.DataFrame:
    """Build n synthetic 1-min bars centered on `price` with high-low spread
    equal to `range_pct`% of price."""
    half = price * range_pct / 100.0 / 2.0
    return pd.DataFrame({
        "open":   [price] * n,
        "high":   [price + half] * n,
        "low":    [price - half] * n,
        "close":  [price] * n,
        "volume": [100_000] * n,
    })


def _cfg_with_atr(use=True, mult=1.5, floor=0.3, ceiling=1.5,
                   stop_loss_pct=0.5, rr=2.0):
    return {
        "risk": {
            "use_atr_stops":     use,
            "atr_period":        14,
            "atr_multiplier":    mult,
            "stop_floor_pct":    floor,
            "stop_ceiling_pct":  ceiling,
            "stop_loss_pct":     stop_loss_pct,
            "reward_risk_ratio": rr,
            "max_risk_per_trade_pct": 1.0,
        },
        "account": {"balance": 200.0, "max_position_pct": 0.95},
    }


def test_atr_stop_used_when_bars_provided():
    rm = RiskManager(_cfg_with_atr())
    bars = _bars_around(100.0, range_pct=1.0)   # 1.0% range → ATR ≈ 1.0
    sig = {"entry_price": 100.0, "symbol": "TEST"}
    pos = rm.calculate_position(sig, account_balance=10_000.0, bars_df=bars)
    assert pos is not None
    assert pos["atr_stop_used"] is True
    assert pos["atr_value"] is not None
    # ATR ~ 1.0, multiplier 1.5 → stop_distance ~ 1.5 (1.5% of 100)
    # 1.5% is at the ceiling, so should equal 1.5
    assert pos["stop_distance_pct"] == pytest.approx(1.5, abs=0.05)


def test_atr_stop_floor_clamps_low_volatility():
    rm = RiskManager(_cfg_with_atr(floor=0.3, ceiling=1.5))
    # Very tight range — ATR will be tiny
    bars = _bars_around(100.0, range_pct=0.05)  # ATR ~ 0.05 → 0.05*1.5 = 0.075
    sig = {"entry_price": 100.0, "symbol": "TEST"}
    pos = rm.calculate_position(sig, account_balance=10_000.0, bars_df=bars)
    assert pos is not None
    assert pos["atr_stop_used"] is True
    # Should have been clamped to the 0.3% floor
    assert pos["stop_distance_pct"] == pytest.approx(0.3, abs=0.01)


def test_atr_stop_ceiling_clamps_high_volatility():
    rm = RiskManager(_cfg_with_atr(floor=0.3, ceiling=1.5))
    # Wide range — ATR will be large
    bars = _bars_around(100.0, range_pct=5.0)   # ATR ~ 5.0 → 5.0*1.5 = 7.5
    sig = {"entry_price": 100.0, "symbol": "TEST"}
    pos = rm.calculate_position(sig, account_balance=10_000.0, bars_df=bars)
    assert pos is not None
    assert pos["atr_stop_used"] is True
    # Should have been clamped to the 1.5% ceiling
    assert pos["stop_distance_pct"] == pytest.approx(1.5, abs=0.01)


def test_falls_back_to_fixed_when_bars_df_missing():
    rm = RiskManager(_cfg_with_atr(use=True, stop_loss_pct=0.5))
    sig = {"entry_price": 100.0, "symbol": "TEST"}
    pos = rm.calculate_position(sig, account_balance=10_000.0, bars_df=None)
    assert pos is not None
    assert pos["atr_stop_used"] is False
    assert pos["atr_value"] is None
    assert pos["stop_distance_pct"] == pytest.approx(0.5, abs=0.01)


def test_falls_back_to_fixed_when_too_few_bars():
    rm = RiskManager(_cfg_with_atr(use=True, stop_loss_pct=0.5))
    # Only 5 bars — ATR(14) will return None
    bars = _bars_around(100.0, range_pct=1.0, n=5)
    sig = {"entry_price": 100.0, "symbol": "TEST"}
    pos = rm.calculate_position(sig, account_balance=10_000.0, bars_df=bars)
    assert pos is not None
    assert pos["atr_stop_used"] is False
    assert pos["stop_distance_pct"] == pytest.approx(0.5, abs=0.01)


def test_disabled_uses_fixed_even_with_bars():
    rm = RiskManager(_cfg_with_atr(use=False, stop_loss_pct=0.5))
    bars = _bars_around(100.0, range_pct=1.0)
    sig = {"entry_price": 100.0, "symbol": "TEST"}
    pos = rm.calculate_position(sig, account_balance=10_000.0, bars_df=bars)
    assert pos is not None
    assert pos["atr_stop_used"] is False
    assert pos["stop_distance_pct"] == pytest.approx(0.5, abs=0.01)


def test_share_count_responds_to_wider_atr_stop():
    """Wider stop → smaller position (fewer shares for same risk amount)."""
    bars_tight = _bars_around(100.0, range_pct=0.4)  # ATR ~ 0.4 → 0.6 stop
    bars_wide  = _bars_around(100.0, range_pct=1.0)  # ATR ~ 1.0 → ceiling 1.5

    rm = RiskManager(_cfg_with_atr())
    sig = {"entry_price": 100.0, "symbol": "TEST"}
    pos_tight = rm.calculate_position(sig, account_balance=10_000.0, bars_df=bars_tight)
    pos_wide  = rm.calculate_position(sig, account_balance=10_000.0, bars_df=bars_wide)

    assert pos_tight["shares"] > pos_wide["shares"]
