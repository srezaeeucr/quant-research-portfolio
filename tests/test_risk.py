"""
Tests for RiskManager.
Run with:  pytest tests/test_risk.py -v
"""
import sys
from pathlib import Path
from datetime import datetime

import pytz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.risk.manager import RiskManager

_ET = pytz.timezone("America/New_York")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return {
        "risk": {
            "max_risk_per_trade_pct": 1.0,
            "daily_loss_limit_pct":   2.0,
            "reward_risk_ratio":      2.0,
        },
        "account": {"balance": 200.0},
    }


@pytest.fixture
def rm(config):
    return RiskManager(config)


def _signal(entry: float = 659.95, symbol: str = "SPY") -> dict:
    return {"symbol": symbol, "entry_price": entry, "signal": "BUY"}


# ---------------------------------------------------------------------------
# calculate_position
# ---------------------------------------------------------------------------

def test_position_shares_and_levels(rm):
    """$200 balance, entry $659.95 — risk-based fractional sizing, no cap."""
    # risk=$2.00, stop_distance=$659.95×0.005=$3.29975 → shares=round(2.00/3.29975,2)=0.61
    pos = rm.calculate_position(_signal(659.95), 200.0)
    assert pos is not None, "Expected a valid fractional position"
    assert pos["shares"] == 0.61
    assert pos["fractional"] is True
    # risk_amount = stop_distance × shares = 3.29975 × 0.61 = 2.0128 (actual dollars at risk)
    assert abs(pos["risk_amount"] - 2.01) < 0.01
    assert abs(pos["reward"] - pos["risk_amount"] * 2.0) < 1e-4   # reward is always exactly 2× actual risk
    assert pos["stop_loss"]  == round(659.95 - 659.95 * 0.005, 4)
    assert pos["target"]     == round(659.95 + 659.95 * 0.005 * 2.0, 4)


def test_position_cheap_entry(rm):
    """$10 entry, $200 balance — risk-based sizing, no cap."""
    entry = 10.0
    pos   = rm.calculate_position(_signal(entry), 200.0)
    assert pos is not None, "Expected a valid position for cheap entry"

    stop_distance = entry * 0.005   # $0.05
    # shares = round(2.00/0.05, 2) = 40.0 (whole number → fractional=False)
    assert pos["shares"] == 40.0
    assert pos["fractional"] is False
    assert abs(pos["stop_loss"] - (entry - stop_distance)) < 1e-6
    assert abs(pos["target"]   - (entry + stop_distance * 2.0)) < 1e-6
    assert pos["symbol"] == "SPY"


def test_position_returns_none_when_too_small(rm):
    """Shares < 0.01 is the only condition that returns None."""
    # balance=$0.10 → risk=$0.001; entry=$100 → stop=$0.50
    # shares = round(0.001/0.50, 2) = 0.0 < 0.01 → None
    pos = rm.calculate_position(_signal(100.0), 0.10)
    assert pos is None


def test_position_cost_reflects_full_risk_sizing(rm):
    """position_cost = shares × entry; no artificial cap applied."""
    # entry=$1, stop=$0.005, shares=round(2.00/0.005,2)=400.0, cost=$400
    pos = rm.calculate_position(_signal(1.0), 200.0)
    assert pos is not None
    assert pos["shares"] == 400.0
    assert abs(pos["position_cost"] - 400.0) < 1e-4


def test_position_whole_shares_fractional_false(rm):
    """fractional=False when shares is a whole number."""
    pos = rm.calculate_position(_signal(10.0), 200.0)   # shares=40.0
    assert pos is not None
    assert pos["fractional"] is False


def test_shares_rounds_to_two_decimal_places(rm):
    """Shares must be rounded to exactly 2 decimal places."""
    # entry=$7.77 → stop=$0.03885 → shares=round(2.00/0.03885,2)=51.48
    pos = rm.calculate_position(_signal(7.77), 200.0)
    assert pos is not None
    shares = pos["shares"]
    assert shares == round(shares, 2), "Shares must have at most 2 decimal places"
    assert isinstance(shares, float)


def test_skip_only_when_shares_below_0_01(rm):
    """Any result >= 0.01 shares should produce a position, not None."""
    # entry=$1.00 → shares=400.0 → accepted
    pos = rm.calculate_position(_signal(1.0), 200.0)
    assert pos is not None, "Should accept positions with shares >= 0.01"

    # True skip: balance=$0.10, entry=$100 → risk=$0.001, stop=$0.50 → shares=0.0 < 0.01
    pos_skip = rm.calculate_position(_signal(100.0), 0.10)
    assert pos_skip is None, "Should skip when shares rounds to < 0.01"


def test_fractional_flag_true_for_non_whole_shares(rm):
    """fractional key is True when shares is not a whole number."""
    pos = rm.calculate_position(_signal(659.95), 200.0)
    assert pos is not None
    assert pos["fractional"] is True


def test_fractional_flag_false_for_whole_shares(rm):
    """fractional key is False when shares is a whole number after rounding."""
    # entry=$10, capped to 4.0 shares — whole number
    pos = rm.calculate_position(_signal(10.0), 200.0)
    assert pos is not None
    assert pos["fractional"] is False


def test_position_reward_ratio(rm):
    """reward should equal 2× risk_amount (for 2:1 R/R and non-clamped position)."""
    entry = 10.0
    pos   = rm.calculate_position(_signal(entry), 200.0)
    assert pos is not None
    # reward = stop_distance * 2.0 * shares
    stop_distance = entry * 0.005
    expected_reward = round(stop_distance * 2.0 * pos["shares"], 4)
    assert abs(pos["reward"] - expected_reward) < 1e-4


# ---------------------------------------------------------------------------
# check_daily_loss_limit
# ---------------------------------------------------------------------------

def test_daily_loss_limit_triggered(rm):
    """Exactly 2% loss on $200 = $4.00 → should halt."""
    assert rm.check_daily_loss_limit(-4.00, 200.0) is True


def test_daily_loss_limit_not_triggered(rm):
    """$3.99 loss is just under 2% of $200 → should not halt."""
    assert rm.check_daily_loss_limit(-3.99, 200.0) is False


def test_daily_loss_limit_profit(rm):
    """Positive P&L never triggers the limit."""
    assert rm.check_daily_loss_limit(10.0, 200.0) is False


# ---------------------------------------------------------------------------
# check_exit
# ---------------------------------------------------------------------------

def _pos(entry: float = 10.0) -> dict:
    stop_distance = entry * 0.005
    return {
        "symbol":      "SPY",
        "shares":      10,
        "entry_price": entry,
        "stop_loss":   round(entry - stop_distance, 4),
        "target":      round(entry + stop_distance * 2.0, 4),
    }


def _dt(h: int, m: int) -> datetime:
    return _ET.localize(datetime(2024, 1, 15, h, m))


def test_exit_stop_loss(rm):
    pos = _pos(10.0)
    result = rm.check_exit(pos, pos["stop_loss"] - 0.01, _dt(10, 0))
    assert result is not None
    assert result["reason"] == "stop_loss"
    assert result["action"] == "CLOSE"


def test_exit_target_hit(rm):
    pos = _pos(10.0)
    result = rm.check_exit(pos, pos["target"] + 0.01, _dt(10, 0))
    assert result is not None
    assert result["reason"] == "target_hit"


def test_exit_eod(rm):
    pos = _pos(10.0)
    mid_price = (pos["stop_loss"] + pos["target"]) / 2
    result = rm.check_exit(pos, mid_price, _dt(15, 55))
    assert result is not None
    assert result["reason"] == "eod_exit"


def test_no_exit_mid_trade(rm):
    pos = _pos(10.0)
    mid_price = (pos["stop_loss"] + pos["target"]) / 2
    result = rm.check_exit(pos, mid_price, _dt(10, 0))
    assert result is None


def test_exit_price_recorded(rm):
    pos = _pos(10.0)
    exit_price = pos["stop_loss"] - 0.05
    result = rm.check_exit(pos, exit_price, _dt(10, 0))
    assert result["exit_price"] == exit_price


# ---------------------------------------------------------------------------
# Buying power guard
# ---------------------------------------------------------------------------

def test_buying_power_clamps_position_cost(rm):
    """With buying_power=$190, position_cost must not exceed $190."""
    # entry=$659.95 → unclamped cost = 0.61 × $659.95 = $402.57 > $190
    pos = rm.calculate_position(_signal(659.95), 200.0, buying_power=190.0)
    assert pos is not None
    assert pos["position_cost"] <= 190.0 + 1e-4
    assert pos["buying_power_clamped"] is True


def test_buying_power_clamp_preserves_rr_ratio(rm):
    """After buying power clamp, reward should still be ~2× risk_amount."""
    pos = rm.calculate_position(_signal(659.95), 200.0, buying_power=190.0)
    assert pos is not None
    # reward = stop_distance × reward_risk_ratio × shares = 2 × actual_risk by construction
    assert abs(pos["reward"] - pos["risk_amount"] * 2.0) < 1e-4


def test_no_buying_power_clamp_when_not_needed(rm):
    """buying_power_clamped=False when cost already fits within buying_power."""
    # entry=$10, shares=40, cost=$400; buying_power=$500 → no clamp
    pos = rm.calculate_position(_signal(10.0), 200.0, buying_power=500.0)
    assert pos is not None
    assert pos["buying_power_clamped"] is False


def test_buying_power_clamp_returns_none_when_too_small(rm):
    """If buying_power is so small shares drop below 0.01, return None."""
    # buying_power=$0.05, entry=$659.95 → shares=round(0.05/659.95,2)=0.0 → None
    pos = rm.calculate_position(_signal(659.95), 200.0, buying_power=0.05)
    assert pos is None
