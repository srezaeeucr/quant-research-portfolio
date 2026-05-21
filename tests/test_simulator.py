"""
Tests for SimulatorBroker.
Run with:  pytest tests/test_simulator.py -v
"""
import sys
from pathlib import Path
from datetime import datetime

import pytz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.broker.simulator import SimulatorBroker

_ET = pytz.timezone("America/New_York")

_CFG = {
    "mode":    "backtest",
    "account": {"balance": 200.0},
}

_REQUIRED_TRADE_FIELDS = {
    "order_id", "symbol", "shares", "entry_price", "exit_price",
    "pnl", "pnl_pct", "reason", "entry_time", "exit_time",
    "duration_minutes", "mode",
}


def _position(entry: float = 100.0, shares: float = 1.0) -> dict:
    cost = round(shares * entry, 4)
    return {
        "symbol":        "SPY",
        "shares":        shares,
        "entry_price":   entry,
        "stop_loss":     round(entry * 0.995, 4),
        "target":        round(entry * 1.010, 4),
        "position_cost": cost,
        "entry_time":    _ET.localize(datetime(2024, 1, 15, 9, 45)),
    }


@pytest.fixture
def broker():
    return SimulatorBroker(_CFG)


# ------------------------------------------------------------------
# submit_order
# ------------------------------------------------------------------

def test_submit_order_returns_order_dict(broker):
    pos   = _position()
    order = broker.submit_order(pos)
    assert isinstance(order, dict)
    assert order["symbol"]      == "SPY"
    assert order["entry_price"] == 100.0


def test_submit_order_deducts_cash(broker):
    pos = _position(entry=100.0, shares=1.0)
    broker.submit_order(pos)
    assert broker.get_account_balance() == pytest.approx(200.0 - 100.0)


def test_submit_order_sets_open_position(broker):
    broker.submit_order(_position())
    assert broker.open_position is not None


def test_submit_order_raises_if_already_open(broker):
    broker.submit_order(_position())
    with pytest.raises(RuntimeError):
        broker.submit_order(_position())


# ------------------------------------------------------------------
# close_position
# ------------------------------------------------------------------

def test_close_position_profit(broker):
    pos   = _position(entry=100.0, shares=1.0)
    order = broker.submit_order(pos)
    exit_pos = {**pos, "exit_time": _ET.localize(datetime(2024, 1, 15, 9, 55))}
    trade = broker.close_position(exit_pos, exit_price=101.0, reason="target_hit")

    assert trade["pnl"] == pytest.approx(1.0)
    assert trade["reason"] == "target_hit"


def test_close_position_loss(broker):
    pos   = _position(entry=100.0, shares=1.0)
    broker.submit_order(pos)
    exit_pos = {**pos, "exit_time": _ET.localize(datetime(2024, 1, 15, 9, 55))}
    trade = broker.close_position(exit_pos, exit_price=99.0, reason="stop_loss")

    assert trade["pnl"] == pytest.approx(-1.0)
    assert trade["reason"] == "stop_loss"


def test_close_position_updates_cash(broker):
    pos   = _position(entry=100.0, shares=2.0)
    broker.submit_order(pos)
    exit_pos = {**pos, "exit_time": _ET.localize(datetime(2024, 1, 15, 9, 55))}
    broker.close_position(exit_pos, exit_price=105.0, reason="target_hit")
    # cash: 200 - 200 (cost) + 210 (proceeds) = 210
    assert broker.get_account_balance() == pytest.approx(210.0)


def test_close_position_clears_open_position(broker):
    pos = _position()
    broker.submit_order(pos)
    exit_pos = {**pos, "exit_time": _ET.localize(datetime(2024, 1, 15, 9, 55))}
    broker.close_position(exit_pos, exit_price=101.0, reason="target_hit")
    assert broker.open_position is None


def test_close_position_pnl_pct(broker):
    pos = _position(entry=100.0, shares=1.0)
    broker.submit_order(pos)
    exit_pos = {**pos, "exit_time": _ET.localize(datetime(2024, 1, 15, 9, 55))}
    trade = broker.close_position(exit_pos, exit_price=102.0, reason="target_hit")
    assert trade["pnl_pct"] == pytest.approx(2.0)


def test_trade_record_has_required_fields(broker):
    pos = _position()
    broker.submit_order(pos)
    exit_pos = {**pos, "exit_time": _ET.localize(datetime(2024, 1, 15, 9, 55))}
    trade = broker.close_position(exit_pos, exit_price=101.0, reason="target_hit")
    missing = _REQUIRED_TRADE_FIELDS - set(trade.keys())
    assert not missing, f"Missing fields: {missing}"


def test_trade_record_duration_minutes(broker):
    pos = _position()
    broker.submit_order(pos)
    exit_pos = {**pos, "exit_time": _ET.localize(datetime(2024, 1, 15, 9, 55))}
    trade = broker.close_position(exit_pos, exit_price=101.0, reason="target_hit")
    assert trade["duration_minutes"] == 10   # 9:45 → 9:55


# ------------------------------------------------------------------
# daily_pnl and reset
# ------------------------------------------------------------------

def test_daily_pnl_accumulates(broker):
    for _ in range(2):
        pos = _position(entry=100.0, shares=1.0)
        broker.submit_order(pos)
        exit_pos = {**pos, "exit_time": _ET.localize(datetime(2024, 1, 15, 9, 55))}
        broker.close_position(exit_pos, exit_price=101.0, reason="target_hit")
        broker.open_position = None   # force clear for second iteration

    assert broker.daily_pnl == pytest.approx(2.0)


def test_reset_daily_clears_pnl(broker):
    pos = _position(entry=100.0, shares=1.0)
    broker.submit_order(pos)
    exit_pos = {**pos, "exit_time": _ET.localize(datetime(2024, 1, 15, 9, 55))}
    broker.close_position(exit_pos, exit_price=101.0, reason="target_hit")
    broker.reset_daily()
    assert broker.daily_pnl == 0.0
