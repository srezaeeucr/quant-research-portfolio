"""
Tests for TradeLogger.
Run with:  pytest tests/test_trade_log.py -v
"""
import sys
import tempfile
from pathlib import Path
from datetime import datetime

import pytz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.logger.trade_log import TradeLogger

_ET = pytz.timezone("America/New_York")


def _trade(symbol: str = "SPY", pnl: float = 1.0,
           entry_date: str = "2024-01-15") -> dict:
    return {
        "order_id":         "test-order-id-001",
        "symbol":           symbol,
        "shares":           0.61,
        "entry_price":      659.95,
        "exit_price":       666.55,
        "pnl":              pnl,
        "pnl_pct":          round(pnl / (0.61 * 659.95) * 100, 4),
        "reason":           "target_hit" if pnl > 0 else "stop_loss",
        "entry_time":       _ET.localize(datetime.strptime(
                                f"{entry_date} 09:45", "%Y-%m-%d %H:%M")),
        "exit_time":        _ET.localize(datetime.strptime(
                                f"{entry_date} 09:55", "%Y-%m-%d %H:%M")),
        "duration_minutes": 10,
        "mode":             "backtest",
    }


@pytest.fixture
def logger_tmp():
    """TradeLogger backed by a fresh temp database for each test."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = {"logging": {"db_path": str(Path(tmp) / "trades.db")}}
        yield TradeLogger(cfg)


# ------------------------------------------------------------------
# log_trade / get_all_trades
# ------------------------------------------------------------------

def test_log_and_retrieve_trade(logger_tmp):
    logger_tmp.log_trade(_trade())
    df = logger_tmp.get_all_trades()
    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "SPY"
    assert df.iloc[0]["pnl"] == pytest.approx(1.0)


def test_all_columns_present(logger_tmp):
    logger_tmp.log_trade(_trade())
    df = logger_tmp.get_all_trades()
    required = {"order_id", "symbol", "shares", "entry_price", "exit_price",
                "pnl", "pnl_pct", "reason", "entry_time", "exit_time",
                "duration_minutes", "mode", "created_at"}
    assert required.issubset(set(df.columns))


def test_multiple_trades_inserted(logger_tmp):
    for i in range(3):
        t = _trade(pnl=float(i + 1))
        t["order_id"] = f"order-{i}"
        logger_tmp.log_trade(t)
    df = logger_tmp.get_all_trades()
    assert len(df) == 3


def test_get_all_trades_empty(logger_tmp):
    df = logger_tmp.get_all_trades()
    assert df.empty


# ------------------------------------------------------------------
# get_daily_summary
# ------------------------------------------------------------------

def test_daily_summary_win_rate(logger_tmp):
    """2 wins, 1 loss → win_rate = 66.67%"""
    logger_tmp.log_trade(_trade(pnl=1.0))
    t2 = _trade(pnl=2.0); t2["order_id"] = "o2"
    logger_tmp.log_trade(t2)
    t3 = _trade(pnl=-0.5); t3["order_id"] = "o3"
    logger_tmp.log_trade(t3)

    summary = logger_tmp.get_daily_summary("2024-01-15")
    assert summary["total_trades"] == 3
    assert summary["wins"]         == 2
    assert summary["losses"]       == 1
    assert summary["win_rate"]     == pytest.approx(66.67)


def test_daily_summary_total_pnl(logger_tmp):
    logger_tmp.log_trade(_trade(pnl=1.5))
    t2 = _trade(pnl=-0.5); t2["order_id"] = "o2"
    logger_tmp.log_trade(t2)
    summary = logger_tmp.get_daily_summary("2024-01-15")
    assert summary["total_pnl"] == pytest.approx(1.0)


def test_daily_summary_no_trades(logger_tmp):
    summary = logger_tmp.get_daily_summary("2024-01-15")
    assert summary["total_trades"] == 0
    assert summary["win_rate"]     == 0.0


def test_daily_summary_filters_by_date(logger_tmp):
    """Trades on different dates should not bleed into each other's summary."""
    logger_tmp.log_trade(_trade(entry_date="2024-01-15", pnl=1.0))
    t2 = _trade(entry_date="2024-01-16", pnl=2.0)
    t2["order_id"] = "o2"
    logger_tmp.log_trade(t2)

    s1 = logger_tmp.get_daily_summary("2024-01-15")
    s2 = logger_tmp.get_daily_summary("2024-01-16")
    assert s1["total_trades"] == 1
    assert s2["total_trades"] == 1


def test_clear_removes_all_trades(logger_tmp):
    """clear() must delete all rows so a fresh run cannot see stale data."""
    for i in range(3):
        t = _trade(pnl=float(i + 1))
        t["order_id"] = f"order-{i}"
        logger_tmp.log_trade(t)

    assert len(logger_tmp.get_all_trades()) == 3
    logger_tmp.clear()
    assert logger_tmp.get_all_trades().empty
    assert logger_tmp.get_daily_summary("2024-01-15")["total_trades"] == 0
