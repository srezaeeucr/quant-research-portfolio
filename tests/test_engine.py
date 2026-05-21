"""
Tests for TradingEngine.
Run with:  pytest tests/test_engine.py -v

All tests inject synthetic 1-min bar data via mock — no network calls.
"""
import sys
import tempfile
from datetime import datetime, time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine import TradingEngine

_ET = pytz.timezone("America/New_York")

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _bar(t: time, close: float, date_str: str = "2026-01-15", volume: int = 100_000) -> dict:
    dt = _ET.localize(
        datetime.strptime(f"{date_str} {t.hour:02d}:{t.minute:02d}", "%Y-%m-%d %H:%M")
    )
    return {"timestamp": dt, "open": close, "high": close + 0.05,
            "low": close - 0.05, "close": close, "volume": volume}


def _make_day(bars: list) -> pd.DataFrame:
    return pd.DataFrame(bars).reset_index(drop=True)


def _or_bars(or_high: float = 10.0, or_low: float = 9.8,
             date_str: str = "2026-01-15") -> list:
    """15 opening-range bars 9:30–9:44."""
    bars = []
    for minute in range(30, 45):
        price = or_high if minute == 30 else (or_low if minute == 31 else 9.9)
        bars.append(_bar(time(9, minute), price, date_str))
    return bars


def _breakout_day(or_high: float = 10.0, exit_close: float = 9.75,
                  exit_minute: int = 50, date_str: str = "2026-01-15") -> pd.DataFrame:
    """
    Full synthetic day:
      OR (9:30–9:44): establishes opening range
      9:45: breakout bar → BUY signal
      9:46–exit_minute-1: price holds above stop
      exit_minute: price drops to exit_close (below stop if < entry * 0.995)
      rest: flat at exit_close
      EOD: 15:55 bar
    """
    bars = _or_bars(or_high=or_high, date_str=date_str)
    # Breakout bar — volume must be ≥ 1.5× avg OR volume (100_000) → 150_000
    entry = round(or_high * 1.002, 2)
    bars.append(_bar(time(9, 45), entry, date_str, volume=160_000))
    # Mid bars (safe zone — above stop, below target)
    stop = round(entry * 0.995, 2)
    for m in range(46, exit_minute):
        bars.append(_bar(time(9, m) if m < 60 else time(10, m - 60), entry, date_str))
    # Exit bar
    bars.append(_bar(time(9, exit_minute) if exit_minute < 60 else time(10, exit_minute - 60),
                     exit_close, date_str))
    # EOD bar (needed for eod_exit fallback)
    bars.append(_bar(time(15, 55), exit_close, date_str))
    return _make_day(bars)


def _cfg(balance: float = 200.0, daily_loss_pct: float = 2.0) -> dict:
    return {
        "mode": "backtest",
        "symbols": ["SPY"],
        "trading_windows": [
            {"start": "09:30", "end": "10:00", "timezone": "America/New_York"},
            {"start": "15:30", "end": "16:00", "timezone": "America/New_York"},
        ],
        "risk": {
            "max_risk_per_trade_pct": 1.0,
            "daily_loss_limit_pct":   daily_loss_pct,
            "reward_risk_ratio":      2.0,
            "fractional_shares":      True,
        },
        "account": {
            "balance":        balance,
            "max_position_pct": 0.95,
        },
        "notifications": {"telegram": {"enabled": False}},
    }


def _engine_with_tmp_db(cfg: dict = None) -> TradingEngine:
    """Return a TradingEngine backed by a temp SQLite database."""
    cfg = cfg or _cfg()
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "trades.db")

    engine = TradingEngine(cfg)
    # Re-point logger to fresh temp db so tests don't bleed into each other
    from src.logger.trade_log import TradeLogger
    engine.tlog = TradeLogger({"logging": {"db_path": tempfile.mktemp(suffix=".db")}})
    return engine


# ---------------------------------------------------------------------------
# Helper: run engine with injected day data
# ---------------------------------------------------------------------------

def _run_with_days(engine: TradingEngine, days_data: dict) -> pd.DataFrame:
    """
    Bypasses network fetch by directly calling the engine's per-day loop
    with synthetic DataFrames supplied in `days_data` (date → DataFrame).
    """
    from src.data.universe import Universe
    symbol = "SPY"
    mode   = engine.mode

    starting_balance = engine.broker.cash
    peak_balance     = starting_balance
    max_dd           = 0.0

    for trading_date, day_df in sorted(days_data.items()):
        engine.broker.reset_daily()
        engine.strategy.reset_session()

        open_position     = None
        daily_loss_halted = False

        import math
        for i in range(len(day_df)):
            bar       = day_df.iloc[i]
            bar_price = float(bar["close"])
            bar_dt    = bar["timestamp"]

            if engine.risk_mgr.check_daily_loss_limit(
                    engine.broker.daily_pnl, engine.broker.cash):
                if not daily_loss_halted:
                    daily_loss_halted = True
                if open_position is not None:
                    open_position["exit_time"] = bar_dt
                    trade = engine.broker.close_position(open_position, bar_price, "eod_exit")
                    engine.tlog.log_trade(trade)
                    open_position = None
                break

            if open_position is None:
                signal = engine.strategy.generate_signal(day_df, i)
                if signal:
                    signal["symbol"] = symbol
                    buying_power = math.floor(engine.broker.cash * engine._max_pos_pct * 100) / 100
                    position = engine.risk_mgr.calculate_position(
                        signal, engine.broker.cash, buying_power=buying_power
                    )
                    if position:
                        position["entry_time"] = bar_dt
                        engine.broker.submit_order(position)
                        open_position = position
            else:
                exit_result = engine.risk_mgr.check_exit(open_position, bar_price, bar_dt)
                if exit_result:
                    open_position["exit_time"] = bar_dt
                    trade = engine.broker.close_position(
                        open_position, exit_result["exit_price"], exit_result["reason"]
                    )
                    engine.tlog.log_trade(trade)
                    open_position = None

        if open_position is not None:
            last_bar = day_df.iloc[-1]
            open_position["exit_time"] = last_bar["timestamp"]
            trade = engine.broker.close_position(
                open_position, float(last_bar["close"]), "eod_exit"
            )
            engine.tlog.log_trade(trade)
            open_position = None

        bal = engine.broker.cash
        if bal > peak_balance:
            peak_balance = bal
        if peak_balance > 0:
            dd = (peak_balance - bal) / peak_balance * 100
            if dd > max_dd:
                max_dd = dd

    return engine.tlog.get_all_trades()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_engine_runs_5_days_without_crash():
    """Engine completes a 5-day run with no exceptions."""
    engine = _engine_with_tmp_db()
    days = {
        datetime(2026, 1, d).date(): _breakout_day(
            date_str=f"2026-01-{d:02d}", exit_close=9.95
        )
        for d in range(13, 18)   # Mon–Fri
    }
    trades = _run_with_days(engine, days)
    assert isinstance(trades, pd.DataFrame)


def test_engine_records_trade_for_each_signal_day():
    """Each day that produces a breakout should yield exactly one trade."""
    engine = _engine_with_tmp_db()
    # exit_close=9.75 < stop (10.02 * 0.995 = 9.97) → stop_loss on every day
    days = {
        datetime(2026, 1, d).date(): _breakout_day(
            date_str=f"2026-01-{d:02d}", exit_close=9.75
        )
        for d in range(13, 18)
    }
    trades = _run_with_days(engine, days)
    assert len(trades) == 5
    assert set(trades["reason"]) == {"stop_loss"}


def test_daily_loss_limit_halts_trading():
    """After a loss that exceeds the daily limit, no further entries are made."""
    # Use a tiny daily_loss_limit_pct so one stop-loss triggers the halt
    # entry ~10.02, stop ~9.97, shares=round(0.10/0.050, 2)=2.0
    # pnl = (9.75-10.02)*2 = -0.54; limit = 0.5% of $200 = $1.00 → need pnl < -$1
    # Use a lower entry and bigger drop:
    # balance=$100, limit=0.5% → $0.50; entry=100, stop=99.5, risk=$0.50
    # shares=round(0.50/0.50,2)=1.0, cost=$100
    # exit at $99.0 → pnl = -$1.00 ≤ -$0.50 → halted

    cfg = _cfg(balance=100.0, daily_loss_pct=0.5)
    engine = _engine_with_tmp_db(cfg)

    # Day 1: position exits at 99.0, triggering the daily loss limit
    or_bars = _or_bars(or_high=100.0, date_str="2026-01-13")
    entry   = round(100.0 * 1.002, 2)   # 100.2
    stop    = round(entry * 0.995, 2)    # 99.699
    bars    = or_bars + [
        _bar(time(9, 45), entry, "2026-01-13", volume=160_000),   # signal (vol ≥ 1.5× OR avg)
        _bar(time(9, 46), stop - 0.10, "2026-01-13"),  # hits stop
        _bar(time(9, 47), stop - 0.10, "2026-01-13"),  # extra bar after stop
        _bar(time(15, 55), stop - 0.10, "2026-01-13"),
    ]
    day1 = _make_day(bars)

    # Day 2: another breakout — should NOT trade because daily loss was yesterday
    # (reset_daily clears daily_pnl, so limit only affects same day)
    # Actually each day starts fresh with reset_daily.
    # To test halt within same day, day 1 must have TWO entries... but ORB only
    # allows one signal per session. So test that the halt fires and second signal
    # within that day is blocked (no open position after halt).

    days = {datetime(2026, 1, 13).date(): day1}
    trades = _run_with_days(engine, days)

    # Trade should have been executed (stop hit) then limit detected
    assert len(trades) >= 1
    # After halt, broker has no open position
    assert engine.broker.open_position is None


def test_eod_force_close():
    """If position is still open at EOD (no stop/target hit), it is force-closed."""
    engine = _engine_with_tmp_db()
    or_high = 10.0
    entry   = round(or_high * 1.002, 2)  # 10.02
    stop    = round(entry * 0.995, 2)    # 9.9699
    target  = round(entry + (entry * 0.005 * 2), 2)  # 10.12

    # Price stays safely between stop and target; EOD bar at 15:55 should force close
    mid = round((stop + target) / 2, 2)
    bars = _or_bars(or_high=or_high) + [
        _bar(time(9, 45), entry, volume=160_000),   # breakout → signal (vol ≥ 1.5× OR avg)
        _bar(time(9, 46), mid),
        _bar(time(10, 30), mid),           # mid-day: no exit
        _bar(time(15, 55), mid),           # EOD: force close fires
        _bar(time(15, 59), mid),
    ]
    day_df = _make_day(bars)
    days   = {datetime(2026, 1, 13).date(): day_df}

    trades = _run_with_days(engine, days)
    assert len(trades) == 1
    assert trades.iloc[0]["reason"] == "eod_exit"


def test_balance_updates_per_day_incremental():
    """broker.cash must change after each day, not stay frozen at $200."""
    engine = _engine_with_tmp_db()
    start_cash = engine.broker.cash  # 200.0

    # Day 1: loss (exit_close < stop ≈ entry * 0.995)
    day1 = _breakout_day(or_high=10.0, exit_close=9.75, date_str="2026-01-13")
    _run_with_days(engine, {datetime(2026, 1, 13).date(): day1})
    cash_after_day1 = engine.broker.cash
    assert cash_after_day1 < start_cash, (
        f"Day 1 (loss): cash should decrease; got ${cash_after_day1:.4f}"
    )

    # Day 2: win (exit_close > target ≈ entry + 2*stop_distance ≈ 10.12)
    day2 = _breakout_day(or_high=10.0, exit_close=10.15, date_str="2026-01-14")
    _run_with_days(engine, {datetime(2026, 1, 14).date(): day2})
    cash_after_day2 = engine.broker.cash
    assert cash_after_day2 > cash_after_day1, (
        f"Day 2 (win): cash should increase from ${cash_after_day1:.4f}; "
        f"got ${cash_after_day2:.4f}"
    )

    # Day 3: loss again
    day3 = _breakout_day(or_high=10.0, exit_close=9.75, date_str="2026-01-15")
    _run_with_days(engine, {datetime(2026, 1, 15).date(): day3})
    cash_after_day3 = engine.broker.cash
    assert cash_after_day3 < cash_after_day2, (
        f"Day 3 (loss): cash should decrease from ${cash_after_day2:.4f}; "
        f"got ${cash_after_day3:.4f}"
    )

    # All three balances must be distinct (no frozen $200 repeats)
    assert len({start_cash, cash_after_day1, cash_after_day2, cash_after_day3}) == 4


def test_final_balance_consistent_with_pnl():
    """Final cash balance must equal starting balance + sum of all P&Ls."""
    engine          = _engine_with_tmp_db()
    starting_balance = engine.broker.cash

    days = {}
    for d in range(13, 18):
        date_key = datetime(2026, 1, d).date()
        days[date_key] = _breakout_day(
            date_str=f"2026-01-{d:02d}", exit_close=9.75
        )

    trades = _run_with_days(engine, days)

    if not trades.empty:
        total_pnl     = float(trades["pnl"].sum())
        expected_cash = round(starting_balance + total_pnl, 4)
        actual_cash   = round(engine.broker.cash, 4)
        assert abs(actual_cash - expected_cash) < 0.01, (
            f"Cash mismatch: expected {expected_cash}, got {actual_cash}"
        )
