"""
Tests for Notifier.
Run with:  pytest tests/test_notifier.py -v
"""
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.notifications.notifier import Notifier

_ET = pytz.timezone("America/New_York")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _cfg(enabled: bool = True) -> dict:
    return {
        "mode": "backtest",
        "notifications": {"telegram": {"enabled": enabled}},
    }


def _notifier(enabled: bool = True, token: str = "fake-token",
              chat_id: str = "12345") -> Notifier:
    with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": token,
                                   "TELEGRAM_CHAT_ID": chat_id}):
        return Notifier(_cfg(enabled))


def _position() -> dict:
    return {
        "symbol":        "SPY",
        "shares":        0.28,
        "entry_price":   659.95,
        "stop_loss":     656.63,
        "target":        666.27,
        "risk_amount":   0.93,
        "reward":        1.85,
        "position_cost": 184.79,
        "entry_time":    _ET.localize(datetime(2026, 3, 23, 9, 53)),
    }


def _trade_record() -> dict:
    return {
        "symbol":           "SPY",
        "shares":           0.28,
        "entry_price":      659.95,
        "exit_price":       656.63,
        "pnl":              -0.93,
        "pnl_pct":          -0.50,
        "reason":           "stop_loss",
        "entry_time":       _ET.localize(datetime(2026, 3, 23, 9, 53)),
        "exit_time":        _ET.localize(datetime(2026, 3, 23, 11, 56)),
        "duration_minutes": 123,
        "mode":             "backtest",
    }


def _summary() -> dict:
    return {
        "date":         "2026-03-23",
        "total_trades": 1,
        "wins":         0,
        "losses":       1,
        "total_pnl":    -0.93,
        "win_rate":     0.0,
    }


# ---------------------------------------------------------------------------
# enabled=False — no HTTP calls
# ---------------------------------------------------------------------------

def test_disabled_no_request_trade_opened():
    n = _notifier(enabled=False)
    with patch("requests.post") as mock_post:
        n.notify_trade_opened(_position())
        mock_post.assert_not_called()


def test_disabled_no_request_trade_closed():
    n = _notifier(enabled=False)
    with patch("requests.post") as mock_post:
        n.notify_trade_closed(_trade_record(), 199.07)
        mock_post.assert_not_called()


def test_disabled_no_request_daily_summary():
    n = _notifier(enabled=False)
    with patch("requests.post") as mock_post:
        n.notify_daily_summary(_summary(), 199.07)
        mock_post.assert_not_called()


def test_disabled_no_request_halt():
    n = _notifier(enabled=False)
    with patch("requests.post") as mock_post:
        n.notify_halt("Daily loss limit reached", -4.0)
        mock_post.assert_not_called()


def test_disabled_no_request_heartbeat():
    n = _notifier(enabled=False)
    with patch("requests.post") as mock_post:
        n.notify_heartbeat("2026-03-23", 200.0)
        mock_post.assert_not_called()


def test_disabled_no_request_error():
    n = _notifier(enabled=False)
    with patch("requests.post") as mock_post:
        n.notify_error("Something went wrong")
        mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# Missing credentials → silently disabled
# ---------------------------------------------------------------------------

def test_missing_token_disables_notifier():
    with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": "123"}):
        n = Notifier(_cfg(enabled=True))
    assert n.enabled is False


def test_missing_chat_id_disables_notifier():
    with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_CHAT_ID": ""}):
        n = Notifier(_cfg(enabled=True))
    assert n.enabled is False


# ---------------------------------------------------------------------------
# Message content
# ---------------------------------------------------------------------------

def _captured_text(mock_post) -> str:
    """Extract the 'text' field from the first requests.post call."""
    call_kwargs = mock_post.call_args
    return call_kwargs.kwargs.get("json", call_kwargs.args[1] if len(call_kwargs.args) > 1 else {}).get("text", "")


def test_trade_opened_message_contains_symbol():
    n = _notifier()
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        n.notify_trade_opened(_position())
        text = _captured_text(mock_post)
    assert "SPY" in text
    assert "TRADE OPENED" in text
    assert "659.95" in text
    assert "[BACKTEST]" in text


def test_trade_closed_message_contains_pnl():
    n = _notifier()
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        n.notify_trade_closed(_trade_record(), 199.07)
        text = _captured_text(mock_post)
    assert "TRADE CLOSED" in text
    assert "656.63" in text
    assert "Stop Loss Hit" in text
    assert "199.07" in text


def test_daily_summary_message_format():
    n = _notifier()
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        n.notify_daily_summary(_summary(), 199.07)
        text = _captured_text(mock_post)
    assert "DAILY SUMMARY" in text
    assert "2026-03-23" in text
    assert "Win Rate" in text


def test_halt_message_format():
    n = _notifier()
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        n.notify_halt("Daily loss limit reached", -4.0)
        text = _captured_text(mock_post)
    assert "TRADING HALTED" in text
    assert "Daily loss limit reached" in text
    assert "paused" in text


def test_heartbeat_message_format():
    n = _notifier()
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        n.notify_heartbeat("2026-03-23", 200.0)
        text = _captured_text(mock_post)
    assert "HEARTBEAT" in text
    assert "200.00" in text
    assert "Running" in text


def test_error_message_format():
    n = _notifier()
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        n.notify_error("Connection timeout on data feed")
        text = _captured_text(mock_post)
    assert "ERROR" in text
    assert "Connection timeout" in text


def test_mode_tag_paper():
    with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_CHAT_ID": "1"}):
        n = Notifier({"mode": "paper",
                      "notifications": {"telegram": {"enabled": True}}})
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        n.notify_heartbeat("2026-03-23", 200.0)
        text = _captured_text(mock_post)
    assert "[PAPER]" in text


# ---------------------------------------------------------------------------
# API failure does not raise
# ---------------------------------------------------------------------------

def test_api_failure_does_not_raise():
    n = _notifier()
    with patch("requests.post", side_effect=Exception("network error")):
        # Must not raise — failure is swallowed
        n.notify_trade_opened(_position())


def test_api_non_200_does_not_raise():
    n = _notifier()
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=500, text="Internal Server Error")
        n.notify_trade_opened(_position())   # no exception


def test_api_failure_retries_once():
    """On failure, _send should attempt exactly 2 calls before giving up."""
    n = _notifier()
    with patch("requests.post", side_effect=Exception("timeout")) as mock_post, \
         patch("time.sleep"):   # skip the 1-second wait in tests
        n.notify_trade_opened(_position())
        assert mock_post.call_count == 2


def test_parse_mode_is_html():
    """All messages must use parse_mode='HTML'."""
    n = _notifier()
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        n.notify_heartbeat("2026-03-23", 200.0)
        payload = mock_post.call_args.kwargs.get("json", {})
    assert payload.get("parse_mode") == "HTML"
