"""
Tests for DataFetcher.fetch_historical_alpaca() and source auto-selection.

All tests mock requests.get — no real HTTP calls are made.
Run with:  pytest tests/test_fetcher_alpaca.py -v
"""
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pandas as pd
import pytest
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fetcher import DataFetcher

_ET = pytz.timezone("America/New_York")

# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

# Two bars inside the morning trading window (09:30 ET = 13:30 UTC)
_BAR_WINDOW_1 = {
    "t": "2025-10-01T13:30:00Z",  # 09:30 ET — inside window
    "o": 580.00, "h": 581.00, "l": 579.50, "c": 580.50, "v": 1_500_000,
}
_BAR_WINDOW_2 = {
    "t": "2025-10-01T13:45:00Z",  # 09:45 ET — inside window
    "o": 580.50, "h": 582.00, "l": 580.00, "c": 581.50, "v": 1_200_000,
}
# One bar outside any trading window (12:00 ET = 16:00 UTC)
_BAR_OUTSIDE = {
    "t": "2025-10-01T16:00:00Z",  # 12:00 ET — outside window
    "o": 582.00, "h": 583.00, "l": 581.50, "c": 582.50, "v": 800_000,
}

_FAKE_ENV = {"ALPACA_API_KEY": "TESTKEY123", "ALPACA_SECRET_KEY": "TESTSECRET456"}


def _mock_resp(bars: list, next_token=None) -> MagicMock:
    """Build a mock requests.Response for a single Alpaca page."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "bars":             bars,
        "symbol":           "SPY",
        "next_page_token":  next_token,
    }
    return resp


def _cfg() -> dict:
    return {
        "mode": "backtest",
        "trading_windows": [
            {"start": "09:30", "end": "10:00", "timezone": "America/New_York"},
            {"start": "15:30", "end": "16:00", "timezone": "America/New_York"},
        ],
    }


# ---------------------------------------------------------------------------
# Basic fetch
# ---------------------------------------------------------------------------

@patch.dict(os.environ, _FAKE_ENV)
@patch("src.data.fetcher.requests.get")
def test_alpaca_returns_dataframe(mock_get):
    """fetch_historical_alpaca returns a non-empty DataFrame."""
    mock_get.return_value = _mock_resp([_BAR_WINDOW_1, _BAR_WINDOW_2])

    fetcher = DataFetcher(_cfg())
    df = fetcher.fetch_historical_alpaca("SPY", "2025-10-01", "2025-10-02")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


@patch.dict(os.environ, _FAKE_ENV)
@patch("src.data.fetcher.requests.get")
def test_required_columns_present(mock_get):
    """Returned DataFrame has exactly the expected columns."""
    mock_get.return_value = _mock_resp([_BAR_WINDOW_1])

    df = DataFetcher(_cfg()).fetch_historical_alpaca("SPY", "2025-10-01", "2025-10-02")

    assert set(df.columns) == {"timestamp", "open", "high", "low", "close", "volume"}


@patch.dict(os.environ, _FAKE_ENV)
@patch("src.data.fetcher.requests.get")
def test_timestamps_are_et_timezone_aware(mock_get):
    """Timestamps must be timezone-aware and in US/Eastern."""
    mock_get.return_value = _mock_resp([_BAR_WINDOW_1])

    df = DataFetcher(_cfg()).fetch_historical_alpaca("SPY", "2025-10-01", "2025-10-02")

    assert df["timestamp"].dt.tz is not None
    tz_name = str(df["timestamp"].dt.tz)
    assert "America/New_York" in tz_name or "US/Eastern" in tz_name or "EDT" in tz_name or "EST" in tz_name


@patch.dict(os.environ, _FAKE_ENV)
@patch("src.data.fetcher.requests.get")
def test_empty_response_returns_empty_dataframe(mock_get):
    """Empty bars list → empty DataFrame with correct columns."""
    mock_get.return_value = _mock_resp([])

    df = DataFetcher(_cfg()).fetch_historical_alpaca("SPY", "2025-10-01", "2025-10-02")

    assert df.empty
    assert set(df.columns) == {"timestamp", "open", "high", "low", "close", "volume"}


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

@patch.dict(os.environ, _FAKE_ENV)
@patch("src.data.fetcher.requests.get")
def test_pagination_fetches_all_pages(mock_get):
    """Two pages are fetched and bars from both are included."""
    mock_get.side_effect = [
        _mock_resp([_BAR_WINDOW_1], next_token="PAGE2TOKEN"),
        _mock_resp([_BAR_WINDOW_2], next_token=None),
    ]

    df = DataFetcher(_cfg()).fetch_historical_alpaca("SPY", "2025-10-01", "2025-10-03")

    assert mock_get.call_count == 2
    assert len(df) == 2


@patch.dict(os.environ, _FAKE_ENV)
@patch("src.data.fetcher.requests.get")
def test_page_token_passed_in_second_request(mock_get):
    """next_page_token from page 1 is forwarded as page_token in page 2."""
    mock_get.side_effect = [
        _mock_resp([_BAR_WINDOW_1], next_token="MYTOKEN"),
        _mock_resp([_BAR_WINDOW_2], next_token=None),
    ]

    DataFetcher(_cfg()).fetch_historical_alpaca("SPY", "2025-10-01", "2025-10-03")

    _, kwargs2 = mock_get.call_args_list[1]
    assert kwargs2["params"]["page_token"] == "MYTOKEN"


@patch.dict(os.environ, _FAKE_ENV)
@patch("src.data.fetcher.requests.get")
def test_null_next_page_token_stops_fetching(mock_get):
    """next_page_token=None on first page means only one request is made."""
    mock_get.return_value = _mock_resp([_BAR_WINDOW_1, _BAR_WINDOW_2], next_token=None)

    DataFetcher(_cfg()).fetch_historical_alpaca("SPY", "2025-10-01", "2025-10-02")

    assert mock_get.call_count == 1


# ---------------------------------------------------------------------------
# Window filter
# ---------------------------------------------------------------------------

@patch.dict(os.environ, _FAKE_ENV)
@patch("src.data.fetcher.requests.get")
def test_window_filter_removes_out_of_window_bars(mock_get):
    """Bars outside trading windows are filtered out."""
    mock_get.return_value = _mock_resp(
        [_BAR_WINDOW_1, _BAR_OUTSIDE, _BAR_WINDOW_2]
    )

    df = DataFetcher(_cfg()).fetch_historical_alpaca("SPY", "2025-10-01", "2025-10-02")

    # _BAR_OUTSIDE (12:00 ET) should be dropped; the two window bars kept
    assert len(df) == 2
    for ts in df["timestamp"]:
        t = ts.astimezone(_ET).time()
        in_window = (
            (ts.hour == 9 and 30 <= ts.minute <= 59) or
            (ts.hour == 15 and 30 <= ts.minute <= 59)
        )
        # Simpler: just check no bar has the out-of-window time
    out_times = df["timestamp"].dt.tz_convert(_ET).dt.hour
    assert (out_times == 12).sum() == 0


# ---------------------------------------------------------------------------
# Auto-selection in fetch_historical()
# ---------------------------------------------------------------------------

@patch.dict(os.environ, {**_FAKE_ENV})
@patch("src.data.fetcher.requests.get")
def test_fetch_historical_uses_alpaca_when_key_present(mock_get):
    """fetch_historical() routes to Alpaca when ALPACA_API_KEY is set."""
    mock_get.return_value = _mock_resp([_BAR_WINDOW_1])

    df = DataFetcher(_cfg()).fetch_historical("SPY", "2025-10-01", "2025-10-02")

    mock_get.assert_called_once()  # Alpaca request was made
    assert not df.empty


@patch.dict(os.environ, {"ALPACA_API_KEY": "", "ALPACA_SECRET_KEY": ""})
@patch("src.data.fetcher.yf.Ticker")
def test_fetch_historical_falls_back_to_yfinance_when_no_key(mock_ticker):
    """fetch_historical() routes to yfinance when ALPACA_API_KEY is absent."""
    import pandas as pd

    # Build a minimal yfinance-style DataFrame
    mock_hist = pd.DataFrame({
        "Datetime": pd.to_datetime(["2026-03-17 09:30:00-04:00"]),
        "Open":  [580.0], "High": [581.0], "Low": [579.5],
        "Close": [580.5], "Volume": [1_500_000],
    })
    mock_ticker.return_value.history.return_value = mock_hist

    df = DataFetcher(_cfg()).fetch_historical("SPY", "2026-03-17", "2026-03-18")

    mock_ticker.assert_called_once_with("SPY")
    assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# filter_windows=False (all-hours path used by TradingEngine)
# ---------------------------------------------------------------------------

@patch.dict(os.environ, _FAKE_ENV)
@patch("src.data.fetcher.requests.get")
def test_filter_windows_false_keeps_out_of_window_bars(mock_get):
    """filter_windows=False returns all bars including those outside trading windows."""
    mock_get.return_value = _mock_resp(
        [_BAR_WINDOW_1, _BAR_OUTSIDE, _BAR_WINDOW_2]
    )

    df = DataFetcher(_cfg()).fetch_historical_alpaca(
        "SPY", "2025-10-01", "2025-10-02", filter_windows=False
    )

    assert len(df) == 3  # all three bars kept


@patch.dict(os.environ, _FAKE_ENV)
@patch("src.data.fetcher.requests.get")
def test_fetch_historical_filter_windows_false_propagates(mock_get):
    """fetch_historical(filter_windows=False) passes the flag through to Alpaca path."""
    mock_get.return_value = _mock_resp(
        [_BAR_WINDOW_1, _BAR_OUTSIDE, _BAR_WINDOW_2]
    )

    df = DataFetcher(_cfg()).fetch_historical(
        "SPY", "2025-10-01", "2025-10-02", filter_windows=False
    )

    assert len(df) == 3


# ---------------------------------------------------------------------------
# Auth guard
# ---------------------------------------------------------------------------

@patch.dict(os.environ, {"ALPACA_API_KEY": "", "ALPACA_SECRET_KEY": ""})
def test_fetch_historical_alpaca_raises_without_credentials():
    """Calling fetch_historical_alpaca() directly without keys raises EnvironmentError."""
    with pytest.raises(EnvironmentError, match="ALPACA_API_KEY"):
        DataFetcher(_cfg()).fetch_historical_alpaca("SPY", "2025-10-01", "2025-10-02")
