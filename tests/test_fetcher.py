"""
Basic tests for DataFetcher and Universe.
Run with:  pytest tests/test_fetcher.py -v
"""
import sys
from pathlib import Path
from datetime import time

import pytest
import pytz

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fetcher import DataFetcher, _parse_windows
from src.data.universe import Universe

_ET = pytz.timezone("America/New_York")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fetcher():
    return DataFetcher()


@pytest.fixture(scope="module")
def spy_df(fetcher):
    return fetcher.fetch_historical("SPY", "2026-03-17", "2026-03-22")


# ---------------------------------------------------------------------------
# DataFetcher tests
# ---------------------------------------------------------------------------

def test_returns_dataframe(spy_df):
    import pandas as pd
    assert isinstance(spy_df, pd.DataFrame), "fetch_historical must return a DataFrame"


def test_required_columns(spy_df):
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(spy_df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_timestamps_are_timezone_aware(spy_df):
    assert spy_df["timestamp"].dt.tz is not None, "timestamp column must be timezone-aware"


def test_timestamps_within_trading_windows(spy_df):
    windows = _parse_windows(DataFetcher().cfg)
    ts_et = spy_df["timestamp"].dt.tz_convert(_ET)

    for _, row_time in ts_et.dt.time.items():
        in_window = any(start <= row_time <= end for start, end in windows)
        assert in_window, f"Timestamp {row_time} is outside all trading windows"


def test_not_empty(spy_df):
    assert len(spy_df) > 0, "Expected at least some bars for SPY in Jan 2024"


# ---------------------------------------------------------------------------
# Universe tests
# ---------------------------------------------------------------------------

def test_universe_returns_configured_symbols():
    """Universe returns whatever symbols are in the live config."""
    u = Universe()
    symbols = u.get_symbols()
    assert isinstance(symbols, list)
    assert len(symbols) >= 1
    assert all(isinstance(s, str) for s in symbols)


def test_validate_symbol_valid():
    u = Universe()
    assert u.validate_symbol("SPY") is True


def test_validate_symbol_empty():
    u = Universe()
    assert u.validate_symbol("") is False
    assert u.validate_symbol("   ") is False


def test_validate_symbol_non_string():
    u = Universe()
    assert u.validate_symbol(123) is False  # type: ignore[arg-type]
