"""
Tests for TradeLogger trade_notes: save_note, get_note, update, persistence.
Run with:  pytest tests/test_trade_notes.py -v
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


def _trade(order_id: str = "order-001") -> dict:
    return {
        "order_id":         order_id,
        "symbol":           "AAPL",
        "shares":           0.50,
        "entry_price":      175.00,
        "exit_price":       176.75,
        "pnl":              0.88,
        "pnl_pct":          1.00,
        "reason":           "target_hit",
        "entry_time":       _ET.localize(datetime(2024, 3, 1, 9, 45)),
        "exit_time":        _ET.localize(datetime(2024, 3, 1, 10, 5)),
        "duration_minutes": 20,
        "mode":             "paper",
    }


@pytest.fixture
def logger_tmp():
    """TradeLogger backed by a fresh temp database for each test."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = {"logging": {"db_path": str(Path(tmp) / "trades.db")}}
        yield TradeLogger(cfg)


# ------------------------------------------------------------------
# save_note / get_note
# ------------------------------------------------------------------

def test_save_and_get_note(logger_tmp):
    """Saving a note and retrieving it returns the same text."""
    logger_tmp.log_trade(_trade("ord-1"))
    logger_tmp.save_note("ord-1", "AAPL had earnings news")
    assert logger_tmp.get_note("ord-1") == "AAPL had earnings news"


def test_get_note_returns_empty_when_missing(logger_tmp):
    """get_note on an unknown order_id returns empty string, not None."""
    result = logger_tmp.get_note("nonexistent-id")
    assert result == ""


def test_update_note_overwrites(logger_tmp):
    """Calling save_note twice updates rather than duplicates the note."""
    logger_tmp.log_trade(_trade("ord-2"))
    logger_tmp.save_note("ord-2", "first note")
    logger_tmp.save_note("ord-2", "updated note")
    assert logger_tmp.get_note("ord-2") == "updated note"


def test_note_persists_across_instances(logger_tmp):
    """A note saved in one TradeLogger instance is visible in a new instance
    pointing at the same db file."""
    logger_tmp.log_trade(_trade("ord-3"))
    logger_tmp.save_note("ord-3", "VIX spike before entry")

    # Open a second instance to the same db
    cfg2 = {"logging": {"db_path": logger_tmp.db_path}}
    logger2 = TradeLogger(cfg2)
    assert logger2.get_note("ord-3") == "VIX spike before entry"


def test_get_all_notes_returns_dict(logger_tmp):
    """get_all_notes returns a dict keyed by order_id."""
    logger_tmp.save_note("ord-a", "note A")
    logger_tmp.save_note("ord-b", "note B")
    notes = logger_tmp.get_all_notes()
    assert isinstance(notes, dict)
    assert notes["ord-a"] == "note A"
    assert notes["ord-b"] == "note B"


def test_get_all_notes_empty_db(logger_tmp):
    """get_all_notes on a fresh db returns empty dict."""
    assert logger_tmp.get_all_notes() == {}


def test_save_note_does_not_require_trade_row(logger_tmp):
    """Notes can be saved for any order_id, even without a matching trade row."""
    logger_tmp.save_note("phantom-order", "orphan note")
    assert logger_tmp.get_note("phantom-order") == "orphan note"


def test_note_table_created_alongside_trades_table(logger_tmp):
    """trade_notes table must exist in the db (created by _init_db)."""
    import sqlite3
    conn = sqlite3.connect(logger_tmp.db_path)
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    conn.close()
    assert "trade_notes" in tables
    assert "trades" in tables
