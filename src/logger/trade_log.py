import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DB = _ROOT / "logs" / "trades.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS trades (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id         TEXT    NOT NULL,
    symbol           TEXT    NOT NULL,
    shares           REAL    NOT NULL,
    entry_price      REAL    NOT NULL,
    exit_price       REAL    NOT NULL,
    pnl              REAL    NOT NULL,
    pnl_pct          REAL    NOT NULL,
    reason           TEXT    NOT NULL,
    entry_time       TEXT    NOT NULL,
    exit_time        TEXT    NOT NULL,
    duration_minutes INTEGER NOT NULL,
    mode             TEXT    NOT NULL,
    created_at       TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

_CREATE_NOTES_TABLE = """
CREATE TABLE IF NOT EXISTS trade_notes (
    order_id   TEXT PRIMARY KEY,
    note       TEXT,
    created_at TEXT,
    updated_at TEXT
);
"""

_INSERT = """
INSERT INTO trades
    (order_id, symbol, shares, entry_price, exit_price, pnl, pnl_pct,
     reason, entry_time, exit_time, duration_minutes, mode)
VALUES
    (:order_id, :symbol, :shares, :entry_price, :exit_price, :pnl, :pnl_pct,
     :reason, :entry_time, :exit_time, :duration_minutes, :mode);
"""


def _dt_str(value) -> str:
    """Coerce datetime / string to ISO-format string."""
    if isinstance(value, str):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


class TradeLogger:
    """Persists trade records to a SQLite database."""

    def __init__(self, config: Optional[Dict] = None):
        db_path = _DEFAULT_DB
        if config:
            db_path = Path(config.get("logging", {}).get("db_path", _DEFAULT_DB))

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_NOTES_TABLE)
        logger.debug("Trade database ready: %s", self.db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_trade(self, trade_record: Dict) -> None:
        """Insert a trade record into the database."""
        row = {
            "order_id":         trade_record["order_id"],
            "symbol":           trade_record["symbol"],
            "shares":           float(trade_record["shares"]),
            "entry_price":      float(trade_record["entry_price"]),
            "exit_price":       float(trade_record["exit_price"]),
            "pnl":              float(trade_record["pnl"]),
            "pnl_pct":          float(trade_record["pnl_pct"]),
            "reason":           trade_record["reason"],
            "entry_time":       _dt_str(trade_record["entry_time"]),
            "exit_time":        _dt_str(trade_record["exit_time"]),
            "duration_minutes": int(trade_record["duration_minutes"]),
            "mode":             trade_record["mode"],
        }
        with self._connect() as conn:
            conn.execute(_INSERT, row)
        logger.info("Trade logged: %s  pnl=$%.4f", trade_record["order_id"], row["pnl"])

    def get_all_trades(self) -> pd.DataFrame:
        """Return all trades as a DataFrame."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM trades ORDER BY id").fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])

    def clear(self) -> None:
        """Delete all trade records (used to reset state at the start of a new run)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM trades")
        logger.debug("Trade log cleared")

    def save_note(self, order_id: str, note: str) -> None:
        """Insert or update a note for the given order_id."""
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT created_at FROM trade_notes WHERE order_id = ?", (order_id,)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE trade_notes SET note = ?, updated_at = ? WHERE order_id = ?",
                    (note, now, order_id),
                )
            else:
                conn.execute(
                    "INSERT INTO trade_notes (order_id, note, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (order_id, note, now, now),
                )
        logger.debug("Note saved for order_id=%s", order_id)

    def get_note(self, order_id: str) -> str:
        """Return note text for the given order_id, or '' if none exists."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT note FROM trade_notes WHERE order_id = ?", (order_id,)
            ).fetchone()
        if row:
            return row["note"] or ""
        return ""

    def get_all_notes(self) -> dict:
        """Return {order_id: note} for all stored notes."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT order_id, note FROM trade_notes"
            ).fetchall()
        return {r["order_id"]: r["note"] or "" for r in rows}

    def get_daily_summary(self, date: str) -> Dict:
        """Return trade stats for a calendar date (YYYY-MM-DD).

        Returns
        -------
        dict with keys: date, total_trades, wins, losses, total_pnl, win_rate
        """
        sql = """
            SELECT pnl, reason
            FROM   trades
            WHERE  date(entry_time) = ?
            ORDER  BY id
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (date,)).fetchall()

        total   = len(rows)
        wins    = sum(1 for r in rows if r["pnl"] > 0)
        losses  = sum(1 for r in rows if r["pnl"] <= 0)
        pnl     = sum(r["pnl"] for r in rows)
        win_rate = (wins / total * 100.0) if total else 0.0

        return {
            "date":         date,
            "total_trades": total,
            "wins":         wins,
            "losses":       losses,
            "total_pnl":    round(pnl, 4),
            "win_rate":     round(win_rate, 2),
        }
