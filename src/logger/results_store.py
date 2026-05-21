"""
ResultsStore — persists grid-search backtest runs to logs/results.db.

Separate from trades.db so normal backtests are not polluted.
"""
import hashlib
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_ROOT        = Path(__file__).resolve().parents[2]
_DEFAULT_DB  = _ROOT / "logs" / "results.db"

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS backtest_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT    UNIQUE NOT NULL,
    strategy            TEXT    NOT NULL,
    symbol              TEXT    NOT NULL,
    period_label        TEXT    NOT NULL,
    start_date          TEXT    NOT NULL,
    end_date            TEXT    NOT NULL,
    stop_loss_pct       REAL    NOT NULL,
    reward_risk         REAL    NOT NULL,
    volume_mult         REAL    NOT NULL,
    regime_filter       INTEGER NOT NULL,
    entry_window_start  TEXT    NOT NULL DEFAULT '',
    entry_window_end    TEXT    NOT NULL DEFAULT '',
    or_duration_min     INTEGER NOT NULL DEFAULT 15,
    vix_threshold       REAL    NOT NULL DEFAULT 0,
    max_trades_per_day  INTEGER NOT NULL DEFAULT 1,
    afternoon_entries   INTEGER NOT NULL DEFAULT 0,
    total_trades        INTEGER NOT NULL DEFAULT 0,
    win_rate            REAL    NOT NULL DEFAULT 0,
    total_pnl           REAL    NOT NULL DEFAULT 0,
    max_drawdown        REAL    NOT NULL DEFAULT 0,
    profit_factor       REAL    NOT NULL DEFAULT 0,
    avg_duration_min    REAL    NOT NULL DEFAULT 0,
    sharpe_ratio        REAL    NOT NULL DEFAULT 0,
    final_balance       REAL    NOT NULL DEFAULT 0,
    ran_at              TEXT    NOT NULL
);
"""

_CREATE_TRADES = """
CREATE TABLE IF NOT EXISTS run_trades (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT    NOT NULL,
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
    mode             TEXT    NOT NULL
);
"""

_INSERT_RUN = """
INSERT OR IGNORE INTO backtest_runs
    (run_id, strategy, symbol, period_label, start_date, end_date,
     stop_loss_pct, reward_risk, volume_mult, regime_filter,
     entry_window_start, entry_window_end, or_duration_min,
     vix_threshold, max_trades_per_day, afternoon_entries,
     total_trades, win_rate, total_pnl, max_drawdown,
     profit_factor, avg_duration_min, sharpe_ratio, final_balance, ran_at)
VALUES
    (:run_id, :strategy, :symbol, :period_label, :start_date, :end_date,
     :stop_loss_pct, :reward_risk, :volume_mult, :regime_filter,
     :entry_window_start, :entry_window_end, :or_duration_min,
     :vix_threshold, :max_trades_per_day, :afternoon_entries,
     :total_trades, :win_rate, :total_pnl, :max_drawdown,
     :profit_factor, :avg_duration_min, :sharpe_ratio, :final_balance, :ran_at)
"""

_INSERT_TRADE = """
INSERT INTO run_trades
    (run_id, order_id, symbol, shares, entry_price, exit_price,
     pnl, pnl_pct, reason, entry_time, exit_time, duration_minutes, mode)
VALUES
    (:run_id, :order_id, :symbol, :shares, :entry_price, :exit_price,
     :pnl, :pnl_pct, :reason, :entry_time, :exit_time, :duration_minutes, :mode)
"""


def make_run_id(params: Dict) -> str:
    """Return a 16-char deterministic hex ID from a sorted JSON hash of params."""
    payload = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(payload.encode()).hexdigest()[:16]


def _dt_str(value) -> str:
    if isinstance(value, str):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


class ResultsStore:
    """SQLite store for grid-search backtest results."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = str(db_path or _DEFAULT_DB)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        # 30s busy timeout — gives concurrent writers time to acquire the lock
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            # WAL journal mode allows concurrent readers + writers (one writer
            # at a time). Critical for parallel batch runners.
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute(_CREATE_RUNS)
            conn.execute(_CREATE_TRADES)
            # Migration: add afternoon_entries column to existing DBs
            cols = [
                row[1] for row in conn.execute(
                    "PRAGMA table_info(backtest_runs)"
                ).fetchall()
            ]
            if "afternoon_entries" not in cols:
                conn.execute(
                    "ALTER TABLE backtest_runs ADD COLUMN "
                    "afternoon_entries INTEGER NOT NULL DEFAULT 0"
                )
        logger.debug("ResultsStore ready: %s", self.db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_run(self, run_config: Dict, metrics: Dict,
                 trades: List[Dict]) -> str:
        """Persist a completed backtest run.

        Parameters
        ----------
        run_config : dict with keys matching backtest_runs columns
                     (strategy, symbol, period_label, start_date, end_date,
                      stop_loss_pct, reward_risk, volume_mult, regime_filter,
                      entry_window_start, entry_window_end, or_duration_min,
                      vix_threshold, max_trades_per_day)
        metrics    : dict with keys matching metrics columns
                     (total_trades, win_rate, total_pnl, max_drawdown,
                      profit_factor, avg_duration_min, sharpe_ratio, final_balance)
        trades     : list of trade dicts (same schema as trades table)

        Returns
        -------
        run_id string.
        """
        run_id = run_config.get("run_id") or make_run_id(run_config)
        ran_at = datetime.utcnow().isoformat()

        row = {
            "run_id":             run_id,
            "strategy":           run_config.get("strategy", ""),
            "symbol":             run_config.get("symbol", ""),
            "period_label":       run_config.get("period_label", ""),
            "start_date":         run_config.get("start_date", ""),
            "end_date":           run_config.get("end_date", ""),
            "stop_loss_pct":      float(run_config.get("stop_loss_pct", 0.5)),
            "reward_risk":        float(run_config.get("reward_risk", 2.0)),
            "volume_mult":        float(run_config.get("volume_mult", 1.2)),
            "regime_filter":      int(bool(run_config.get("regime_filter", False))),
            "entry_window_start": run_config.get("entry_window_start", ""),
            "entry_window_end":   run_config.get("entry_window_end", ""),
            "or_duration_min":    int(run_config.get("or_duration_min", 15)),
            "vix_threshold":      float(run_config.get("vix_threshold", 0)),
            "max_trades_per_day": int(run_config.get("max_trades_per_day", 1)),
            "afternoon_entries":  int(bool(run_config.get("afternoon_entries", False))),
            "total_trades":       int(metrics.get("total_trades", 0)),
            "win_rate":           float(metrics.get("win_rate", 0.0)),
            "total_pnl":          float(metrics.get("total_pnl", 0.0)),
            "max_drawdown":       float(metrics.get("max_drawdown", 0.0)),
            "profit_factor":      float(metrics.get("profit_factor", 0.0)),
            "avg_duration_min":   float(metrics.get("avg_duration_min", 0.0)),
            "sharpe_ratio":       float(metrics.get("sharpe_ratio", 0.0)),
            "final_balance":      float(metrics.get("final_balance", 0.0)),
            "ran_at":             ran_at,
        }

        with self._connect() as conn:
            conn.execute(_INSERT_RUN, row)
            # Only insert trades for new runs (INSERT OR IGNORE skips dupes)
            if not self.run_exists(run_id) or not trades:
                # Check if run was actually inserted (not ignored)
                n = conn.execute(
                    "SELECT COUNT(*) FROM run_trades WHERE run_id=?", (run_id,)
                ).fetchone()[0]
                if n == 0:
                    for t in trades:
                        conn.execute(_INSERT_TRADE, {
                            "run_id":          run_id,
                            "order_id":        t.get("order_id", ""),
                            "symbol":          t.get("symbol", ""),
                            "shares":          float(t.get("shares", 0)),
                            "entry_price":     float(t.get("entry_price", 0)),
                            "exit_price":      float(t.get("exit_price", 0)),
                            "pnl":             float(t.get("pnl", 0)),
                            "pnl_pct":         float(t.get("pnl_pct", 0)),
                            "reason":          t.get("reason", ""),
                            "entry_time":      _dt_str(t.get("entry_time", "")),
                            "exit_time":       _dt_str(t.get("exit_time", "")),
                            "duration_minutes": int(t.get("duration_minutes", 0)),
                            "mode":            t.get("mode", "backtest"),
                        })

        logger.info("Run saved: %s  trades=%d  pnl=%.4f",
                    run_id, row["total_trades"], row["total_pnl"])
        return run_id

    def run_exists(self, run_id: str) -> bool:
        """Return True if run_id is already in backtest_runs."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM backtest_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        return row is not None

    def get_all_runs(self) -> pd.DataFrame:
        """Return all backtest_runs as a DataFrame, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM backtest_runs ORDER BY id DESC"
            ).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])

    def get_run_trades(self, run_id: str) -> pd.DataFrame:
        """Return all trades for a specific run_id."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM run_trades WHERE run_id=? ORDER BY id",
                (run_id,),
            ).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])

    def get_best_runs(self, metric: str = "profit_factor",
                      n: int = 10) -> pd.DataFrame:
        """Return the top N runs sorted descending by metric."""
        allowed = {
            "profit_factor", "total_pnl", "win_rate", "sharpe_ratio",
            "total_trades", "final_balance",
        }
        if metric not in allowed:
            raise ValueError(f"metric must be one of {allowed}")
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM backtest_runs ORDER BY {metric} DESC LIMIT ?",
                (n,),
            ).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])
