"""Tests for ResultsStore (src/logger/results_store.py)."""
import tempfile
from pathlib import Path

import pytest

from src.logger.results_store import ResultsStore, make_run_id


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    """Fresh ResultsStore backed by a temp SQLite file."""
    return ResultsStore(db_path=tmp_path / "test_results.db")


def _sample_config(override=None):
    cfg = {
        "strategy":           "orb",
        "symbol":             "SPY",
        "period_label":       "bull_2023",
        "start_date":         "2023-01-01",
        "end_date":           "2023-12-31",
        "stop_loss_pct":      0.5,
        "reward_risk":        2.0,
        "volume_mult":        1.2,
        "regime_filter":      True,
        "entry_window_start": "",
        "entry_window_end":   "",
        "or_duration_min":    15,
        "vix_threshold":      20,
        "max_trades_per_day": 1,
    }
    if override:
        cfg.update(override)
    return cfg


def _sample_metrics():
    return {
        "total_trades":    10,
        "win_rate":        60.0,
        "total_pnl":       12.34,
        "max_drawdown":    1.5,
        "profit_factor":   2.1,
        "avg_duration_min": 45.0,
        "sharpe_ratio":    0.8,
        "final_balance":   212.34,
    }


def _sample_trades(run_id, n=3):
    return [
        {
            "run_id":          run_id,
            "order_id":        f"ord-{i}",
            "symbol":          "SPY",
            "shares":          0.5,
            "entry_price":     400.0 + i,
            "exit_price":      402.0 + i,
            "pnl":             1.0,
            "pnl_pct":        0.5,
            "reason":          "target_hit",
            "entry_time":      "2023-03-01T09:45:00",
            "exit_time":       "2023-03-01T10:30:00",
            "duration_minutes": 45,
            "mode":            "backtest",
        }
        for i in range(n)
    ]


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_save_run_inserts_correctly(store):
    """save_run persists config + metrics columns accurately."""
    cfg = _sample_config()
    metrics = _sample_metrics()
    run_id = make_run_id(cfg)
    cfg["run_id"] = run_id

    store.save_run(cfg, metrics, [])

    df = store.get_all_runs()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["run_id"] == run_id
    assert row["strategy"] == "orb"
    assert row["symbol"] == "SPY"
    assert row["period_label"] == "bull_2023"
    assert float(row["total_pnl"]) == pytest.approx(12.34)
    assert float(row["win_rate"]) == pytest.approx(60.0)
    assert int(row["total_trades"]) == 10


def test_run_exists_true_after_save(store):
    """run_exists returns True after a run has been saved."""
    cfg = _sample_config()
    run_id = make_run_id(cfg)
    cfg["run_id"] = run_id

    assert store.run_exists(run_id) is False
    store.save_run(cfg, _sample_metrics(), [])
    assert store.run_exists(run_id) is True


def test_get_best_runs_returns_top_n(store):
    """get_best_runs returns at most N rows sorted by metric descending."""
    for i, pnl in enumerate([5.0, 20.0, 10.0, 1.0]):
        cfg = _sample_config({"symbol": f"SYM{i}", "period_label": f"p{i}"})
        run_id = make_run_id(cfg)
        cfg["run_id"] = run_id
        metrics = _sample_metrics()
        metrics["total_pnl"] = pnl
        store.save_run(cfg, metrics, [])

    best = store.get_best_runs(metric="total_pnl", n=2)
    assert len(best) == 2
    assert float(best.iloc[0]["total_pnl"]) == pytest.approx(20.0)
    assert float(best.iloc[1]["total_pnl"]) == pytest.approx(10.0)


def test_get_run_trades_returns_correct_trades(store):
    """get_run_trades returns only trades belonging to the requested run_id."""
    cfg_a = _sample_config({"symbol": "SPY"})
    cfg_b = _sample_config({"symbol": "QQQ"})
    run_id_a = make_run_id(cfg_a)
    run_id_b = make_run_id(cfg_b)
    cfg_a["run_id"] = run_id_a
    cfg_b["run_id"] = run_id_b

    store.save_run(cfg_a, _sample_metrics(), _sample_trades(run_id_a, n=3))
    store.save_run(cfg_b, _sample_metrics(), _sample_trades(run_id_b, n=5))

    trades_a = store.get_run_trades(run_id_a)
    trades_b = store.get_run_trades(run_id_b)

    assert len(trades_a) == 3
    assert len(trades_b) == 5
    assert all(trades_a["run_id"] == run_id_a)
    assert all(trades_b["run_id"] == run_id_b)


def test_duplicate_run_id_is_skipped(store):
    """Saving the same run_id twice does not duplicate the row (idempotent)."""
    cfg = _sample_config()
    run_id = make_run_id(cfg)
    cfg["run_id"] = run_id
    metrics = _sample_metrics()

    store.save_run(cfg, metrics, _sample_trades(run_id, n=2))
    # Save again — should be a no-op
    metrics["total_pnl"] = 999.0  # different value; should NOT overwrite
    store.save_run(cfg, metrics, _sample_trades(run_id, n=2))

    df = store.get_all_runs()
    assert len(df) == 1
    assert float(df.iloc[0]["total_pnl"]) == pytest.approx(12.34)  # original value kept

    trades = store.get_run_trades(run_id)
    assert len(trades) == 2  # not doubled
