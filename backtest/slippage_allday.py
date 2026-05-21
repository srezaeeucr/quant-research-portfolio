#!/usr/bin/env python3
"""
Slippage stress test on all-day live config combos.
Tests 4 slippage levels on ORB/NVDA and ORB/AMD with all-day entries.
"""
import os
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.engine import TradingEngine

_ROOT = Path(__file__).resolve().parents[1]

COMBOS = [
    # (symbol, sl, rr, label)
    ("NVDA", 0.3, 2.5, "NVDA live config"),
    ("AMD", 1.0, 4.0, "AMD live config"),
    ("AMD", 0.75, 5.0, "AMD best holdout"),
    ("NVDA", 0.5, 3.0, "NVDA wider stop"),
]

SLIPPAGE_LEVELS = [0.0, 0.02, 0.05, 0.10, 0.20]

PERIODS = [
    ("2023-01-01", "2025-01-01", "full_2yr"),
    ("2022-01-01", "2022-12-31", "holdout_2022"),
    ("2025-06-01", "2026-03-31", "bear_2025"),
]


def main():
    with open(_ROOT / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["mode"] = "backtest"
    cfg["account"] = {"balance": 500.0, "max_position_pct": 0.95}

    engine = TradingEngine(cfg)

    print(f"{'Combo':<22} {'Period':<14} {'Slip%':<7} {'Trades':<7} {'WR%':<7} {'PnL':>9} {'PF':>7} {'DD%':>6}")
    print("-" * 85)

    for symbol, sl, rr, label in COMBOS:
        for start, end, period_label in PERIODS:
            for slip in SLIPPAGE_LEVELS:
                try:
                    metrics = engine.run_backtest_config(
                        config_override={
                            "active_strategy": "orb",
                            "stop_loss_pct": sl,
                            "reward_risk": rr,
                            "volume_mult": 1.2,
                            "regime_filter": True,
                            "vix_threshold": 30,
                            "max_trades_per_day": 3,
                            "afternoon_entries": True,
                            "entry_window_minutes": 360,
                            "slippage_pct": slip,
                        },
                        start_date=start,
                        end_date=end,
                        symbol=symbol,
                    )
                    trades = metrics.get("total_trades", 0)
                    wr = metrics.get("win_rate", 0)
                    pnl = metrics.get("total_pnl", 0)
                    pf = metrics.get("profit_factor", 0)
                    dd = metrics.get("max_drawdown", 0)
                    print(f"{label:<22} {period_label:<14} {slip:<7.2f} {trades:<7} {wr:<7.1f} ${pnl:>+8.2f} {pf:>7.2f} {dd:>5.1f}%")
                except Exception as e:
                    print(f"{label:<22} {period_label:<14} {slip:<7.2f} ERROR: {e}")
            print()


if __name__ == "__main__":
    main()
