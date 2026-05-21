#!/usr/bin/env python3
"""
Backtest entry point.

Usage:
    python backtest/run_backtest.py --start 2025-10-01 --end 2026-03-01
    python backtest/run_backtest.py --start 2026-03-01 --end 2026-03-31 --symbol QQQ
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

_ROOT = Path(__file__).resolve().parents[1]


_VALID_STRATEGIES = ("orb", "vwap_reversion", "gap_fill", "ema_crossover", "momentum")


def parse_args():
    p = argparse.ArgumentParser(description="Run trading-bot backtest")
    p.add_argument("--start",    required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end",      required=True, help="End date   YYYY-MM-DD (exclusive)")
    p.add_argument("--symbol",   default=None,  help="Override symbol (default: from config)")
    p.add_argument("--strategy", default=None,
                   choices=_VALID_STRATEGIES,
                   help="Override active_strategy from config")
    return p.parse_args()


def main():
    args = parse_args()

    with open(_ROOT / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Ensure backtest mode regardless of config setting
    cfg["mode"] = "backtest"

    # CLI strategy override — set both new top-level key and legacy strategy key
    if args.strategy:
        cfg["active_strategy"] = args.strategy
        cfg.setdefault("strategy", {})["active_strategy"] = args.strategy

    from src.engine import TradingEngine

    engine  = TradingEngine(cfg)
    symbols = [args.symbol] if args.symbol else None

    trades = engine.run_backtest(
        start_date=args.start,
        end_date=args.end,
        symbols=symbols,
    )

    if not trades.empty:
        print("  Trade log (all trades):")
        print("  " + "-" * 76)
        cols = ["symbol", "shares", "entry_price", "exit_price",
                "pnl", "pnl_pct", "reason", "duration_minutes"]
        print(trades[cols].to_string(index=False, col_space=12))
        print()


if __name__ == "__main__":
    main()
