#!/usr/bin/env python3
"""
Study #1: Fine-grained SL/RR sweep on AMD with EXTENDED recent data.

The original V4 grid tested SL ∈ {0.3, 0.5, 0.75, 1.0}. Given AMD's
parabolic move ($220→$348 in 3 weeks), tight stops are getting clipped.
Test the full SL range to find the sweet spot for current regime.

Test ALL strategies (ORB, EMA, Momentum) since the strategy might also
need to change.
"""
import os, sys, json, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import pandas as pd
import pytz
import yaml
from src.data.fetcher import DataFetcher
from src.engine import TradingEngine

_ET = pytz.timezone("America/New_York")
_ROOT = Path(__file__).resolve().parents[1]

SYMBOL = "AMD"
START = "2026-01-01"
END   = "2026-04-26"

SL_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.25, 1.5, 2.0]
RR_VALUES = [2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
STRATEGIES = ["orb", "ema_crossover", "momentum"]


def main():
    print(f"=== AMD fine-grained SL/RR sweep ===")
    print(f"Period: {START} → {END}")
    print(f"Grid: {len(SL_VALUES)} SL × {len(RR_VALUES)} RR × {len(STRATEGIES)} strats = "
          f"{len(SL_VALUES) * len(RR_VALUES) * len(STRATEGIES)} configs")
    print()

    # Fetch bars once
    fetcher = DataFetcher({})
    print(f"Fetching {SYMBOL} bars {START}→{END}...", flush=True)
    bars = fetcher.fetch_historical_alpaca(SYMBOL, START, END, filter_windows=False)
    print(f"  {len(bars)} bars fetched")

    cfg = yaml.safe_load(open(_ROOT / "config/config.yaml"))
    cfg["mode"] = "backtest"
    cfg["account"] = {"balance": 500.0, "max_position_pct": 0.95}
    engine = TradingEngine(cfg)

    results = []
    for strat in STRATEGIES:
        for sl in SL_VALUES:
            for rr in RR_VALUES:
                override = {
                    "active_strategy": strat,
                    "stop_loss_pct": sl,
                    "reward_risk": rr,
                    "volume_mult": 1.2,
                    "regime_filter": True,
                    "vix_threshold": 25,
                    "max_trades_per_day": 100,
                    "afternoon_entries": True,
                    "entry_window_minutes": 360,
                }
                try:
                    res = engine.run_backtest_config(
                        config_override=override,
                        start_date=START, end_date=END,
                        symbol=SYMBOL,
                        cached_bars=bars, cached_sma={}, cached_vix={},
                    )
                    results.append({
                        "strategy": strat, "sl": sl, "rr": rr,
                        "trades": res["total_trades"],
                        "pf": round(res["profit_factor"], 3),
                        "wr": round(res["win_rate"], 1),
                        "pnl": round(res["total_pnl"], 2),
                        "max_dd": round(res["max_drawdown"], 2),
                    })
                except Exception as e:
                    results.append({"strategy": strat, "sl": sl, "rr": rr, "error": str(e)[:80]})

    # Sort by PnL
    valid = [r for r in results if "error" not in r and r.get("trades", 0) >= 5]
    valid.sort(key=lambda r: -r["pnl"])

    print(f"\n=== TOP 15 (by PnL, with ≥5 trades) ===")
    print(f"  {'strategy':<14} {'sl':<5} {'rr':<5} {'trades':<7} {'pf':<6} {'wr':<6} {'pnl':<8} {'maxdd':<7}")
    for r in valid[:15]:
        print(f"  {r['strategy']:<14} {r['sl']:<5} {r['rr']:<5} {r['trades']:<7} "
              f"{r['pf']:<6.2f} {r['wr']:<6.1f} ${r['pnl']:<+7.2f} {r['max_dd']:<6.1f}%")

    print(f"\n=== BOTTOM 5 (worst) ===")
    for r in valid[-5:]:
        print(f"  {r['strategy']:<14} {r['sl']:<5} {r['rr']:<5} {r['trades']:<7} "
              f"{r['pf']:<6.2f} {r['wr']:<6.1f} ${r['pnl']:<+7.2f}")

    # Heatmap-like summary by strategy
    print(f"\n=== PER-STRATEGY: best PnL ===")
    for s in STRATEGIES:
        s_results = [r for r in valid if r["strategy"] == s]
        if s_results:
            best = max(s_results, key=lambda r: r["pnl"])
            print(f"  {s}: best={best['sl']}/{best['rr']}, PnL=${best['pnl']:+.2f}, PF={best['pf']:.2f}, trades={best['trades']}")
        else:
            print(f"  {s}: no valid results")

    out = _ROOT / "logs/research/studies_2026_04_26/amd_finegrained.json"
    with open(out, "w") as f:
        json.dump({"period": [START, END], "results": results}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
