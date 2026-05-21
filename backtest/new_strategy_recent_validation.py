#!/usr/bin/env python3
"""
Reasonable-grid validation of new strategies on RECENT data only.

Skipping VWAP Bands (too correlated with ORB — not a real diversifier).
Validating RSI Reversion (best diversifier) and MACD Crossover.

Reasonable params (small grid to avoid overfitting):
  - RSI:    oversold ∈ {25, 30}, period={14}, SL ∈ {0.5, 0.75, 1.0}, RR ∈ {2, 3, 4}
  - MACD:   textbook (12, 26, 9), SL ∈ {0.5, 0.75, 1.0}, RR ∈ {2, 3, 4}

Period: Jan 1 → Apr 26 2026 (extended recent).
Symbols: AMD, GOOGL.
"""
import os, sys, json, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import pandas as pd, pytz, yaml
from src.data.fetcher import DataFetcher
from src.engine import TradingEngine

_ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ["AMD", "GOOGL"]
START = "2026-01-01"
END   = "2026-04-26"

# Reasonable grid (small, not exhaustive)
GRID = []
for sl in [0.5, 0.75, 1.0]:
    for rr in [2.0, 3.0, 4.0]:
        # MACD with textbook params
        GRID.append({"strategy": "macd_crossover", "sl": sl, "rr": rr,
                     "params": {"fast_period": 12, "slow_period": 26, "signal_period": 9}})
        # RSI with 2 oversold thresholds
        for ov in [25.0, 30.0]:
            GRID.append({"strategy": "rsi_reversion", "sl": sl, "rr": rr,
                         "params": {"rsi_period": 14, "oversold_threshold": ov}})


def main():
    fetcher = DataFetcher({})
    cfg = yaml.safe_load(open(_ROOT / "config/config.yaml"))
    cfg["mode"] = "backtest"
    cfg["account"] = {"balance": 500.0, "max_position_pct": 0.95}

    print(f"=== New strategy recent validation ===")
    print(f"Period: {START} → {END}")
    print(f"Configs: {len(GRID)} × {len(SYMBOLS)} symbols = {len(GRID) * len(SYMBOLS)}")
    print()

    results = []
    for sym in SYMBOLS:
        print(f"Fetching {sym} bars...", flush=True)
        bars = fetcher.fetch_historical_alpaca(sym, START, END, filter_windows=False)
        if bars is None or bars.empty:
            print(f"  no bars")
            continue
        print(f"  {len(bars)} bars")

        for cfg_item in GRID:
            engine = TradingEngine({**cfg, "active_strategy": cfg_item["strategy"],
                                    "strategy_params": {cfg_item["strategy"]: cfg_item["params"]}})
            override = {
                "active_strategy": cfg_item["strategy"],
                "stop_loss_pct": cfg_item["sl"],
                "reward_risk":   cfg_item["rr"],
                "volume_mult":   1.2,
                "regime_filter": True,
                "vix_threshold": 25,
                "max_trades_per_day": 100,
                "afternoon_entries": True,
                "entry_window_minutes": 360,
            }
            try:
                res = engine.run_backtest_config(
                    config_override=override, start_date=START, end_date=END,
                    symbol=sym, cached_bars=bars, cached_sma={}, cached_vix={},
                )
                if res["total_trades"] >= 5:
                    results.append({
                        "strategy": cfg_item["strategy"],
                        "symbol":   sym,
                        "sl":       cfg_item["sl"],
                        "rr":       cfg_item["rr"],
                        "params":   cfg_item["params"],
                        "trades":   res["total_trades"],
                        "pf":       round(res["profit_factor"], 3),
                        "wr":       round(res["win_rate"], 1),
                        "pnl":      round(res["total_pnl"], 2),
                        "max_dd":   round(res["max_drawdown"], 2),
                    })
            except Exception as e:
                print(f"  ERROR {cfg_item['strategy']} {sym}: {e}")

    # Sort by PnL
    results.sort(key=lambda r: -r["pnl"])
    print(f"\n=== TOP 15 (by PnL, ≥5 trades) ===")
    print(f"  {'strategy':<16} {'sym':<6} {'sl':<5} {'rr':<5} {'trades':<7} {'pf':<6} {'wr':<6} {'pnl':<8}  params")
    for r in results[:15]:
        params_str = ', '.join(f"{k}={v}" for k, v in r['params'].items())
        print(f"  {r['strategy']:<16} {r['symbol']:<6} {r['sl']:<5} {r['rr']:<5} "
              f"{r['trades']:<7} {r['pf']:<6.2f} {r['wr']:<6.1f} ${r['pnl']:<+7.2f}  ({params_str})")

    # Best per (strategy, symbol)
    print(f"\n=== Best per (strategy, symbol) ===")
    from collections import defaultdict
    by_ss = defaultdict(list)
    for r in results:
        by_ss[(r["strategy"], r["symbol"])].append(r)
    for (s, sy), rs in sorted(by_ss.items()):
        best = max(rs, key=lambda r: r["pnl"])
        params_str = ', '.join(f"{k}={v}" for k, v in best['params'].items())
        print(f"  {s}/{sy}: SL={best['sl']} RR={best['rr']}  trades={best['trades']} "
              f"pf={best['pf']:.2f} pnl=${best['pnl']:+.2f}  ({params_str})")

    # Profitable summary
    profitable = [r for r in results if r["pf"] > 1.0]
    print(f"\nTotal valid configs: {len(results)}")
    print(f"Profitable (PF > 1.0): {len(profitable)} ({100*len(profitable)/max(len(results),1):.0f}%)")

    out = _ROOT / "logs/research/studies_2026_04_26/new_strategy_recent.json"
    with open(out, "w") as f: json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
