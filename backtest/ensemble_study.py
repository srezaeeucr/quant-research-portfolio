#!/usr/bin/env python3
"""
Study #5: Multi-strategy ensemble validation.

GPU Mega Research found "5/5 strategies agree → 100% profitable". Validate
this on actual recent data: for each (symbol, day, bar), count strategies
firing entry signals. Test "trade only when N≥k strategies agree" rules.
"""
import os, sys, json, glob, pickle, itertools
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
for p in [Path(__file__).resolve().parents[1] / ".env",
          Path.home() / "projects" / "dl_course" / ".env"]:
    if p.exists():
        load_dotenv(p); break

import pandas as pd, pytz, yaml
import numpy as np
_ET = pytz.timezone("America/New_York")
_ROOT = Path(__file__).resolve().parents[1]

# Test on "robust" symbols + a few historical winners
SYMBOLS = ["AMD", "GOOGL", "META", "TSLA", "COIN", "NVDA"]
STRATEGIES = ["orb", "ema_crossover", "momentum"]


def load_cached(cache_dir, sym, start, end):
    for ext in ["parquet", "pkl"]:
        p = cache_dir / f"{sym}_{start}_{end}.{ext}"
        if p.exists():
            if ext == "parquet": return pd.read_parquet(p)
            with open(p, "rb") as f: return pickle.load(f)
    return None


def main():
    from src.engine import TradingEngine
    cache_dir = _ROOT / "logs" / "research" / "bar_cache"
    cfg_path = _ROOT / "config" / "config.yaml"
    with open(cfg_path) as f: cfg = yaml.safe_load(f)
    cfg["mode"] = "backtest"
    cfg.setdefault("account", {})["balance"] = 500.0

    print(f"Testing ensemble on {SYMBOLS} (full_2yr 2023-2024)")
    print()

    PERIOD = ("2023-01-01", "2025-01-01")

    # For each symbol: run each strategy at one fixed config, collect signal times
    results = []
    for sym in SYMBOLS:
        bars = load_cached(cache_dir, sym, PERIOD[0], PERIOD[1])
        if bars is None or bars.empty:
            print(f"  {sym}: no bars, skipping")
            continue

        # Get signal times for each strategy
        signal_days = defaultdict(set)  # day -> set of strategies that fired entry
        for strat in STRATEGIES:
            engine = TradingEngine({**cfg, "active_strategy": strat})
            override = {
                "active_strategy": strat, "stop_loss_pct": 0.5, "reward_risk": 3.0,
                "volume_mult": 1.2, "regime_filter": True, "vix_threshold": 25,
                "max_trades_per_day": 100, "afternoon_entries": True,
                "entry_window_minutes": 360,
            }
            try:
                res = engine.run_backtest_config(
                    config_override=override, start_date=PERIOD[0], end_date=PERIOD[1],
                    symbol=sym, cached_bars=bars, cached_sma={}, cached_vix={},
                )
                trades = res.get("trades_list", [])
                for t in trades:
                    et = pd.to_datetime(t["entry_time"])
                    if et.tz is None: et = et.tz_localize("UTC").tz_convert(_ET)
                    else: et = et.tz_convert(_ET)
                    signal_days[et.date()].add(strat)
            except Exception as e:
                print(f"  {sym}/{strat}: error {e}")

        print(f"\n=== {sym} ===")
        # Distribution: how many days had N strategies fire?
        agreement_dist = defaultdict(int)
        for day, strats in signal_days.items():
            agreement_dist[len(strats)] += 1
        for n in sorted(agreement_dist):
            print(f"  {n} strategies: {agreement_dist[n]} days")

        # Outcome by agreement level: re-run each strategy and tag trades by agreement
        # Simplification: count how many trades happened on multi-agreement days
        for strat in STRATEGIES:
            engine = TradingEngine({**cfg, "active_strategy": strat})
            override = {
                "active_strategy": strat, "stop_loss_pct": 0.5, "reward_risk": 3.0,
                "volume_mult": 1.2, "regime_filter": True, "vix_threshold": 25,
                "max_trades_per_day": 100, "afternoon_entries": True,
                "entry_window_minutes": 360,
            }
            try:
                res = engine.run_backtest_config(
                    config_override=override, start_date=PERIOD[0], end_date=PERIOD[1],
                    symbol=sym, cached_bars=bars, cached_sma={}, cached_vix={},
                )
                trades = res.get("trades_list", [])
                # Bucket trades by agreement count on entry day
                bucket = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})
                for t in trades:
                    et = pd.to_datetime(t["entry_time"])
                    if et.tz is None: et = et.tz_localize("UTC").tz_convert(_ET)
                    else: et = et.tz_convert(_ET)
                    n_agree = len(signal_days.get(et.date(), set()))
                    bucket[n_agree]["n"] += 1
                    bucket[n_agree]["pnl"] += float(t["pnl"])
                    if float(t["pnl"]) > 0: bucket[n_agree]["wins"] += 1
                print(f"  {strat:<14} trades by agreement (1=alone, 3=all 3 agree):")
                for n in sorted(bucket):
                    b = bucket[n]
                    wr = 100 * b["wins"] / b["n"] if b["n"] else 0
                    print(f"    {n} agreed: {b['n']:>3} trades  pnl=${b['pnl']:+8.2f}  wr={wr:.0f}%")
                results.append({
                    "symbol": sym, "strategy": strat, "agreement_buckets": dict(bucket),
                })
            except: pass

    out = _ROOT / "logs/research/studies_2026_04_26/ensemble_study.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f: json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
