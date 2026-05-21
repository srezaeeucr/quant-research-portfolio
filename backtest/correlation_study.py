#!/usr/bin/env python3
"""
Correlation Study — How do the 4 live configs interact?

Runs each of the 4 live configs on full_2yr (2023-01-01 → 2025-01-01) and
computes:
  1. Daily PnL correlation matrix between symbols
  2. Signal co-occurrence (how often do 2+ symbols fire on the same bar)
  3. Capital concurrency (how often are N positions open simultaneously)
"""
import os, sys, pickle, json
from pathlib import Path
from collections import defaultdict
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import yaml
import pandas as pd
import pytz
from src.engine import TradingEngine

_ET = pytz.timezone("America/New_York")
_ROOT = Path(__file__).resolve().parents[1]
CACHE = _ROOT / "logs" / "research" / "bar_cache"

LIVE_CONFIGS = [
    {"strategy": "momentum",      "symbol": "AMD",  "sl": 0.3,  "rr": 1.5},
    {"strategy": "momentum",      "symbol": "META", "sl": 0.75, "rr": 4.0},
    {"strategy": "orb",           "symbol": "COIN", "sl": 1.0,  "rr": 5.0},
    {"strategy": "ema_crossover", "symbol": "TSLA", "sl": 0.75, "rr": 3.0},
]
PERIOD_START = "2023-01-01"
PERIOD_END = "2025-01-01"


def load_cached(sym, start, end):
    pq = CACHE / f"{sym}_{start}_{end}.parquet"
    if pq.exists(): return pd.read_parquet(pq)
    p = CACHE / f"{sym}_{start}_{end}.pkl"
    if p.exists():
        with open(p, "rb") as f: return pickle.load(f)
    return None


def load_sma(sym):
    jf = CACHE / f"{sym}_sma.json"
    if jf.exists():
        return {date.fromisoformat(k): v for k, v in json.load(open(jf)).items()}
    p = CACHE / f"{sym}_sma.pkl"
    if p.exists():
        with open(p, "rb") as f: return pickle.load(f)
    return {}


def run_one(cfg_item):
    cfg_path = _ROOT / "config" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["mode"] = "backtest"
    cfg["account"] = {"balance": 500.0, "max_position_pct": 0.95}
    engine = TradingEngine(cfg)

    bars = load_cached(cfg_item["symbol"], PERIOD_START, PERIOD_END)
    if bars is None:
        print(f"  NO DATA for {cfg_item['symbol']}")
        return None

    override = {
        "active_strategy": cfg_item["strategy"],
        "stop_loss_pct": cfg_item["sl"],
        "reward_risk": cfg_item["rr"],
        "volume_mult": 1.2,
        "regime_filter": True,
        "vix_threshold": 25,
        "max_trades_per_day": 100,
        "afternoon_entries": True,
        "entry_window_minutes": 360,
    }
    result = engine.run_backtest_config(
        config_override=override,
        start_date=PERIOD_START, end_date=PERIOD_END,
        symbol=cfg_item["symbol"],
        cached_bars=bars, cached_sma=load_sma(cfg_item["symbol"]), cached_vix={},
    )
    trades = result.get("trades_list", [])
    return trades


def compute_daily_pnl(trades_by_key):
    """Return DataFrame with columns per (strat/symbol), rows per day."""
    all_days = set()
    per_key = {}
    for key, trades in trades_by_key.items():
        day_pnl = defaultdict(float)
        for t in trades:
            et = pd.to_datetime(t["entry_time"])
            if et.tz is None: et = et.tz_localize("UTC").tz_convert(_ET)
            else: et = et.tz_convert(_ET)
            day = et.date()
            day_pnl[day] += float(t["pnl"])
            all_days.add(day)
        per_key[key] = day_pnl

    # Build DataFrame
    days_sorted = sorted(all_days)
    data = {}
    for key, day_pnl in per_key.items():
        data[key] = [day_pnl.get(d, 0.0) for d in days_sorted]
    df = pd.DataFrame(data, index=days_sorted)
    return df


def compute_concurrency(trades_by_key):
    """For each minute, count how many positions are open across all 4 strategies."""
    events = []  # (time, +1 or -1)
    for key, trades in trades_by_key.items():
        for t in trades:
            et = pd.to_datetime(t["entry_time"])
            xt = pd.to_datetime(t["exit_time"])
            if et.tz is None: et = et.tz_localize("UTC").tz_convert(_ET)
            else: et = et.tz_convert(_ET)
            if xt.tz is None: xt = xt.tz_localize("UTC").tz_convert(_ET)
            else: xt = xt.tz_convert(_ET)
            events.append((et, key, +1))
            events.append((xt, key, -1))
    events.sort()

    # Walk through events, maintain open-set per bar
    open_count_dist = defaultdict(int)  # count of minutes at each concurrency level
    current_open = 0
    prev_t = None
    for t, key, delta in events:
        if prev_t is not None:
            minutes = int((t - prev_t).total_seconds() / 60)
            open_count_dist[current_open] += max(minutes, 0)
        current_open += delta
        prev_t = t
    return open_count_dist


def compute_signal_cooccurrence(trades_by_key):
    """For each trading day, which symbols fired entries?"""
    day_entries = defaultdict(set)  # day -> set of symbols that entered
    for key, trades in trades_by_key.items():
        for t in trades:
            et = pd.to_datetime(t["entry_time"])
            if et.tz is None: et = et.tz_localize("UTC").tz_convert(_ET)
            else: et = et.tz_convert(_ET)
            day_entries[et.date()].add(key)

    # Count days by number of symbols firing
    count_dist = defaultdict(int)
    for day, syms in day_entries.items():
        count_dist[len(syms)] += 1
    return count_dist, day_entries


def main():
    print("=" * 70)
    print("CORRELATION STUDY — 4 Live Configs on full_2yr (2023-2024)")
    print("=" * 70)

    trades_by_key = {}
    for cfg in LIVE_CONFIGS:
        key = f'{cfg["strategy"][:5]}/{cfg["symbol"]}'
        print(f"\nRunning {cfg['strategy']}/{cfg['symbol']} SL={cfg['sl']} RR={cfg['rr']} ...", flush=True)
        trades = run_one(cfg)
        if trades:
            n = len(trades)
            pnl = sum(float(t["pnl"]) for t in trades)
            print(f"  {n} trades, total PnL=${pnl:+.2f}")
            trades_by_key[key] = trades

    if len(trades_by_key) < 2:
        print("Not enough data to compute correlations")
        return

    # 1. Daily PnL correlation
    print("\n" + "=" * 70)
    print("1. DAILY PNL CORRELATION (Pearson)")
    print("=" * 70)
    df = compute_daily_pnl(trades_by_key)
    df_traded = df[(df != 0).any(axis=1)]  # only days with any trade
    print(f"Days with any trade: {len(df_traded)}")
    print()
    corr = df_traded.corr()
    print(corr.round(3).to_string())

    # 2. Signal co-occurrence
    print("\n" + "=" * 70)
    print("2. SIGNAL CO-OCCURRENCE (days with N symbols firing entries)")
    print("=" * 70)
    co_dist, day_entries = compute_signal_cooccurrence(trades_by_key)
    total_days = sum(co_dist.values())
    for n in sorted(co_dist.keys()):
        pct = 100 * co_dist[n] / total_days
        print(f"  {n} symbols firing: {co_dist[n]:>4} days  ({pct:5.1f}%)")

    # Pair-wise co-fire counts
    print("\n  Pair-wise co-fire counts (days both entered):")
    keys = list(trades_by_key.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            both = sum(1 for day, syms in day_entries.items() if keys[i] in syms and keys[j] in syms)
            print(f"    {keys[i]:<14} & {keys[j]:<14}  {both:>3} days")

    # 3. Capital concurrency
    print("\n" + "=" * 70)
    print("3. CAPITAL CONCURRENCY (% of trading time with N positions open)")
    print("=" * 70)
    conc = compute_concurrency(trades_by_key)
    total_min = sum(conc.values())
    for n in sorted(conc.keys()):
        pct = 100 * conc[n] / total_min if total_min else 0
        print(f"  {n} positions open: {conc[n]:>8,} min  ({pct:5.1f}%)")

    # Save
    out = _ROOT / "logs" / "research" / "correlation_study.json"
    output = {
        "period": f"{PERIOD_START} to {PERIOD_END}",
        "configs": LIVE_CONFIGS,
        "trade_counts": {k: len(v) for k, v in trades_by_key.items()},
        "pnl_correlation": corr.to_dict(),
        "signal_cooccurrence": dict(co_dist),
        "capital_concurrency": dict(conc),
    }
    with open(out, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
