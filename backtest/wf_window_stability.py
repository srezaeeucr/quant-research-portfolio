#!/usr/bin/env python3
"""
Regime Stability Study (simplified WF variant).

For each of the 100 V4 passers, test the config's SL/RR on 12 quarterly
windows across 2022-2024. Output: PF per quarter, count of profitable quarters.

A robust config should win a majority of quarters — not just average out well.
If a config wins only 4/12 quarters but has a great average because of one
outlier, it's regime-dependent, not robust.

Uses the existing per-symbol cached bar files (TSLA_2022-01-01_2022-12-31,
TSLA_2023-01-01_2025-01-01, etc).
"""
import os
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import sys
import time as _time
import json
import glob
import pickle
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_ROOT = Path(__file__).resolve().parents[1]

from dotenv import load_dotenv
for p in [_ROOT / ".env", Path.home() / "projects" / "dl_course" / ".env",
          Path.home() / "projects" / "day-trading-bot" / ".env"]:
    if p.exists():
        load_dotenv(p)
        break

import yaml
import pandas as pd
import pytz

_ET = pytz.timezone("America/New_York")

# 12 quarterly test windows (2022 Q1 to 2024 Q4)
QUARTERS = [
    ("2022-01-01", "2022-03-31", "2022Q1"),
    ("2022-04-01", "2022-06-30", "2022Q2"),
    ("2022-07-01", "2022-09-30", "2022Q3"),
    ("2022-10-01", "2022-12-31", "2022Q4"),
    ("2023-01-01", "2023-03-31", "2023Q1"),
    ("2023-04-01", "2023-06-30", "2023Q2"),
    ("2023-07-01", "2023-09-30", "2023Q3"),
    ("2023-10-01", "2023-12-31", "2023Q4"),
    ("2024-01-01", "2024-03-31", "2024Q1"),
    ("2024-04-01", "2024-06-30", "2024Q2"),
    ("2024-07-01", "2024-09-30", "2024Q3"),
    ("2024-10-01", "2024-12-31", "2024Q4"),
]

# Preferred cache files per quarter (largest available that contains the quarter)
QUARTER_TO_CACHE = {
    "2022Q1": ("2022-01-01", "2022-12-31"),
    "2022Q2": ("2022-01-01", "2022-12-31"),
    "2022Q3": ("2022-01-01", "2022-12-31"),
    "2022Q4": ("2022-01-01", "2022-12-31"),
    "2023Q1": ("2023-01-01", "2025-01-01"),
    "2023Q2": ("2023-01-01", "2025-01-01"),
    "2023Q3": ("2023-01-01", "2025-01-01"),
    "2023Q4": ("2023-01-01", "2025-01-01"),
    "2024Q1": ("2023-01-01", "2025-01-01"),
    "2024Q2": ("2023-01-01", "2025-01-01"),
    "2024Q3": ("2023-01-01", "2025-01-01"),
    "2024Q4": ("2023-01-01", "2025-01-01"),
}

# Alternates if primary cache not found
CACHE_FALLBACKS = {
    ("2022-01-01", "2022-12-31"): [
        ("2022-01-01", "2022-09-30"),
        ("2022-04-01", "2022-12-31"),
    ],
    ("2023-01-01", "2025-01-01"): [
        ("2022-10-01", "2023-06-30"),
        ("2023-01-01", "2023-09-30"),
        ("2023-04-01", "2023-12-31"),
        ("2023-07-01", "2024-03-31"),
        ("2023-10-01", "2024-06-30"),
        ("2024-01-01", "2024-09-30"),
        ("2024-04-01", "2024-12-31"),
    ],
}


def load_cached(cache_dir, sym, start, end):
    for ext in ["parquet", "pkl"]:
        p = cache_dir / f"{sym}_{start}_{end}.{ext}"
        if p.exists():
            if ext == "parquet":
                return pd.read_parquet(p)
            with open(p, "rb") as f:
                return pickle.load(f)
    return None


def load_sma(cache_dir, sym):
    from datetime import date
    jf = cache_dir / f"{sym}_sma.json"
    if jf.exists():
        data = json.load(open(jf))
        return {date.fromisoformat(k): v for k, v in data.items()}
    p = cache_dir / f"{sym}_sma.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return {}


def get_bars_for_quarter(cache_dir, sym, quarter_label):
    """Load bars that contain the given quarter."""
    primary = QUARTER_TO_CACHE[quarter_label]
    bars = load_cached(cache_dir, sym, primary[0], primary[1])
    if bars is not None and not bars.empty:
        return bars
    # Fallbacks
    for start, end in CACHE_FALLBACKS.get(primary, []):
        bars = load_cached(cache_dir, sym, start, end)
        if bars is not None and not bars.empty:
            # Check if it contains the quarter
            qs, qe, _ = next(q for q in QUARTERS if q[2] == quarter_label)
            ts = pd.to_datetime(bars["timestamp"])
            if ts.dt.tz is not None:
                ts = ts.dt.tz_convert(_ET)
            if ts.min() <= qs and ts.max() >= qe:
                return bars
    return None


def validate_passer(args):
    passer, cfg_path, cache_dir_str, out_dir_str = args

    from src.engine import TradingEngine

    cache_dir = Path(cache_dir_str)
    out_dir = Path(out_dir_str)

    strat = passer["strategy"]
    sym = passer["symbol"]
    sl = passer["sl"]
    rr = passer["rr"]

    out_path = out_dir / f"{strat}_{sym}_{sl}_{rr}.json"
    if out_path.exists():
        try:
            return json.load(open(out_path))
        except:
            pass

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["mode"] = "backtest"
    cfg.setdefault("account", {})["balance"] = 500.0
    engine = TradingEngine(cfg)

    sma = load_sma(cache_dir, sym)

    base = {
        "active_strategy": strat, "stop_loss_pct": sl, "reward_risk": rr,
        "volume_mult": 1.2, "regime_filter": True, "vix_threshold": 25,
        "max_trades_per_day": 100, "afternoon_entries": True,
        "entry_window_minutes": 360,
    }

    result = {
        "strategy": strat, "symbol": sym, "sl": sl, "rr": rr,
        "quarters": {},
    }

    # Cache bars per cache range (so we only load each range once)
    range_cache = {}

    for q_start, q_end, q_label in QUARTERS:
        cache_key = QUARTER_TO_CACHE[q_label]
        if cache_key not in range_cache:
            range_cache[cache_key] = get_bars_for_quarter(cache_dir, sym, q_label)
        bars = range_cache[cache_key]
        if bars is None or bars.empty:
            result["quarters"][q_label] = {"error": "no bars"}
            continue
        try:
            m = engine.run_backtest_config(
                config_override=base,
                start_date=q_start, end_date=q_end, symbol=sym,
                cached_bars=bars, cached_sma=sma, cached_vix={},
            )
            result["quarters"][q_label] = {
                "pf": round(m.get("profit_factor", 0), 3),
                "trades": m.get("total_trades", 0),
                "wr": round(m.get("win_rate", 0), 2),
                "pnl": round(m.get("total_pnl", 0), 2),
            }
        except Exception as e:
            result["quarters"][q_label] = {"error": str(e)[:100]}

    # Compute stability stats
    valid = [q for q in result["quarters"].values() if "pf" in q and q.get("trades", 0) >= 3]
    n_quarters = len(valid)
    wins = sum(1 for q in valid if q["pf"] > 1.0)
    pnls = [q["pnl"] for q in valid]
    result["stability"] = {
        "n_valid_quarters": n_quarters,
        "quarters_profitable": wins,
        "win_rate_quarterly": round(wins / n_quarters, 3) if n_quarters else 0,
        "avg_pf": round(sum(q["pf"] for q in valid) / n_quarters, 3) if n_quarters else 0,
        "total_pnl": round(sum(pnls), 2),
        "min_quarterly_pnl": round(min(pnls), 2) if pnls else 0,
        "max_quarterly_pnl": round(max(pnls), 2) if pnls else 0,
    }
    # Stable if > 8/12 quarters profitable AND min PF > 0.7
    stable = (n_quarters >= 10
              and wins / n_quarters >= 0.67
              and min((q["pf"] for q in valid), default=0) >= 0.7)
    result["robust"] = stable

    json.dump(result, open(out_path, "w"))
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=19)
    parser.add_argument("--passers-dir", default=str(_ROOT / "logs" / "research" / "val_v4"))
    args = parser.parse_args()

    cache_dir = _ROOT / "logs" / "research" / "bar_cache"
    out_dir = _ROOT / "logs" / "research" / "wf_stability"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the 100 V4 passers
    passers_src = Path(args.passers_dir)
    passers = []
    for f in passers_src.glob("*.json"):
        try:
            r = json.load(open(f))
        except:
            continue
        if r.get("full_2yr_trades", 0) < 10:
            continue
        slip = (r.get("slip_full_2yr", {}) or {}).get("0.05", 0)
        if r.get("wf_wins", 0) >= 6 and r.get("mc_p_value", 1.0) <= 0.05 and slip >= 1.0:
            passers.append(r)

    print(f"Loaded {len(passers)} V4 passers", flush=True)
    print(f"Testing on {len(QUARTERS)} quarterly windows: 2022Q1 - 2024Q4", flush=True)
    print(f"Workers: {args.workers}", flush=True)

    cfg_path = _ROOT / "config" / "config.yaml"
    tasks = [(p, str(cfg_path), str(cache_dir), str(out_dir)) for p in passers]

    start = _time.time()
    with mp.Pool(args.workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(validate_passer, tasks)):
            results.append(r)
            if (i + 1) % 10 == 0:
                elapsed = (_time.time() - start) / 60
                print(f"  [{i+1}/{len(tasks)}] {elapsed:.1f}min elapsed", flush=True)

    # Summary
    robust = [r for r in results if r.get("robust", False)]
    print(f"\n{'='*70}")
    print(f"ROBUST (>= 67% quarters profitable, min PF >= 0.7): {len(robust)}/{len(results)}")
    print(f"{'='*70}")

    by_sym = defaultdict(lambda: {"robust": 0, "fail": 0})
    for r in results:
        if r.get("robust", False):
            by_sym[r["symbol"]]["robust"] += 1
        else:
            by_sym[r["symbol"]]["fail"] += 1
    print("\nPer-symbol:")
    for s in sorted(by_sym):
        d = by_sym[s]
        total = d["robust"] + d["fail"]
        print(f"  {s:<8} {d['robust']}/{total} robust")

    # Top robust configs
    robust.sort(key=lambda r: -r.get("stability", {}).get("win_rate_quarterly", 0))
    print("\nTop 10 robust configs (ranked by quarterly win rate):")
    print(f"  {'strategy':<14} {'sym':<6} {'sl':<5} {'rr':<5} {'Q_wins':<8} {'avg_PF':<8} {'total_pnl':<10} {'worst_q':<8}")
    for r in robust[:10]:
        s = r["stability"]
        print(f"  {r['strategy']:<14} {r['symbol']:<6} {r['sl']:<5} {r['rr']:<5} "
              f"{s['quarters_profitable']}/{s['n_valid_quarters']:<6} {s['avg_pf']:<8.2f} "
              f"${s['total_pnl']:<+10.2f} ${s['min_quarterly_pnl']:<+8.2f}")

    elapsed = (_time.time() - start) / 60
    print(f"\nTotal elapsed: {elapsed:.1f}min", flush=True)


if __name__ == "__main__":
    main()
