#!/usr/bin/env python3
"""
Research Wave 1 — three high-priority experiments.

  --experiment slippage   Slippage stress test (top combos × 4 slippage levels)
  --experiment sizing     Position sizing sweep (top combos × 6 risk levels)
  --experiment recent     Most recent 3 months out-of-sample test

All three pull the same "top N candidates" set from results.db so they're
comparable. The top candidates are the unique (strategy, symbol, sl, rr,
vol, regime, vix, max_t, pm) tuples with the highest in-sample composite
profit factor across the 4 standard periods, with ≥30 trades.

Designed to be run on superpower in parallel (one process per experiment).
Output: a CSV per experiment under logs/research/.
"""
import os
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import argparse
import csv
import sqlite3
import sys
import time as _time_mod
from datetime import date, timedelta
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[1]

# In-sample periods (used for the top-candidate selection)
_IN_SAMPLE_PERIODS = ("bull_2023", "bull_2024", "bear_2025", "full_2yr")


# ─────────────────────────────────────────────────────────────────────────
# Top candidate selection
# ─────────────────────────────────────────────────────────────────────────

def select_top_candidates(n: int = 10) -> List[Dict]:
    """Return top-N (strategy, symbol, sl, rr, vol, regime, vix, max_t, pm)
    tuples by average in-sample PF across periods (≥30 trades, ≥3 periods)."""
    conn = sqlite3.connect(str(_ROOT / "logs" / "results.db"), timeout=30)
    df = pd.read_sql_query(
        "SELECT * FROM backtest_runs "
        "WHERE total_trades >= 30 AND profit_factor > 0",
        conn
    )
    conn.close()

    df["pf_safe"] = df["profit_factor"].replace([float("inf")], 5.0)
    df = df[df["period_label"].isin(_IN_SAMPLE_PERIODS)]

    # Group by core combo and average PF across periods
    keys = ["strategy", "symbol", "stop_loss_pct", "reward_risk",
            "volume_mult", "regime_filter", "vix_threshold",
            "max_trades_per_day", "afternoon_entries"]
    agg = df.groupby(keys).agg(
        n_periods=("period_label", "nunique"),
        avg_pf=("pf_safe", "mean"),
        avg_pnl=("total_pnl", "mean"),
        max_dd=("max_drawdown", "max"),
        total_trades=("total_trades", "sum"),
    ).reset_index()

    # Require profitable in at least 3 periods
    agg = agg[(agg["n_periods"] >= 3) & (agg["avg_pf"] > 1.2)]
    agg = agg.sort_values("avg_pf", ascending=False)

    # Dedupe at the (strategy, symbol, sl, rr, pm) level — pick the BEST
    # variant of each core strategy/symbol/RR. This avoids the top N being
    # filled with filter-dimension duplicates.
    dedupe_keys = ["strategy", "symbol", "stop_loss_pct", "reward_risk",
                   "afternoon_entries"]
    agg = agg.drop_duplicates(subset=dedupe_keys, keep="first")

    # Top by PF first (we'll prepend the must-include candidates after)
    agg = agg.head(n)

    # MUST-INCLUDE candidates with hardcoded parameters — these are the
    # current live config and the validated 2nd candidate. They may not be
    # the absolute top by PF (they were picked for holdout robustness), but
    # we need experiments to speak directly to them.
    must_include = [
        {  # Live config: EMA/AAPL/R:R 4.0
            "strategy": "ema_crossover", "symbol": "AAPL",
            "stop_loss_pct": 0.5, "reward_risk": 4.0,
            "volume_mult": 1.0, "regime_filter": True,
            "vix_threshold": 30, "max_trades_per_day": 2,
            "afternoon_entries": False,
            "_avg_pf": 0.0, "_avg_pnl": 0.0,
            "_label_prefix": "[LIVE] ",
        },
        {  # Validated 2nd candidate: ORB/NVDA/R:R 2.5
            "strategy": "orb", "symbol": "NVDA",
            "stop_loss_pct": 0.3, "reward_risk": 2.5,
            "volume_mult": 1.2, "regime_filter": True,
            "vix_threshold": 30, "max_trades_per_day": 1,
            "afternoon_entries": False,
            "_avg_pf": 0.0, "_avg_pnl": 0.0,
            "_label_prefix": "[2ND] ",
        },
    ]

    candidates = list(must_include)
    for _, row in agg.iterrows():
        candidates.append({
            "strategy":           row["strategy"],
            "symbol":             row["symbol"],
            "stop_loss_pct":      float(row["stop_loss_pct"]),
            "reward_risk":        float(row["reward_risk"]),
            "volume_mult":        float(row["volume_mult"]),
            "regime_filter":      bool(row["regime_filter"]),
            "vix_threshold":      int(row["vix_threshold"]),
            "max_trades_per_day": int(row["max_trades_per_day"]),
            "afternoon_entries":  bool(row["afternoon_entries"]),
            "_avg_pf":            float(row["avg_pf"]),
            "_avg_pnl":           float(row["avg_pnl"]),
            "_label_prefix":      "",
        })
    return candidates[:n + 2]   # +2 to keep both must-include alongside top N


def candidate_label(c: Dict) -> str:
    pm = "PM" if c["afternoon_entries"] else "AM"
    prefix = c.get("_label_prefix", "")
    return (f"{prefix}{c['strategy']:<14} {c['symbol']:<5} "
            f"sl={c['stop_loss_pct']:.2f}% rr={c['reward_risk']:.1f} "
            f"vol={c['volume_mult']:.1f} reg={'T' if c['regime_filter'] else 'F'} "
            f"vix={c['vix_threshold']} mt={c['max_trades_per_day']} {pm}")


def base_override(c: Dict) -> Dict:
    return {
        "active_strategy":    c["strategy"],
        "stop_loss_pct":      c["stop_loss_pct"],
        "reward_risk":        c["reward_risk"],
        "volume_mult":        c["volume_mult"],
        "regime_filter":      c["regime_filter"],
        "vix_threshold":      c["vix_threshold"],
        "max_trades_per_day": c["max_trades_per_day"],
        "afternoon_entries":  c["afternoon_entries"],
    }


# ─────────────────────────────────────────────────────────────────────────
# Experiment 1: SLIPPAGE STRESS TEST
# ─────────────────────────────────────────────────────────────────────────

def run_slippage(engine, candidates, period_label, start, end):
    """Run each candidate at 4 slippage levels on the given period."""
    SLIPPAGE_LEVELS = [0.0, 0.05, 0.10, 0.20]   # in percent
    bars_cache = {}
    rows = []
    for ci, c in enumerate(candidates, 1):
        cache_key = (c["symbol"], start, end)
        if cache_key not in bars_cache:
            try:
                bars_cache[cache_key] = engine.fetcher.fetch_historical(
                    c["symbol"], start, end, filter_windows=False
                )
            except Exception:
                bars_cache[cache_key] = None
        bars = bars_cache[cache_key]
        if bars is None or bars.empty:
            print(f"  [{ci}/{len(candidates)}] {candidate_label(c)}  → NO DATA")
            continue

        for slip in SLIPPAGE_LEVELS:
            override = base_override(c)
            override["slippage_pct"] = slip
            t0 = _time_mod.time()
            m = engine.run_backtest_config(
                config_override=override,
                start_date=start, end_date=end,
                symbol=c["symbol"], cached_bars=bars,
            )
            elapsed = _time_mod.time() - t0
            pf = m["profit_factor"]
            if pf == float("inf"):
                pf = 999.0
            rows.append({
                "candidate":   candidate_label(c),
                "strategy":    c["strategy"],
                "symbol":      c["symbol"],
                "rr":          c["reward_risk"],
                "sl":          c["stop_loss_pct"],
                "slippage":    slip,
                "trades":      m["total_trades"],
                "win_rate":    round(m["win_rate"], 2),
                "pf":          round(pf, 3),
                "pnl":         round(m["total_pnl"], 2),
                "dd":          round(m["max_drawdown"], 2),
                "period":      period_label,
            })
        baseline = next(r for r in rows[-len(SLIPPAGE_LEVELS):] if r["slippage"] == 0)
        worst   = next(r for r in rows[-len(SLIPPAGE_LEVELS):] if r["slippage"] == 0.20)
        print(f"  [{ci:>2}/{len(candidates)}] {candidate_label(c):<70} "
              f"PF {baseline['pf']:.2f} → {worst['pf']:.2f}  "
              f"P&L {baseline['pnl']:+.2f} → {worst['pnl']:+.2f}")
    return rows


# ─────────────────────────────────────────────────────────────────────────
# Experiment 2: POSITION SIZING SWEEP
# ─────────────────────────────────────────────────────────────────────────

def run_sizing(engine, candidates, period_label, start, end):
    SIZE_LEVELS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    bars_cache = {}
    rows = []
    for ci, c in enumerate(candidates, 1):
        cache_key = (c["symbol"], start, end)
        if cache_key not in bars_cache:
            try:
                bars_cache[cache_key] = engine.fetcher.fetch_historical(
                    c["symbol"], start, end, filter_windows=False
                )
            except Exception:
                bars_cache[cache_key] = None
        bars = bars_cache[cache_key]
        if bars is None or bars.empty:
            print(f"  [{ci}/{len(candidates)}] {candidate_label(c)}  → NO DATA")
            continue

        for size in SIZE_LEVELS:
            override = base_override(c)
            override["max_risk_per_trade_pct"] = size
            m = engine.run_backtest_config(
                config_override=override,
                start_date=start, end_date=end,
                symbol=c["symbol"], cached_bars=bars,
            )
            pf = m["profit_factor"]
            if pf == float("inf"):
                pf = 999.0
            rows.append({
                "candidate":   candidate_label(c),
                "strategy":    c["strategy"],
                "symbol":      c["symbol"],
                "risk_pct":    size,
                "trades":      m["total_trades"],
                "win_rate":    round(m["win_rate"], 2),
                "pf":          round(pf, 3),
                "pnl":         round(m["total_pnl"], 2),
                "dd":          round(m["max_drawdown"], 2),
                "period":      period_label,
            })
        # Print quick summary line
        slice_ = rows[-len(SIZE_LEVELS):]
        best = max(slice_, key=lambda r: r["pf"])
        print(f"  [{ci:>2}/{len(candidates)}] {candidate_label(c):<70} "
              f"best risk={best['risk_pct']}% → PF {best['pf']:.2f}, P&L {best['pnl']:+.2f}")
    return rows


# ─────────────────────────────────────────────────────────────────────────
# Experiment 3: RECENT 3 MONTHS OUT-OF-SAMPLE TEST
# ─────────────────────────────────────────────────────────────────────────

def run_recent(engine, candidates):
    """Test top candidates on the most recent 90 days available from Alpaca."""
    today = date.today()
    end_date = today.isoformat()
    start_date = (today - timedelta(days=90)).isoformat()
    print(f"  date range: {start_date} → {end_date}")

    bars_cache = {}
    rows = []
    for ci, c in enumerate(candidates, 1):
        cache_key = (c["symbol"], start_date, end_date)
        if cache_key not in bars_cache:
            try:
                bars_cache[cache_key] = engine.fetcher.fetch_historical(
                    c["symbol"], start_date, end_date, filter_windows=False
                )
            except Exception as exc:
                print(f"  fetch failed for {c['symbol']}: {exc}")
                bars_cache[cache_key] = None
        bars = bars_cache[cache_key]
        if bars is None or bars.empty:
            print(f"  [{ci}/{len(candidates)}] {candidate_label(c)}  → NO DATA")
            continue

        override = base_override(c)
        m = engine.run_backtest_config(
            config_override=override,
            start_date=start_date, end_date=end_date,
            symbol=c["symbol"], cached_bars=bars,
        )
        pf = m["profit_factor"]
        if pf == float("inf"):
            pf = 999.0
        row = {
            "candidate":   candidate_label(c),
            "strategy":    c["strategy"],
            "symbol":      c["symbol"],
            "rr":          c["reward_risk"],
            "in_sample_pf": round(c["_avg_pf"], 3),
            "trades":      m["total_trades"],
            "win_rate":    round(m["win_rate"], 2),
            "pf":          round(pf, 3),
            "pnl":         round(m["total_pnl"], 2),
            "dd":          round(m["max_drawdown"], 2),
            "period":      f"{start_date}_{end_date}",
        }
        rows.append(row)
        print(f"  [{ci:>2}/{len(candidates)}] {candidate_label(c):<70} "
              f"in-sample PF {c['_avg_pf']:.2f} → recent PF {pf:.2f} "
              f"({row['trades']} trades)")
    return rows


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True,
                        choices=["slippage", "sizing", "recent"])
    parser.add_argument("--n", type=int, default=10,
                        help="Number of top candidates to test")
    parser.add_argument("--period", default="full_2yr",
                        help="Period for slippage/sizing experiments")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end",   default="2025-01-01")
    args = parser.parse_args()

    print(f"\n=== Wave 1 :: {args.experiment.upper()} ===")
    print(f"Selecting top {args.n} candidates from results.db ...")
    candidates = select_top_candidates(args.n)
    print(f"Got {len(candidates)} candidates:")
    for i, c in enumerate(candidates, 1):
        print(f"  #{i}  avg PF {c['_avg_pf']:.2f}  pnl ${c['_avg_pnl']:+.2f}  "
              f"{candidate_label(c)}")

    if not candidates:
        print("No candidates — aborting.")
        return

    with open(_ROOT / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["mode"] = "backtest"

    from src.engine import TradingEngine
    engine = TradingEngine(cfg)

    print(f"\nRunning {args.experiment} experiment...")
    t0 = _time_mod.time()

    if args.experiment == "slippage":
        rows = run_slippage(engine, candidates, args.period, args.start, args.end)
    elif args.experiment == "sizing":
        rows = run_sizing(engine, candidates, args.period, args.start, args.end)
    else:
        rows = run_recent(engine, candidates)

    elapsed = _time_mod.time() - t0
    print(f"\n  Done in {elapsed:.1f}s ({elapsed/60:.1f}m)")

    # Write CSV
    out_dir = _ROOT / "logs" / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"wave1_{args.experiment}.csv"
    if rows:
        keys = list(rows[0].keys())
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"  CSV: {out_path}")
        print(f"  Rows: {len(rows)}")


if __name__ == "__main__":
    main()
