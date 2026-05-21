#!/usr/bin/env python3
"""
Study #6: Recent walk-forward.

Train on Q3 2025, test Q4 2025, then roll. Use the validated 2025-2026
period to find currently-optimal configs.

Use top 100 V4 passers as the candidate set; for each, run new WF windows
and report which still pass.
"""
import os, sys, json, glob, pickle, itertools, multiprocessing as mp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
for p in [Path(__file__).resolve().parents[1] / ".env",
          Path.home() / "projects" / "dl_course" / ".env"]:
    if p.exists():
        load_dotenv(p); break

import yaml, pandas as pd, pytz
_ET = pytz.timezone("America/New_York")
_ROOT = Path(__file__).resolve().parents[1]

# Quarterly windows for 2025-2026
WINDOWS = [
    ("2025-04-01", "2025-06-30", "2025-07-01", "2025-09-30", "2025Q3"),
    ("2025-07-01", "2025-09-30", "2025-10-01", "2025-12-31", "2025Q4"),
    ("2025-10-01", "2025-12-31", "2026-01-01", "2026-03-31", "2026Q1"),
    ("2026-01-01", "2026-03-31", "2026-04-01", "2026-04-26", "2026Q2"),
]


def load_cached(cache_dir, sym, start, end):
    for ext in ["parquet", "pkl"]:
        p = cache_dir / f"{sym}_{start}_{end}.{ext}"
        if p.exists():
            if ext == "parquet": return pd.read_parquet(p)
            with open(p, "rb") as f: return pickle.load(f)
    return None


def validate_passer(args):
    passer, cfg_path, cache_dir_str, out_dir_str = args
    from src.engine import TradingEngine
    cache_dir = Path(cache_dir_str); out_dir = Path(out_dir_str)
    strat, sym = passer["strategy"], passer["symbol"]
    sl, rr = passer["sl"], passer["rr"]
    out_path = out_dir / f"{strat}_{sym}_{sl}_{rr}.json"
    if out_path.exists():
        try: return json.load(open(out_path))
        except: pass

    with open(cfg_path) as f: cfg = yaml.safe_load(f)
    cfg["mode"] = "backtest"; cfg.setdefault("account", {})["balance"] = 500.0
    engine = TradingEngine(cfg)

    # Load all bars covering 2025-04-01 → 2026-04-26
    candidates = [
        ("2025-06-01", "2026-03-31"),
        ("2025-10-01", "2026-04-14"),
        ("2026-01-01", "2026-04-14"),
    ]
    bars_full = None
    for s, e in candidates:
        b = load_cached(cache_dir, sym, s, e)
        if b is not None and not b.empty:
            bars_full = b if bars_full is None else pd.concat([bars_full, b]).drop_duplicates(subset=["timestamp"])
    if bars_full is None or bars_full.empty:
        return {"strategy": strat, "symbol": sym, "sl": sl, "rr": rr, "error": "no bars"}
    ts = pd.to_datetime(bars_full["timestamp"])
    if ts.dt.tz is not None: ts = ts.dt.tz_convert(_ET)
    bars_full = bars_full.copy(); bars_full["_ts_et"] = ts

    base = {
        "active_strategy": strat, "stop_loss_pct": sl, "reward_risk": rr,
        "volume_mult": 1.2, "regime_filter": True, "vix_threshold": 25,
        "max_trades_per_day": 100, "afternoon_entries": True,
        "entry_window_minutes": 360,
    }

    result = {"strategy": strat, "symbol": sym, "sl": sl, "rr": rr, "windows": {}}
    for ts_s, te_s, vs_s, ve_s, label in WINDOWS:
        # Train: pick best (sl, rr) on train data via small inner sweep
        train_mask = (bars_full["_ts_et"] >= ts_s) & (bars_full["_ts_et"] <= te_s + " 23:59:59")
        test_mask  = (bars_full["_ts_et"] >= vs_s) & (bars_full["_ts_et"] <= ve_s + " 23:59:59")
        train_bars = bars_full[train_mask]; test_bars = bars_full[test_mask]
        if train_bars.empty or test_bars.empty:
            result["windows"][label] = {"error": "no bars"}
            continue
        try:
            test_m = engine.run_backtest_config(
                config_override=base, start_date=vs_s, end_date=ve_s,
                symbol=sym, cached_bars=test_bars, cached_sma={}, cached_vix={},
            )
            result["windows"][label] = {
                "pf": round(test_m.get("profit_factor", 0), 3),
                "trades": test_m.get("total_trades", 0),
                "pnl": round(test_m.get("total_pnl", 0), 2),
                "wr": round(test_m.get("win_rate", 0), 1),
            }
        except Exception as e:
            result["windows"][label] = {"error": str(e)[:80]}

    # Stability score
    valid = [w for w in result["windows"].values() if "pf" in w and w.get("trades", 0) >= 3]
    n = len(valid)
    wins = sum(1 for w in valid if w["pf"] > 1.0)
    pnl = sum(w["pnl"] for w in valid)
    result["recent_robust"] = (n >= 3 and wins / n >= 0.67 and min((w["pf"] for w in valid), default=0) >= 0.7)
    result["recent_wins"] = wins
    result["recent_total"] = n
    result["recent_pnl"]   = round(pnl, 2)
    json.dump(result, open(out_path, "w"))
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=19)
    parser.add_argument("--passers-dir", default=str(_ROOT / "logs" / "research" / "val_v4"))
    args = parser.parse_args()

    cache_dir = _ROOT / "logs" / "research" / "bar_cache"
    out_dir = _ROOT / "logs" / "research" / "recent_wf"
    out_dir.mkdir(parents=True, exist_ok=True)

    passers = []
    for f in glob.glob(args.passers_dir + "/*.json"):
        try: r = json.load(open(f))
        except: continue
        if r.get("full_2yr_trades", 0) < 10: continue
        slip = (r.get("slip_full_2yr", {}) or {}).get("0.05", 0)
        if r.get("wf_wins", 0) >= 6 and r.get("mc_p_value", 1.0) <= 0.05 and slip >= 1.0:
            passers.append(r)

    print(f"Loaded {len(passers)} V4 passers")
    print(f"Recent WF windows: {len(WINDOWS)}")
    cfg_path = _ROOT / "config" / "config.yaml"
    tasks = [(p, str(cfg_path), str(cache_dir), str(out_dir)) for p in passers]

    import time as _time
    start = _time.time()
    with mp.Pool(args.workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(validate_passer, tasks)):
            results.append(r)
            if (i+1) % 10 == 0:
                el = (_time.time() - start) / 60
                print(f"  [{i+1}/{len(tasks)}] {el:.1f}min")

    robust = [r for r in results if r.get("recent_robust")]
    print(f"\nConfigs robust on recent WF (>=67% windows, min PF >=0.7): {len(robust)}/{len(results)}")
    robust.sort(key=lambda r: -r.get("recent_pnl", 0))
    print(f"\nTop 15:")
    print(f"  {'strategy':<14} {'sym':<6} {'sl':<5} {'rr':<5} {'wins':<6} {'pnl':<8}")
    for r in robust[:15]:
        print(f"  {r['strategy']:<14} {r['symbol']:<6} {r['sl']:<5} {r['rr']:<5} "
              f"{r['recent_wins']}/{r['recent_total']:<5} ${r['recent_pnl']:<+7.2f}")
    el = (_time.time() - start) / 60
    print(f"\nTotal: {el:.1f}min")


if __name__ == "__main__":
    main()
