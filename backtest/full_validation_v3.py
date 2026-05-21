#!/usr/bin/env python3
"""
Full Validation Chain V3 — subprocess workers with pre-cached data.
Step 1: Run prefetch_data.py to cache all bars to disk
Step 2: Run this — workers read from cache, no API calls, full parallelism

Usage:
  python backtest/prefetch_data.py                              # run once
  python backtest/full_validation_v3.py --slice 1/2 --workers 50  # Superpower
  python backtest/full_validation_v3.py --slice 2/2 --workers 80  # GCP
  python backtest/full_validation_v3.py --dry-run
"""
import os
import sys
import time as _time
import json
import subprocess
import itertools
from pathlib import Path
from collections import deque

_ROOT = Path(__file__).resolve().parents[1]

SYMBOLS = ["AMD", "META", "COIN", "TSLA", "NVDA", "SPY", "QQQ", "AAPL",
           "AMZN", "MSFT", "GOOGL", "JPM", "MSTR", "IWM",
           "XLK", "XLF", "XLE", "XLV", "XLI"]
STRATEGIES = ["orb", "ema_crossover", "momentum"]
STOP_LOSSES = [0.3, 0.5, 0.75, 1.0]
REWARD_RISKS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

WORKER_SCRIPT = '''
import os, sys, json, pickle
from pathlib import Path

PROJECT_ROOT = os.environ["PROJECT_ROOT"]
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
for p in [Path(PROJECT_ROOT) / ".env",
          Path.home() / "projects" / "dl_course" / ".env",
          Path.home() / "projects" / "day-trading-bot" / ".env"]:
    if p.exists():
        load_dotenv(p)
        break

import yaml
import itertools
import numpy as np
from src.engine import TradingEngine

strategy = os.environ["STRATEGY"]
symbol = os.environ["SYMBOL"]
sl = float(os.environ["SL"])
rr = float(os.environ["RR"])
out_path = os.environ["OUT_PATH"]
cache_dir = Path(os.environ["CACHE_DIR"])
cfg_path = os.environ["CFG_PATH"]

def load_cached(sym, start, end):
    """Load bars from cache. Try parquet first (cross-version), then pkl."""
    import pandas as pd
    pq = cache_dir / f"{sym}_{start}_{end}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    p = cache_dir / f"{sym}_{start}_{end}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None

def load_sma(sym):
    jf = cache_dir / f"{sym}_sma.json"
    if jf.exists():
        import json
        from datetime import date
        data = json.load(open(jf))
        return {date.fromisoformat(k): v for k, v in data.items()}
    p = cache_dir / f"{sym}_sma.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return {}

with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
cfg["mode"] = "backtest"
cfg.setdefault("account", {})["balance"] = 500.0

try:
    engine = TradingEngine(cfg)
except Exception as e:
    json.dump({"strategy":strategy,"symbol":symbol,"sl":sl,"rr":rr,"error":str(e)}, open(out_path,"w"))
    sys.exit(0)

base = {
    "active_strategy": strategy, "stop_loss_pct": sl, "reward_risk": rr,
    "volume_mult": 1.2, "regime_filter": True, "vix_threshold": 25,
    "max_trades_per_day": 100, "afternoon_entries": True, "entry_window_minutes": 360,
}

result = {
    "strategy": strategy, "symbol": symbol, "sl": sl, "rr": rr,
    "wf_wins": 0, "wf_total": 0, "wf_avg_test_pf": 0,
    "mc_real_pf": 0, "mc_random_pf": 0, "mc_p_value": 1.0,
    "slip_full_2yr": {}, "slip_holdout": {}, "slip_bear": {},
    "full_2yr_trades": 0, "full_2yr_pf": 0, "full_2yr_wr": 0,
    "full_2yr_avg_dur": 0,
}

# Load cached data
bars_2yr = load_cached(symbol, "2023-01-01", "2025-01-01")
sma = load_sma(symbol)

if bars_2yr is None or bars_2yr.empty:
    json.dump(result, open(out_path, "w"))
    sys.exit(0)

# Quick check
try:
    quick = engine.run_backtest_config(config_override=base, start_date="2023-01-01", end_date="2025-01-01",
                                        symbol=symbol, cached_bars=bars_2yr, cached_sma=sma, cached_vix={})
    result["full_2yr_trades"] = quick.get("total_trades", 0)
    result["full_2yr_pf"] = quick.get("profit_factor", 0)
    result["full_2yr_wr"] = quick.get("win_rate", 0)
    result["full_2yr_avg_dur"] = quick.get("avg_duration_min", 0)
    if quick.get("total_trades", 0) < 10:
        json.dump(result, open(out_path, "w"))
        sys.exit(0)
except Exception as e:
    result["error"] = str(e)
    json.dump(result, open(out_path, "w"))
    sys.exit(0)

# Walk-forward
WF_WINDOWS = [
    ("2022-01-01","2022-06-30","2022-07-01","2022-09-30"),
    ("2022-04-01","2022-09-30","2022-10-01","2022-12-31"),
    ("2022-07-01","2022-12-31","2023-01-01","2023-03-31"),
    ("2022-10-01","2023-03-31","2023-04-01","2023-06-30"),
    ("2023-01-01","2023-06-30","2023-07-01","2023-09-30"),
    ("2023-04-01","2023-09-30","2023-10-01","2023-12-31"),
    ("2023-07-01","2023-12-31","2024-01-01","2024-03-31"),
    ("2023-10-01","2024-03-31","2024-04-01","2024-06-30"),
    ("2024-01-01","2024-06-30","2024-07-01","2024-09-30"),
    ("2024-04-01","2024-09-30","2024-10-01","2024-12-31"),
]
# Map WF windows to cache date ranges
WF_CACHE_MAP = [
    ("2022-01-01","2022-09-30"),
    ("2022-04-01","2022-12-31"),
    ("2022-07-01","2023-03-31"),
    ("2022-10-01","2023-06-30"),
    ("2023-01-01","2023-09-30"),
    ("2023-04-01","2023-12-31"),
    ("2023-07-01","2024-03-31"),
    ("2023-10-01","2024-06-30"),
    ("2024-01-01","2024-09-30"),
    ("2024-04-01","2024-12-31"),
]
WF_PARAMS = list(itertools.product([0.3,0.5,0.75,1.0],[1.0,1.5,2.0,2.5,3.0,4.0,5.0],[1.0,1.2]))
wf_pfs = []
for idx, (ts,te,vs,ve) in enumerate(WF_WINDOWS):
    cache_start, cache_end = WF_CACHE_MAP[idx]
    all_bars = load_cached(symbol, cache_start, cache_end)
    if all_bars is None or all_bars.empty:
        continue
    import pytz
    _ET = pytz.timezone("America/New_York")
    all_bars_ts = all_bars["timestamp"].dt.tz_convert(_ET) if all_bars["timestamp"].dt.tz is not None else all_bars["timestamp"]
    train_mask = (all_bars_ts >= ts) & (all_bars_ts < te + " 23:59:59")
    test_mask = (all_bars_ts >= vs) & (all_bars_ts < ve + " 23:59:59")
    train_bars = all_bars[train_mask]
    test_bars = all_bars[test_mask]
    if train_bars.empty or test_bars.empty:
        continue
    best_pf, best_p = 0, None
    for p_sl,p_rr,p_vol in WF_PARAMS:
        try:
            m = engine.run_backtest_config(
                config_override={**base,"stop_loss_pct":p_sl,"reward_risk":p_rr,"volume_mult":p_vol},
                start_date=ts,end_date=te,symbol=symbol,cached_bars=train_bars,cached_sma=sma,cached_vix={})
            if m.get("profit_factor",0) > best_pf and m.get("total_trades",0) >= 5:
                best_pf = m["profit_factor"]; best_p = (p_sl,p_rr,p_vol)
        except: pass
    if best_p is None: continue
    try:
        tm = engine.run_backtest_config(
            config_override={**base,"stop_loss_pct":best_p[0],"reward_risk":best_p[1],"volume_mult":best_p[2]},
            start_date=vs,end_date=ve,symbol=symbol,cached_bars=test_bars,cached_sma=sma,cached_vix={})
        wf_pfs.append(tm.get("profit_factor",0))
    except: pass
result["wf_total"] = len(wf_pfs)
result["wf_wins"] = sum(1 for p in wf_pfs if p > 1.0)
result["wf_avg_test_pf"] = sum(wf_pfs)/len(wf_pfs) if wf_pfs else 0

# Monte Carlo
try:
    result["mc_real_pf"] = result["full_2yr_pf"]
    rand_pfs = []
    for i in range(200):
        try:
            rm = engine.run_backtest_config(
                config_override={**base,"volume_mult":0.0,"regime_filter":False,"vix_threshold":999,
                    "max_trades_per_day":1,"afternoon_entries":False,"entry_window_minutes":60},
                start_date="2023-01-01",end_date="2025-01-01",symbol=symbol,
                cached_bars=bars_2yr,cached_sma=sma,cached_vix={})
            if rm.get("profit_factor",0) > 0: rand_pfs.append(rm["profit_factor"])
        except: pass
    if rand_pfs:
        result["mc_random_pf"] = round(sum(rand_pfs)/len(rand_pfs),3)
        beats = sum(1 for p in rand_pfs if result["mc_real_pf"] > p)
        result["mc_p_value"] = round(1 - beats/len(rand_pfs),3)
except: pass

# Slippage
SLIP_PERIODS = [
    ("2023-01-01","2025-01-01","full_2yr"),
    ("2022-01-01","2022-12-31","holdout"),
    ("2025-06-01","2026-03-31","bear_2025"),
]
for start,end,plabel in SLIP_PERIODS:
    cached = load_cached(symbol, start, end)
    sr = {}
    for slip in [0.0,0.02,0.05,0.10,0.20]:
        try:
            kw = {"config_override":{**base,"slippage_pct":slip},"start_date":start,"end_date":end,"symbol":symbol}
            if cached is not None and not cached.empty:
                kw["cached_bars"] = cached
                kw["cached_sma"] = sma
                kw["cached_vix"] = {}
            m = engine.run_backtest_config(**kw)
            sr[str(slip)] = round(m.get("profit_factor",0),3)
        except: sr[str(slip)] = 0
    if plabel == "full_2yr": result["slip_full_2yr"] = sr
    elif plabel == "holdout": result["slip_holdout"] = sr
    else: result["slip_bear"] = sr

json.dump(result, open(out_path, "w"))
'''


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", default="1/1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cache_dir = _ROOT / "logs" / "research" / "bar_cache"
    if not cache_dir.exists() or len(list(cache_dir.glob("*.pkl"))) < 10:
        print("ERROR: Run prefetch_data.py first to cache bar data!")
        print(f"  python backtest/prefetch_data.py")
        sys.exit(1)

    all_configs = list(itertools.product(STRATEGIES, SYMBOLS, STOP_LOSSES, REWARD_RISKS))
    print(f"Total configs: {len(all_configs)}")
    print(f"Cache dir: {cache_dir} ({len(list(cache_dir.glob('*.pkl')))} files)")

    if args.dry_run:
        return

    slice_n, slice_m = map(int, args.slice.split("/"))
    chunk = len(all_configs) // slice_m
    start_idx = (slice_n - 1) * chunk
    end_idx = start_idx + chunk if slice_n < slice_m else len(all_configs)
    configs = all_configs[start_idx:end_idx]
    print(f"Slice {slice_n}/{slice_m}: {len(configs)} configs, {args.workers} workers")

    worker_file = _ROOT / "backtest" / "_validation_worker.py"
    with open(worker_file, "w") as f:
        f.write(WORKER_SCRIPT)

    cfg_path = str(_ROOT / "config" / "config.yaml")
    out_dir = _ROOT / "logs" / "research" / f"val_v3_s{slice_n}"
    out_dir.mkdir(parents=True, exist_ok=True)

    python = str(_ROOT / "venv" / "bin" / "python")
    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(_ROOT)
    env["CFG_PATH"] = cfg_path
    env["CACHE_DIR"] = str(cache_dir)

    t0 = _time.time()
    active = deque()
    results = []
    submitted = 0
    completed = 0
    skipped = 0

    config_iter = iter(configs)
    done_iter = False

    while not done_iter or active:
        while len(active) < args.workers and not done_iter:
            try:
                strat, sym, sl_val, rr_val = next(config_iter)
            except StopIteration:
                done_iter = True
                break

            out_path = str(out_dir / f"{strat}_{sym}_{sl_val}_{rr_val}.json")
            submitted += 1

            if os.path.exists(out_path):
                try:
                    r = json.load(open(out_path))
                    results.append(r)
                    completed += 1
                    skipped += 1
                except:
                    pass
                continue

            proc_env = env.copy()
            proc_env["STRATEGY"] = strat
            proc_env["SYMBOL"] = sym
            proc_env["SL"] = str(sl_val)
            proc_env["RR"] = str(rr_val)
            proc_env["OUT_PATH"] = out_path

            proc = subprocess.Popen(
                [python, str(worker_file)],
                env=proc_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            active.append((proc, out_path))

        still_active = deque()
        for proc, out_path in active:
            if proc.poll() is not None:
                completed += 1
                try:
                    r = json.load(open(out_path))
                    results.append(r)
                except:
                    pass
            else:
                still_active.append((proc, out_path))
        active = still_active

        if completed > 0 and completed % 20 == 0:
            elapsed = (_time.time() - t0) / 60
            tested = sum(1 for r in results if r.get("full_2yr_trades", 0) >= 10)
            wf_p = sum(1 for r in results if r.get("wf_wins", 0) >= 6)
            mc_p = sum(1 for r in results if r.get("mc_p_value", 1) < 0.05)
            sl_p = sum(1 for r in results if r.get("slip_full_2yr", {}).get("0.05", 0) > 1.0)
            all_p = sum(1 for r in results
                        if r.get("wf_wins", 0) >= 6
                        and r.get("mc_p_value", 1) < 0.05
                        and r.get("slip_full_2yr", {}).get("0.05", 0) > 1.0)
            remaining = len(configs) - completed
            rate = completed / elapsed if elapsed > 0 else 0
            eta = remaining / rate if rate > 0 else 0
            print(f"  [{completed}/{len(configs)}] {elapsed:.1f}min ETA={eta:.0f}min "
                  f"active={len(active)} skip={skipped} tested={tested} "
                  f"WF={wf_p} MC={mc_p} Slip={sl_p} ALL={all_p}")

        _time.sleep(0.3)

    combined = _ROOT / "logs" / "research" / f"validation_v3_s{slice_n}.json"
    with open(combined, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = (_time.time() - t0) / 60
    tested = [r for r in results if r.get("full_2yr_trades", 0) >= 10]
    all_pass = [r for r in tested
                if r.get("wf_wins", 0) >= 6
                and r.get("mc_p_value", 1) < 0.05
                and r.get("slip_full_2yr", {}).get("0.05", 0) > 1.0]

    print(f"\n{'='*70}")
    print(f"  VALIDATION V3 COMPLETE — {completed} configs in {elapsed:.1f} min")
    print(f"{'='*70}")
    print(f"  Tested (≥10 trades): {len(tested)}/{completed}")
    print(f"  ★ ALL THREE PASS: {len(all_pass)}")

    if all_pass:
        print(f"\n  FULLY VALIDATED:")
        print(f"  {'Strategy':<16} {'Symbol':<8} {'SL':<5} {'RR':<5} {'WF':>5} {'MC p':>7} {'Slip':>6} {'Trades':>7} {'PF':>6} {'Dur':>5}")
        print(f"  {'-'*75}")
        for x in sorted(all_pass, key=lambda r: -r.get("slip_full_2yr", {}).get("0.05", 0)):
            print(f"  {x['strategy']:<16} {x['symbol']:<8} {x['sl']:<5} {x['rr']:<5} "
                  f"{x['wf_wins']}/{x['wf_total']:>2}  p={x['mc_p_value']:.3f} "
                  f"PF={x['slip_full_2yr'].get('0.05', 0):.2f} "
                  f"{x['full_2yr_trades']:>7} {x['full_2yr_pf']:>6.2f} {x['full_2yr_avg_dur']:>4.0f}m")

    try:
        worker_file.unlink()
    except:
        pass


if __name__ == "__main__":
    main()
