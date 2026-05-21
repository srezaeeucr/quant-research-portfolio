#!/usr/bin/env python3
"""
Extended SL/RR Validation — widens the grid beyond V4's SL 0.3-1.0 and RR 1.0-5.0.

Runs ONLY on the 5 symbols that produced V4 passers (TSLA, AMD, META, GOOGL, MSFT)
plus NVDA (one more look at the "unvalidated" star symbol).

New SL range: 1.25, 1.5, 2.0 (wider stops — previously only hinted at with AMD SL=1.5)
New RR range: 6.0, 7.0, 8.0 (letting winners run)

Full V4 pipeline: quick → WF (10 windows) → MC (200 perms) → Slippage (5 x 3 periods).

Total: 6 symbols x 3 strat x 3 new SL x 3 new RR = 162 new configs.
Plus cross-combos: 6 symbols x 3 strat x 3 new SL x 4 old RR = 216
Plus: 6 symbols x 3 strat x 4 old SL x 3 new RR = 216
Grand total: 594 configs.
"""
import os
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import sys, time as _time, json, pickle, itertools, multiprocessing as mp
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

import yaml, numpy as np, pandas as pd, pytz
_ET = pytz.timezone("America/New_York")

# Focus on 6 symbols
SYMBOLS = ["AMD", "TSLA", "META", "GOOGL", "MSFT", "NVDA"]
STRATEGIES = ["orb", "ema_crossover", "momentum"]

# Extended ranges: any combo where EITHER sl is new OR rr is new
OLD_SL = [0.3, 0.5, 0.75, 1.0]
NEW_SL = [1.25, 1.5, 2.0]
OLD_RR = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
NEW_RR = [6.0, 7.0, 8.0]

ALL_SL = OLD_SL + NEW_SL
ALL_RR = OLD_RR + NEW_RR

# Existing V4 covered (old_sl, old_rr). We skip those.
V4_COVERED = set((sl, rr) for sl in OLD_SL for rr in OLD_RR)
EXTENDED_PARAMS = [(sl, rr) for sl in ALL_SL for rr in ALL_RR if (sl, rr) not in V4_COVERED]


WF_WINDOWS = [
    ("2022-01-01", "2022-06-30", "2022-07-01", "2022-09-30"),
    ("2022-04-01", "2022-09-30", "2022-10-01", "2022-12-31"),
    ("2022-07-01", "2022-12-31", "2023-01-01", "2023-03-31"),
    ("2022-10-01", "2023-03-31", "2023-04-01", "2023-06-30"),
    ("2023-01-01", "2023-06-30", "2023-07-01", "2023-09-30"),
    ("2023-04-01", "2023-09-30", "2023-10-01", "2023-12-31"),
    ("2023-07-01", "2023-12-31", "2024-01-01", "2024-03-31"),
    ("2023-10-01", "2024-03-31", "2024-04-01", "2024-06-30"),
    ("2024-01-01", "2024-06-30", "2024-07-01", "2024-09-30"),
    ("2024-04-01", "2024-09-30", "2024-10-01", "2024-12-31"),
]
WF_CACHE_RANGES = [
    ("2022-01-01", "2022-09-30"), ("2022-04-01", "2022-12-31"),
    ("2022-07-01", "2023-03-31"), ("2022-10-01", "2023-06-30"),
    ("2023-01-01", "2023-09-30"), ("2023-04-01", "2023-12-31"),
    ("2023-07-01", "2024-03-31"), ("2023-10-01", "2024-06-30"),
    ("2024-01-01", "2024-09-30"), ("2024-04-01", "2024-12-31"),
]
WF_PARAMS = list(itertools.product([0.3, 0.5, 0.75, 1.0], [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0], [1.0, 1.2]))  # V4-style inner opt sweep (56 combos)
SLIPPAGE_LEVELS = [0.0, 0.02, 0.05, 0.10, 0.20]
SLIP_PERIODS = [
    ("2023-01-01", "2025-01-01", "full_2yr"),
    ("2022-01-01", "2022-12-31", "holdout"),
    ("2025-06-01", "2026-03-31", "bear_2025"),
]


def load_cached(cache_dir, sym, start, end):
    for ext in ["parquet", "pkl"]:
        p = cache_dir / f"{sym}_{start}_{end}.{ext}"
        if p.exists():
            if ext == "parquet":
                return pd.read_parquet(p)
            with open(p, "rb") as f: return pickle.load(f)
    return None


def load_sma(cache_dir, sym):
    from datetime import date
    jf = cache_dir / f"{sym}_sma.json"
    if jf.exists():
        return {date.fromisoformat(k): v for k, v in json.load(open(jf)).items()}
    p = cache_dir / f"{sym}_sma.pkl"
    if p.exists():
        with open(p, "rb") as f: return pickle.load(f)
    return {}


def validate_symbol(args):
    sym, cfg_path, cache_dir_str, out_dir_str = args
    from src.engine import TradingEngine

    cache_dir = Path(cache_dir_str)
    out_dir = Path(out_dir_str)

    with open(cfg_path) as f: cfg = yaml.safe_load(f)
    cfg["mode"] = "backtest"
    cfg.setdefault("account", {})["balance"] = 500.0
    engine = TradingEngine(cfg)

    bars_2yr = load_cached(cache_dir, sym, "2023-01-01", "2025-01-01")
    sma = load_sma(cache_dir, sym)
    if bars_2yr is None or bars_2yr.empty:
        return []

    wf_bar_cache = {}
    for idx, (ts, te, vs, ve) in enumerate(WF_WINDOWS):
        cs, ce = WF_CACHE_RANGES[idx]
        all_bars = load_cached(cache_dir, sym, cs, ce)
        if all_bars is None or all_bars.empty: continue
        bar_ts = all_bars["timestamp"]
        if bar_ts.dt.tz is not None:
            bar_ts = bar_ts.dt.tz_convert(_ET)
        train = all_bars[(bar_ts >= ts) & (bar_ts < te + " 23:59:59")]
        test = all_bars[(bar_ts >= vs) & (bar_ts < ve + " 23:59:59")]
        if not train.empty and not test.empty:
            wf_bar_cache[idx] = (train, test)

    slip_bars = {}
    for start, end, label in SLIP_PERIODS:
        b = load_cached(cache_dir, sym, start, end)
        if b is not None and not b.empty:
            slip_bars[label] = b

    results = []
    configs = [(strat, sl, rr) for strat in STRATEGIES for sl, rr in EXTENDED_PARAMS]

    for strat, sl, rr in configs:
        out_path = out_dir / f"{strat}_{sym}_{sl}_{rr}.json"
        if out_path.exists():
            try: results.append(json.load(open(out_path))); continue
            except: pass

        base = {
            "active_strategy": strat, "stop_loss_pct": sl, "reward_risk": rr,
            "volume_mult": 1.2, "regime_filter": True, "vix_threshold": 25,
            "max_trades_per_day": 100, "afternoon_entries": True, "entry_window_minutes": 360,
        }
        result = {
            "strategy": strat, "symbol": sym, "sl": sl, "rr": rr,
            "wf_wins": 0, "wf_total": 0, "wf_avg_test_pf": 0,
            "mc_real_pf": 0, "mc_random_pf": 0, "mc_p_value": 1.0,
            "slip_full_2yr": {}, "slip_holdout": {}, "slip_bear": {},
            "full_2yr_trades": 0, "full_2yr_pf": 0, "full_2yr_wr": 0,
            "full_2yr_avg_dur": 0,
        }

        try:
            quick = engine.run_backtest_config(
                config_override=base, start_date="2023-01-01", end_date="2025-01-01",
                symbol=sym, cached_bars=bars_2yr, cached_sma=sma, cached_vix={})
            result["full_2yr_trades"] = quick.get("total_trades", 0)
            result["full_2yr_pf"] = quick.get("profit_factor", 0)
            result["full_2yr_wr"] = quick.get("win_rate", 0)
            result["full_2yr_avg_dur"] = quick.get("avg_duration_min", 0)
            if quick.get("total_trades", 0) < 10:
                json.dump(result, open(out_path, "w")); results.append(result); continue
        except:
            json.dump(result, open(out_path, "w")); results.append(result); continue

        # Walk-forward
        wf_pfs = []
        for idx in range(len(WF_WINDOWS)):
            if idx not in wf_bar_cache: continue
            train_bars, test_bars = wf_bar_cache[idx]
            ts_s = WF_WINDOWS[idx][0]; te_s = WF_WINDOWS[idx][1]
            vs_s = WF_WINDOWS[idx][2]; ve_s = WF_WINDOWS[idx][3]
            best_pf, best_p = 0, None
            for p_sl, p_rr, p_vol in WF_PARAMS:
                try:
                    m = engine.run_backtest_config(
                        config_override={**base, "stop_loss_pct": p_sl, "reward_risk": p_rr, "volume_mult": p_vol},
                        start_date=ts_s, end_date=te_s, symbol=sym,
                        cached_bars=train_bars, cached_sma=sma, cached_vix={})
                    pf = m.get("profit_factor", 0)
                    if pf > best_pf and m.get("total_trades", 0) >= 5:
                        best_pf = pf; best_p = (p_sl, p_rr, p_vol)
                except: pass
            if best_p:
                try:
                    tm = engine.run_backtest_config(
                        config_override={**base, "stop_loss_pct": best_p[0], "reward_risk": best_p[1], "volume_mult": best_p[2]},
                        start_date=vs_s, end_date=ve_s, symbol=sym,
                        cached_bars=test_bars, cached_sma=sma, cached_vix={})
                    wf_pfs.append(tm.get("profit_factor", 0))
                except: pass
        result["wf_total"] = len(wf_pfs)
        result["wf_wins"] = sum(1 for p in wf_pfs if p > 1.0)
        result["wf_avg_test_pf"] = sum(wf_pfs) / len(wf_pfs) if wf_pfs else 0

        # EARLY TERMINATION: skip MC and Slippage if WF clearly fails
        if result["wf_total"] > 0 and result["wf_wins"] < 5:
            json.dump(result, open(out_path, "w")); results.append(result); continue

        # MC
        try:
            result["mc_real_pf"] = result["full_2yr_pf"]
            rand_pfs = []
            for i in range(200):
                try:
                    rm = engine.run_backtest_config(
                        config_override={**base, "volume_mult": 0.0, "regime_filter": False,
                                         "vix_threshold": 999, "max_trades_per_day": 1,
                                         "afternoon_entries": False, "entry_window_minutes": 60},
                        start_date="2023-01-01", end_date="2025-01-01", symbol=sym,
                        cached_bars=bars_2yr, cached_sma=sma, cached_vix={})
                    pf = rm.get("profit_factor", 0)
                    if pf > 0: rand_pfs.append(pf)
                except: pass
            if rand_pfs:
                result["mc_random_pf"] = round(sum(rand_pfs)/len(rand_pfs), 3)
                beats = sum(1 for p in rand_pfs if result["mc_real_pf"] > p)
                result["mc_p_value"] = round(1 - beats/len(rand_pfs), 3)
        except: pass

        # EARLY TERMINATION: skip Slippage if MC clearly fails
        if result.get("mc_p_value", 1.0) > 0.1:
            json.dump(result, open(out_path, "w")); results.append(result); continue

        # Slip
        for start, end, plabel in SLIP_PERIODS:
            sr = {}
            cached = slip_bars.get(plabel)
            for slip in SLIPPAGE_LEVELS:
                try:
                    kw = {"config_override": {**base, "slippage_pct": slip},
                          "start_date": start, "end_date": end, "symbol": sym}
                    if cached is not None:
                        kw["cached_bars"] = cached; kw["cached_sma"] = sma; kw["cached_vix"] = {}
                    sr[str(slip)] = round(engine.run_backtest_config(**kw).get("profit_factor", 0), 3)
                except: sr[str(slip)] = 0
            if plabel == "full_2yr": result["slip_full_2yr"] = sr
            elif plabel == "holdout": result["slip_holdout"] = sr
            else: result["slip_bear"] = sr

        json.dump(result, open(out_path, "w"))
        results.append(result)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    cache_dir = _ROOT / "logs" / "research" / "bar_cache"
    out_dir = _ROOT / "logs" / "research" / "val_extended_slrr"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_configs = len(SYMBOLS) * len(STRATEGIES) * len(EXTENDED_PARAMS)
    print(f"Total: {total_configs} NEW configs across {len(SYMBOLS)} focused symbols")
    print(f"Extended params (SL, RR): {len(EXTENDED_PARAMS)} new combos per (strat, sym)")
    print(f"Output: {out_dir}")
    existing = len(list(out_dir.glob("*.json")))
    print(f"Existing: {existing} (will skip)")

    cfg_path = _ROOT / "config" / "config.yaml"
    tasks = [(sym, str(cfg_path), str(cache_dir), str(out_dir)) for sym in SYMBOLS]

    start = _time.time()
    with mp.Pool(args.workers) as pool:
        symbol_results = []
        for i, results in enumerate(pool.imap_unordered(validate_symbol, tasks)):
            symbol_results.extend(results)
            elapsed = (_time.time() - start) / 60
            print(f"  [{i+1}/{len(tasks)}] symbol done, {elapsed:.1f}min elapsed  results={len(results)}")

    # Summary — new passers
    print(f"\n{'='*70}")
    pass_count = 0
    for r in symbol_results:
        if r.get("full_2yr_trades", 0) < 10: continue
        slip = (r.get("slip_full_2yr", {}) or {}).get("0.05", 0)
        if r.get("wf_wins", 0) >= 6 and r.get("mc_p_value", 1.0) <= 0.05 and slip >= 1.0:
            pass_count += 1
    print(f"NEW CONFIGS PASSING ALL 3 TESTS: {pass_count} / {len(symbol_results)}")
    elapsed = (_time.time() - start) / 60
    print(f"Total elapsed: {elapsed:.1f}min")


if __name__ == "__main__":
    main()
