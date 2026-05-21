#!/usr/bin/env python3
"""
Full Validation V4 — Optimized for speed. GCP ONLY.
Same MC=200, same WF=56 params — identical results, just faster.

Optimizations:
1. Single engine instance reused across all configs
2. Bars loaded once per symbol, cached in memory
3. WF bars pre-sliced per window (no repeated filtering)
4. Multiprocessing per symbol (not per config) — fewer subprocess spawns
5. Results written incrementally

Usage:
  python backtest/full_validation_v4.py --workers 24
"""
import os
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import sys
import time as _time
import json
import pickle
import itertools
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_ROOT = Path(__file__).resolve().parents[1]

# Load .env
from dotenv import load_dotenv
for p in [_ROOT / ".env", Path.home() / "projects" / "dl_course" / ".env",
          Path.home() / "projects" / "day-trading-bot" / ".env"]:
    if p.exists():
        load_dotenv(p)
        break

import yaml
import numpy as np
import pytz

_ET = pytz.timezone("America/New_York")

SYMBOLS = ["AMD", "META", "COIN", "TSLA", "NVDA", "SPY", "QQQ", "AAPL",
           "AMZN", "MSFT", "GOOGL", "JPM", "MSTR", "IWM",
           "XLK", "XLF", "XLE", "XLV", "XLI"]
STRATEGIES = ["orb", "ema_crossover", "momentum"]
STOP_LOSSES = [0.3, 0.5, 0.75, 1.0]
REWARD_RISKS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

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
    ("2022-01-01", "2022-09-30"),
    ("2022-04-01", "2022-12-31"),
    ("2022-07-01", "2023-03-31"),
    ("2022-10-01", "2023-06-30"),
    ("2023-01-01", "2023-09-30"),
    ("2023-04-01", "2023-12-31"),
    ("2023-07-01", "2024-03-31"),
    ("2023-10-01", "2024-06-30"),
    ("2024-01-01", "2024-09-30"),
    ("2024-04-01", "2024-12-31"),
]

WF_PARAMS = list(itertools.product(
    [0.3, 0.5, 0.75, 1.0],
    [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    [1.0, 1.2],
))

SLIPPAGE_LEVELS = [0.0, 0.02, 0.05, 0.10, 0.20]
SLIP_PERIODS = [
    ("2023-01-01", "2025-01-01", "full_2yr"),
    ("2022-01-01", "2022-12-31", "holdout"),
    ("2025-06-01", "2026-03-31", "bear_2025"),
]


def load_cached(cache_dir, sym, start, end):
    import pandas as pd
    pq = cache_dir / f"{sym}_{start}_{end}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    p = cache_dir / f"{sym}_{start}_{end}.pkl"
    if p.exists():
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


def validate_symbol(args):
    """Validate ALL configs for ONE symbol. Returns list of results."""
    sym, cfg_path, cache_dir_str, out_dir_str = args

    import yaml
    from src.engine import TradingEngine
    from datetime import date

    cache_dir = Path(cache_dir_str)
    out_dir = Path(out_dir_str)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["mode"] = "backtest"
    cfg.setdefault("account", {})["balance"] = 500.0

    engine = TradingEngine(cfg)

    # Load ALL bars for this symbol ONCE
    bars_2yr = load_cached(cache_dir, sym, "2023-01-01", "2025-01-01")
    sma = load_sma(cache_dir, sym)

    if bars_2yr is None or bars_2yr.empty:
        return []

    # Pre-load and pre-slice WF bars
    wf_bar_cache = {}
    for idx, (ts, te, vs, ve) in enumerate(WF_WINDOWS):
        cache_start, cache_end = WF_CACHE_RANGES[idx]
        all_bars = load_cached(cache_dir, sym, cache_start, cache_end)
        if all_bars is None or all_bars.empty:
            continue
        bar_ts = all_bars["timestamp"]
        if bar_ts.dt.tz is not None:
            bar_ts = bar_ts.dt.tz_convert(_ET)
        train = all_bars[(bar_ts >= ts) & (bar_ts < te + " 23:59:59")]
        test = all_bars[(bar_ts >= vs) & (bar_ts < ve + " 23:59:59")]
        if not train.empty and not test.empty:
            wf_bar_cache[idx] = (train, test)

    # Pre-load slippage bars
    slip_bars = {}
    for start, end, label in SLIP_PERIODS:
        b = load_cached(cache_dir, sym, start, end)
        if b is not None and not b.empty:
            slip_bars[label] = b

    results = []
    configs = list(itertools.product(STRATEGIES, STOP_LOSSES, REWARD_RISKS))

    for strat, sl, rr in configs:
        out_path = out_dir / f"{strat}_{sym}_{sl}_{rr}.json"
        if out_path.exists():
            # Skip if already done
            try:
                results.append(json.load(open(out_path)))
            except:
                pass
            continue

        base = {
            "active_strategy": strat, "stop_loss_pct": sl, "reward_risk": rr,
            "volume_mult": 1.2, "regime_filter": True, "vix_threshold": 25,
            "max_trades_per_day": 100, "afternoon_entries": True,
            "entry_window_minutes": 360,
        }

        result = {
            "strategy": strat, "symbol": sym, "sl": sl, "rr": rr,
            "wf_wins": 0, "wf_total": 0, "wf_avg_test_pf": 0,
            "mc_real_pf": 0, "mc_random_pf": 0, "mc_p_value": 1.0,
            "slip_full_2yr": {}, "slip_holdout": {}, "slip_bear": {},
            "full_2yr_trades": 0, "full_2yr_pf": 0, "full_2yr_wr": 0,
            "full_2yr_avg_dur": 0,
        }

        # Quick check
        try:
            quick = engine.run_backtest_config(
                config_override=base,
                start_date="2023-01-01", end_date="2025-01-01",
                symbol=sym, cached_bars=bars_2yr, cached_sma=sma, cached_vix={},
            )
            result["full_2yr_trades"] = quick.get("total_trades", 0)
            result["full_2yr_pf"] = quick.get("profit_factor", 0)
            result["full_2yr_wr"] = quick.get("win_rate", 0)
            result["full_2yr_avg_dur"] = quick.get("avg_duration_min", 0)
            if quick.get("total_trades", 0) < 10:
                json.dump(result, open(out_path, "w"))
                results.append(result)
                continue
        except:
            json.dump(result, open(out_path, "w"))
            results.append(result)
            continue

        # Walk-forward (using pre-sliced bars)
        wf_pfs = []
        for idx in range(len(WF_WINDOWS)):
            if idx not in wf_bar_cache:
                continue
            train_bars, test_bars = wf_bar_cache[idx]
            ts = WF_WINDOWS[idx][0]
            te = WF_WINDOWS[idx][1]
            vs = WF_WINDOWS[idx][2]
            ve = WF_WINDOWS[idx][3]

            best_pf, best_p = 0, None
            for p_sl, p_rr, p_vol in WF_PARAMS:
                try:
                    m = engine.run_backtest_config(
                        config_override={**base, "stop_loss_pct": p_sl,
                                         "reward_risk": p_rr, "volume_mult": p_vol},
                        start_date=ts, end_date=te, symbol=sym,
                        cached_bars=train_bars, cached_sma=sma, cached_vix={},
                    )
                    pf = m.get("profit_factor", 0)
                    if pf > best_pf and m.get("total_trades", 0) >= 5:
                        best_pf = pf
                        best_p = (p_sl, p_rr, p_vol)
                except:
                    pass

            if best_p:
                try:
                    tm = engine.run_backtest_config(
                        config_override={**base, "stop_loss_pct": best_p[0],
                                         "reward_risk": best_p[1], "volume_mult": best_p[2]},
                        start_date=vs, end_date=ve, symbol=sym,
                        cached_bars=test_bars, cached_sma=sma, cached_vix={},
                    )
                    wf_pfs.append(tm.get("profit_factor", 0))
                except:
                    pass

        result["wf_total"] = len(wf_pfs)
        result["wf_wins"] = sum(1 for p in wf_pfs if p > 1.0)
        result["wf_avg_test_pf"] = sum(wf_pfs) / len(wf_pfs) if wf_pfs else 0

        # Monte Carlo (reusing bars_2yr)
        try:
            result["mc_real_pf"] = result["full_2yr_pf"]
            rand_pfs = []
            for i in range(200):
                try:
                    rm = engine.run_backtest_config(
                        config_override={**base, "volume_mult": 0.0,
                                         "regime_filter": False, "vix_threshold": 999,
                                         "max_trades_per_day": 1,
                                         "afternoon_entries": False,
                                         "entry_window_minutes": 60},
                        start_date="2023-01-01", end_date="2025-01-01",
                        symbol=sym, cached_bars=bars_2yr, cached_sma=sma,
                        cached_vix={},
                    )
                    pf = rm.get("profit_factor", 0)
                    if pf > 0:
                        rand_pfs.append(pf)
                except:
                    pass
            if rand_pfs:
                result["mc_random_pf"] = round(sum(rand_pfs) / len(rand_pfs), 3)
                beats = sum(1 for p in rand_pfs if result["mc_real_pf"] > p)
                result["mc_p_value"] = round(1 - beats / len(rand_pfs), 3)
        except:
            pass

        # Slippage (using pre-loaded bars)
        for start, end, plabel in SLIP_PERIODS:
            sr = {}
            cached = slip_bars.get(plabel)
            for slip in SLIPPAGE_LEVELS:
                try:
                    kw = {"config_override": {**base, "slippage_pct": slip},
                          "start_date": start, "end_date": end, "symbol": sym}
                    if cached is not None:
                        kw["cached_bars"] = cached
                        kw["cached_sma"] = sma
                        kw["cached_vix"] = {}
                    sr[str(slip)] = round(
                        engine.run_backtest_config(**kw).get("profit_factor", 0), 3)
                except:
                    sr[str(slip)] = 0
            if plabel == "full_2yr":
                result["slip_full_2yr"] = sr
            elif plabel == "holdout":
                result["slip_holdout"] = sr
            else:
                result["slip_bear"] = sr

        # Save immediately
        json.dump(result, open(out_path, "w"))
        results.append(result)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=19)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cache_dir = _ROOT / "logs" / "research" / "bar_cache"
    out_dir = _ROOT / "logs" / "research" / "val_v4"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_configs = len(SYMBOLS) * len(STRATEGIES) * len(STOP_LOSSES) * len(REWARD_RISKS)
    print(f"Total: {total_configs} configs across {len(SYMBOLS)} symbols")
    print(f"Workers: {args.workers} (1 per symbol in parallel)")
    print(f"Cache: {cache_dir} ({len(list(cache_dir.glob('*')))} files)")
    print(f"Output: {out_dir}")

    existing = len(list(out_dir.glob("*.json")))
    print(f"Existing results: {existing} (will skip)")

    if args.dry_run:
        return

    cfg_path = str(_ROOT / "config" / "config.yaml")
    work = [(sym, cfg_path, str(cache_dir), str(out_dir)) for sym in SYMBOLS]

    t0 = _time.time()

    # Use fork on Linux (inherits env vars)
    ctx = mp.get_context("fork" if sys.platform == "linux" else "spawn")
    with ctx.Pool(args.workers) as pool:
        all_results = []
        for sym_results in pool.imap_unordered(validate_symbol, work):
            all_results.extend(sym_results)
            elapsed = (_time.time() - t0) / 60
            tested = sum(1 for r in all_results if r.get("full_2yr_trades", 0) >= 10)
            all_pass = sum(1 for r in all_results
                           if r.get("wf_wins", 0) >= 6
                           and r.get("mc_p_value", 1) < 0.05
                           and r.get("slip_full_2yr", {}).get("0.05", 0) > 1.0)
            print(f"  [{len(all_results)}/{total_configs}] {elapsed:.1f}min "
                  f"tested={tested} ALL_PASS={all_pass}")

    # Save combined
    combined = _ROOT / "logs" / "research" / "validation_v4.json"
    with open(combined, "w") as f:
        json.dump(all_results, f, indent=2)

    elapsed = (_time.time() - t0) / 60
    tested = [r for r in all_results if r.get("full_2yr_trades", 0) >= 10]
    all_pass = [r for r in tested
                if r.get("wf_wins", 0) >= 6
                and r.get("mc_p_value", 1) < 0.05
                and r.get("slip_full_2yr", {}).get("0.05", 0) > 1.0]

    print(f"\n{'='*70}")
    print(f"  V4 COMPLETE — {len(all_results)} configs in {elapsed:.1f} min")
    print(f"{'='*70}")
    print(f"  Tested: {len(tested)}")
    print(f"  ★ ALL PASS: {len(all_pass)}")

    if all_pass:
        print(f"\n  VALIDATED:")
        from collections import defaultdict
        by_sym = defaultdict(list)
        for r in all_pass:
            by_sym[r["symbol"]].append(r)
        for sym in sorted(by_sym.keys()):
            configs = sorted(by_sym[sym],
                             key=lambda x: -x.get("slip_full_2yr", {}).get("0.05", 0))
            best = configs[0]
            print(f"  {sym} ({len(configs)}): {best['strategy']} SL={best['sl']} "
                  f"RR={best['rr']} Slip={best['slip_full_2yr'].get('0.05', 0):.2f} "
                  f"WF={best['wf_wins']}/{best['wf_total']}")


if __name__ == "__main__":
    main()
