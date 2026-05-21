#!/usr/bin/env python3
"""
Full Validation Chain — runs walk-forward + Monte Carlo + slippage
for ALL configs that passed holdout (PF > 1.0, ≥15 trades).

Usage:
  python backtest/full_validation_chain.py --slice 1/3 --workers 8   # Mac Mini
  python backtest/full_validation_chain.py --slice 2/3 --workers 27  # Superpower
  python backtest/full_validation_chain.py --slice 3/3 --workers 7   # GCP
  python backtest/full_validation_chain.py --dry-run
"""
import os
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import argparse
import itertools
import multiprocessing as mp
import sys
import time as _time
import sqlite3
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_ROOT = Path(__file__).resolve().parents[1]

WALK_FORWARD_WINDOWS = [
    ("2022-01-01", "2022-06-30", "2022-07-01", "2022-09-30", "W1"),
    ("2022-04-01", "2022-09-30", "2022-10-01", "2022-12-31", "W2"),
    ("2022-07-01", "2022-12-31", "2023-01-01", "2023-03-31", "W3"),
    ("2022-10-01", "2023-03-31", "2023-04-01", "2023-06-30", "W4"),
    ("2023-01-01", "2023-06-30", "2023-07-01", "2023-09-30", "W5"),
    ("2023-04-01", "2023-09-30", "2023-10-01", "2023-12-31", "W6"),
    ("2023-07-01", "2023-12-31", "2024-01-01", "2024-03-31", "W7"),
    ("2023-10-01", "2024-03-31", "2024-04-01", "2024-06-30", "W8"),
    ("2024-01-01", "2024-06-30", "2024-07-01", "2024-09-30", "W9"),
    ("2024-04-01", "2024-09-30", "2024-10-01", "2024-12-31", "W10"),
]

WF_PARAM_GRID = list(itertools.product(
    [0.3, 0.5, 0.75, 1.0],  # SL
    [2.0, 3.0, 4.0, 5.0],    # RR
    [1.0, 1.2],               # vol
))

SLIPPAGE_LEVELS = [0.0, 0.02, 0.05, 0.10, 0.20]
SLIPPAGE_PERIODS = [
    ("2023-01-01", "2025-01-01", "full_2yr"),
    ("2022-01-01", "2022-12-31", "holdout"),
    ("2025-06-01", "2026-03-31", "bear_2025"),
]

MC_PERMUTATIONS = 200


def _validate_one(args):
    """Run full validation chain for one (strategy, symbol, sl, rr) config."""
    strategy, symbol, sl, rr, cfg_path = args

    import yaml
    from src.engine import TradingEngine

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["mode"] = "backtest"
    cfg.setdefault("account", {})["balance"] = 500.0

    try:
        engine = TradingEngine(cfg)
    except Exception as e:
        return {"config": f"{strategy}/{symbol}/SL={sl}/RR={rr}", "error": str(e)}

    base_override = {
        "active_strategy": strategy, "stop_loss_pct": sl, "reward_risk": rr,
        "volume_mult": 1.2, "regime_filter": True, "vix_threshold": 25,
        "max_trades_per_day": 3, "afternoon_entries": True, "entry_window_minutes": 360,
    }

    result = {
        "strategy": strategy, "symbol": symbol, "sl": sl, "rr": rr,
        "wf_wins": 0, "wf_total": 0, "wf_avg_test_pf": 0,
        "mc_real_pf": 0, "mc_random_pf": 0, "mc_p_value": 1.0,
        "slip_full_2yr": {}, "slip_holdout": {}, "slip_bear": {},
    }

    # ═══ 1. WALK-FORWARD ═══
    wf_test_pfs = []
    for ts, te, vs, ve, label in WALK_FORWARD_WINDOWS:
        try:
            train_bars = engine.fetcher.fetch_historical(symbol, ts, te, filter_windows=False)
            test_bars = engine.fetcher.fetch_historical(symbol, vs, ve, filter_windows=False)
        except Exception:
            continue
        if train_bars is None or train_bars.empty or test_bars is None or test_bars.empty:
            continue

        try:
            sma = engine._compute_daily_sma(symbol, ts, ve, 20)
        except Exception:
            sma = {}

        # Find best params on train
        best_train_pf = 0
        best_p = None
        for p_sl, p_rr, p_vol in WF_PARAM_GRID:
            try:
                m = engine.run_backtest_config(
                    config_override={**base_override, "stop_loss_pct": p_sl, "reward_risk": p_rr, "volume_mult": p_vol},
                    start_date=ts, end_date=te, symbol=symbol,
                    cached_bars=train_bars, cached_sma=sma, cached_vix={},
                )
                pf = m.get("profit_factor", 0)
                if pf > best_train_pf and m.get("total_trades", 0) >= 5:
                    best_train_pf = pf
                    best_p = (p_sl, p_rr, p_vol)
            except Exception:
                pass

        if best_p is None:
            continue

        # Test on unseen period
        try:
            tm = engine.run_backtest_config(
                config_override={**base_override, "stop_loss_pct": best_p[0], "reward_risk": best_p[1], "volume_mult": best_p[2]},
                start_date=vs, end_date=ve, symbol=symbol,
                cached_bars=test_bars, cached_sma=sma, cached_vix={},
            )
            test_pf = tm.get("profit_factor", 0)
            wf_test_pfs.append(test_pf)
        except Exception:
            pass

    result["wf_total"] = len(wf_test_pfs)
    result["wf_wins"] = sum(1 for p in wf_test_pfs if p > 1.0)
    result["wf_avg_test_pf"] = sum(wf_test_pfs) / len(wf_test_pfs) if wf_test_pfs else 0

    # ═══ 2. MONTE CARLO ═══
    try:
        real_m = engine.run_backtest_config(
            config_override=base_override,
            start_date="2023-01-01", end_date="2025-01-01", symbol=symbol,
        )
        real_pf = real_m.get("profit_factor", 0)
        result["mc_real_pf"] = real_pf

        bars = engine.fetcher.fetch_historical(symbol, "2023-01-01", "2025-01-01", filter_windows=False)
        try:
            sma = engine._compute_daily_sma(symbol, "2023-01-01", "2025-01-01", 20)
        except Exception:
            sma = {}

        random_pfs = []
        for i in range(MC_PERMUTATIONS):
            try:
                rm = engine.run_backtest_config(
                    config_override={**base_override, "volume_mult": 0.0, "regime_filter": False,
                                     "vix_threshold": 999, "max_trades_per_day": 1,
                                     "afternoon_entries": False, "entry_window_minutes": 60},
                    start_date="2023-01-01", end_date="2025-01-01", symbol=symbol,
                    cached_bars=bars, cached_sma=sma, cached_vix={},
                )
                pf = rm.get("profit_factor", 0)
                if pf > 0:
                    random_pfs.append(pf)
            except Exception:
                pass

        if random_pfs:
            result["mc_random_pf"] = sum(random_pfs) / len(random_pfs)
            beats = sum(1 for p in random_pfs if real_pf > p)
            result["mc_p_value"] = 1 - beats / len(random_pfs)
    except Exception:
        pass

    # ═══ 3. SLIPPAGE ═══
    for start, end, plabel in SLIPPAGE_PERIODS:
        slip_results = {}
        for slip in SLIPPAGE_LEVELS:
            try:
                m = engine.run_backtest_config(
                    config_override={**base_override, "slippage_pct": slip},
                    start_date=start, end_date=end, symbol=symbol,
                )
                slip_results[str(slip)] = round(m.get("profit_factor", 0), 3)
            except Exception:
                slip_results[str(slip)] = 0
        if plabel == "full_2yr":
            result["slip_full_2yr"] = slip_results
        elif plabel == "holdout":
            result["slip_holdout"] = slip_results
        else:
            result["slip_bear"] = slip_results

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", default="1/1")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-pf", type=float, default=1.0, help="Min holdout PF to validate")
    parser.add_argument("--min-trades", type=int, default=15)
    args = parser.parse_args()

    # Get configs to validate from DB
    merged = _ROOT / "logs" / "results_merged.db"
    db_path = merged if merged.exists() else _ROOT / "logs" / "results.db"
    c = sqlite3.connect(str(db_path), timeout=30)
    configs = c.execute(f"""
        SELECT DISTINCT strategy, symbol, stop_loss_pct, reward_risk
        FROM backtest_runs
        WHERE period_label = 'holdout_2022' AND profit_factor > {args.min_pf}
              AND total_trades >= {args.min_trades} AND profit_factor < 100
        ORDER BY profit_factor DESC
    """).fetchall()
    c.close()

    print(f"Total configs to validate: {len(configs)}")

    if args.dry_run:
        # Estimate time: ~3 min per config (WF + MC + slippage)
        print(f"Est. time per config: ~3 min")
        print(f"Est. total: ~{len(configs) * 3 / 60:.0f} hours")
        return

    # Slice
    slice_n, slice_m = map(int, args.slice.split("/"))
    chunk = len(configs) // slice_m
    start_idx = (slice_n - 1) * chunk
    end_idx = start_idx + chunk if slice_n < slice_m else len(configs)
    configs = configs[start_idx:end_idx]

    print(f"Slice {slice_n}/{slice_m}: {len(configs)} configs")

    cfg_path = str(_ROOT / "config" / "config.yaml")
    work = [(s, sym, sl, rr, cfg_path) for s, sym, sl, rr in configs]

    n_workers = args.workers or min(8, max(1, mp.cpu_count() - 2))
    print(f"Workers: {n_workers}")

    t0 = _time.time()
    all_results = []
    done = 0

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as pool:
        for r in pool.map(_validate_one, work):
            all_results.append(r)
            done += 1
            if done % 10 == 0:
                elapsed = (_time.time() - t0) / 60
                eta = (len(configs) - done) / (done / elapsed) if done > 0 else 0
                # Count passes
                wf_pass = sum(1 for x in all_results if x.get("wf_wins", 0) >= 6)
                mc_pass = sum(1 for x in all_results if x.get("mc_p_value", 1) < 0.05)
                slip_pass = sum(1 for x in all_results if x.get("slip_full_2yr", {}).get("0.05", 0) > 1.0)
                all_pass = sum(1 for x in all_results
                               if x.get("wf_wins", 0) >= 6
                               and x.get("mc_p_value", 1) < 0.05
                               and x.get("slip_full_2yr", {}).get("0.05", 0) > 1.0)
                print(f"  [{done}/{len(configs)}] {elapsed:.1f}min ETA={eta:.0f}min | "
                      f"WF pass={wf_pass} MC pass={mc_pass} Slip pass={slip_pass} ALL PASS={all_pass}")

    # Save results
    out_path = _ROOT / "logs" / "research" / f"validation_chain_s{slice_n}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    elapsed = (_time.time() - t0) / 60
    print(f"\n{'='*70}")
    print(f"  VALIDATION CHAIN COMPLETE — {done} configs in {elapsed:.1f} min")
    print(f"{'='*70}")

    wf_pass = [x for x in all_results if x.get("wf_wins", 0) >= 6]
    mc_pass = [x for x in all_results if x.get("mc_p_value", 1) < 0.05]
    slip_pass = [x for x in all_results if x.get("slip_full_2yr", {}).get("0.05", 0) > 1.0]
    all_pass = [x for x in all_results
                if x.get("wf_wins", 0) >= 6
                and x.get("mc_p_value", 1) < 0.05
                and x.get("slip_full_2yr", {}).get("0.05", 0) > 1.0]

    print(f"  Walk-forward pass (≥6/10): {len(wf_pass)}/{done}")
    print(f"  Monte Carlo pass (p<0.05): {len(mc_pass)}/{done}")
    print(f"  Slippage pass (PF>1 @ 0.05%): {len(slip_pass)}/{done}")
    print(f"  ★ ALL THREE PASS: {len(all_pass)}/{done}")

    if all_pass:
        print(f"\n  FULLY VALIDATED CONFIGS:")
        print(f"  {'Strategy':<16} {'Symbol':<8} {'SL':<5} {'RR':<5} {'WF':>5} {'MC p':>7} {'Slip 5bp':>9}")
        print(f"  {'-'*58}")
        for x in sorted(all_pass, key=lambda r: -r.get("slip_full_2yr", {}).get("0.05", 0)):
            print(f"  {x['strategy']:<16} {x['symbol']:<8} {x['sl']:<5} {x['rr']:<5} "
                  f"{x['wf_wins']}/{x['wf_total']:>2}  p={x['mc_p_value']:.3f} "
                  f"PF={x['slip_full_2yr'].get('0.05', 0):.2f}")
    else:
        print(f"\n  No configs passed all three tests.")

    # Also print configs that passed 2/3
    two_pass = [x for x in all_results
                if sum([x.get("wf_wins", 0) >= 6,
                        x.get("mc_p_value", 1) < 0.05,
                        x.get("slip_full_2yr", {}).get("0.05", 0) > 1.0]) == 2]
    if two_pass:
        print(f"\n  PASSED 2/3 TESTS ({len(two_pass)} configs):")
        for x in sorted(two_pass, key=lambda r: -r.get("wf_avg_test_pf", 0))[:15]:
            wf = "✓" if x.get("wf_wins", 0) >= 6 else "✗"
            mc = "✓" if x.get("mc_p_value", 1) < 0.05 else "✗"
            sp = "✓" if x.get("slip_full_2yr", {}).get("0.05", 0) > 1.0 else "✗"
            print(f"  {x['strategy']:<16} {x['symbol']:<8} SL={x['sl']} RR={x['rr']} "
                  f"WF={wf}({x['wf_wins']}/{x['wf_total']}) MC={mc}(p={x['mc_p_value']:.3f}) "
                  f"Slip={sp}(PF={x['slip_full_2yr'].get('0.05', 0):.2f})")


if __name__ == "__main__":
    main()
