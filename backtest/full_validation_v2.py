#!/usr/bin/env python3
"""
Full Validation Chain V2 — tests ALL symbol/strategy/param combos
including low R:R (1.0-2.5) that were previously excluded.

No holdout PF filter — tests EVERYTHING.

Usage:
  python backtest/full_validation_v2.py --slice 1/3 --workers 8   # Mac Mini
  python backtest/full_validation_v2.py --slice 2/3 --workers 27  # Superpower
  python backtest/full_validation_v2.py --slice 3/3 --workers 7   # GCP
  python backtest/full_validation_v2.py --dry-run
"""
import os
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

# Load .env BEFORE anything else — must be in os.environ for spawn workers
from pathlib import Path as _P
from dotenv import load_dotenv
for _envpath in [_P(__file__).resolve().parents[1] / ".env",
                 _P.home() / "projects" / "dl_course" / ".env",
                 _P.home() / "projects" / "day-trading-bot" / ".env"]:
    if _envpath.exists():
        load_dotenv(_envpath)
        break

import argparse
import itertools
import multiprocessing as mp
import sys
import time as _time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_ROOT = Path(__file__).resolve().parents[1]

# ALL symbols, strategies, and params — no filtering
SYMBOLS = ["AMD", "META", "COIN", "TSLA", "NVDA", "SPY", "QQQ", "AAPL",
           "AMZN", "MSFT", "GOOGL", "JPM", "MSTR", "IWM",
           "XLK", "XLF", "XLE", "XLV", "XLI"]
STRATEGIES = ["orb", "ema_crossover", "momentum"]
STOP_LOSSES = [0.3, 0.5, 0.75, 1.0]
REWARD_RISKS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

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
    [0.3, 0.5, 0.75, 1.0],
    [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    [1.0, 1.2],
))

SLIPPAGE_LEVELS = [0.0, 0.02, 0.05, 0.10, 0.20]
SLIPPAGE_PERIODS = [
    ("2023-01-01", "2025-01-01", "full_2yr"),
    ("2022-01-01", "2022-12-31", "holdout"),
    ("2025-06-01", "2026-03-31", "bear_2025"),
]

MC_PERMUTATIONS = 200


def _validate_one(args):
    strategy, symbol, sl, rr, cfg_path = args

    # Load .env in worker process (spawn context doesn't inherit env)
    from pathlib import Path as _WP
    from dotenv import load_dotenv as _ld
    _env = _WP(cfg_path).resolve().parent.parent / ".env"
    _ld(_env)
    import os
    if not os.environ.get("ALPACA_API_KEY"):
        # Fallback: try common locations
        for p in [_WP.home() / "projects" / "dl_course" / ".env",
                  _WP.home() / "projects" / "day-trading-bot" / ".env"]:
            if p.exists():
                _ld(p)
                break

    import yaml
    from src.engine import TradingEngine

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["mode"] = "backtest"
    cfg.setdefault("account", {})["balance"] = 500.0

    try:
        engine = TradingEngine(cfg)
    except Exception as e:
        return {"strategy": strategy, "symbol": symbol, "sl": sl, "rr": rr, "error": str(e)}

    base = {
        "active_strategy": strategy, "stop_loss_pct": sl, "reward_risk": rr,
        "volume_mult": 1.2, "regime_filter": True, "vix_threshold": 25,
        "max_trades_per_day": 100, "afternoon_entries": True, "entry_window_minutes": 360,
    }

    result = {
        "strategy": strategy, "symbol": symbol, "sl": sl, "rr": rr,
        "wf_wins": 0, "wf_total": 0, "wf_avg_test_pf": 0, "wf_test_pfs": [],
        "mc_real_pf": 0, "mc_random_pf": 0, "mc_p_value": 1.0,
        "slip_full_2yr": {}, "slip_holdout": {}, "slip_bear": {},
        "full_2yr_trades": 0, "full_2yr_pf": 0, "full_2yr_wr": 0,
        "full_2yr_avg_dur": 0,
    }

    # Quick check: does this combo have any trades?
    try:
        quick = engine.run_backtest_config(
            config_override=base,
            start_date="2023-01-01", end_date="2025-01-01", symbol=symbol,
        )
        result["full_2yr_trades"] = quick.get("total_trades", 0)
        result["full_2yr_pf"] = quick.get("profit_factor", 0)
        result["full_2yr_wr"] = quick.get("win_rate", 0)
        result["full_2yr_avg_dur"] = quick.get("avg_duration_min", 0)
        if quick.get("total_trades", 0) < 10:
            return result  # Not enough trades, skip validation
    except Exception as e:
        result["error"] = str(e)
        return result

    # ═══ 1. WALK-FORWARD ═══
    wf_pfs = []
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

        best_pf = 0
        best_p = None
        for p_sl, p_rr, p_vol in WF_PARAM_GRID:
            try:
                m = engine.run_backtest_config(
                    config_override={**base, "stop_loss_pct": p_sl, "reward_risk": p_rr, "volume_mult": p_vol},
                    start_date=ts, end_date=te, symbol=symbol,
                    cached_bars=train_bars, cached_sma=sma, cached_vix={},
                )
                pf = m.get("profit_factor", 0)
                if pf > best_pf and m.get("total_trades", 0) >= 5:
                    best_pf = pf
                    best_p = (p_sl, p_rr, p_vol)
            except Exception:
                pass

        if best_p is None:
            continue

        try:
            tm = engine.run_backtest_config(
                config_override={**base, "stop_loss_pct": best_p[0], "reward_risk": best_p[1], "volume_mult": best_p[2]},
                start_date=vs, end_date=ve, symbol=symbol,
                cached_bars=test_bars, cached_sma=sma, cached_vix={},
            )
            wf_pfs.append(tm.get("profit_factor", 0))
        except Exception:
            pass

    result["wf_total"] = len(wf_pfs)
    result["wf_wins"] = sum(1 for p in wf_pfs if p > 1.0)
    result["wf_avg_test_pf"] = sum(wf_pfs) / len(wf_pfs) if wf_pfs else 0
    result["wf_test_pfs"] = [round(p, 2) for p in wf_pfs]

    # ═══ 2. MONTE CARLO ═══
    try:
        real_pf = result["full_2yr_pf"]
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
                    config_override={**base, "volume_mult": 0.0, "regime_filter": False,
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
            result["mc_random_pf"] = round(sum(random_pfs) / len(random_pfs), 3)
            beats = sum(1 for p in random_pfs if real_pf > p)
            result["mc_p_value"] = round(1 - beats / len(random_pfs), 3)
    except Exception:
        pass

    # ═══ 3. SLIPPAGE ═══
    for start, end, plabel in SLIPPAGE_PERIODS:
        slip_results = {}
        for slip in SLIPPAGE_LEVELS:
            try:
                m = engine.run_backtest_config(
                    config_override={**base, "slippage_pct": slip},
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
    args = parser.parse_args()

    # Generate ALL combos
    all_configs = list(itertools.product(STRATEGIES, SYMBOLS, STOP_LOSSES, REWARD_RISKS))
    print(f"Total configs: {len(all_configs)}")

    if args.dry_run:
        print(f"Est. ~3 min per config → ~{len(all_configs) * 3 / 60:.0f} hours total")
        return

    slice_n, slice_m = map(int, args.slice.split("/"))
    chunk = len(all_configs) // slice_m
    start_idx = (slice_n - 1) * chunk
    end_idx = start_idx + chunk if slice_n < slice_m else len(all_configs)
    configs = all_configs[start_idx:end_idx]

    print(f"Slice {slice_n}/{slice_m}: {len(configs)} configs")

    cfg_path = str(_ROOT / "config" / "config.yaml")
    work = [(s, sym, sl, rr, cfg_path) for s, sym, sl, rr in configs]

    n_workers = args.workers or min(8, max(1, mp.cpu_count() - 2))
    print(f"Workers: {n_workers}")

    t0 = _time.time()
    all_results = []
    done = 0

    # Use 'fork' on Linux (inherits env vars), 'spawn' on macOS (avoids fork crash)
    ctx = mp.get_context("fork" if sys.platform == "linux" else "spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        for r in pool.map(_validate_one, work):
            all_results.append(r)
            done += 1
            if done % 10 == 0:
                elapsed = (_time.time() - t0) / 60
                eta = (len(configs) - done) / (done / elapsed) if done > 0 else 0
                wf_pass = sum(1 for x in all_results if x.get("wf_wins", 0) >= 6)
                mc_pass = sum(1 for x in all_results if x.get("mc_p_value", 1) < 0.05)
                slip_pass = sum(1 for x in all_results if x.get("slip_full_2yr", {}).get("0.05", 0) > 1.0)
                all_pass = sum(1 for x in all_results
                               if x.get("wf_wins", 0) >= 6
                               and x.get("mc_p_value", 1) < 0.05
                               and x.get("slip_full_2yr", {}).get("0.05", 0) > 1.0)
                print(f"  [{done}/{len(configs)}] {elapsed:.1f}min ETA={eta:.0f}min | "
                      f"WF={wf_pass} MC={mc_pass} Slip={slip_pass} ALL={all_pass}")

    # Save
    out_path = _ROOT / "logs" / "research" / f"validation_v2_s{slice_n}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    elapsed = (_time.time() - t0) / 60
    print(f"\n{'='*70}")
    print(f"  VALIDATION V2 COMPLETE — {done} configs in {elapsed:.1f} min")
    print(f"{'='*70}")

    tested = [x for x in all_results if x.get("full_2yr_trades", 0) >= 10]
    wf_pass = [x for x in tested if x.get("wf_wins", 0) >= 6]
    mc_pass = [x for x in tested if x.get("mc_p_value", 1) < 0.05]
    slip_pass = [x for x in tested if x.get("slip_full_2yr", {}).get("0.05", 0) > 1.0]
    all_pass = [x for x in tested
                if x.get("wf_wins", 0) >= 6
                and x.get("mc_p_value", 1) < 0.05
                and x.get("slip_full_2yr", {}).get("0.05", 0) > 1.0]

    print(f"  Tested (≥10 trades): {len(tested)}/{done}")
    print(f"  Walk-forward pass: {len(wf_pass)}")
    print(f"  Monte Carlo pass:  {len(mc_pass)}")
    print(f"  Slippage pass:     {len(slip_pass)}")
    print(f"  ★ ALL THREE: {len(all_pass)}")

    if all_pass:
        print(f"\n  FULLY VALIDATED:")
        print(f"  {'Strategy':<16} {'Symbol':<8} {'SL':<5} {'RR':<5} {'WF':>5} {'MC p':>7} {'Slip':>6} {'Trades':>7} {'PF':>6} {'WR%':>5} {'Dur':>5}")
        print(f"  {'-'*80}")
        for x in sorted(all_pass, key=lambda r: -r.get("slip_full_2yr", {}).get("0.05", 0)):
            print(f"  {x['strategy']:<16} {x['symbol']:<8} {x['sl']:<5} {x['rr']:<5} "
                  f"{x['wf_wins']}/{x['wf_total']:>2}  p={x['mc_p_value']:.3f} "
                  f"PF={x['slip_full_2yr'].get('0.05', 0):.2f} "
                  f"{x['full_2yr_trades']:>7} {x['full_2yr_pf']:>6.2f} {x['full_2yr_wr']:>4.0f}% {x['full_2yr_avg_dur']:>4.0f}m")

    # Also print top by PF regardless of validation
    print(f"\n  TOP 20 BY SLIPPAGE-ADJUSTED PF (regardless of WF/MC):")
    by_slip = sorted(tested, key=lambda x: -x.get("slip_full_2yr", {}).get("0.05", 0))
    for x in by_slip[:20]:
        wf = "✓" if x.get("wf_wins", 0) >= 6 else "✗"
        mc = "✓" if x.get("mc_p_value", 1) < 0.05 else "✗"
        print(f"  {x['strategy']:<16} {x['symbol']:<8} SL={x['sl']} RR={x['rr']} "
              f"WF={wf}({x['wf_wins']}/{x['wf_total']}) MC={mc}(p={x['mc_p_value']:.3f}) "
              f"Slip={x['slip_full_2yr'].get('0.05', 0):.2f} "
              f"trades={x['full_2yr_trades']} PF={x['full_2yr_pf']:.2f} dur={x['full_2yr_avg_dur']:.0f}m")


if __name__ == "__main__":
    main()
