#!/usr/bin/env python3
"""
Loosened Restrictions Grid — comprehensive exploration of wider entry windows,
higher max_trades, VIX loosening, and afternoon entries.

New dimensions not covered in earlier grids:
  - entry_window_minutes: [15, 30, 45, 75] → cutoffs at 10:00, 10:15, 10:30, 11:00 ET
  - max_trades_per_day: [1, 2, 3]
  - vix_threshold: [30, 40] (40 = much looser than 30)
  - afternoon_entries: [False, True]
  - balance: $500 (up from $200)

~69,120 combos. Uses the parallel batch runner framework. ~23 hours on
superpower (28 cores, 56 threads).

Usage:
    python backtest/loosened_grid.py                    # full run
    python backtest/loosened_grid.py --dry-run           # count only
    python backtest/loosened_grid.py --workers 27        # custom worker count
"""
import os
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import argparse
import itertools
import multiprocessing as mp
import sys
import time as _time_mod
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

_ROOT = Path(__file__).resolve().parents[1]

LOOSENED_GRID = {
    "strategies":         ["orb", "ema_crossover"],
    "symbols":            ["NVDA", "AMD", "AAPL", "SPY"],
    "periods": [
        {"label": "bull_2023",    "start": "2023-01-01", "end": "2023-12-31"},
        {"label": "bull_2024",    "start": "2024-01-01", "end": "2024-12-31"},
        {"label": "bear_2025",    "start": "2025-06-01", "end": "2026-03-31"},
        {"label": "holdout_2022", "start": "2022-01-01", "end": "2022-12-31"},
    ],
    "entry_window_minutes": [15, 30, 45, 75],  # minutes after 9:45 → cutoff at 10:00/10:15/10:30/11:00
    "stop_loss_pct":        [0.3, 0.5, 0.75],
    "reward_risk":          [2.0, 2.5, 3.0, 4.0, 5.0],
    "volume_mult":          [1.0, 1.2],
    "regime_filter":        [True, False],
    "vix_threshold":        [30, 40],
    "max_trades_per_day":   [1, 2, 3],
    "afternoon_entries":    [False, True],
}


def _generate_combinations(grid):
    hyper_keys = ["entry_window_minutes", "stop_loss_pct", "reward_risk",
                  "volume_mult", "regime_filter", "vix_threshold",
                  "max_trades_per_day", "afternoon_entries"]
    hyper_vals = [grid[k] for k in hyper_keys]
    combos = []
    for strategy, symbol, period in itertools.product(
        grid["strategies"], grid["symbols"], grid["periods"]
    ):
        for combo in itertools.product(*hyper_vals):
            params = dict(zip(hyper_keys, combo))
            params.update({
                "strategy":     strategy,
                "symbol":       symbol,
                "period_label": period["label"],
                "start_date":   period["start"],
                "end_date":     period["end"],
            })
            combos.append(params)
    return combos


def _worker(args_tuple):
    symbol, period_label, combos_for_pair, base_cfg_path = args_tuple

    import yaml as _yaml
    from src.engine import TradingEngine
    from src.logger.results_store import ResultsStore, make_run_id

    t0 = _time_mod.time()
    result = {
        "symbol": symbol, "period": period_label,
        "done": 0, "skipped": 0, "errors": 0, "no_data": False,
        "elapsed": 0.0,
    }
    if not combos_for_pair:
        return result

    with open(base_cfg_path) as f:
        base_cfg = _yaml.safe_load(f)
    base_cfg["mode"] = "backtest"
    # Use the new $500 balance
    base_cfg.setdefault("account", {})["balance"] = 500.0
    sma_days = base_cfg.get("filters", base_cfg.get("strategy", {})).get("regime_sma_days", 20)

    try:
        engine = TradingEngine(base_cfg)
    except Exception as exc:
        result["errors"] = len(combos_for_pair)
        return result

    store = ResultsStore()
    start_date = combos_for_pair[0]["start_date"]
    end_date   = combos_for_pair[0]["end_date"]

    try:
        bars = engine.fetcher.fetch_historical(
            symbol, start_date, end_date, filter_windows=False
        )
    except Exception:
        result["no_data"] = True
        result["skipped"] = len(combos_for_pair)
        result["elapsed"] = _time_mod.time() - t0
        return result

    if bars is None or bars.empty:
        result["no_data"] = True
        result["skipped"] = len(combos_for_pair)
        result["elapsed"] = _time_mod.time() - t0
        return result

    try:
        cached_sma = engine._compute_daily_sma(symbol, start_date, end_date, sma_days)
    except Exception:
        cached_sma = {}
    try:
        cached_vix = engine._fetch_vix_data(start_date, end_date)
    except Exception:
        cached_vix = {}

    for params in combos_for_pair:
        run_id = make_run_id(params)
        if store.run_exists(run_id):
            result["skipped"] += 1
            continue
        try:
            config_override = {
                "active_strategy":      params["strategy"],
                "stop_loss_pct":        params["stop_loss_pct"],
                "reward_risk":          params["reward_risk"],
                "volume_mult":          params["volume_mult"],
                "regime_filter":        params["regime_filter"],
                "vix_threshold":        params["vix_threshold"],
                "max_trades_per_day":   params["max_trades_per_day"],
                "afternoon_entries":    bool(params["afternoon_entries"]),
                "entry_window_minutes": params["entry_window_minutes"],
            }
            metrics = engine.run_backtest_config(
                config_override=config_override,
                start_date=start_date, end_date=end_date,
                symbol=symbol, cached_bars=bars,
                cached_sma=cached_sma, cached_vix=cached_vix,
            )
            trades = metrics.pop("trades_list", [])
            run_config = dict(params)
            run_config["run_id"] = run_id
            store.save_run(run_config, metrics, trades)
            result["done"] += 1
        except Exception as exc:
            result["errors"] += 1

    result["elapsed"] = _time_mod.time() - t0
    return result


def main():
    parser = argparse.ArgumentParser(description="Loosened restrictions grid")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    combos = _generate_combinations(LOOSENED_GRID)
    total = len(combos)

    print(f"\n  Loosened Restrictions Grid")
    print(f"  Strategies   : {LOOSENED_GRID['strategies']}")
    print(f"  Symbols      : {LOOSENED_GRID['symbols']}")
    print(f"  Entry windows: {LOOSENED_GRID['entry_window_minutes']} min after 9:45")
    print(f"  Max trades   : {LOOSENED_GRID['max_trades_per_day']}")
    print(f"  VIX          : {LOOSENED_GRID['vix_threshold']}")
    print(f"  PM entries   : {LOOSENED_GRID['afternoon_entries']}")
    print(f"  Balance      : $500")
    print(f"  Total combos : {total:,}")

    if args.dry_run:
        return

    n_workers = args.workers if args.workers is not None else max(1, mp.cpu_count() - 1)
    print(f"  Workers      : {n_workers} (cpu_count={mp.cpu_count()})")

    groups = {}
    for p in combos:
        key = (p["symbol"], p["period_label"])
        groups.setdefault(key, []).append(p)
    n_units = len(groups)
    print(f"  Work units   : {n_units}")
    print(f"  Avg per unit : {total // max(n_units, 1):,}\n")

    base_cfg_path = str(_ROOT / "config" / "config.yaml")
    work_units = [
        (sym, per, combo_list, base_cfg_path)
        for (sym, per), combo_list in groups.items()
    ]

    done = skipped = errors = 0
    completed = 0
    t0 = _time_mod.time()
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        futures = {ex.submit(_worker, wu): wu for wu in work_units}
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception as exc:
                print(f"  WORKER FAILED: {exc}")
                continue
            done    += res["done"]
            skipped += res["skipped"]
            errors  += res["errors"]
            completed += 1
            elapsed = _time_mod.time() - t0
            eta_s = (elapsed / completed * (n_units - completed)
                     if completed > 0 else 0)
            nd = " (NO DATA)" if res.get("no_data") else ""
            er = f" err={res['errors']}" if res["errors"] else ""
            print(
                f"  [{completed:>3}/{n_units}] {res['symbol']:<5} {res['period']:<13} "
                f"done={res['done']:>5} skip={res['skipped']:>5}{er}"
                f"  ({res['elapsed']:.0f}s){nd}"
                f"  ETA {int(eta_s)//60}m{int(eta_s)%60:02d}s"
            )

    elapsed = _time_mod.time() - t0
    print(f"\n  Done.  ran={done}  skipped={skipped}  errors={errors}"
          f"  elapsed={elapsed:.1f}s ({elapsed/60:.1f}m / {elapsed/3600:.1f}h)")


if __name__ == "__main__":
    main()
