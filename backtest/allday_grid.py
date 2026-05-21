#!/usr/bin/env python3
"""
All-Day Entry Window Grid — test strategies with entry windows extending
through the entire trading day (9:45 → 15:45 ET).

Two modes:
  --scope quick   : EMA only, 3 symbols, holdout+full_2yr, ~500 combos (Mac Mini)
  --scope full    : EMA+ORB+Momentum, 4 symbols, 4 periods, ~14,000 combos (superpower)

The key new dimension is entry_window_minutes=[120, 180, 240, 360] which
translates to entry cutoffs at 11:45, 12:45, 13:45, 15:45 ET — filling
the dead zone between the morning and afternoon windows.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

_ROOT = Path(__file__).resolve().parents[1]

QUICK_GRID = {
    "strategies":          ["ema_crossover"],
    "symbols":             ["NVDA", "AMD", "AAPL"],
    "periods": [
        {"label": "holdout_2022", "start": "2022-01-01", "end": "2022-12-31"},
        {"label": "full_2yr",     "start": "2023-01-01", "end": "2025-01-01"},
    ],
    "entry_window_minutes": [120, 180, 240, 360],
    "stop_loss_pct":        [0.3, 0.5, 1.0],
    "reward_risk":          [2.0, 2.5, 4.0],
    "volume_mult":          [1.0, 1.2],
    "regime_filter":        [True, False],
    "vix_threshold":        [30],
    "max_trades_per_day":   [1, 2, 3],
    "afternoon_entries":    [True],
}

FULL_GRID = {
    "strategies":          ["ema_crossover", "orb", "momentum"],
    "symbols":             ["NVDA", "AMD", "AAPL", "SPY"],
    "periods": [
        {"label": "bull_2023",    "start": "2023-01-01", "end": "2023-12-31"},
        {"label": "bull_2024",    "start": "2024-01-01", "end": "2024-12-31"},
        {"label": "bear_2025",    "start": "2025-06-01", "end": "2026-03-31"},
        {"label": "holdout_2022", "start": "2022-01-01", "end": "2022-12-31"},
    ],
    "entry_window_minutes": [120, 180, 240, 360],
    "stop_loss_pct":        [0.3, 0.5, 1.0],
    "reward_risk":          [2.0, 2.5, 3.0, 4.0],
    "volume_mult":          [1.0, 1.2],
    "regime_filter":        [True, False],
    "vix_threshold":        [30, 40],
    "max_trades_per_day":   [1, 2, 3],
    "afternoon_entries":    [True],
}


def _generate_combos(grid):
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
                "strategy": strategy, "symbol": symbol,
                "period_label": period["label"],
                "start_date": period["start"], "end_date": period["end"],
            })
            combos.append(params)
    return combos


def _worker(args_tuple):
    symbol, period_label, combos_for_pair, base_cfg_path = args_tuple
    import yaml as _yaml
    from src.engine import TradingEngine
    from src.logger.results_store import ResultsStore, make_run_id

    t0 = _time_mod.time()
    result = {"symbol": symbol, "period": period_label,
              "done": 0, "skipped": 0, "errors": 0, "no_data": False, "elapsed": 0.0}
    if not combos_for_pair:
        return result

    with open(base_cfg_path) as f:
        cfg = _yaml.safe_load(f)
    cfg["mode"] = "backtest"
    cfg.setdefault("account", {})["balance"] = 500.0
    sma_days = cfg.get("filters", cfg.get("strategy", {})).get("regime_sma_days", 20)

    try:
        engine = TradingEngine(cfg)
    except Exception:
        result["errors"] = len(combos_for_pair)
        return result

    store = ResultsStore()
    start_date = combos_for_pair[0]["start_date"]
    end_date = combos_for_pair[0]["end_date"]

    try:
        bars = engine.fetcher.fetch_historical(symbol, start_date, end_date, filter_windows=False)
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
            metrics = engine.run_backtest_config(
                config_override={
                    "active_strategy":      params["strategy"],
                    "stop_loss_pct":        params["stop_loss_pct"],
                    "reward_risk":          params["reward_risk"],
                    "volume_mult":          params["volume_mult"],
                    "regime_filter":        params["regime_filter"],
                    "vix_threshold":        params["vix_threshold"],
                    "max_trades_per_day":   params["max_trades_per_day"],
                    "afternoon_entries":    bool(params["afternoon_entries"]),
                    "entry_window_minutes": params["entry_window_minutes"],
                },
                start_date=start_date, end_date=end_date,
                symbol=symbol, cached_bars=bars,
                cached_sma=cached_sma, cached_vix=cached_vix,
            )
            trades = metrics.pop("trades_list", [])
            run_config = dict(params)
            run_config["run_id"] = run_id
            store.save_run(run_config, metrics, trades)
            result["done"] += 1
        except Exception:
            result["errors"] += 1

    result["elapsed"] = _time_mod.time() - t0
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default="quick", choices=["quick", "full"])
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    grid = QUICK_GRID if args.scope == "quick" else FULL_GRID
    combos = _generate_combos(grid)
    total = len(combos)

    print(f"\n  All-Day Entry Grid ({args.scope})")
    print(f"  Strategies      : {grid['strategies']}")
    print(f"  Symbols         : {grid['symbols']}")
    print(f"  Entry windows   : {grid['entry_window_minutes']} min after 9:45")
    cutoffs = [f"{(9*60+45+m)//60}:{(9*60+45+m)%60:02d}" for m in grid['entry_window_minutes']]
    print(f"                  → cutoffs at {cutoffs} ET")
    print(f"  Max trades/day  : {grid['max_trades_per_day']}")
    print(f"  Balance         : $500")
    print(f"  Total combos    : {total:,}")

    if args.dry_run:
        return

    n_workers = args.workers or max(1, mp.cpu_count() - 1)
    groups = {}
    for p in combos:
        key = (p["symbol"], p["period_label"])
        groups.setdefault(key, []).append(p)

    print(f"  Workers         : {n_workers}")
    print(f"  Work units      : {len(groups)}\n")

    base_cfg_path = str(_ROOT / "config" / "config.yaml")
    work_units = [(s, p, cl, base_cfg_path) for (s, p), cl in groups.items()]

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
            done += res["done"]
            skipped += res["skipped"]
            errors += res["errors"]
            completed += 1
            elapsed = _time_mod.time() - t0
            n_units = len(work_units)
            eta_s = (elapsed / completed * (n_units - completed)) if completed else 0
            nd = " (NO DATA)" if res.get("no_data") else ""
            print(f"  [{completed:>3}/{n_units}] {res['symbol']:<5} {res['period']:<13} "
                  f"done={res['done']:>5} skip={res['skipped']:>5}{nd}  "
                  f"({res['elapsed']:.0f}s)  ETA {int(eta_s)//60}m{int(eta_s)%60:02d}s")

    elapsed = _time_mod.time() - t0
    print(f"\n  Done.  ran={done}  skipped={skipped}  errors={errors}"
          f"  elapsed={elapsed:.1f}s ({elapsed/60:.1f}m / {elapsed/3600:.1f}h)")


if __name__ == "__main__":
    main()
