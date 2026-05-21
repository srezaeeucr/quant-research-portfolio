#!/usr/bin/env python3
"""
Focused holdout_2022 backfill — runs only the holdout period for every
existing (strategy, symbol) pair so the dashboard scorecard's overfitting
metric can compute real ratios for everything in results.db.

Designed to be run on the superpower server then rsync'd back.

Usage:
    python backtest/holdout_backfill.py            # parallel (cpu_count - 1)
    python backtest/holdout_backfill.py --workers 27
    python backtest/holdout_backfill.py --dry-run

Grid (PM=False only — afternoon entries get a separate validation later):
    strategies × symbols × stop_loss × reward_risk × vol × regime × vix × max_t
        5      ×    11    ×     3     ×      5      ×  2  ×   2    ×  2  ×  2
                                                                   = 13,200 combos
"""
import os
# Set BEFORE any pandas/numpy imports to allow safe forking on macOS.
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

# ── Grid ──────────────────────────────────────────────────────────────────
GRID = {
    "strategies": ["orb", "vwap_reversion", "gap_fill", "ema_crossover", "momentum"],
    "symbols":    ["SPY", "QQQ", "IWM", "AAPL", "TSLA",
                   "MSFT", "NVDA", "AMZN", "META", "GOOGL", "JPM"],
    "period":     {"label": "holdout_2022",
                   "start": "2022-01-01", "end": "2022-12-31"},
    "stop_loss_pct":      [0.3, 0.5, 0.75],
    "reward_risk":        [2.0, 2.5, 3.0, 4.0, 5.0],
    "volume_mult":        [1.0, 1.2],
    "regime_filter":      [True, False],
    "vix_threshold":      [25, 30],
    "max_trades_per_day": [1, 2],
    "afternoon_entries":  [False],  # PM kept off for the backfill
}


def _generate_combos():
    out = []
    hyper_keys = ["stop_loss_pct", "reward_risk", "volume_mult",
                  "regime_filter", "vix_threshold", "max_trades_per_day",
                  "afternoon_entries"]
    hyper_vals = [GRID[k] for k in hyper_keys]
    for strategy, symbol in itertools.product(GRID["strategies"], GRID["symbols"]):
        for combo in itertools.product(*hyper_vals):
            params = dict(zip(hyper_keys, combo))
            params.update({
                "strategy":     strategy,
                "symbol":       symbol,
                "period_label": GRID["period"]["label"],
                "start_date":   GRID["period"]["start"],
                "end_date":     GRID["period"]["end"],
            })
            out.append(params)
    return out


def _worker(args_tuple):
    """Run all combos for one (symbol, period) pair — same shape as
    batch_runner._run_symbol_period_unit."""
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
    sma_days = base_cfg.get("filters", base_cfg.get("strategy", {})).get("regime_sma_days", 20)

    try:
        engine = TradingEngine(base_cfg)
    except Exception as exc:
        result["errors"] = len(combos_for_pair)
        result["error_msg"] = f"engine init: {exc}"
        return result

    store = ResultsStore()
    start_date = combos_for_pair[0]["start_date"]
    end_date   = combos_for_pair[0]["end_date"]

    try:
        bars = engine.fetcher.fetch_historical(
            symbol, start_date, end_date, filter_windows=False
        )
    except Exception as exc:
        result["no_data"] = True
        result["skipped"] = len(combos_for_pair)
        result["error_msg"] = f"fetch failed: {exc}"
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
                    "active_strategy":    params["strategy"],
                    "stop_loss_pct":      params["stop_loss_pct"],
                    "reward_risk":        params["reward_risk"],
                    "volume_mult":        params["volume_mult"],
                    "regime_filter":      params["regime_filter"],
                    "vix_threshold":      params["vix_threshold"],
                    "max_trades_per_day": params["max_trades_per_day"],
                    "afternoon_entries":  bool(params.get("afternoon_entries", False)),
                },
                start_date=start_date,
                end_date=end_date,
                symbol=symbol,
                cached_bars=bars,
                cached_sma=cached_sma,
                cached_vix=cached_vix,
            )
            trades = metrics.pop("trades_list", [])
            run_config = dict(params)
            run_config["run_id"] = run_id
            store.save_run(run_config, metrics, trades)
            result["done"] += 1
        except Exception as exc:
            result["errors"] += 1
            if "error_msg" not in result:
                result["error_msg"] = f"{params['strategy']}: {exc}"

    result["elapsed"] = _time_mod.time() - t0
    return result


def parse_args():
    p = argparse.ArgumentParser(description="Holdout 2022 backfill runner")
    p.add_argument("--workers", type=int, default=None,
                   help="Parallel worker count (default: cpu_count - 1)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print combo count and exit")
    return p.parse_args()


def main():
    args = parse_args()
    combos = _generate_combos()
    total = len(combos)

    print(f"\n  Holdout 2022 backfill")
    print(f"  Strategies   : {len(GRID['strategies'])}")
    print(f"  Symbols      : {len(GRID['symbols'])}")
    print(f"  Combos/symbol: {total // len(GRID['symbols'])}")
    print(f"  Total combos : {total:,}")

    if args.dry_run:
        return

    n_workers = args.workers if args.workers is not None else max(1, mp.cpu_count() - 1)
    print(f"  Workers      : {n_workers} (cpu_count={mp.cpu_count()})")

    # Group by (symbol, period) — one work unit per group
    groups = {}
    for p in combos:
        key = (p["symbol"], p["period_label"])
        groups.setdefault(key, []).append(p)
    print(f"  Work units   : {len(groups)} (one per symbol)")
    print(f"  Avg per unit : {total // len(groups):,}\n")

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
            n_units = len(work_units)
            eta_s = (elapsed / completed * (n_units - completed)
                     if completed > 0 else 0)
            no_data_tag = " (NO DATA)" if res.get("no_data") else ""
            err_tag = f" err={res['errors']}" if res["errors"] else ""

            print(
                f"  [{completed:>3}/{n_units}] {res['symbol']:<5} "
                f"done={res['done']:>4} skip={res['skipped']:>4}{err_tag}"
                f"  ({res['elapsed']:.0f}s){no_data_tag}"
                f"  ETA {int(eta_s)//60}m{int(eta_s)%60:02d}s"
            )

    elapsed = _time_mod.time() - t0
    print(f"\n  Done.  ran={done}  skipped={skipped}  errors={errors}"
          f"  elapsed={elapsed:.1f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
