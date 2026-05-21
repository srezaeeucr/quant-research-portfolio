#!/usr/bin/env python3
"""
Expanded Parameter Search — explores untested parameter regions.
Designed to split across machines via --slice N/M.

New parameters tested:
- Stop loss: 1.5%, 2.0%, 3.0% (wide stops for volatile symbols)
- R:R: 1.0, 1.5 (low targets, high WR) + 6.0, 8.0 (lottery tickets)
- Volume mult: 0.0 (no filter), 0.5, 1.5, 2.0
- OR duration: 5, 10, 20, 30 min
- Max trades: 4, 5
- VIX: 15, 20 (calm-only)
- Strategies: orb, ema_crossover, momentum (top 3)
- Symbols: NVDA, AMD, SPY, QQQ, AAPL, COIN, META (validated + promising)

Usage:
  python backtest/expanded_params.py --slice 1/4 --workers 8   # Mac Mini
  python backtest/expanded_params.py --slice 2/4 --workers 27  # Superpower
  python backtest/expanded_params.py --slice 3/4 --workers 4   # MacBook Air
  python backtest/expanded_params.py --slice 4/4 --workers 7   # GCP
  python backtest/expanded_params.py --dry-run                 # count combos
"""
import os
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import argparse
import itertools
import multiprocessing as mp
import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_ROOT = Path(__file__).resolve().parents[1]

GRID = {
    "symbols": ["NVDA", "AMD", "SPY", "QQQ", "AAPL", "COIN", "META",
                "AMZN", "MSFT", "GOOGL", "JPM", "MSTR", "IWM", "TSLA",
                "XLK", "XLF", "XLE", "XLV", "XLI"],
    "strategies": ["orb", "ema_crossover", "momentum"],
    "periods": [
        {"label": "bull_2023", "start": "2023-01-01", "end": "2023-12-31"},
        {"label": "bull_2024", "start": "2024-01-01", "end": "2024-12-31"},
        {"label": "bear_2025", "start": "2025-06-01", "end": "2026-03-31"},
        {"label": "full_2yr", "start": "2023-01-01", "end": "2025-01-01"},
        {"label": "holdout_2022", "start": "2022-01-01", "end": "2022-12-31"},
    ],
    # NEW parameter values (skip already-tested values)
    "stop_loss_pct": [1.5, 2.0, 3.0],
    "reward_risk": [1.0, 1.5, 6.0, 8.0],
    "volume_mult": [0.0, 0.5, 1.5, 2.0],
    "or_duration_min": [5, 10, 20, 30],
    "max_trades_per_day": [4, 5],
    "vix_threshold": [15, 20],
    "regime_filter": [True],
    "afternoon_entries": [True],
    "entry_window_minutes": [360],
}


def _generate_combos(grid):
    combos = []
    for sym, strat, period, sl, rr, vol, or_dur, mt, vix in itertools.product(
        grid["symbols"], grid["strategies"], grid["periods"],
        grid["stop_loss_pct"], grid["reward_risk"], grid["volume_mult"],
        grid["or_duration_min"], grid["max_trades_per_day"], grid["vix_threshold"],
    ):
        combos.append({
            "symbol": sym, "strategy": strat,
            "period_label": period["label"],
            "start_date": period["start"], "end_date": period["end"],
            "stop_loss_pct": sl, "reward_risk": rr,
            "volume_mult": vol, "or_duration_min": or_dur,
            "max_trades_per_day": mt, "vix_threshold": vix,
            "regime_filter": True, "afternoon_entries": True,
            "entry_window_minutes": 360,
        })
    return combos


def _worker(args):
    symbol, period_label, combos_for_pair, cfg_path = args

    import yaml as _yaml
    from src.engine import TradingEngine
    from src.logger.results_store import ResultsStore, make_run_id

    result = {"symbol": symbol, "period": period_label,
              "done": 0, "skipped": 0, "errors": 0}

    if not combos_for_pair:
        return result

    with open(cfg_path) as f:
        cfg = _yaml.safe_load(f)
    cfg["mode"] = "backtest"
    cfg.setdefault("account", {})["balance"] = 500.0

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
        result["errors"] = len(combos_for_pair)
        return result

    if bars is None or bars.empty:
        result["skipped"] = len(combos_for_pair)
        return result

    try:
        cached_sma = engine._compute_daily_sma(symbol, start_date, end_date, 20)
    except Exception:
        cached_sma = {}

    for params in combos_for_pair:
        run_id = make_run_id(params)
        if store.run_exists(run_id):
            result["skipped"] += 1
            continue
        try:
            metrics = engine.run_backtest_config(
                config_override={
                    "active_strategy": params["strategy"],
                    "stop_loss_pct": params["stop_loss_pct"],
                    "reward_risk": params["reward_risk"],
                    "volume_mult": params["volume_mult"],
                    "regime_filter": params["regime_filter"],
                    "vix_threshold": params["vix_threshold"],
                    "max_trades_per_day": params["max_trades_per_day"],
                    "afternoon_entries": params["afternoon_entries"],
                    "entry_window_minutes": params["entry_window_minutes"],
                },
                start_date=start_date, end_date=end_date,
                symbol=symbol, cached_bars=bars,
                cached_sma=cached_sma, cached_vix={},
            )
            trades = metrics.pop("trades_list", [])
            run_config = dict(params)
            run_config["run_id"] = run_id
            store.save_run(run_config, metrics, trades)
            result["done"] += 1
        except Exception:
            result["errors"] += 1

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", default="1/1", help="N/M — run slice N of M")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    slice_n, slice_m = map(int, args.slice.split("/"))

    all_combos = _generate_combos(GRID)
    print(f"Total combos: {len(all_combos):,}")

    if args.dry_run:
        print(f"Slice {slice_n}/{slice_m}: {len(all_combos)//slice_m:,} combos")
        return

    # Group by (symbol, period) for data caching
    from collections import defaultdict
    groups = defaultdict(list)
    for c in all_combos:
        groups[(c["symbol"], c["period_label"])].append(c)

    work_units = list(groups.items())
    # Slice
    chunk = len(work_units) // slice_m
    start_idx = (slice_n - 1) * chunk
    end_idx = start_idx + chunk if slice_n < slice_m else len(work_units)
    work_units = work_units[start_idx:end_idx]

    total_combos = sum(len(v) for _, v in work_units)
    print(f"Slice {slice_n}/{slice_m}: {len(work_units)} (symbol,period) pairs, {total_combos:,} combos")

    cfg_path = str(_ROOT / "config" / "config.yaml")
    work = [(k[0], k[1], v, cfg_path) for k, v in work_units]

    n_workers = args.workers or max(1, mp.cpu_count() - 2)
    print(f"Workers: {n_workers}")

    t0 = _time.time()
    total_done = 0
    total_skip = 0
    total_err = 0

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as pool:
        for r in pool.map(_worker, work):
            total_done += r["done"]
            total_skip += r["skipped"]
            total_err += r["errors"]
            elapsed = _time.time() - t0
            processed = total_done + total_skip + total_err
            if processed % 1000 < 100:
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total_combos - processed) / rate / 60 if rate > 0 else 0
                print(f"  [{processed:,}/{total_combos:,}] done={total_done} skip={total_skip} "
                      f"err={total_err} rate={rate:.0f}/s ETA={eta:.0f}min")

    elapsed = (_time.time() - t0) / 60
    print(f"\nCOMPLETE in {elapsed:.1f} min")
    print(f"  Done: {total_done:,} | Skipped: {total_skip:,} | Errors: {total_err:,}")


if __name__ == "__main__":
    main()
