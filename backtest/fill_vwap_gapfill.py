#!/usr/bin/env python3
"""
Fill VWAP Reversion + Gap Fill grid — the two most underexplored strategies.
Most symbol×strategy combos have only 816-1920 rows vs 20k+ for ORB/EMA.

19 symbols × 2 strategies × 5 periods × 7 SL × 9 RR × 2 vol × 2 vix × 2 mt × 2 ew
= 191,520 combos. Skip-if-exists handles dedup.

Mac Mini, ~12 hours at 8 workers.
"""
import os
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import argparse
import itertools
import multiprocessing as mp
import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_ROOT = Path(__file__).resolve().parents[1]

GRID = {
    "symbols": ["NVDA", "AMD", "SPY", "QQQ", "AAPL", "COIN", "META",
                "AMZN", "MSFT", "GOOGL", "JPM", "MSTR", "IWM", "TSLA",
                "XLK", "XLF", "XLE", "XLV", "XLI"],
    "strategies": ["orb", "ema_crossover", "momentum", "vwap_reversion", "gap_fill"],
    "periods": [
        {"label": "bull_2023", "start": "2023-01-01", "end": "2023-12-31"},
        {"label": "bull_2024", "start": "2024-01-01", "end": "2024-12-31"},
        {"label": "bear_2025", "start": "2025-06-01", "end": "2026-03-31"},
        {"label": "full_2yr", "start": "2023-01-01", "end": "2025-01-01"},
        {"label": "holdout_2022", "start": "2022-01-01", "end": "2022-12-31"},
    ],
    "stop_loss_pct": [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    "reward_risk": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0],
    "volume_mult": [1.0, 1.2],
    "vix_threshold": [25, 30],
    "max_trades_per_day": [2, 3],
    "afternoon_entries": [True],
    "entry_window_minutes": [180, 360],
}


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
                    "regime_filter": True,
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
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    g = GRID
    all_combos = []
    for sym, strat, period, sl, rr, vol, vix, mt, ew in itertools.product(
        g["symbols"], g["strategies"], g["periods"],
        g["stop_loss_pct"], g["reward_risk"], g["volume_mult"],
        g["vix_threshold"], g["max_trades_per_day"], g["entry_window_minutes"],
    ):
        all_combos.append({
            "symbol": sym, "strategy": strat,
            "period_label": period["label"],
            "start_date": period["start"], "end_date": period["end"],
            "stop_loss_pct": sl, "reward_risk": rr,
            "volume_mult": vol, "vix_threshold": vix,
            "max_trades_per_day": mt, "afternoon_entries": True,
            "entry_window_minutes": ew, "regime_filter": True,
        })

    print(f"Total combos: {len(all_combos):,}")
    if args.dry_run:
        return

    groups = defaultdict(list)
    for c in all_combos:
        groups[(c["symbol"], c["period_label"])].append(c)

    work_units = list(groups.items())
    total_combos = sum(len(v) for _, v in work_units)
    print(f"{len(work_units)} (symbol,period) pairs, {total_combos:,} combos")

    cfg_path = str(_ROOT / "config" / "config.yaml")
    work = [(k[0], k[1], v, cfg_path) for k, v in work_units]

    n_workers = args.workers or max(1, mp.cpu_count() - 2)
    print(f"Workers: {n_workers}")

    t0 = _time.time()
    total_done = total_skip = total_err = 0

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as pool:
        for r in pool.map(_worker, work):
            total_done += r["done"]
            total_skip += r["skipped"]
            total_err += r["errors"]
            processed = total_done + total_skip + total_err
            if processed % 5000 < 500:
                elapsed = _time.time() - t0
                rate = total_done / elapsed if elapsed > 0 else 0
                remaining = total_combos - processed
                eta = remaining / (processed / elapsed) / 60 if processed > 0 else 0
                print(f"  [{processed:,}/{total_combos:,}] done={total_done:,} skip={total_skip:,} "
                      f"err={total_err} rate={rate:.1f} new/s ETA={eta:.0f}min")

    elapsed = (_time.time() - t0) / 60
    print(f"\nCOMPLETE in {elapsed:.1f} min ({elapsed/60:.1f} hours)")
    print(f"  Done: {total_done:,} | Skipped: {total_skip:,} | Errors: {total_err:,}")


if __name__ == "__main__":
    main()
