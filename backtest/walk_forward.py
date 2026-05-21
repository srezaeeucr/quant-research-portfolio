#!/usr/bin/env python3
"""
S1: Walk-Forward Optimization — the gold standard for proving an edge.

Splits data into 8 rolling windows:
  Train on 6 months → test on next 3 months → roll forward.

For each window: find the best hyperparameters on the TRAIN period,
then measure performance on the TEST period. If the test-period PF
is consistently > 1.0 across all windows → the edge is REAL and
doesn't depend on knowing the future.

Designed to run on superpower (parallel across windows).
"""
import os
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import argparse
import itertools
import multiprocessing as mp
import sys
import time as _time_mod
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

_ROOT = Path(__file__).resolve().parents[1]

# Rolling windows (train 6 months, test 3 months)
WINDOWS = [
    {"train_start": "2022-01-01", "train_end": "2022-06-30", "test_start": "2022-07-01", "test_end": "2022-09-30", "label": "W1"},
    {"train_start": "2022-04-01", "train_end": "2022-09-30", "test_start": "2022-10-01", "test_end": "2022-12-31", "label": "W2"},
    {"train_start": "2022-07-01", "train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-03-31", "label": "W3"},
    {"train_start": "2022-10-01", "train_end": "2023-03-31", "test_start": "2023-04-01", "test_end": "2023-06-30", "label": "W4"},
    {"train_start": "2023-01-01", "train_end": "2023-06-30", "test_start": "2023-07-01", "test_end": "2023-09-30", "label": "W5"},
    {"train_start": "2023-04-01", "train_end": "2023-09-30", "test_start": "2023-10-01", "test_end": "2023-12-31", "label": "W6"},
    {"train_start": "2023-07-01", "train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-03-31", "label": "W7"},
    {"train_start": "2023-10-01", "train_end": "2024-03-31", "test_start": "2024-04-01", "test_end": "2024-06-30", "label": "W8"},
    {"train_start": "2024-01-01", "train_end": "2024-06-30", "test_start": "2024-07-01", "test_end": "2024-09-30", "label": "W9"},
    {"train_start": "2024-04-01", "train_end": "2024-09-30", "test_start": "2024-10-01", "test_end": "2024-12-31", "label": "W10"},
]

PARAM_GRID = {
    "stop_loss_pct": [0.3, 0.5, 1.0],
    "reward_risk":   [2.0, 2.5, 4.0],
    "volume_mult":   [1.0, 1.2],
}


def _run_window(args_tuple):
    """Worker: optimize on train, evaluate on test for one (symbol, window)."""
    symbol, window, base_cfg_path = args_tuple

    import yaml as _yaml
    from src.engine import TradingEngine

    with open(base_cfg_path) as f:
        cfg = _yaml.safe_load(f)
    cfg["mode"] = "backtest"
    cfg.setdefault("account", {})["balance"] = 500.0

    try:
        engine = TradingEngine(cfg)
    except Exception as exc:
        return {"error": str(exc)}

    # Fetch bars for both train and test periods
    try:
        train_bars = engine.fetcher.fetch_historical(
            symbol, window["train_start"], window["train_end"], filter_windows=False
        )
        test_bars = engine.fetcher.fetch_historical(
            symbol, window["test_start"], window["test_end"], filter_windows=False
        )
    except Exception:
        return {"symbol": symbol, "window": window["label"], "error": "data fetch failed"}

    if train_bars is None or train_bars.empty or test_bars is None or test_bars.empty:
        return {"symbol": symbol, "window": window["label"], "error": "no data"}

    # Phase 1: find best params on TRAIN period
    best_pf = -1
    best_params = None
    for sl, rr, vol in itertools.product(
        PARAM_GRID["stop_loss_pct"], PARAM_GRID["reward_risk"], PARAM_GRID["volume_mult"]
    ):
        m = engine.run_backtest_config(
            config_override={
                "active_strategy": "orb", "stop_loss_pct": sl,
                "reward_risk": rr, "volume_mult": vol,
                "regime_filter": True, "vix_threshold": 30,
                "max_trades_per_day": 1, "afternoon_entries": False,
            },
            start_date=window["train_start"], end_date=window["train_end"],
            symbol=symbol, cached_bars=train_bars,
        )
        pf = m["profit_factor"]
        if pf == float("inf"): pf = 999
        if pf > best_pf and m["total_trades"] >= 5:
            best_pf = pf
            best_params = {"sl": sl, "rr": rr, "vol": vol}

    if best_params is None:
        return {"symbol": symbol, "window": window["label"],
                "train_pf": 0, "test_pf": 0, "error": "no valid train params"}

    # Phase 2: evaluate best params on TEST period (out-of-sample)
    test_m = engine.run_backtest_config(
        config_override={
            "active_strategy": "orb",
            "stop_loss_pct": best_params["sl"],
            "reward_risk": best_params["rr"],
            "volume_mult": best_params["vol"],
            "regime_filter": True, "vix_threshold": 30,
            "max_trades_per_day": 1, "afternoon_entries": False,
        },
        start_date=window["test_start"], end_date=window["test_end"],
        symbol=symbol, cached_bars=test_bars,
    )
    test_pf = test_m["profit_factor"]
    if test_pf == float("inf"): test_pf = 999

    return {
        "symbol": symbol, "window": window["label"],
        "train_start": window["train_start"], "train_end": window["train_end"],
        "test_start": window["test_start"], "test_end": window["test_end"],
        "best_sl": best_params["sl"], "best_rr": best_params["rr"],
        "best_vol": best_params["vol"],
        "train_pf": round(best_pf, 3),
        "train_trades": 0,  # not tracked for speed
        "test_pf": round(test_pf, 3),
        "test_trades": test_m["total_trades"],
        "test_wr": round(test_m["win_rate"], 1),
        "test_pnl": round(test_m["total_pnl"], 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--symbols", nargs="+", default=["NVDA", "AMD", "AAPL", "SPY"])
    args = parser.parse_args()

    n_workers = args.workers or max(1, mp.cpu_count() - 1)
    work_units = [(sym, w, str(_ROOT / "config" / "config.yaml"))
                  for sym in args.symbols for w in WINDOWS]

    print(f"\nS1: WALK-FORWARD OPTIMIZATION")
    print(f"  Symbols: {args.symbols}")
    print(f"  Windows: {len(WINDOWS)} (train 6 months → test 3 months)")
    print(f"  Params per window: {len(list(itertools.product(*PARAM_GRID.values())))}")
    print(f"  Workers: {n_workers}")
    print(f"  Total work units: {len(work_units)}\n")

    t0 = _time_mod.time()
    results = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        futures = {ex.submit(_run_window, wu): wu for wu in work_units}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            if "error" not in r:
                print(f"  {r['symbol']:<5} {r['window']:<4} "
                      f"train PF={r['train_pf']:.2f} → test PF={r['test_pf']:.2f}  "
                      f"(sl={r['best_sl']} rr={r['best_rr']})  "
                      f"test: {r['test_trades']} trades, WR={r['test_wr']:.0f}%, "
                      f"P&L=${r['test_pnl']:+.2f}")

    elapsed = _time_mod.time() - t0
    print(f"\n  Done in {elapsed:.0f}s ({elapsed/60:.1f}m)\n")

    # Summary per symbol
    for sym in args.symbols:
        sym_results = [r for r in results if r.get("symbol") == sym and "error" not in r]
        if not sym_results:
            print(f"  {sym}: no valid results")
            continue
        test_pfs = [r["test_pf"] for r in sym_results]
        train_pfs = [r["train_pf"] for r in sym_results]
        wins = sum(1 for pf in test_pfs if pf > 1.0)
        import numpy as np
        print(f"  {sym}: {len(sym_results)} windows")
        print(f"    Train PF: avg={np.mean(train_pfs):.2f}")
        print(f"    Test PF:  avg={np.mean(test_pfs):.2f}  "
              f"median={np.median(test_pfs):.2f}  "
              f"wins={wins}/{len(test_pfs)}")
        if wins >= len(test_pfs) * 0.6:
            print(f"    ✅ PASSES walk-forward: profitable in {wins}/{len(test_pfs)} windows")
        else:
            print(f"    ❌ FAILS walk-forward: only profitable in {wins}/{len(test_pfs)} windows")


if __name__ == "__main__":
    main()
