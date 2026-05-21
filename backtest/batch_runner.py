#!/usr/bin/env python3
"""
Batch backtest runner — pre-computes a curated parameter grid and stores
every result in logs/results.db via ResultsStore.

Usage:
    # Dry run — show all combinations without running
    python backtest/batch_runner.py --dry-run

    # Subset by strategy / symbol / period
    python backtest/batch_runner.py --strategy orb --symbol SPY --period bull_2023

    # Full grid (long — use sparingly)
    python backtest/batch_runner.py
"""
import os
# Must be set BEFORE any imports that touch Objective-C runtime (pandas, etc.)
# Otherwise fork-based multiprocessing crashes on macOS.
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import argparse
import itertools
import logging
import multiprocessing as mp
import sys
import time as _time_mod
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

_ROOT = Path(__file__).resolve().parents[1]

# ── Curated search grid (legacy, kept for backward compat) ────────────────
CURATED_GRID = {
    "strategies": ["orb", "vwap_reversion", "gap_fill", "ema_crossover", "momentum"],
    "symbols":    ["SPY", "QQQ", "IWM", "AAPL", "TSLA"],
    "periods": [
        {"label": "bull_2023", "start": "2023-01-01", "end": "2023-12-31"},
        {"label": "bull_2024", "start": "2024-01-01", "end": "2024-12-31"},
        {"label": "bear_2025", "start": "2025-06-01", "end": "2026-03-31"},
        {"label": "full_2yr",  "start": "2023-01-01", "end": "2025-01-01"},
    ],
    "stop_loss_pct":      [0.3, 0.5, 0.75],
    "reward_risk":        [1.5, 2.0, 2.5],
    "volume_mult":        [1.0, 1.2],
    "regime_filter":      [True, False],
    "vix_threshold":      [25, 30],
    "max_trades_per_day": [1, 2],
}

# ── Symbol groups for filtering the expanded grid ──────────────────────────
SYMBOL_GROUPS = {
    "original": ["SPY", "QQQ", "IWM", "AAPL", "TSLA"],
    "sp500":    ["MSFT", "NVDA", "AMZN", "META", "GOOGL", "JPM", "UNH"],
    "etfs":     ["XLK", "XLF", "XLE", "XLV", "XLI"],
    "volatile": ["AMD", "COIN", "MSTR"],
}
SYMBOL_GROUPS["all"] = (
    SYMBOL_GROUPS["original"]
    + SYMBOL_GROUPS["sp500"]
    + SYMBOL_GROUPS["etfs"]
    + SYMBOL_GROUPS["volatile"]
)

# ── Expanded search grid (20 symbols + afternoon_entries dimension) ────────
EXPANDED_GRID = {
    "strategies": ["orb", "vwap_reversion", "gap_fill", "ema_crossover", "momentum"],
    "symbols":    SYMBOL_GROUPS["all"],
    "periods": [
        {"label": "bull_2023", "start": "2023-01-01", "end": "2023-12-31"},
        {"label": "bull_2024", "start": "2024-01-01", "end": "2024-12-31"},
        {"label": "bear_2025", "start": "2025-06-01", "end": "2026-03-31"},
        {"label": "full_2yr",  "start": "2023-01-01", "end": "2025-01-01"},
    ],
    "stop_loss_pct":      [0.3, 0.5, 0.75],
    "reward_risk":        [2.0, 2.5, 3.0, 4.0, 5.0],
    "volume_mult":        [1.0, 1.2],
    "regime_filter":      [True, False],
    "vix_threshold":      [25, 30],
    "max_trades_per_day": [1, 2],
    "afternoon_entries":  [False, True],
}


# ── Focused R:R comparison grid ────────────────────────────────────────────
RR_COMPARISON_GRID = {
    "strategies": ["ema_crossover", "momentum", "orb"],
    "symbols":    ["AAPL", "SPY"],
    "periods": [
        {"label": "bull_2023",    "start": "2023-01-01", "end": "2023-12-31"},
        {"label": "bull_2024",    "start": "2024-01-01", "end": "2024-12-31"},
        {"label": "bear_2025",    "start": "2025-06-01", "end": "2026-03-31"},
        {"label": "holdout_2022", "start": "2022-01-01", "end": "2022-12-31"},
    ],
    "stop_loss_pct":      [0.3, 0.5, 0.75],
    "reward_risk":        [2.0, 2.5, 3.0, 4.0, 5.0],   # expanded R:R range
    "volume_mult":        [1.0, 1.2],
    "regime_filter":      [True, False],
    "vix_threshold":      [25, 30],
    "max_trades_per_day": [1, 2],
}


def _generate_combinations(grid: dict, strategy_filter: str = None,
                            symbol_filter: str = None,
                            period_filter: str = None,
                            symbol_list: Optional[list] = None) -> list:
    """Return a list of parameter combo dicts, optionally filtered."""
    strategies = [s for s in grid["strategies"]
                  if strategy_filter is None or s == strategy_filter]
    if symbol_list is not None:
        symbols = [s for s in symbol_list
                   if symbol_filter is None or s == symbol_filter]
    else:
        symbols = [s for s in grid["symbols"]
                   if symbol_filter is None or s == symbol_filter]
    periods    = [p for p in grid["periods"]
                  if period_filter is None or p["label"] == period_filter]

    combos = []
    # Hyperparameter keys — afternoon_entries only included if grid has it
    hyper_keys = ["stop_loss_pct", "reward_risk", "volume_mult",
                  "regime_filter", "vix_threshold", "max_trades_per_day"]
    if "afternoon_entries" in grid:
        hyper_keys.append("afternoon_entries")
    hyper_vals = [grid[k] for k in hyper_keys]

    for strategy, symbol, period in itertools.product(strategies, symbols, periods):
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


def _format_combo(idx: int, total: int, p: dict) -> str:
    regime = "regime=ON" if p["regime_filter"] else "regime=OFF"
    pm = "pm=ON " if p.get("afternoon_entries") else "pm=OFF"
    return (
        f"[{idx:>5}/{total}]  "
        f"{p['strategy']:<16} {p['symbol']:<5}  "
        f"{p['period_label']:<11}  "
        f"sl={p['stop_loss_pct']:.2f}%  "
        f"rr={p['reward_risk']:.1f}  "
        f"vol={p['volume_mult']:.1f}  "
        f"{regime}  "
        f"vix={p['vix_threshold']}  "
        f"max_t={p['max_trades_per_day']}  "
        f"{pm}"
    )


def _run_symbol_period_unit(
    args_tuple: Tuple[str, str, list, str],
) -> dict:
    """Worker function: run all combos for one (symbol, period) pair.

    Designed to be picklable so it can be dispatched to ProcessPoolExecutor.
    Each worker:
      1. Loads config and instantiates its own TradingEngine + ResultsStore
      2. Fetches bars + SMA + VIX once for the (symbol, period)
      3. Iterates all combos for the pair, skipping ones already in DB
      4. Returns aggregate stats for parent progress reporting

    Returns
    -------
    dict with keys: symbol, period, done, skipped, errors, no_data, elapsed
    """
    symbol, period_label, combos_for_pair, base_cfg_path = args_tuple

    # Re-import inside worker (necessary for fork-safe multiprocessing)
    import yaml as _yaml
    from src.engine import TradingEngine
    from src.logger.results_store import ResultsStore, make_run_id

    t0 = _time_mod.time()
    result = {
        "symbol":  symbol,
        "period":  period_label,
        "done":    0,
        "skipped": 0,
        "errors":  0,
        "no_data": False,
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
        result["error_msg"] = f"engine init failed: {exc}"
        return result

    store = ResultsStore()

    # Pull start/end from the first combo (all combos in this group share them)
    start_date = combos_for_pair[0]["start_date"]
    end_date   = combos_for_pair[0]["end_date"]

    # Fetch bars once for this (symbol, period)
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

    # Pre-compute SMA + VIX once for this (symbol, period)
    try:
        cached_sma = engine._compute_daily_sma(symbol, start_date, end_date, sma_days)
    except Exception:
        cached_sma = {}
    try:
        cached_vix = engine._fetch_vix_data(start_date, end_date)
    except Exception:
        cached_vix = {}

    # Iterate combos
    for params in combos_for_pair:
        run_id = make_run_id(params)
        if store.run_exists(run_id):
            result["skipped"] += 1
            continue
        try:
            config_override = {
                "active_strategy":    params["strategy"],
                "stop_loss_pct":      params["stop_loss_pct"],
                "reward_risk":        params["reward_risk"],
                "volume_mult":        params["volume_mult"],
                "regime_filter":      params["regime_filter"],
                "vix_threshold":      params["vix_threshold"],
                "max_trades_per_day": params["max_trades_per_day"],
                "afternoon_entries":  bool(params.get("afternoon_entries", False)),
            }
            metrics = engine.run_backtest_config(
                config_override=config_override,
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
    p = argparse.ArgumentParser(description="Batch backtest runner")
    p.add_argument("--strategy", default=None,
                   help="Run only this strategy")
    p.add_argument("--symbol",   default=None,
                   help="Run only this symbol")
    p.add_argument("--period",   default=None,
                   help="Run only this time period")
    p.add_argument("--dry-run",  action="store_true",
                   help="Print combinations without running backtests")
    p.add_argument("--mode",     default="expanded",
                   choices=["full", "rr_comparison", "expanded"],
                   help="Grid: expanded (default, 20 symbols + PM dim), "
                        "full (legacy 14,400), rr_comparison (5,760)")
    p.add_argument("--symbol_group", default=None,
                   choices=list(SYMBOL_GROUPS.keys()),
                   help="Symbol group to run: original | sp500 | etfs | volatile | all")
    p.add_argument("--workers", type=int, default=None,
                   help="Parallel worker processes (default: cpu_count-1; "
                        "set to 1 for serial mode)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "rr_comparison":
        grid = RR_COMPARISON_GRID
    elif args.mode == "full":
        grid = CURATED_GRID
    else:
        grid = EXPANDED_GRID

    # Resolve symbol list from --symbol_group if provided
    symbol_list = SYMBOL_GROUPS[args.symbol_group] if args.symbol_group else None

    combos = _generate_combinations(
        grid,
        strategy_filter=args.strategy,
        symbol_filter=args.symbol,
        period_filter=args.period,
        symbol_list=symbol_list,
    )
    total = len(combos)

    if total == 0:
        print("No combinations matched the filters.")
        return

    # ── Dry run ────────────────────────────────────────────────────────
    if args.dry_run:
        for i, p in enumerate(combos, 1):
            print(_format_combo(i, total, p))
        print(f"\n  Total combinations: {total:,}")
        est_sec = total * 2
        print(f"  Estimated runtime : ~{est_sec//60}m {est_sec%60}s "
              f"(assuming 2s per combo, varies by data availability)")
        return

    # ── Real run ───────────────────────────────────────────────────────
    n_workers = args.workers if args.workers is not None else max(1, mp.cpu_count() - 1)

    print(f"\n  Batch runner  —  {total:,} combinations")
    print(f"  Results DB    :  logs/results.db")
    print(f"  Filters       :  strategy={args.strategy or 'all'}  "
          f"symbol={args.symbol or 'all'}  "
          f"period={args.period or 'all'}")
    print(f"  Workers       :  {n_workers} (cpu_count={mp.cpu_count()})")
    est = total * 2 // max(n_workers, 1)
    print(f"  Est. runtime  :  ~{est//60}m {est%60}s\n")

    # ── Parallel path: dispatch (symbol, period) groups to a worker pool ──
    if n_workers > 1:
        # Group combos by (symbol, period) — one work unit per group
        groups: dict = {}
        for params in combos:
            key = (params["symbol"], params["period_label"])
            groups.setdefault(key, []).append(params)

        base_cfg_path = str(_ROOT / "config" / "config.yaml")
        work_units = [
            (sym, per, combo_list, base_cfg_path)
            for (sym, per), combo_list in groups.items()
        ]
        n_units = len(work_units)
        print(f"  Work units    :  {n_units} (symbol × period pairs)")
        print(f"  Avg per unit  :  {total // max(n_units, 1)} combos\n")

        done = skipped = errors = 0
        completed_units = 0
        t0 = _time_mod.time()

        # 'spawn' is the only safe context on macOS — fork crashes when
        # parent has loaded numpy/pandas/Objective-C frameworks. Workers
        # take ~10–15s to cold-start because they re-import the module.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=ctx
        ) as executor:
            futures = {
                executor.submit(_run_symbol_period_unit, wu): wu
                for wu in work_units
            }
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                except Exception as exc:
                    print(f"  WORKER FAILED: {exc}")
                    continue

                done    += res["done"]
                skipped += res["skipped"]
                errors  += res["errors"]
                completed_units += 1

                elapsed = _time_mod.time() - t0
                pct = completed_units / n_units * 100
                eta_s = (elapsed / completed_units * (n_units - completed_units)
                         if completed_units > 0 else 0)
                no_data_tag = " (NO DATA)" if res.get("no_data") else ""
                err_tag = (f" err={res['errors']}" if res["errors"] else "")

                print(
                    f"  [{completed_units:>4}/{n_units}] "
                    f"{res['symbol']:<5} {res['period']:<11} "
                    f"done={res['done']:>4} skip={res['skipped']:>4}{err_tag}"
                    f"  ({res['elapsed']:.0f}s){no_data_tag}"
                    f"  ETA {int(eta_s)//60}m{int(eta_s)%60:02d}s"
                )

        elapsed = _time_mod.time() - t0
        print(f"\n  Done.  ran={done}  skipped={skipped}  errors={errors}"
              f"  elapsed={elapsed:.1f}s  ({elapsed/60:.1f}m)")
        return

    # ── Serial path (n_workers == 1, original behavior) ───────────────
    with open(_ROOT / "config" / "config.yaml") as f:
        base_cfg = yaml.safe_load(f)
    base_cfg["mode"] = "backtest"

    from src.engine import TradingEngine
    from src.logger.results_store import ResultsStore, make_run_id

    store  = ResultsStore()
    engine = TradingEngine(base_cfg)

    # Cache bars, SMA, and VIX to avoid re-fetching for each combo
    # bars + SMA keyed by (symbol, period); VIX keyed by period_label only
    bar_cache: dict = {}
    sma_cache: dict = {}
    vix_cache: dict = {}

    sma_days = base_cfg.get("filters", base_cfg.get("strategy", {})).get("regime_sma_days", 20)

    done = skipped = errors = 0
    t0 = _time_mod.time()

    for idx, params in enumerate(combos, 1):
        run_id = make_run_id(params)

        if store.run_exists(run_id):
            skipped += 1
            continue

        # Fetch bars once per (symbol, period)
        cache_key   = (params["symbol"], params["period_label"])
        period_key  = params["period_label"]
        if cache_key not in bar_cache:
            try:
                bars = engine.fetcher.fetch_historical(
                    params["symbol"],
                    params["start_date"],
                    params["end_date"],
                    filter_windows=False,
                )
            except Exception as exc:
                logger.warning("Fetch failed %s/%s: %s",
                               params["symbol"], params["period_label"], exc)
                bars = __import__("pandas").DataFrame()
            bar_cache[cache_key] = bars

            # Pre-compute SMA once per (symbol, period)
            try:
                sma_cache[cache_key] = engine._compute_daily_sma(
                    params["symbol"],
                    params["start_date"],
                    params["end_date"],
                    sma_days,
                )
            except Exception as exc:
                logger.warning("SMA compute failed %s/%s: %s",
                               params["symbol"], params["period_label"], exc)
                sma_cache[cache_key] = {}

        # Pre-compute VIX once per period (VIX is market-wide, not symbol-specific)
        if period_key not in vix_cache:
            try:
                vix_cache[period_key] = engine._fetch_vix_data(
                    params["start_date"], params["end_date"]
                )
            except Exception as exc:
                logger.warning("VIX fetch failed %s: %s", period_key, exc)
                vix_cache[period_key] = {}

        cached     = bar_cache[cache_key]
        cached_sma = sma_cache.get(cache_key, {})
        cached_vix = vix_cache.get(period_key, {})

        # Data availability check — skip if no bars (e.g. COIN/MSTR pre-IPO)
        if cached is None or cached.empty:
            skipped += 1
            if skipped % 50 == 1:
                logger.warning(
                    "No data for %s/%s — skipping",
                    params["symbol"], params["period_label"],
                )
            continue

        try:
            config_override = {
                "active_strategy":    params["strategy"],
                "stop_loss_pct":      params["stop_loss_pct"],
                "reward_risk":        params["reward_risk"],
                "volume_mult":        params["volume_mult"],
                "regime_filter":      params["regime_filter"],
                "vix_threshold":      params["vix_threshold"],
                "max_trades_per_day": params["max_trades_per_day"],
                "afternoon_entries":  bool(params.get("afternoon_entries", False)),
            }
            metrics = engine.run_backtest_config(
                config_override=config_override,
                start_date=params["start_date"],
                end_date=params["end_date"],
                symbol=params["symbol"],
                cached_bars=cached,
                cached_sma=cached_sma,
                cached_vix=cached_vix,
            )
        except Exception as exc:
            errors += 1
            print(f"  ERROR {_format_combo(idx, total, params)}  →  {exc}")
            continue

        trades = metrics.pop("trades_list", [])
        run_config = dict(params)
        run_config["run_id"] = run_id
        store.save_run(run_config, metrics, trades)
        done += 1

        wr  = metrics["win_rate"]
        pnl = metrics["total_pnl"]
        nt  = metrics["total_trades"]
        elapsed = _time_mod.time() - t0
        rate    = done / elapsed if elapsed > 0 else 0
        eta_s   = int((total - idx) / rate) if rate > 0 else 0

        print(
            f"  {_format_combo(idx, total, params)}"
            f"  →  {nt} trades  {wr:.1f}% WR  "
            f"{'+'if pnl>=0 else ''}${pnl:.2f}"
            f"  (ETA {eta_s//60}m{eta_s%60:02d}s)"
        )

    elapsed = _time_mod.time() - t0
    print(f"\n  Done.  ran={done}  skipped={skipped}  errors={errors}"
          f"  elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
