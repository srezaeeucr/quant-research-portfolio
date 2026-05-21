#!/usr/bin/env python3
"""
ATR vs Fixed Stop comparison backtest.

For each (strategy, symbol, period) combo, runs two backtests using
engine.run_backtest_config() with the same params except for use_atr_stops.
Prints a side-by-side comparison table.

Designed to be run on superpower then rsync'd back.

Usage:
    python backtest/atr_comparison.py            # default scope
    python backtest/atr_comparison.py --quick    # just EMA/AAPL/2022-2024
"""
import os
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import argparse
import sys
import time as _time_mod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

_ROOT = Path(__file__).resolve().parents[1]


SCOPES = {
    "quick": [
        # (strategy, symbol, period_label, start, end, sl_pct, rr)
        ("ema_crossover", "AAPL", "2023-2024", "2023-01-01", "2024-12-31", 0.5, 4.0),
    ],
    "default": [
        # The original prompt's combo
        ("ema_crossover", "AAPL", "2023-2024", "2023-01-01", "2024-12-31", 0.5, 4.0),
        # Holdout 2022 — true out-of-sample
        ("ema_crossover", "AAPL", "holdout_2022", "2022-01-01", "2022-12-31", 0.5, 4.0),
        # The validated 2nd candidate
        ("orb",           "NVDA", "2023-2024", "2023-01-01", "2024-12-31", 0.3, 2.5),
        ("orb",           "NVDA", "holdout_2022", "2022-01-01", "2022-12-31", 0.3, 2.5),
        # Different symbol — does ATR generalize?
        ("ema_crossover", "SPY",  "2023-2024", "2023-01-01", "2024-12-31", 0.5, 4.0),
        ("ema_crossover", "SPY",  "holdout_2022", "2022-01-01", "2022-12-31", 0.5, 4.0),
        # Bear period stress
        ("ema_crossover", "AAPL", "bear_2025",  "2025-06-01", "2026-03-31", 0.5, 4.0),
    ],
}


def parse_args():
    p = argparse.ArgumentParser(description="ATR vs fixed stop comparison")
    p.add_argument("--scope", default="default", choices=list(SCOPES.keys()))
    p.add_argument("--atr-period",     type=int,   default=14)
    p.add_argument("--atr-multiplier", type=float, default=1.5)
    p.add_argument("--floor",          type=float, default=0.3)
    p.add_argument("--ceiling",        type=float, default=1.5)
    return p.parse_args()


def run_one(engine, strategy, symbol, start, end, sl_pct, rr,
            use_atr, atr_period, atr_mult, floor, ceiling, cached_bars=None):
    override = {
        "active_strategy":    strategy,
        "stop_loss_pct":      sl_pct,
        "reward_risk":        rr,
        "volume_mult":        1.0,
        "regime_filter":      True,
        "vix_threshold":      30,
        "max_trades_per_day": 2,
        "afternoon_entries":  False,
        "use_atr_stops":      use_atr,
        "atr_period":         atr_period,
        "atr_multiplier":     atr_mult,
        "stop_floor_pct":     floor,
        "stop_ceiling_pct":   ceiling,
    }
    return engine.run_backtest_config(
        config_override=override,
        start_date=start,
        end_date=end,
        symbol=symbol,
        cached_bars=cached_bars,
    )


def main():
    args = parse_args()
    combos = SCOPES[args.scope]

    with open(_ROOT / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["mode"] = "backtest"

    from src.engine import TradingEngine
    engine = TradingEngine(cfg)

    print(f"\nATR comparison — scope={args.scope}")
    print(f"  atr_period={args.atr_period}  atr_multiplier={args.atr_multiplier}")
    print(f"  floor={args.floor}%  ceiling={args.ceiling}%\n")

    results = []
    bars_cache = {}
    for strategy, symbol, label, start, end, sl, rr in combos:
        print(f"  → {strategy:<14} {symbol:<5} {label:<14} (sl={sl}% rr={rr}) ...")
        cache_key = (symbol, start, end)
        if cache_key not in bars_cache:
            try:
                bars_cache[cache_key] = engine.fetcher.fetch_historical(
                    symbol, start, end, filter_windows=False
                )
            except Exception as exc:
                print(f"     fetch failed: {exc}")
                bars_cache[cache_key] = None
        bars = bars_cache[cache_key]

        t0 = _time_mod.time()
        m_fix = run_one(engine, strategy, symbol, start, end, sl, rr,
                        use_atr=False, atr_period=args.atr_period,
                        atr_mult=args.atr_multiplier, floor=args.floor,
                        ceiling=args.ceiling, cached_bars=bars)
        m_atr = run_one(engine, strategy, symbol, start, end, sl, rr,
                        use_atr=True,  atr_period=args.atr_period,
                        atr_mult=args.atr_multiplier, floor=args.floor,
                        ceiling=args.ceiling, cached_bars=bars)
        elapsed = _time_mod.time() - t0
        print(f"     done in {elapsed:.1f}s "
              f"(fixed: {m_fix['total_trades']} trades, "
              f"atr: {m_atr['total_trades']} trades)")
        results.append((strategy, symbol, label, m_fix, m_atr))

    # ── Comparison report ────────────────────────────────────────────
    def fmt_pf(pf):
        if pf == float("inf"):
            return "  ∞ "
        return f"{pf:>5.2f}"

    print("\n" + "=" * 110)
    print("ATR vs FIXED STOP COMPARISON")
    print("=" * 110)
    header = (f"{'Strategy':<14} {'Sym':<5} {'Period':<12} | "
              f"{'Trades':>13} | {'WR%':>13} | {'PF':>13} | {'P&L':>17} | {'MaxDD%':>13}")
    sep    = (f"{'':14} {'':5} {'':12} | "
              f"{'fixed':>6} {'atr':>6} | {'fixed':>6} {'atr':>6} | "
              f"{'fixed':>6} {'atr':>6} | {'fixed':>8} {'atr':>8} | "
              f"{'fixed':>6} {'atr':>6}")
    print(header)
    print(sep)
    print("-" * 110)
    for strategy, symbol, label, mf, ma in results:
        print(
            f"{strategy:<14} {symbol:<5} {label:<12} | "
            f"{mf['total_trades']:>6} {ma['total_trades']:>6} | "
            f"{mf['win_rate']:>5.1f}% {ma['win_rate']:>5.1f}% | "
            f"{fmt_pf(mf['profit_factor'])} {fmt_pf(ma['profit_factor'])} | "
            f"{mf['total_pnl']:>+8.2f} {ma['total_pnl']:>+8.2f} | "
            f"{mf['max_drawdown']:>5.1f}% {ma['max_drawdown']:>5.1f}%"
        )

    # ── Aggregate verdict ─────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("AGGREGATE — ATR vs Fixed deltas")
    print("=" * 110)
    pf_wins = pnl_wins = dd_wins = 0
    pf_diffs = []
    pnl_diffs = []
    dd_diffs = []
    for _, _, _, mf, ma in results:
        pf_diff = ma["profit_factor"] - mf["profit_factor"]
        pnl_diff = ma["total_pnl"] - mf["total_pnl"]
        dd_diff = ma["max_drawdown"] - mf["max_drawdown"]
        if ma["profit_factor"] > mf["profit_factor"]:
            pf_wins += 1
        if ma["total_pnl"] > mf["total_pnl"]:
            pnl_wins += 1
        if ma["max_drawdown"] < mf["max_drawdown"]:
            dd_wins += 1
        # Avoid inf in diffs
        if ma["profit_factor"] != float("inf") and mf["profit_factor"] != float("inf"):
            pf_diffs.append(pf_diff)
        pnl_diffs.append(pnl_diff)
        dd_diffs.append(dd_diff)

    n = len(results)
    print(f"  PF:    ATR wins {pf_wins}/{n}   "
          f"avg ΔPF = {sum(pf_diffs)/max(len(pf_diffs),1):+.3f}")
    print(f"  P&L:   ATR wins {pnl_wins}/{n}   "
          f"avg ΔP&L = {sum(pnl_diffs)/n:+.2f}")
    print(f"  DD:    ATR wins {dd_wins}/{n}   "
          f"avg ΔDD = {sum(dd_diffs)/n:+.2f}%")


if __name__ == "__main__":
    main()
