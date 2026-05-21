#!/usr/bin/env python3
"""
R:R Comparison Analysis — reads results.db and prints performance by R:R ratio
for each strategy/symbol combination.

Usage:
    python backtest/rr_analysis.py
    python backtest/rr_analysis.py --strategy ema_crossover --symbol AAPL
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.logger.results_store import ResultsStore

# Colors for terminal output
_GRN = "\033[92m"
_RED = "\033[91m"
_AMB = "\033[93m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RST = "\033[0m"

# R:R values in the comparison grid
RR_VALUES = [2.0, 2.5, 3.0, 4.0, 5.0]

# Break-even win rate for each R:R
# Break-even = 1 / (1 + R:R)  (simplified: if you risk $1 to make $R, you need
# to win 1/(1+R) of the time to break even)
BREAK_EVEN_WR = {rr: 100.0 / (1.0 + rr) for rr in RR_VALUES}


def load_rr_data(strategy_filter=None, symbol_filter=None):
    """Load runs from results.db, filtered to R:R comparison values."""
    store = ResultsStore()
    df = store.get_all_runs()
    if df.empty:
        return df

    # Filter to R:R comparison strategies and symbols
    rr_strategies = ["ema_crossover", "momentum", "orb"]
    rr_symbols = ["AAPL", "SPY"]

    df = df[df["strategy"].isin(rr_strategies)]
    df = df[df["symbol"].isin(rr_symbols)]
    df = df[df["reward_risk"].isin(RR_VALUES)]

    if strategy_filter:
        df = df[df["strategy"] == strategy_filter]
    if symbol_filter:
        df = df[df["symbol"] == symbol_filter]

    return df


def print_combo_table(df, strategy, symbol, period_label):
    """Print a comparison table for one strategy/symbol/period."""
    sub = df[(df["strategy"] == strategy) &
             (df["symbol"] == symbol) &
             (df["period_label"] == period_label)]

    if sub.empty:
        return

    # For each R:R, average across other hyperparams (sl, vol_mult, regime, etc.)
    # to show the typical performance at that R:R level
    rows = []
    for rr in RR_VALUES:
        rr_df = sub[sub["reward_risk"] == rr]
        if rr_df.empty:
            continue
        # Use the best combo (highest PF) at this R:R level
        best = rr_df.loc[rr_df["profit_factor"].replace([float("inf")], 0).idxmax()]
        be_wr = BREAK_EVEN_WR[rr]
        wr = float(best["win_rate"])
        pf = float(best["profit_factor"])
        pnl = float(best["total_pnl"])
        dd = float(best["max_drawdown"])
        trades = int(best["total_trades"])
        # Highlight: win_rate > break_even AND profit_factor > 1.2
        good = wr > be_wr and pf > 1.2
        rows.append({
            "rr": rr, "trades": trades, "win_rate": wr, "pnl": pnl,
            "pf": pf, "max_dd": dd, "be_wr": be_wr, "good": good,
            "sl": float(best["stop_loss_pct"]),
        })

    if not rows:
        return

    # Header
    strat_name = {"ema_crossover": "EMA Crossover", "momentum": "Momentum",
                  "orb": "ORB"}.get(strategy, strategy)
    print(f"\n{_BOLD}{strat_name} / {symbol} / {period_label}{_RST}")
    print(f"{'R:R':>5}  {'Trades':>6}  {'WinRate':>8}  {'P&L':>8}  "
          f"{'PF':>6}  {'MaxDD':>6}  {'BE-WR':>6}  {'SL%':>5}  Status")
    print(f"{'─'*5}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*8}")

    best_pf = max(r["pf"] for r in rows)

    for r in rows:
        color = _GRN if r["good"] else (_RED if r["pf"] < 1.0 else _DIM)
        sweet = " << SWEET SPOT" if r["pf"] == best_pf and r["good"] else ""
        status = f"{_GRN}PASS{_RST}" if r["good"] else f"{_DIM}---{_RST}"

        print(
            f"{color}{r['rr']:>5.1f}  {r['trades']:>6}  "
            f"{r['win_rate']:>7.1f}%  "
            f"{'+'if r['pnl']>=0 else ''}{r['pnl']:>7.2f}  "
            f"{r['pf']:>6.2f}  "
            f"{r['max_dd']:>5.1f}%  "
            f"{r['be_wr']:>5.1f}%  "
            f"{r['sl']:>5.2f}"
            f"  {status}{sweet}{_RST}"
        )


def print_sweet_spots(df):
    """Print overall sweet-spot summary for each strategy/symbol."""
    strategies = sorted(df["strategy"].unique())
    symbols = sorted(df["symbol"].unique())

    print(f"\n{'='*72}")
    print(f"{_BOLD}SWEET SPOT SUMMARY — Best R:R per Strategy/Symbol{_RST}")
    print(f"{'='*72}")
    print(f"{'Strategy':<16} {'Symbol':<6} {'Best R:R':>8} "
          f"{'PF':>6} {'WR':>6} {'P&L':>8} {'DD':>6} {'Period':<14}")
    print(f"{'─'*16} {'─'*6} {'─'*8} {'─'*6} {'─'*6} {'─'*8} {'─'*6} {'─'*14}")

    for strategy in strategies:
        for symbol in symbols:
            sub = df[(df["strategy"] == strategy) &
                     (df["symbol"] == symbol) &
                     (df["reward_risk"].isin(RR_VALUES)) &
                     (df["total_trades"] >= 10)]
            if sub.empty:
                continue
            # Best PF across all R:R values and periods
            best = sub.loc[sub["profit_factor"].replace([float("inf")], 0).idxmax()]
            strat_name = {"ema_crossover": "EMA Crossover",
                          "momentum": "Momentum",
                          "orb": "ORB"}.get(strategy, strategy)
            pf = float(best["profit_factor"])
            color = _GRN if pf > 1.2 else (_AMB if pf > 1.0 else _RED)
            print(
                f"{color}{strat_name:<16} {symbol:<6} "
                f"{float(best['reward_risk']):>8.1f} "
                f"{pf:>6.2f} "
                f"{float(best['win_rate']):>5.1f}% "
                f"{'+'if best['total_pnl']>=0 else ''}{float(best['total_pnl']):>7.2f} "
                f"{float(best['max_drawdown']):>5.1f}% "
                f"{best['period_label']:<14}{_RST}"
            )


def print_holdout_comparison(df):
    """Print holdout_2022 results separately per R:R for comparison."""
    holdout = df[df["period_label"] == "holdout_2022"]
    if holdout.empty:
        print(f"\n{_AMB}No holdout_2022 results found.{_RST}")
        return

    print(f"\n{'='*72}")
    print(f"{_BOLD}HOLDOUT 2022 (Out-of-Sample) — Performance by R:R{_RST}")
    print(f"{'='*72}")

    strategies = sorted(holdout["strategy"].unique())
    symbols = sorted(holdout["symbol"].unique())

    for strategy in strategies:
        for symbol in symbols:
            print_combo_table(df, strategy, symbol, "holdout_2022")


def main():
    parser = argparse.ArgumentParser(description="R:R Comparison Analysis")
    parser.add_argument("--strategy", default=None,
                        choices=["ema_crossover", "momentum", "orb"])
    parser.add_argument("--symbol", default=None,
                        choices=["AAPL", "SPY"])
    args = parser.parse_args()

    df = load_rr_data(args.strategy, args.symbol)
    if df.empty:
        print("No R:R comparison data found in results.db.")
        print("Run: python backtest/batch_runner.py --mode rr_comparison")
        return

    total = len(df)
    strategies = sorted(df["strategy"].unique())
    symbols = sorted(df["symbol"].unique())
    periods = sorted(df["period_label"].unique())

    print(f"\n{_BOLD}R:R COMPARISON ANALYSIS{_RST}")
    print(f"  Runs loaded  : {total}")
    print(f"  Strategies   : {', '.join(strategies)}")
    print(f"  Symbols      : {', '.join(symbols)}")
    print(f"  Periods      : {', '.join(periods)}")
    print(f"  R:R values   : {', '.join(str(r) for r in RR_VALUES)}")
    print(f"  Break-even WR: " + ", ".join(
        f"{rr}={BREAK_EVEN_WR[rr]:.1f}%" for rr in RR_VALUES
    ))

    # In-sample results (all non-holdout periods)
    in_sample_periods = [p for p in periods if p != "holdout_2022"]
    for strategy in strategies:
        for symbol in symbols:
            for period in in_sample_periods:
                print_combo_table(df, strategy, symbol, period)

    # Holdout results
    print_holdout_comparison(df)

    # Sweet spot summary
    # Use non-holdout periods for sweet spot
    in_sample = df[df["period_label"] != "holdout_2022"]
    if not in_sample.empty:
        print_sweet_spots(in_sample)


if __name__ == "__main__":
    main()
