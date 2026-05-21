#!/usr/bin/env python3
"""Run a grid of strategy combos on today's data to see what would have happened."""
import os
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import itertools
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.engine import TradingEngine

TODAY = "2026-04-10"
_ROOT = Path(__file__).resolve().parents[1]

GRID = {
    "strategies": ["orb", "ema_crossover", "momentum", "vwap_reversion", "gap_fill"],
    "symbols": ["NVDA", "AMD"],
    "stop_loss_pct": [0.3, 0.5, 1.0],
    "reward_risk": [2.0, 2.5, 3.0, 4.0, 5.0],
    "volume_mult": [1.0, 1.2],
    "entry_window_minutes": [30, 180, 360],
}


def _build_config(strategy, symbol, sl, rr, vol, ew):
    return {
        "mode": "backtest",
        "symbols": [symbol],
        "active_strategy": strategy,
        "account": {"balance": 500.0, "max_position_pct": 0.95},
        "risk": {
            "max_risk_per_trade_pct": 1.0,
            "daily_loss_limit_pct": 2.0,
            "reward_risk_ratio": rr,
            "fractional_shares": True,
            "stop_loss_pct": sl,
            "use_atr_stops": False,
        },
        "filters": {
            "max_trades_per_day": 3,
            "regime_filter": True,
            "regime_sma_days": 20,
            "vix_threshold": 30,
        },
        "strategy_params": {
            "orb": {
                "min_breakout_pct": 0.1,
                "or_duration_min": 15,
                "volume_multiplier": vol,
                "afternoon_entries": True,
                "entry_window_minutes": ew,
            },
            "ema_crossover": {
                "ema_fast": 9, "ema_slow": 21,
                "volume_multiplier": vol,
                "afternoon_entry_enabled": ew > 30,
            },
            "momentum": {
                "momentum_threshold": 0.2,
                "or_duration_min": 15,
                "volume_multiplier": vol,
                "afternoon_entries": ew > 30,
            },
            "vwap_reversion": {
                "deviation_threshold": 0.001,
                "bounce_bars": 1,
                "volume_multiplier": vol,
                "afternoon_entries": ew > 30,
            },
            "gap_fill": {
                "min_gap_pct": 0.2,
                "max_gap_pct": 3.0,
            },
        },
        "strategy": {
            "active_strategy": strategy,
            "volume_multiplier": vol,
            "use_regime_filter": True,
            "regime_sma_days": 20,
            "min_breakout_pct": 0.1,
            "min_gap_pct": 0.2,
            "momentum_threshold_pct": 0.2,
            "vwap_dip_pct": 0.2,
            "ema_fast_period": 9,
            "ema_slow_period": 21,
        },
        "trading_windows": [
            {"start": "09:30", "end": "16:00", "timezone": "America/New_York"},
        ],
    }


def _run_one(args):
    strategy, symbol, sl, rr, vol, ew = args
    cfg = _build_config(strategy, symbol, sl, rr, vol, ew)
    try:
        engine = TradingEngine(cfg)
        metrics = engine.run_backtest_config(
            config_override={
                "active_strategy": strategy,
                "stop_loss_pct": sl,
                "reward_risk": rr,
                "volume_mult": vol,
                "regime_filter": True,
                "vix_threshold": 30,
                "max_trades_per_day": 3,
                "afternoon_entries": ew > 30,
                "entry_window_minutes": ew,
            },
            start_date=TODAY,
            end_date=TODAY,
            symbol=symbol,
        )
        return {
            "strategy": strategy,
            "symbol": symbol,
            "sl": sl,
            "rr": rr,
            "vol": vol,
            "ew": ew,
            "trades": metrics.get("total_trades", 0),
            "wins": metrics.get("wins", 0),
            "pnl": metrics.get("total_pnl", 0),
            "wr": metrics.get("win_rate", 0),
        }
    except Exception as e:
        return {
            "strategy": strategy, "symbol": symbol,
            "sl": sl, "rr": rr, "vol": vol, "ew": ew,
            "trades": 0, "wins": 0, "pnl": 0, "wr": 0,
            "error": str(e),
        }


def main():
    combos = list(itertools.product(
        GRID["strategies"], GRID["symbols"],
        GRID["stop_loss_pct"], GRID["reward_risk"],
        GRID["volume_mult"], GRID["entry_window_minutes"],
    ))
    print(f"Running {len(combos)} combos on {TODAY}...")

    results = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_run_one, c): c for c in combos}
        done = 0
        for fut in as_completed(futures):
            done += 1
            r = fut.result()
            results.append(r)
            if done % 100 == 0:
                print(f"  {done}/{len(combos)}")

    # Sort by P&L descending
    results.sort(key=lambda x: x["pnl"], reverse=True)

    # Summary
    traded = [r for r in results if r["trades"] > 0]
    winners = [r for r in traded if r["pnl"] > 0]
    losers = [r for r in traded if r["pnl"] < 0]
    no_trade = [r for r in results if r["trades"] == 0]

    print(f"\n{'='*80}")
    print(f"  TODAY'S GRID RESULTS — {TODAY}")
    print(f"{'='*80}")
    print(f"  Total combos : {len(results)}")
    print(f"  Traded       : {len(traded)}  ({len(traded)/len(results)*100:.0f}%)")
    print(f"  Winners      : {len(winners)}")
    print(f"  Losers       : {len(losers)}")
    print(f"  No signal    : {len(no_trade)}")

    if winners:
        print(f"\n  TOP 20 WINNERS:")
        print(f"  {'Strategy':<16} {'Sym':<6} {'SL%':<6} {'RR':<5} {'Vol':<5} {'EW':<5} {'Trades':<7} {'WR%':<6} {'P&L':>8}")
        print(f"  {'-'*72}")
        for r in results[:20]:
            if r["pnl"] > 0:
                print(f"  {r['strategy']:<16} {r['symbol']:<6} {r['sl']:<6} {r['rr']:<5} {r['vol']:<5} {r['ew']:<5} {r['trades']:<7} {r['wr']:<6.1f} ${r['pnl']:>+7.2f}")

    if losers:
        print(f"\n  WORST 10 LOSERS:")
        print(f"  {'Strategy':<16} {'Sym':<6} {'SL%':<6} {'RR':<5} {'Vol':<5} {'EW':<5} {'Trades':<7} {'WR%':<6} {'P&L':>8}")
        print(f"  {'-'*72}")
        for r in results[-10:]:
            print(f"  {r['strategy']:<16} {r['symbol']:<6} {r['sl']:<6} {r['rr']:<5} {r['vol']:<5} {r['ew']:<5} {r['trades']:<7} {r['wr']:<6.1f} ${r['pnl']:>+7.2f}")

    # Per-strategy summary
    print(f"\n  PER-STRATEGY SUMMARY:")
    print(f"  {'Strategy':<16} {'Combos':<8} {'Traded':<8} {'Avg P&L':>9} {'Best P&L':>10} {'Worst P&L':>10}")
    print(f"  {'-'*62}")
    for strat in GRID["strategies"]:
        sr = [r for r in results if r["strategy"] == strat]
        st = [r for r in sr if r["trades"] > 0]
        if st:
            avg_pnl = sum(r["pnl"] for r in st) / len(st)
            best = max(r["pnl"] for r in st)
            worst = min(r["pnl"] for r in st)
            print(f"  {strat:<16} {len(sr):<8} {len(st):<8} ${avg_pnl:>+8.2f} ${best:>+9.2f} ${worst:>+9.2f}")
        else:
            print(f"  {strat:<16} {len(sr):<8} {'0':<8} {'—':>9} {'—':>10} {'—':>10}")

    # Per-symbol summary
    print(f"\n  PER-SYMBOL SUMMARY:")
    for sym in GRID["symbols"]:
        sr = [r for r in traded if r["symbol"] == sym]
        if sr:
            avg = sum(r["pnl"] for r in sr) / len(sr)
            best = max(sr, key=lambda x: x["pnl"])
            print(f"  {sym}: {len(sr)} traded, avg P&L ${avg:+.2f}, "
                  f"best ${best['pnl']:+.2f} ({best['strategy']}/{best['sl']}/{best['rr']}/{best['ew']})")


if __name__ == "__main__":
    main()
