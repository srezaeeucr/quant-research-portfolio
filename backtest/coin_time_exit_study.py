#!/usr/bin/env python3
"""
COIN Time-Exit Study — Does exiting sooner beat hold-to-EOD?

For each COIN trade in the cached backtest history, look up the bar at
entry_time + N minutes and compute hypothetical PnL if we had exited there.
Compare to the actual exit (stop_loss / target_hit / eod_exit).

Tests hold limits: 30, 60, 90, 120 minutes, and a 1%/0.5% trailing stop.
"""
import os, sys, itertools, pickle, json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import yaml
import pandas as pd
import pytz
from src.engine import TradingEngine

_ET = pytz.timezone("America/New_York")
_ROOT = Path(__file__).resolve().parents[1]
CACHE = _ROOT / "logs" / "research" / "bar_cache"


def load_cached(sym, start, end):
    pq = CACHE / f"{sym}_{start}_{end}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    p = CACHE / f"{sym}_{start}_{end}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def load_sma(sym):
    jf = CACHE / f"{sym}_sma.json"
    if jf.exists():
        from datetime import date
        data = json.load(open(jf))
        return {date.fromisoformat(k): v for k, v in data.items()}
    p = CACHE / f"{sym}_sma.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return {}


def simulate_alt_exits(trade, bars_by_day):
    """Given a trade, compute hypothetical PnL under alternative exit rules."""
    entry_dt = pd.to_datetime(trade["entry_time"])
    if entry_dt.tz is None:
        entry_dt = entry_dt.tz_localize("UTC").tz_convert(_ET)
    else:
        entry_dt = entry_dt.tz_convert(_ET)
    exit_dt = pd.to_datetime(trade["exit_time"])
    if exit_dt.tz is None:
        exit_dt = exit_dt.tz_localize("UTC").tz_convert(_ET)
    else:
        exit_dt = exit_dt.tz_convert(_ET)

    day = entry_dt.date()
    day_bars = bars_by_day.get(day)
    if day_bars is None:
        return {}

    entry_px = float(trade["entry_price"])
    shares = float(trade["shares"])
    stop_px = entry_px * (1 - trade["_sl_frac"])
    target_px = entry_px * (1 + trade["_sl_frac"] * trade["_rr"])

    # Bars from entry to EOD
    mask = (day_bars["ts_et"] >= entry_dt) & (day_bars["ts_et"] <= exit_dt)
    window = day_bars[mask]
    if window.empty:
        return {}

    results = {}
    # 1) Time-based exits at 30/60/90/120 min (unless stop/target hit first)
    for hold in [30, 60, 90, 120]:
        peak = entry_px
        trough = entry_px
        exit_px = None
        reason = None
        for _, row in window.iterrows():
            high = float(row["high"]); low = float(row["low"]); close = float(row["close"])
            t = row["ts_et"]
            minute = int((t - entry_dt).total_seconds() / 60)
            # Stop/target take precedence
            if low <= stop_px:
                exit_px = stop_px; reason = "stop_loss"; break
            if high >= target_px:
                exit_px = target_px; reason = "target_hit"; break
            if minute >= hold:
                exit_px = close; reason = f"time_{hold}"; break
        if exit_px is None:
            # never hit stop/target/time — hold to EOD
            exit_px = float(window.iloc[-1]["close"])
            reason = "eod_exit"
        results[f"time_{hold}"] = {
            "pnl": (exit_px - entry_px) * shares,
            "reason": reason,
        }

    # 2) Trailing stops
    for pullback in [0.005, 0.01]:
        peak = entry_px
        exit_px = None
        reason = None
        for _, row in window.iterrows():
            high = float(row["high"]); low = float(row["low"]); close = float(row["close"])
            # Stop/target first
            if low <= stop_px:
                exit_px = stop_px; reason = "stop_loss"; break
            if high >= target_px:
                exit_px = target_px; reason = "target_hit"; break
            peak = max(peak, high)
            # Activate trailing only after +0.5% from entry
            if peak >= entry_px * 1.005 and low <= peak * (1 - pullback):
                exit_px = peak * (1 - pullback); reason = f"trail_{pullback*100:.1f}"; break
        if exit_px is None:
            exit_px = float(window.iloc[-1]["close"])
            reason = "eod_exit"
        results[f"trail_{pullback*100:.1f}pct"] = {
            "pnl": (exit_px - entry_px) * shares,
            "reason": reason,
        }

    return results


def run_backtest_and_analyze(strategy, sl_pct, rr, period_start, period_end, label):
    print(f"\n=== {strategy}/COIN SL={sl_pct}% RR={rr} {label} ({period_start} to {period_end}) ===")

    bars = load_cached("COIN", period_start, period_end)
    if bars is None:
        print(f"  ❌ No cached bars for COIN {period_start}_{period_end}")
        return None

    # Timezone-normalize
    ts = pd.to_datetime(bars["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    bars = bars.copy()
    bars["ts_et"] = ts.dt.tz_convert(_ET)

    # Group bars by day
    bars_by_day = {d: g for d, g in bars.groupby(bars["ts_et"].dt.date)}

    # Run backtest
    cfg_path = _ROOT / "config" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["mode"] = "backtest"
    cfg["account"] = {"balance": 500.0, "max_position_pct": 0.95}
    engine = TradingEngine(cfg)

    override = {
        "active_strategy": strategy,
        "stop_loss_pct": sl_pct,
        "reward_risk": rr,
        "volume_mult": 1.2,
        "regime_filter": True,
        "vix_threshold": 25,
        "max_trades_per_day": 100,
        "afternoon_entries": True,
        "entry_window_minutes": 360,
    }
    result = engine.run_backtest_config(
        config_override=override,
        start_date=period_start, end_date=period_end,
        symbol="COIN",
        cached_bars=bars, cached_sma=load_sma("COIN"), cached_vix={},
    )
    trades = result.get("trades_list", [])
    print(f"  Actual: {len(trades)} trades  PF={result['profit_factor']:.2f}  WR={result['win_rate']:.1f}%  PnL=${result['total_pnl']:+.2f}")

    # Inject SL/RR so alt-exit sim can compute stop/target levels
    sl_frac = sl_pct / 100
    for t in trades:
        t["_sl_frac"] = sl_frac
        t["_rr"] = rr

    # Simulate alternative exits for each trade
    rules = ["time_30", "time_60", "time_90", "time_120", "trail_0.5pct", "trail_1.0pct"]
    totals = {r: 0.0 for r in rules}
    wins = {r: 0 for r in rules}
    reasons = {r: defaultdict(int) for r in rules}
    skipped = 0

    for trade in trades:
        alt = simulate_alt_exits(trade, bars_by_day)
        if not alt:
            skipped += 1
            continue
        for r in rules:
            if r in alt:
                totals[r] += alt[r]["pnl"]
                if alt[r]["pnl"] > 0: wins[r] += 1
                reasons[r][alt[r]["reason"]] += 1

    n = len(trades) - skipped
    if n == 0:
        print("  ❌ No simulatable trades")
        return None

    # Compute PF per alt rule by re-accumulating gross wins/losses
    def pf_for(rule):
        pos = 0.0; neg = 0.0; w = 0
        for t in trades:
            alt = simulate_alt_exits(t, bars_by_day)
            if rule not in alt: continue
            p = alt[rule]["pnl"]
            if p > 0: pos += p; w += 1
            elif p < 0: neg += abs(p)
        return (pos / neg if neg > 0 else float('inf')), w

    # First recompute with single pass (faster than double loop)
    def pf_from_pnls(pnls):
        pos = sum(p for p in pnls if p > 0)
        neg = sum(-p for p in pnls if p < 0)
        return pos / neg if neg > 0 else float('inf')

    # Cache alt results so we don't re-simulate
    per_trade_alts = []
    for t in trades:
        alt = simulate_alt_exits(t, bars_by_day)
        if alt:
            per_trade_alts.append(alt)

    print(f"\n  {'Rule':<18} {'Total PnL':>11} {'Trades':>7} {'WR%':>6} {'PF':>6}  Exit-reason breakdown")
    print(f"  {'-'*80}")
    # actual row
    actual_pnls = [(tr["exit_price"] - tr["entry_price"]) * tr["shares"] for tr in trades]
    print(f"  {'ACTUAL (hold/EOD)':<18} {sum(actual_pnls):>+11.2f} {len(actual_pnls):>7} {100*sum(1 for p in actual_pnls if p>0)/max(len(actual_pnls),1):>5.1f}% {pf_from_pnls(actual_pnls):>6.2f}")
    for rule in rules:
        pnls = [a[rule]["pnl"] for a in per_trade_alts if rule in a]
        rtotal = sum(pnls)
        wr = 100 * sum(1 for p in pnls if p > 0) / max(len(pnls), 1)
        pf = pf_from_pnls(pnls)
        reason_str = ", ".join(f"{k}:{v}" for k, v in sorted(reasons[rule].items(), key=lambda x: -x[1])[:4])
        tag = "  ⬆" if rtotal > sum(actual_pnls) else ("  ⬇" if rtotal < sum(actual_pnls) else "")
        print(f"  {rule:<18} {rtotal:>+11.2f} {len(pnls):>7} {wr:>5.1f}% {pf:>6.2f}  {reason_str}{tag}")

    return {
        "strategy": strategy, "sl": sl_pct, "rr": rr, "label": label,
        "actual_pnl": sum(actual_pnls),
        "alt_totals": {r: sum(a[r]["pnl"] for a in per_trade_alts if r in a) for r in rules},
        "n_trades": len(trades),
    }


def main():
    # Test across multiple COIN configs and periods
    configs = [
        # (strategy, sl, rr)
        ("orb",      1.0, 5.0),    # COIN ORB top performer
        ("orb",      1.0, 2.5),    # alt COIN ORB
        ("momentum", 1.0, 3.0),    # COIN Momentum
    ]
    periods = [
        ("2023-01-01", "2025-01-01", "full_2yr"),
        ("2025-06-01", "2026-03-31", "bear_2025"),
    ]

    all_results = []
    for strat, sl, rr in configs:
        for start, end, label in periods:
            r = run_backtest_and_analyze(strat, sl, rr, start, end, label)
            if r: all_results.append(r)

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY — Did time-based exits beat hold-to-EOD on COIN?")
    print("=" * 90)
    print(f"{'Config':<35} {'Actual':>10} {'time_30':>9} {'time_60':>9} {'time_90':>9} {'time_120':>9} {'trail_0.5':>10} {'trail_1.0':>10}")
    for r in all_results:
        label = f"{r['strategy']}/SL={r['sl']}/RR={r['rr']}/{r['label']}"
        vals = r['alt_totals']
        print(f"{label:<35} {r['actual_pnl']:>+10.2f} {vals['time_30']:>+9.2f} {vals['time_60']:>+9.2f} {vals['time_90']:>+9.2f} {vals['time_120']:>+9.2f} {vals['trail_0.5pct']:>+10.2f} {vals['trail_1.0pct']:>+10.2f}")

    # Save
    out = _ROOT / "logs" / "research" / "coin_time_exit_study.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
