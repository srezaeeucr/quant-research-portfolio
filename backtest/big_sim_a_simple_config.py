#!/usr/bin/env python3
"""
Big Simulation A — Simple Three-Slot Configuration.

HYPOTHESIS:
  A small, intentionally simple 3-slot portfolio configuration —
  AMD/ORB (SL=1.0%, RR=1.5), AMD/RSI (SL=0.5%, RR=2.0), GOOGL/ORB
  (SL=0.75%, RR=2.0) — is profitable on BOTH the in-sample period
  (Jan 2023 – Jan 2025) AND the out-of-sample period (Jan – May 2026).

  This configuration was selected by the anti-overfit reversion documented
  in `docs/case-studies/anti-overfit.md` — mid-range parameters chosen
  for cross-regime stability rather than period-specific optimization.

PASS CRITERIA (stated BEFORE running):
  - Combined PF ≥ 1.2 on in-sample
  - Combined PF ≥ 1.0 on out-of-sample (the more important test)
  - At least one slot profitable in each period

Runs each slot independently as a backtest, then aggregates trade lists into
a shared-capital portfolio with max-2-concurrent @ 50% allocation (the
policy that won in `backtest/portfolio_backtest.py`).
"""
import os, sys, json, pickle, math
from pathlib import Path
from collections import defaultdict
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import pandas as pd, pytz, yaml
from src.data.fetcher import DataFetcher
from src.engine import TradingEngine

_ET = pytz.timezone("America/New_York")
_ROOT = Path(__file__).resolve().parents[1]
CACHE = _ROOT / "logs" / "research" / "bar_cache"

LIVE_SLOTS = [
    {"strategy": "orb",           "symbol": "AMD",   "sl": 1.0,  "rr": 1.5, "name": "AMD/ORB"},
    {"strategy": "rsi_reversion", "symbol": "AMD",   "sl": 0.5,  "rr": 2.0, "name": "AMD/RSI"},
    {"strategy": "orb",           "symbol": "GOOGL", "sl": 0.75, "rr": 2.0, "name": "GOOGL/ORB"},
]

PERIODS = [
    ("full_2yr", "2023-01-01", "2025-01-01", "cached"),
    ("recent",   "2026-01-01", "2026-05-30", "fetch"),
]

START_CASH = 500.0


def load_cached(sym, s, e):
    for ext in ["parquet", "pkl"]:
        p = CACHE / f"{sym}_{s}_{e}.{ext}"
        if p.exists():
            if ext == "parquet": return pd.read_parquet(p)
            with open(p, "rb") as f: return pickle.load(f)
    return None


def load_sma(sym):
    jf = CACHE / f"{sym}_sma.json"
    if jf.exists():
        return {date.fromisoformat(k): v for k, v in json.load(open(jf)).items()}
    p = CACHE / f"{sym}_sma.pkl"
    if p.exists():
        with open(p, "rb") as f: return pickle.load(f)
    return {}


def run_one(slot, bars, sma, start, end):
    cfg = yaml.safe_load(open(_ROOT / "config/config.yaml"))
    cfg["mode"] = "backtest"
    cfg["account"] = {"balance": START_CASH, "max_position_pct": 0.95}
    cfg["active_strategy"] = slot["strategy"]
    engine = TradingEngine(cfg)
    override = {
        "active_strategy": slot["strategy"], "stop_loss_pct": slot["sl"],
        "reward_risk": slot["rr"], "volume_mult": 1.2, "regime_filter": True,
        "vix_threshold": 25, "max_trades_per_day": 100,
        "afternoon_entries": True, "entry_window_minutes": 360,
    }
    return engine.run_backtest_config(
        config_override=override, start_date=start, end_date=end,
        symbol=slot["symbol"], cached_bars=bars, cached_sma=sma, cached_vix={},
    )


def portfolio_simulate(trades_by_slot, max_concurrent=2, per_slot_pct=0.50):
    """Simulate trades on shared START_CASH with max N concurrent positions."""
    events = []
    for slot_name, trades in trades_by_slot.items():
        for t in trades:
            et = pd.to_datetime(t["entry_time"])
            xt = pd.to_datetime(t["exit_time"])
            if et.tz is None: et = et.tz_localize("UTC").tz_convert(_ET)
            else: et = et.tz_convert(_ET)
            if xt.tz is None: xt = xt.tz_localize("UTC").tz_convert(_ET)
            else: xt = xt.tz_convert(_ET)
            entry_px = float(t["entry_price"])
            exit_px = float(t.get("exit_price", t.get("entry_price", 0)))
            pnl_pct = (exit_px - entry_px) / entry_px * 100 if entry_px else 0.0
            events.append({
                "slot": slot_name, "entry_time": et, "exit_time": xt,
                "entry_px": entry_px, "pnl_pct": pnl_pct,
            })
    events.sort(key=lambda e: e["entry_time"])

    cash = START_CASH
    peak = START_CASH
    max_dd = 0.0
    open_positions = []
    taken = []
    skipped = 0
    equity = [(None, START_CASH)]

    i = 0
    while i < len(events) or open_positions:
        next_t = events[i]["entry_time"] if i < len(events) else None
        # Close exits before next entry
        open_positions.sort(key=lambda p: p["exit_time"])
        while open_positions and (next_t is None or open_positions[0]["exit_time"] <= next_t):
            pos = open_positions.pop(0)
            pnl = pos["size"] * pos["pnl_pct"] / 100.0
            cash += pos["size"] + pnl
            equity.append((pos["exit_time"], cash))
            peak = max(peak, cash)
            dd = (peak - cash) / peak * 100
            max_dd = max(max_dd, dd)
        if i >= len(events):
            break
        ev = events[i]; i += 1
        if len(open_positions) >= max_concurrent:
            skipped += 1; continue
        size = START_CASH * per_slot_pct
        if size > cash: skipped += 1; continue
        cash -= size
        open_positions.append({**ev, "size": size})
        taken.append({**ev, "size": size})

    final = equity[-1][1] if equity else cash
    pnls = [t["size"] * t["pnl_pct"] / 100 for t in taken]
    pos_sum = sum(p for p in pnls if p > 0)
    neg_sum = abs(sum(p for p in pnls if p <= 0))
    pf = pos_sum / neg_sum if neg_sum > 0 else float("inf")
    wr = 100 * sum(1 for p in pnls if p > 0) / max(len(pnls), 1)

    # Sharpe (daily)
    daily = defaultdict(float)
    for t, p in zip(taken, pnls):
        daily[t["entry_time"].date()] += p
    dseries = list(daily.values())
    if len(dseries) >= 2:
        mu = sum(dseries)/len(dseries)
        var = sum((x-mu)**2 for x in dseries)/len(dseries)
        std = math.sqrt(var) if var > 0 else 0
        sharpe = (mu/std * math.sqrt(252)) if std > 0 else 0.0
    else:
        sharpe = 0.0
    return {
        "trades_taken": len(taken), "trades_skipped": skipped,
        "total_pnl": sum(pnls), "final_cash": final,
        "pf": pf, "wr": wr, "sharpe": sharpe, "max_dd_pct": max_dd,
    }


def main():
    print("=" * 75)
    print("BIG SIMULATION A — Current Live Config on All Data")
    print("=" * 75)
    print()
    print("Slots:")
    for s in LIVE_SLOTS:
        print(f"  {s['name']:<14} {s['strategy']:<14} SL={s['sl']}% RR={s['rr']}")
    print()

    fetcher = DataFetcher({})
    results = {}

    for period_name, start, end, src in PERIODS:
        print("=" * 75)
        print(f"  PERIOD: {period_name} ({start} → {end})")
        print("=" * 75)

        trades_by_slot = {}
        per_slot_metrics = {}
        for slot in LIVE_SLOTS:
            sym = slot["symbol"]
            if src == "cached":
                bars = load_cached(sym, start, end); sma = load_sma(sym)
            else:
                try:
                    bars = fetcher.fetch_historical_alpaca(sym, start, end, filter_windows=False)
                except Exception as e:
                    print(f"  {slot['name']}: fetch error: {e}"); continue
                sma = {}
            if bars is None or bars.empty:
                print(f"  {slot['name']}: no bars"); continue
            try:
                r = run_one(slot, bars, sma, start, end)
                trades = r.get("trades_list", [])
                trades_by_slot[slot["name"]] = trades
                per_slot_metrics[slot["name"]] = {
                    "n": len(trades), "pf": r["profit_factor"],
                    "wr": r["win_rate"], "pnl": r["total_pnl"],
                }
                print(f"  {slot['name']:<14}: {len(trades):4d} trades  "
                      f"PF={r['profit_factor']:.2f}  WR={r['win_rate']:.1f}%  "
                      f"PnL=${r['total_pnl']:+.2f}")
            except Exception as e:
                print(f"  {slot['name']}: error {e}")

        # Portfolio aggregation
        if trades_by_slot:
            port = portfolio_simulate(trades_by_slot, max_concurrent=2, per_slot_pct=0.50)
            print(f"\n  PORTFOLIO (max 2 concurrent @ 50%):")
            print(f"    trades taken: {port['trades_taken']} / skipped: {port['trades_skipped']}")
            print(f"    PnL: ${port['total_pnl']:+.2f}  (final cash ${port['final_cash']:.2f})")
            print(f"    PF: {port['pf']:.2f}  WR: {port['wr']:.1f}%  Sharpe: {port['sharpe']:.2f}  MaxDD: {port['max_dd_pct']:.1f}%")
            results[period_name] = {"per_slot": per_slot_metrics, "portfolio": port}

    # Pass/fail evaluation
    print("\n" + "=" * 75)
    print("HYPOTHESIS EVALUATION")
    print("=" * 75)
    criteria = [
        ("full_2yr PF ≥ 1.2",    "full_2yr", lambda r: r["portfolio"]["pf"] >= 1.2),
        ("recent PF ≥ 1.0",      "recent",   lambda r: r["portfolio"]["pf"] >= 1.0),
        ("full_2yr maxDD ≤ 5%",  "full_2yr", lambda r: r["portfolio"]["max_dd_pct"] <= 5.0),
        ("recent maxDD ≤ 5%",    "recent",   lambda r: r["portfolio"]["max_dd_pct"] <= 5.0),
        ("full_2yr ≥1 slot positive", "full_2yr", lambda r: any(s["pnl"]>0 for s in r["per_slot"].values())),
        ("recent ≥1 slot positive",   "recent",   lambda r: any(s["pnl"]>0 for s in r["per_slot"].values())),
    ]
    passed = 0; total = 0
    for label, period, check in criteria:
        if period in results:
            total += 1
            r = results[period]
            try:
                ok = check(r)
            except: ok = False
            passed += 1 if ok else 0
            mark = "✓" if ok else "✗"
            print(f"  {mark} {label}")
    print(f"\nOVERALL: {passed}/{total} criteria passed")

    out = _ROOT / "logs/research/big_sim_a_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
