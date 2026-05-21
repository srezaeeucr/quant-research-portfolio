#!/usr/bin/env python3
"""
Portfolio Backtest — 4 live configs on SHARED $500 capital.

Compares 4 capital-allocation policies:
  A. FCFS              — first-come-first-served, 95% of capital per trade (current bot)
  B. Equal split 4-way — each trade sized at 25% of capital
  C. Priority order    — AMD > META > COIN > TSLA, first-to-fire wins with 95%
  D. Max-N concurrent  — allow up to N open positions, each sized at 1/N

For each policy, runs every trade that would have fired in the per-symbol
isolated backtest, but skips it if capital constraints prevent entry.
Reports portfolio PF / Sharpe / max DD / utilization.
"""
import os, sys, pickle, json, math
from pathlib import Path
from collections import defaultdict
from datetime import date

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

LIVE_CONFIGS = [
    {"strategy": "momentum",      "symbol": "AMD",  "sl": 0.3,  "rr": 1.5},
    {"strategy": "momentum",      "symbol": "META", "sl": 0.75, "rr": 4.0},
    {"strategy": "orb",           "symbol": "COIN", "sl": 1.0,  "rr": 5.0},
    {"strategy": "ema_crossover", "symbol": "TSLA", "sl": 0.75, "rr": 3.0},
]
PERIOD_START = "2023-01-01"
PERIOD_END = "2025-01-01"
START_CASH = 500.0


def load_cached(sym, s, e):
    pq = CACHE / f"{sym}_{s}_{e}.parquet"
    if pq.exists(): return pd.read_parquet(pq)
    p = CACHE / f"{sym}_{s}_{e}.pkl"
    if p.exists():
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


def run_symbol(cfg_item):
    cfg_path = _ROOT / "config" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["mode"] = "backtest"
    cfg["account"] = {"balance": START_CASH, "max_position_pct": 0.95}
    engine = TradingEngine(cfg)

    bars = load_cached(cfg_item["symbol"], PERIOD_START, PERIOD_END)
    if bars is None: return []

    override = {
        "active_strategy": cfg_item["strategy"],
        "stop_loss_pct": cfg_item["sl"],
        "reward_risk": cfg_item["rr"],
        "volume_mult": 1.2,
        "regime_filter": True,
        "vix_threshold": 25,
        "max_trades_per_day": 100,
        "afternoon_entries": True,
        "entry_window_minutes": 360,
    }
    result = engine.run_backtest_config(
        config_override=override,
        start_date=PERIOD_START, end_date=PERIOD_END,
        symbol=cfg_item["symbol"],
        cached_bars=bars, cached_sma=load_sma(cfg_item["symbol"]), cached_vix={},
    )
    return result.get("trades_list", [])


def normalize_trades(trades, key, priority):
    """Convert engine trades to unified events with timezone-aware times."""
    out = []
    for t in trades:
        et = pd.to_datetime(t["entry_time"])
        xt = pd.to_datetime(t["exit_time"])
        if et.tz is None: et = et.tz_localize("UTC").tz_convert(_ET)
        else: et = et.tz_convert(_ET)
        if xt.tz is None: xt = xt.tz_localize("UTC").tz_convert(_ET)
        else: xt = xt.tz_convert(_ET)
        entry_px = float(t["entry_price"])
        shares = float(t["shares"])
        pnl_abs = float(t["pnl"])       # engine-computed $ PnL (95% of $500 → sized off START_CASH)
        pnl_pct = float(t.get("pnl_pct", 0))  # % move entry→exit
        out.append({
            "key": key,
            "symbol": t["symbol"],
            "priority": priority,
            "entry_time": et,
            "exit_time": xt,
            "entry_px": entry_px,
            "shares": shares,
            "pnl_pct": pnl_pct if pnl_pct != 0 else (float(t["exit_price"]) - entry_px) / entry_px * 100,
            "reason": t.get("reason", ""),
        })
    return out


def simulate(all_trades, policy, max_n=None):
    """Replay trades on shared capital under given policy.

    Policies:
      'fcfs'     - first-come-first-served, position = 95% of available cash
      'equal_4'  - split each trade at 25% of START_CASH max
      'priority' - like FCFS but prefers lower 'priority' number (sorted queue)
      'maxN'     - up to max_n concurrent positions, each sized START_CASH/max_n
    """
    # Build entry events sorted by time
    events = sorted(all_trades, key=lambda x: (x["entry_time"], x["priority"]))

    cash = START_CASH
    equity_curve = [(None, START_CASH)]
    open_positions = []   # list of dicts with keys: exit_time, exit_px_pct, size, key
    peak = START_CASH
    max_dd = 0.0
    taken = []
    skipped = defaultdict(int)
    n_concurrent_hist = defaultdict(int)  # time spent at each concurrency level (in minutes)

    i = 0
    while i < len(events) or open_positions:
        # Close any positions whose exit_time is before next event
        if events[i:] and open_positions:
            next_entry_t = events[i]["entry_time"]
        elif not events[i:]:
            next_entry_t = None  # process remaining exits
        else:
            next_entry_t = events[i]["entry_time"]

        # Exit any positions whose exit_time <= next_entry_t (or any if no more events)
        open_positions.sort(key=lambda p: p["exit_time"])
        while open_positions and (next_entry_t is None or open_positions[0]["exit_time"] <= next_entry_t):
            pos = open_positions.pop(0)
            # Compute PnL: size (cash allocated) moved by pnl_pct%
            pnl = pos["size"] * pos["pnl_pct"] / 100.0
            cash += pos["size"] + pnl
            equity_curve.append((pos["exit_time"], cash))
            peak = max(peak, cash)
            dd = (peak - cash) / peak * 100
            max_dd = max(max_dd, dd)

        if i >= len(events):
            break

        ev = events[i]
        i += 1

        # Determine position size based on policy
        allowed = True
        if policy == 'fcfs':
            size = cash * 0.95
            if size <= 0.01: allowed = False
        elif policy == 'equal_4':
            size = START_CASH * 0.25
            if size > cash: allowed = False
        elif policy == 'priority':
            # Like FCFS — priority only affects event sort order
            size = cash * 0.95
            if size <= 0.01: allowed = False
        elif policy == 'maxN':
            if len(open_positions) >= max_n:
                allowed = False
                size = 0
            else:
                size = START_CASH / max_n
                if size > cash: allowed = False
        else:
            raise ValueError(f"Unknown policy {policy}")

        if not allowed:
            skipped[ev["key"]] += 1
            continue

        # Open position
        cash -= size
        open_positions.append({
            "exit_time": ev["exit_time"],
            "pnl_pct": ev["pnl_pct"],
            "size": size,
            "key": ev["key"],
        })
        taken.append({**ev, "size": size})
        peak_cash_deployed = START_CASH - cash  # unused here but could track
        equity_curve.append((ev["entry_time"], cash + sum(p["size"] for p in open_positions)))

    # Final metrics
    final = equity_curve[-1][1]
    pnls = []
    for t in taken:
        pnls.append(t["size"] * t["pnl_pct"] / 100.0)
    total_pnl = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p <= 0))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    wr = 100 * wins / len(pnls) if pnls else 0

    # Sharpe: daily PnL series
    daily = defaultdict(float)
    for t, p in zip(taken, pnls):
        daily[t["entry_time"].date()] += p
    dseries = list(daily.values())
    if len(dseries) >= 2:
        mu = sum(dseries) / len(dseries)
        var = sum((x - mu) ** 2 for x in dseries) / len(dseries)
        std = math.sqrt(var) if var > 0 else 0.0
        sharpe = (mu / std * math.sqrt(252)) if std > 0 else 0.0
    else:
        sharpe = 0.0

    # Per-symbol contribution
    by_sym = defaultdict(float)
    by_sym_count = defaultdict(int)
    for t, p in zip(taken, pnls):
        by_sym[t["key"]] += p
        by_sym_count[t["key"]] += 1

    return {
        "policy": policy + (f"_{max_n}" if max_n else ""),
        "taken": len(taken),
        "total_submitted": len(events),
        "skipped_by_key": dict(skipped),
        "total_pnl": total_pnl,
        "final_cash": final,
        "wr": wr,
        "pf": pf,
        "sharpe": sharpe,
        "max_dd_pct": max_dd,
        "by_sym_pnl": dict(by_sym),
        "by_sym_count": dict(by_sym_count),
    }


def main():
    print("=" * 75)
    print("PORTFOLIO BACKTEST — 4 Live Configs on Shared $500")
    print(f"Period: {PERIOD_START} → {PERIOD_END}")
    print("=" * 75)

    all_trades = []
    for priority, cfg_item in enumerate(LIVE_CONFIGS):
        key = f'{cfg_item["strategy"][:5]}/{cfg_item["symbol"]}'
        print(f"\nLoading {key} SL={cfg_item['sl']} RR={cfg_item['rr']} ...", flush=True)
        trades = run_symbol(cfg_item)
        norm = normalize_trades(trades, key, priority)
        n = len(norm)
        pnl_iso = sum(t["size"] * t["pnl_pct"] / 100 for t in [{"size": START_CASH * 0.95, "pnl_pct": t["pnl_pct"]} for t in norm])
        print(f"  {n} trades isolated; isolated-PnL @95%size = ${pnl_iso:+.2f}")
        all_trades.extend(norm)

    # Reference: sum of isolated backtests (what our per-symbol numbers show)
    iso_pnl = sum(t["pnl_pct"] / 100 * START_CASH * 0.95 for t in all_trades)
    print(f"\nIsolated-sum PnL (baseline, not realistic): ${iso_pnl:+.2f}")
    print(f"Total trade events across 4 symbols: {len(all_trades)}")

    # Run each policy
    print("\n" + "=" * 75)
    print("POLICY RESULTS")
    print("=" * 75)
    policies = [
        ('fcfs',     None,    'A. FCFS 95%'),
        ('equal_4',  None,    'B. Equal 4-way 25%'),
        ('priority', None,    'C. Priority (AMD>META>COIN>TSLA)'),
        ('maxN',     2,       'D. Max 2 concurrent @50%'),
        ('maxN',     3,       'E. Max 3 concurrent @33%'),
        ('maxN',     4,       'F. Max 4 concurrent @25%'),
    ]
    results = []
    for pol, n, label in policies:
        r = simulate(all_trades, pol, n)
        r["label"] = label
        results.append(r)
        print(f"\n{label}")
        print(f"  Trades taken: {r['taken']}/{r['total_submitted']}  PF={r['pf']:.2f}  WR={r['wr']:.1f}%  Sharpe={r['sharpe']:.2f}  MaxDD={r['max_dd_pct']:.1f}%")
        print(f"  Total PnL: ${r['total_pnl']:+.2f}  (final cash ${r['final_cash']:.2f})")
        print(f"  Skipped by symbol: {r['skipped_by_key']}")
        for sym, pnl in sorted(r['by_sym_pnl'].items(), key=lambda x: -x[1]):
            n_t = r['by_sym_count'][sym]
            print(f"    {sym:<14} {n_t:>3} trades ${pnl:+8.2f}")

    # Summary table
    print("\n" + "=" * 75)
    print("SUMMARY — Ranked by total PnL")
    print("=" * 75)
    results.sort(key=lambda r: -r['total_pnl'])
    print(f"{'Policy':<38} {'Trades':>7} {'PnL':>9} {'PF':>6} {'Sharpe':>7} {'MaxDD':>7}")
    for r in results:
        print(f"{r['label']:<38} {r['taken']:>7} {r['total_pnl']:>+9.2f} {r['pf']:>6.2f} {r['sharpe']:>7.2f} {r['max_dd_pct']:>6.1f}%")

    out = _ROOT / "logs" / "research" / "portfolio_backtest.json"
    with open(out, "w") as f:
        json.dump({"configs": LIVE_CONFIGS, "period": [PERIOD_START, PERIOD_END], "results": results}, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
