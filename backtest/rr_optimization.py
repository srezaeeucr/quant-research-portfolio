#!/usr/bin/env python3
"""
RR-only sweep on each LIVE slot, with MFE distribution analysis.

Hypothesis (from live observation): targets are too far away. Most winners
exit at EOD without hitting target; lower RR might capture more profit.

Tests RR ∈ {1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0} on each current
live slot with its existing SL fixed.

Periods:
  A. full_2yr (2023-01-01 → 2025-01-01) — large sample, historical
  B. recent (2026-01-01 → today) — current regime, smaller sample

Also computes MFE distribution across all trades in the period — what
fraction of winners' MFE peaks fall in each band.
"""
import os, sys, json, pickle
from pathlib import Path
from collections import defaultdict, Counter
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

# Current live slots — sweep RR for each, keeping SL fixed
LIVE_SLOTS = [
    {"strategy": "orb",            "symbol": "AMD",   "sl": 1.0,  "label": "AMD/ORB"},
    {"strategy": "orb",            "symbol": "GOOGL", "sl": 0.75, "label": "GOOGL/ORB"},
    {"strategy": "rsi_reversion",  "symbol": "AMD",   "sl": 0.5,  "label": "AMD/RSI"},
]
RR_VALUES = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

PERIODS = [
    ("full_2yr", "2023-01-01", "2025-01-01", "cached"),
    ("recent",   "2026-01-01", "2026-05-02", "fetch"),
]


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


def compute_mfe_distribution(trades_list, bars):
    """For each trade, compute max favorable excursion (high during trade)."""
    if not trades_list:
        return {}
    bts = pd.to_datetime(bars["timestamp"])
    if bts.dt.tz is None: bts = bts.dt.tz_localize("UTC").tz_convert(_ET)
    else: bts = bts.dt.tz_convert(_ET)
    bars = bars.copy()
    bars["ts_et"] = bts

    bands = {"<0.5%": 0, "0.5-1.0%": 0, "1.0-1.5%": 0, "1.5-2.0%": 0,
             "2.0-3.0%": 0, "3.0-5.0%": 0, ">5.0%": 0}
    for t in trades_list:
        et = pd.to_datetime(t["entry_time"])
        xt = pd.to_datetime(t["exit_time"])
        if et.tz is None: et = et.tz_localize("UTC").tz_convert(_ET)
        else: et = et.tz_convert(_ET)
        if xt.tz is None: xt = xt.tz_localize("UTC").tz_convert(_ET)
        else: xt = xt.tz_convert(_ET)
        ix = bars[(bars["ts_et"] >= et) & (bars["ts_et"] <= xt)]
        if ix.empty: continue
        entry = float(t["entry_price"])
        max_high = float(ix["high"].max())
        mfe_pct = (max_high - entry) / entry * 100
        if   mfe_pct < 0.5:  bands["<0.5%"] += 1
        elif mfe_pct < 1.0:  bands["0.5-1.0%"] += 1
        elif mfe_pct < 1.5:  bands["1.0-1.5%"] += 1
        elif mfe_pct < 2.0:  bands["1.5-2.0%"] += 1
        elif mfe_pct < 3.0:  bands["2.0-3.0%"] += 1
        elif mfe_pct < 5.0:  bands["3.0-5.0%"] += 1
        else:                bands[">5.0%"] += 1
    return bands


def run_one(strat, sym, sl, rr, bars, sma, start, end, cfg):
    cfg2 = {**cfg, "active_strategy": strat}
    engine = TradingEngine(cfg2)
    override = {
        "active_strategy": strat, "stop_loss_pct": sl, "reward_risk": rr,
        "volume_mult": 1.2, "regime_filter": True, "vix_threshold": 25,
        "max_trades_per_day": 100, "afternoon_entries": True,
        "entry_window_minutes": 360,
    }
    return engine.run_backtest_config(
        config_override=override, start_date=start, end_date=end,
        symbol=sym, cached_bars=bars, cached_sma=sma, cached_vix={},
    )


def main():
    cfg = yaml.safe_load(open(_ROOT / "config/config.yaml"))
    cfg["mode"] = "backtest"
    cfg["account"] = {"balance": 500.0, "max_position_pct": 0.95}

    fetcher = DataFetcher({})
    all_results = []
    mfe_summary = {}

    for slot in LIVE_SLOTS:
        for period_name, start, end, src in PERIODS:
            print(f"\n{'='*75}")
            print(f"  {slot['label']} on {period_name} ({start} → {end})")
            print(f"  SL={slot['sl']}% (fixed), RR sweep")
            print(f"{'='*75}")

            if src == "cached":
                bars = load_cached(slot["symbol"], start, end)
                sma = load_sma(slot["symbol"])
            else:
                try:
                    bars = fetcher.fetch_historical_alpaca(slot["symbol"], start, end, filter_windows=False)
                except Exception as e:
                    print(f"  fetch error: {e}"); continue
                sma = {}
            if bars is None or bars.empty:
                print(f"  no bars"); continue
            print(f"  {len(bars)} bars")

            # MFE on baseline (RR=3.0, neutral)
            try:
                base = run_one(slot["strategy"], slot["symbol"], slot["sl"], 3.0,
                               bars, sma, start, end, cfg)
                trades_base = base.get("trades_list", [])
                mfe_dist = compute_mfe_distribution(trades_base, bars)
                key = f"{slot['label']}/{period_name}"
                mfe_summary[key] = {"n_trades": len(trades_base), "bands": mfe_dist}
                print(f"\n  MFE distribution (n={len(trades_base)} trades, baseline RR=3.0):")
                for band, count in mfe_dist.items():
                    bar = "█" * (count * 30 // max(len(trades_base), 1)) if len(trades_base) else ""
                    pct = 100 * count / len(trades_base) if len(trades_base) else 0
                    print(f"    MFE {band:<10} {count:>4}  ({pct:5.1f}%)  {bar}")
            except Exception as e:
                print(f"  MFE error: {e}")

            # RR sweep
            print(f"\n  RR sweep:")
            print(f"    {'RR':<6} {'trades':<7} {'PF':<6} {'WR%':<6} {'PnL':<8} {'avg_win':<8} {'avg_loss':<8} {'maxDD':<6}")
            best = None
            for rr in RR_VALUES:
                try:
                    r = run_one(slot["strategy"], slot["symbol"], slot["sl"], rr,
                                bars, sma, start, end, cfg)
                    n = r["total_trades"]
                    if n < 5: continue
                    trades = r.get("trades_list", [])
                    pos = [float(t["pnl"]) for t in trades if float(t["pnl"]) > 0]
                    neg = [float(t["pnl"]) for t in trades if float(t["pnl"]) <= 0]
                    avg_win = sum(pos)/len(pos) if pos else 0
                    avg_loss = sum(neg)/len(neg) if neg else 0
                    print(f"    {rr:<6} {n:<7} {r['profit_factor']:<6.2f} "
                          f"{r['win_rate']:<6.1f} ${r['total_pnl']:<+7.2f} "
                          f"${avg_win:<+7.2f} ${avg_loss:<+7.2f} {r['max_drawdown']:<5.1f}%")
                    all_results.append({
                        "label": slot["label"], "period": period_name, "rr": rr,
                        "trades": n, "pf": round(r["profit_factor"], 3),
                        "wr": round(r["win_rate"], 1), "pnl": round(r["total_pnl"], 2),
                        "max_dd": round(r["max_drawdown"], 2),
                        "avg_win": round(avg_win, 2), "avg_loss": round(avg_loss, 2),
                    })
                    if best is None or r["total_pnl"] > best[1]:
                        best = (rr, r["total_pnl"], r["profit_factor"])
                except Exception as e:
                    print(f"    {rr}: error {e}")
            if best:
                print(f"    → best by PnL: RR={best[0]} (PnL=${best[1]:+.2f}, PF={best[2]:.2f})")

    # Combined summary table per slot
    print("\n\n" + "="*75)
    print("SUMMARY — best RR per slot per period")
    print("="*75)
    by_slot = defaultdict(list)
    for r in all_results:
        by_slot[(r["label"], r["period"])].append(r)
    for (label, period), rs in sorted(by_slot.items()):
        rs.sort(key=lambda x: -x["pnl"])
        best = rs[0]
        worst = rs[-1]
        # Current setting
        current_rr = 3.0
        current = next((r for r in rs if r["rr"] == current_rr), None)
        cur_pnl = current["pnl"] if current else None
        print(f"\n  {label} / {period}:")
        print(f"    BEST  RR={best['rr']:<5}: PnL=${best['pnl']:+.2f}  PF={best['pf']:.2f}  WR={best['wr']}%  trades={best['trades']}")
        if current and current['rr'] != best['rr']:
            delta = best['pnl'] - cur_pnl
            print(f"    CURR  RR={current_rr:<5}: PnL=${cur_pnl:+.2f}  PF={current['pf']:.2f}  WR={current['wr']}%  → switching gains ${delta:+.2f}")
        elif current:
            print(f"    (current RR=3.0 IS best for this period)")
        print(f"    WORST RR={worst['rr']:<5}: PnL=${worst['pnl']:+.2f}")

    out = _ROOT / "logs/research/studies_2026_04_26/rr_optimization.json"
    with open(out, "w") as f:
        json.dump({"results": all_results, "mfe_distribution": mfe_summary}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
