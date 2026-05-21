#!/usr/bin/env python3
"""
Ensemble strategy validation.

Tests EnsembleStrategy with various sub-strategy lists and thresholds on
both full_2yr (historical) and recent (Jan-Apr 26) data, AMD + GOOGL.

Compares ensemble PF/PnL/WR against the best single sub-strategy.
"""
import os, sys, json, pickle, copy
from pathlib import Path
from collections import defaultdict
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import pandas as pd, pytz, yaml
from src.data.fetcher import DataFetcher
from src.engine import TradingEngine

_ROOT = Path(__file__).resolve().parents[1]
CACHE = _ROOT / "logs" / "research" / "bar_cache"
SYMBOLS = ["AMD", "GOOGL"]


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


def run_test(strat_name, sub_strategies, threshold, sym, sl, rr,
             bars, sma, start, end):
    cfg = yaml.safe_load(open(_ROOT / "config/config.yaml"))
    cfg["mode"] = "backtest"
    cfg["account"] = {"balance": 500.0, "max_position_pct": 0.95}
    if strat_name == "ensemble":
        cfg.setdefault("strategy_params", {})["ensemble"] = {
            "threshold": threshold, "sub_strategies": sub_strategies,
        }
    engine = TradingEngine(cfg)
    override = {
        "active_strategy": strat_name, "stop_loss_pct": sl, "reward_risk": rr,
        "volume_mult": 1.2, "regime_filter": True, "vix_threshold": 25,
        "max_trades_per_day": 100, "afternoon_entries": True,
        "entry_window_minutes": 360,
    }
    res = engine.run_backtest_config(
        config_override=override, start_date=start, end_date=end,
        symbol=sym, cached_bars=bars, cached_sma=sma, cached_vix={},
    )
    return res


def main():
    # 5 sub-strategies covering different families
    subs = ["orb", "rsi_reversion", "macd_crossover", "bollinger_reversal", "donchian_breakout"]
    print(f"=== Ensemble validation ===")
    print(f"Sub-strategies: {subs}")
    print(f"Thresholds: 2, 3, 4, 5")
    print(f"Symbols: {SYMBOLS}\n")

    fetcher = DataFetcher({})
    results = []

    # ------ HISTORICAL ------
    print("=" * 70)
    print("PHASE A — full_2yr (2023-01-01 → 2025-01-01)")
    print("=" * 70)
    PERIOD = ("2023-01-01", "2025-01-01")
    for sym in SYMBOLS:
        bars = load_cached(sym, *PERIOD)
        if bars is None: print(f"{sym}: no bars"); continue
        sma = load_sma(sym)
        print(f"\n{sym} on full_2yr:")
        # Baseline: each sub-strategy alone at SL=0.75 RR=3.0
        base_results = []
        for sub in subs:
            try:
                r = run_test(sub, [], 0, sym, 0.75, 3.0, bars, sma, *PERIOD)
                base_results.append((sub, r))
                print(f"  baseline {sub:<22}: {r['total_trades']:4d} trades  PF={r['profit_factor']:5.2f}  PnL=${r['total_pnl']:+8.2f}  WR={r['win_rate']:.0f}%")
            except Exception as e:
                print(f"  baseline {sub}: ERROR {e}")
        # Ensemble at each threshold
        for th in [2, 3, 4, 5]:
            if th > len(subs): continue
            try:
                r = run_test("ensemble", subs, th, sym, 0.75, 3.0, bars, sma, *PERIOD)
                marker = ""
                # Compare to best baseline PnL
                best_base = max(base_results, key=lambda x: x[1]["total_pnl"])
                if r["total_pnl"] > best_base[1]["total_pnl"]:
                    marker = "  ⭐ beats best single"
                print(f"  ensemble N={th:<23}: {r['total_trades']:4d} trades  PF={r['profit_factor']:5.2f}  PnL=${r['total_pnl']:+8.2f}  WR={r['win_rate']:.0f}%{marker}")
                results.append({
                    "phase": "full_2yr", "symbol": sym, "threshold": th, "subs": subs,
                    "trades": r["total_trades"], "pf": round(r["profit_factor"], 3),
                    "wr": round(r["win_rate"], 1), "pnl": round(r["total_pnl"], 2),
                })
            except Exception as e:
                print(f"  ensemble N={th}: ERROR {e}")

    # ------ RECENT ------
    print("\n" + "=" * 70)
    print("PHASE B — recent (Jan-Apr 26, 2026)")
    print("=" * 70)
    START, END = "2026-01-01", "2026-04-26"
    for sym in SYMBOLS:
        print(f"\n{sym} on recent:")
        try:
            bars = fetcher.fetch_historical_alpaca(sym, START, END, filter_windows=False)
        except Exception as e:
            print(f"  fetch error: {e}"); continue
        if bars is None or bars.empty:
            print(f"  no bars"); continue
        sma = {}
        # Baselines
        base_results = []
        for sub in subs:
            try:
                r = run_test(sub, [], 0, sym, 0.75, 3.0, bars, sma, START, END)
                base_results.append((sub, r))
                print(f"  baseline {sub:<22}: {r['total_trades']:4d} trades  PF={r['profit_factor']:5.2f}  PnL=${r['total_pnl']:+8.2f}  WR={r['win_rate']:.0f}%")
            except Exception as e:
                print(f"  baseline {sub}: ERROR {e}")
        for th in [2, 3, 4, 5]:
            if th > len(subs): continue
            try:
                r = run_test("ensemble", subs, th, sym, 0.75, 3.0, bars, sma, START, END)
                marker = ""
                if base_results:
                    best_base = max(base_results, key=lambda x: x[1]["total_pnl"])
                    if r["total_pnl"] > best_base[1]["total_pnl"]:
                        marker = "  ⭐ beats best single"
                print(f"  ensemble N={th:<23}: {r['total_trades']:4d} trades  PF={r['profit_factor']:5.2f}  PnL=${r['total_pnl']:+8.2f}  WR={r['win_rate']:.0f}%{marker}")
                results.append({
                    "phase": "recent", "symbol": sym, "threshold": th, "subs": subs,
                    "trades": r["total_trades"], "pf": round(r["profit_factor"], 3),
                    "wr": round(r["win_rate"], 1), "pnl": round(r["total_pnl"], 2),
                })
            except Exception as e:
                print(f"  ensemble N={th}: ERROR {e}")

    # Save
    out = _ROOT / "logs/research/studies_2026_04_26/ensemble_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f: json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
