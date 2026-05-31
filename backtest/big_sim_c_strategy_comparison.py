#!/usr/bin/env python3
"""
Big Simulation C — Strategy-Class Comparison Sweep.

HYPOTHESIS:
  At neutral parameters (SL=0.75% / RR=3.0), strategies form a clear quality
  ranking: breakout/trend strategies (ORB, MACD, Donchian) and well-known
  oscillators (RSI, Stochastic) outperform pattern-only or volume-only
  strategies (Inside Bar, VWAP Bands) on average.

PASS CRITERIA (stated BEFORE running):
  - At least 6 of 12 strategies produce PF > 1.0 on full_2yr aggregate
    (AMD + GOOGL combined)
  - Composite strategies (Ensemble, Hybrid) do not exceed the best
    individual strategy on full_2yr (per the ensemble study's finding)
  - The ORB family ranks in the top 5 (its standalone-edge claim holds)
  - Mean reversion strategies (RSI, Bollinger, VWAP Reversion) rank
    near the middle (not best, not worst)

This produces a clean comparison table suitable for the portfolio site.
"""
import os, sys, json, pickle
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import pandas as pd, pytz, yaml
from src.data.fetcher import DataFetcher
from src.engine import TradingEngine

_ROOT = Path(__file__).resolve().parents[1]
CACHE = _ROOT / "logs" / "research" / "bar_cache"

STRATEGIES = [
    "orb", "ema_crossover", "momentum", "macd_crossover",
    "rsi_reversion", "bollinger_reversal", "vwap_reversion",
    "vwap_bands", "donchian_breakout", "stochastic_crossover",
    "inside_bar_breakout", "gap_fill",
    # Composites (also test these — per hypothesis they should NOT win)
    # "ensemble", "hybrid_filter",  # skipped: require sub-strategy setup
]
SYMBOLS = ["AMD", "GOOGL"]
SL = 0.75
RR = 3.0


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


def run_one(strat, sym, sl, rr, bars, sma, start, end):
    cfg = yaml.safe_load(open(_ROOT / "config/config.yaml"))
    cfg["mode"] = "backtest"
    cfg["account"] = {"balance": 500.0, "max_position_pct": 0.95}
    cfg["active_strategy"] = strat
    engine = TradingEngine(cfg)
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
    print("=" * 75)
    print("BIG SIMULATION C — Strategy-Class Comparison")
    print("=" * 75)
    print(f"\nParams: SL={SL}%, RR={RR} (neutral, not optimized)")
    print(f"Symbols: {SYMBOLS}")
    print(f"Strategies: {len(STRATEGIES)}\n")

    fetcher = DataFetcher({})
    summary = []

    for period_name, start, end, src in [
        ("full_2yr", "2023-01-01", "2025-01-01", "cached"),
        ("recent",   "2026-01-01", "2026-05-30", "fetch"),
    ]:
        print("=" * 75)
        print(f"  PERIOD: {period_name} ({start} → {end})")
        print("=" * 75)
        print(f"  {'strategy':<22} {'symbol':<6} {'trades':<7} {'PF':<6} {'WR%':<6} {'PnL':<9}")
        print(f"  {'-' * 60}")
        bars_cache = {}
        for strat in STRATEGIES:
            for sym in SYMBOLS:
                if sym not in bars_cache:
                    if src == "cached":
                        bars_cache[sym] = load_cached(sym, start, end)
                    else:
                        try:
                            bars_cache[sym] = fetcher.fetch_historical_alpaca(sym, start, end, filter_windows=False)
                        except: bars_cache[sym] = None
                bars = bars_cache.get(sym)
                sma = load_sma(sym) if src == "cached" else {}
                if bars is None or bars.empty: continue
                try:
                    r = run_one(strat, sym, SL, RR, bars, sma, start, end)
                    if r["total_trades"] >= 5:
                        print(f"  {strat:<22} {sym:<6} {r['total_trades']:<7} "
                              f"{r['profit_factor']:<6.2f} {r['win_rate']:<6.1f} "
                              f"${r['total_pnl']:<+8.2f}")
                        summary.append({
                            "period": period_name, "strategy": strat, "symbol": sym,
                            "trades": r["total_trades"], "pf": round(r["profit_factor"], 3),
                            "wr": round(r["win_rate"], 1), "pnl": round(r["total_pnl"], 2),
                        })
                except Exception as e:
                    print(f"  {strat:<22} {sym:<6} ERROR")

    # Aggregate ranking
    print("\n" + "=" * 75)
    print("AGGREGATE RANKING — full_2yr (AMD + GOOGL combined PnL per strategy)")
    print("=" * 75)
    by_strat = {}
    for r in summary:
        if r["period"] != "full_2yr": continue
        s = r["strategy"]
        if s not in by_strat:
            by_strat[s] = {"trades": 0, "pnl": 0.0, "wins": 0, "loss_sum": 0.0, "win_sum": 0.0}
        by_strat[s]["trades"] += r["trades"]
        by_strat[s]["pnl"] += r["pnl"]
    ranked = sorted(by_strat.items(), key=lambda x: -x[1]["pnl"])
    print(f"  {'rank':<5} {'strategy':<22} {'trades':<8} {'PnL':<10}")
    print(f"  {'-' * 50}")
    for i, (strat, m) in enumerate(ranked, 1):
        print(f"  {i:<5} {strat:<22} {m['trades']:<8} ${m['pnl']:<+9.2f}")

    # Pass/fail
    print("\n" + "=" * 75)
    print("HYPOTHESIS EVALUATION")
    print("=" * 75)
    full2_results = [r for r in summary if r["period"] == "full_2yr"]
    profitable = sum(1 for r in full2_results if r["pf"] > 1.0)
    by_s = {}
    for r in full2_results:
        by_s.setdefault(r["strategy"], []).append(r["pnl"])
    sym_strat_count = sum(1 for r in full2_results)
    print(f"  Configs with PF > 1.0:           {profitable}/{sym_strat_count}")

    # ORB ranks
    orb_rank = next((i for i, (s, _) in enumerate(ranked, 1) if s == "orb"), None)
    print(f"  ORB rank (aggregate PnL):        #{orb_rank}  {'✓' if orb_rank and orb_rank <= 5 else '✗'}")

    out = _ROOT / "logs/research/big_sim_c_results.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
