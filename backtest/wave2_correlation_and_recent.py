#!/usr/bin/env python3
"""
Wave-2 strategy study: correlation + recent validation in one run.

Strategies in scope (12 total): orb, ema_crossover, momentum, vwap_reversion,
gap_fill, macd_crossover, rsi_reversion, vwap_bands, bollinger_reversal,
donchian_breakout, stochastic_crossover, inside_bar_breakout.

Symbols: AMD, GOOGL.

Phase A — full_2yr correlation matrix (12 strats × 2 syms = 24 series).
Phase B — recent (Jan-Apr 26) validation of the 4 NEW strategies on
          a small reasonable param grid.
"""
import os, sys, json, pickle
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

ALL_STRATS = [
    "orb", "ema_crossover", "momentum", "vwap_reversion", "gap_fill",
    "macd_crossover", "rsi_reversion", "vwap_bands",
    "bollinger_reversal", "donchian_breakout", "stochastic_crossover",
    "inside_bar_breakout",
]
NEW_STRATS = ["bollinger_reversal", "donchian_breakout",
              "stochastic_crossover", "inside_bar_breakout"]
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


def run_backtest(engine, strat_name, sym, sl, rr, bars, sma, start, end):
    override = {
        "active_strategy": strat_name, "stop_loss_pct": sl, "reward_risk": rr,
        "volume_mult": 1.2, "regime_filter": True, "vix_threshold": 25,
        "max_trades_per_day": 100, "afternoon_entries": True, "entry_window_minutes": 360,
    }
    return engine.run_backtest_config(
        config_override=override, start_date=start, end_date=end,
        symbol=sym, cached_bars=bars, cached_sma=sma, cached_vix={},
    )


# =============================================================
# Phase A — full_2yr correlation
# =============================================================
def phase_a():
    print(f"=== Phase A: 12-strategy correlation on full_2yr ===\n")
    PERIOD = ("2023-01-01", "2025-01-01")
    cfg = yaml.safe_load(open(_ROOT / "config/config.yaml"))
    cfg["mode"] = "backtest"
    cfg["account"] = {"balance": 500.0, "max_position_pct": 0.95}
    engine = TradingEngine(cfg)

    daily_pnl_by_key = {}
    for sym in SYMBOLS:
        bars = load_cached(sym, *PERIOD)
        if bars is None:
            print(f"  {sym}: no cached bars, skipping"); continue
        sma = load_sma(sym)
        for strat in ALL_STRATS:
            key = f"{strat}/{sym}"
            try:
                res = run_backtest(engine, strat, sym, 0.75, 3.0, bars, sma, *PERIOD)
                trades = res.get("trades_list", [])
                day_pnl = defaultdict(float)
                for t in trades:
                    et = pd.to_datetime(t["entry_time"])
                    if et.tz is None: et = et.tz_localize("UTC").tz_convert(_ET)
                    else: et = et.tz_convert(_ET)
                    day_pnl[et.date()] += float(t["pnl"])
                daily_pnl_by_key[key] = day_pnl
                print(f"  {key:<32}  {len(trades):4d} trades  ${res['total_pnl']:+8.2f}")
            except Exception as e:
                print(f"  {key}: ERROR {e}")

    all_days = set()
    for d in daily_pnl_by_key.values(): all_days.update(d.keys())
    days_sorted = sorted(all_days)
    df = pd.DataFrame({k: [d.get(day, 0.0) for day in days_sorted]
                       for k, d in daily_pnl_by_key.items()}, index=days_sorted)
    corr = df.corr()

    # NEW vs all OLD: for each new strategy and symbol, show min correlation
    print(f"\n=== NEW strategies vs existing 8 (per symbol) — min/max corr ===")
    for sym in SYMBOLS:
        print(f"\n  {sym}:")
        for new in NEW_STRATS:
            new_key = f"{new}/{sym}"
            if new_key not in corr.columns: continue
            old_corrs = []
            for old in ["orb","ema_crossover","momentum","vwap_reversion","gap_fill",
                        "macd_crossover","rsi_reversion","vwap_bands"]:
                old_key = f"{old}/{sym}"
                if old_key in corr.columns:
                    old_corrs.append((old, corr.loc[new_key, old_key]))
            if not old_corrs: continue
            old_corrs.sort(key=lambda x: abs(x[1]))
            min_pair = old_corrs[0]
            max_pair = old_corrs[-1]
            avg_abs = sum(abs(c) for _,c in old_corrs) / len(old_corrs)
            verdict = "⭐ DIVERSIFIER" if avg_abs < 0.25 else "(similar)" if avg_abs > 0.5 else ""
            print(f"    {new:<22}  min_corr={min_pair[0]}({min_pair[1]:+.2f})  "
                  f"max_corr={max_pair[0]}({max_pair[1]:+.2f})  avg|c|={avg_abs:.2f}  {verdict}")

    out = _ROOT / "logs/research/studies_2026_04_26/wave2_correlation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"strategies": ALL_STRATS, "symbols": SYMBOLS,
                   "correlation": corr.round(4).to_dict()}, f, indent=2, default=str)
    print(f"\nSaved: {out}")


# =============================================================
# Phase B — recent validation, NEW strategies only
# =============================================================
def phase_b():
    print(f"\n\n=== Phase B: NEW strategies on recent (Jan-Apr 26) ===\n")
    START, END = "2026-01-01", "2026-04-26"

    # Reasonable param grid (small, 27 combos × 2 sym × 4 strats = 216)
    GRID = []
    for sl in [0.5, 0.75, 1.0]:
        for rr in [2.0, 3.0, 4.0]:
            # Bollinger: vary period (textbook = 20)
            for p in [15, 20]:
                GRID.append({"strategy": "bollinger_reversal", "sl": sl, "rr": rr,
                             "params": {"period": p, "num_std": 2.0}})
            # Donchian: vary period
            for p in [15, 20]:
                GRID.append({"strategy": "donchian_breakout", "sl": sl, "rr": rr,
                             "params": {"period": p}})
            # Stochastic: textbook (14, 3)
            GRID.append({"strategy": "stochastic_crossover", "sl": sl, "rr": rr,
                         "params": {"k_period": 14, "d_period": 3, "oversold_max": 20.0}})
            # Inside Bar: no params
            GRID.append({"strategy": "inside_bar_breakout", "sl": sl, "rr": rr,
                         "params": {}})

    cfg = yaml.safe_load(open(_ROOT / "config/config.yaml"))
    cfg["mode"] = "backtest"
    cfg["account"] = {"balance": 500.0, "max_position_pct": 0.95}

    fetcher = DataFetcher({})
    results = []
    for sym in SYMBOLS:
        print(f"Fetching {sym} bars...", flush=True)
        bars = fetcher.fetch_historical_alpaca(sym, START, END, filter_windows=False)
        if bars is None or bars.empty:
            print(f"  no bars"); continue
        print(f"  {len(bars)} bars")

        for cfg_item in GRID:
            engine = TradingEngine({**cfg, "active_strategy": cfg_item["strategy"],
                                    "strategy_params": {cfg_item["strategy"]: cfg_item["params"]}})
            try:
                res = run_backtest(engine, cfg_item["strategy"], sym, cfg_item["sl"], cfg_item["rr"],
                                   bars, {}, START, END)
                if res["total_trades"] >= 5:
                    results.append({
                        "strategy": cfg_item["strategy"], "symbol": sym,
                        "sl": cfg_item["sl"], "rr": cfg_item["rr"],
                        "params": cfg_item["params"],
                        "trades": res["total_trades"], "pf": round(res["profit_factor"], 3),
                        "wr": round(res["win_rate"], 1), "pnl": round(res["total_pnl"], 2),
                    })
            except Exception as e:
                pass

    results.sort(key=lambda r: -r["pnl"])
    print(f"\n=== TOP 20 NEW strategies on recent data ===")
    print(f"  {'strategy':<24} {'sym':<6} {'sl':<5} {'rr':<5} {'trades':<7} {'pf':<6} {'wr':<6} {'pnl':<8}")
    for r in results[:20]:
        print(f"  {r['strategy']:<24} {r['symbol']:<6} {r['sl']:<5} {r['rr']:<5} "
              f"{r['trades']:<7} {r['pf']:<6.2f} {r['wr']:<6.1f} ${r['pnl']:<+7.2f}")

    profitable = [r for r in results if r["pf"] > 1.0]
    print(f"\nValid configs (≥5 trades): {len(results)}")
    print(f"Profitable (PF > 1.0): {len(profitable)} ({100*len(profitable)/max(len(results),1):.0f}%)")

    print(f"\n=== Best per (strategy, symbol) ===")
    by_ss = defaultdict(list)
    for r in results: by_ss[(r["strategy"], r["symbol"])].append(r)
    for (s, sy), rs in sorted(by_ss.items()):
        best = max(rs, key=lambda r: r["pnl"])
        params_str = ', '.join(f"{k}={v}" for k, v in best['params'].items())
        print(f"  {s:<22}/{sy}: SL={best['sl']} RR={best['rr']}  trades={best['trades']} "
              f"pf={best['pf']:.2f} pnl=${best['pnl']:+.2f}  ({params_str})")

    out = _ROOT / "logs/research/studies_2026_04_26/wave2_recent.json"
    with open(out, "w") as f: json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    phase_a()
    phase_b()
