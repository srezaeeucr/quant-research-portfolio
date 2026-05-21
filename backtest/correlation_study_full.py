#!/usr/bin/env python3
"""
Full-scope Correlation Study — all symbols × 3 strategies.

For each (strategy, symbol), pick the best-PF config from V4 results
(wf_wins >= 6, MC p <= 0.05, slip05 >= 1.0 preferred; else highest full_2yr PF).
Run backtest, compute daily PnL. Build 57x57 correlation matrix.

Report:
  - 10 most-correlated pairs (worst for diversification)
  - 10 least-correlated pairs (best for diversification)
  - Per-symbol family clustering
"""
import os, sys, pickle, json, glob
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

SYMBOLS = ["AMD", "META", "COIN", "TSLA", "NVDA", "SPY", "QQQ", "AAPL",
           "AMZN", "MSFT", "GOOGL", "JPM", "MSTR", "IWM",
           "XLK", "XLF", "XLE", "XLV", "XLI"]
STRATEGIES = ["orb", "ema_crossover", "momentum"]
PERIOD_START = "2023-01-01"
PERIOD_END = "2025-01-01"


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


def pick_best_configs():
    """Pick one (sl, rr) per (strategy, symbol) using V4 data.

    Priority: configs that PASS V4 all 3 tests, ranked by slip_full_2yr 0.05.
    Fallback: highest full_2yr PF.
    """
    best = {}  # (strat, sym) -> dict with sl, rr, source
    passers = defaultdict(list)
    candidates = defaultdict(list)
    for f in glob.glob('logs/research/val_v4_all/*.json'):
        try: r = json.load(open(f))
        except: continue
        key = (r['strategy'], r['symbol'])
        if r.get('full_2yr_trades', 0) < 5:
            continue
        slip = (r.get('slip_full_2yr', {}) or {}).get('0.05', 0)
        passed = (r.get('wf_wins', 0) >= 6 and r.get('mc_p_value', 1.0) <= 0.05 and slip >= 1.0)
        entry = {
            'sl': r['sl'], 'rr': r['rr'], 'slip05': slip,
            'full_pf': r.get('full_2yr_pf', 0),
            'passed': passed,
        }
        if passed:
            passers[key].append(entry)
        candidates[key].append(entry)

    for key in candidates:
        if passers[key]:
            chosen = max(passers[key], key=lambda e: e['slip05'])
            best[key] = {**chosen, 'source': 'v4_pass'}
        else:
            chosen = max(candidates[key], key=lambda e: e['full_pf'])
            best[key] = {**chosen, 'source': 'fallback_fullpf'}
    return best


def run_one(strat, sym, sl, rr):
    cfg_path = _ROOT / "config" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["mode"] = "backtest"
    cfg["account"] = {"balance": 500.0, "max_position_pct": 0.95}
    engine = TradingEngine(cfg)

    bars = load_cached(sym, PERIOD_START, PERIOD_END)
    if bars is None: return []

    override = {
        "active_strategy": strat,
        "stop_loss_pct": sl, "reward_risk": rr,
        "volume_mult": 1.2,
        "regime_filter": True, "vix_threshold": 25,
        "max_trades_per_day": 100,
        "afternoon_entries": True,
        "entry_window_minutes": 360,
    }
    try:
        result = engine.run_backtest_config(
            config_override=override,
            start_date=PERIOD_START, end_date=PERIOD_END,
            symbol=sym, cached_bars=bars, cached_sma=load_sma(sym), cached_vix={},
        )
        return result.get("trades_list", [])
    except Exception as e:
        print(f"  ERROR: {e}")
        return []


def main():
    print("Picking best configs per (strategy, symbol)...")
    best_configs = pick_best_configs()
    print(f"Picked {len(best_configs)} configs")
    v4_pass_count = sum(1 for v in best_configs.values() if v['source'] == 'v4_pass')
    print(f"  V4-passed:      {v4_pass_count}")
    print(f"  Fallback (no V4 pass for that symbol+strat): {len(best_configs) - v4_pass_count}")

    # Print selected configs
    print("\nSelected configs:")
    print(f'{"strat":<14} {"sym":<6} {"sl":<5} {"rr":<5} {"slip05":<7} {"full_PF":<8} {"source"}')
    for (strat, sym), c in sorted(best_configs.items()):
        print(f'{strat:<14} {sym:<6} {c["sl"]:<5} {c["rr"]:<5} {c["slip05"]:<7.2f} {c["full_pf"]:<8.2f} {c["source"]}')

    # Run backtests
    print("\n" + "=" * 70)
    print("Running backtests (may take ~15-20 min)...")
    print("=" * 70)

    daily_pnl_by_key = {}
    for i, ((strat, sym), c) in enumerate(sorted(best_configs.items())):
        key = f'{strat[:5]}/{sym}'
        print(f"[{i+1}/{len(best_configs)}] {key}  SL={c['sl']} RR={c['rr']}", flush=True)
        trades = run_one(strat, sym, c['sl'], c['rr'])
        # Aggregate daily PnL
        day_pnl = defaultdict(float)
        for t in trades:
            et = pd.to_datetime(t["entry_time"])
            if et.tz is None: et = et.tz_localize("UTC").tz_convert(_ET)
            else: et = et.tz_convert(_ET)
            day_pnl[et.date()] += float(t["pnl"])
        daily_pnl_by_key[key] = day_pnl
        print(f"  trades={len(trades)}")

    # Build DataFrame
    all_days = set()
    for d in daily_pnl_by_key.values():
        all_days.update(d.keys())
    days_sorted = sorted(all_days)
    df = pd.DataFrame({
        k: [d.get(day, 0.0) for day in days_sorted]
        for k, d in daily_pnl_by_key.items()
    }, index=days_sorted)

    # Correlation matrix
    corr = df.corr()

    # Flatten to pairs
    pairs = []
    keys = list(df.columns)
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            pairs.append((keys[i], keys[j], corr.iloc[i, j]))

    pairs.sort(key=lambda p: p[2])

    print("\n" + "=" * 70)
    print("10 LEAST correlated pairs (best diversifiers)")
    print("=" * 70)
    for a, b, c in pairs[:10]:
        print(f"  {a:<14} vs {b:<14}  corr = {c:+.3f}")

    print("\n" + "=" * 70)
    print("10 MOST correlated pairs (worst diversifiers — avoid running both)")
    print("=" * 70)
    for a, b, c in pairs[-10:]:
        print(f"  {a:<14} vs {b:<14}  corr = {c:+.3f}")

    # Summary stats
    n = len(keys)
    total_pairs = n * (n - 1) // 2
    corrs = [p[2] for p in pairs]
    mean_corr = sum(corrs) / len(corrs)
    high_corr = sum(1 for c in corrs if c > 0.3)
    neg_corr = sum(1 for c in corrs if c < -0.1)
    print(f"\n{total_pairs} pairs. Mean corr = {mean_corr:+.3f}. >0.3: {high_corr} ({100*high_corr/total_pairs:.1f}%). <-0.1: {neg_corr} ({100*neg_corr/total_pairs:.1f}%)")

    # Cluster: for each symbol, how correlated are its 3 strategies
    print("\n" + "=" * 70)
    print("Same-symbol strategy clustering (do 3 strategies on same symbol move together?)")
    print("=" * 70)
    symbols = sorted(set(k.split('/')[1] for k in keys))
    for s in symbols:
        same_sym_keys = [k for k in keys if k.endswith('/' + s)]
        if len(same_sym_keys) < 2: continue
        sym_corrs = []
        for i in range(len(same_sym_keys)):
            for j in range(i+1, len(same_sym_keys)):
                sym_corrs.append(corr.loc[same_sym_keys[i], same_sym_keys[j]])
        avg = sum(sym_corrs) / len(sym_corrs)
        print(f"  {s:<8}  {len(same_sym_keys)} strategies  avg within-symbol corr = {avg:+.3f}")

    # Save
    out = _ROOT / "logs" / "research" / "correlation_study_full.json"
    output = {
        "period": [PERIOD_START, PERIOD_END],
        "configs": {f"{s}/{sy}": c for (s, sy), c in best_configs.items()},
        "trade_counts": {k: sum(1 for _ in d.keys()) for k, d in daily_pnl_by_key.items()},
        "correlation_matrix": corr.round(4).to_dict(),
        "least_correlated": [{"a": a, "b": b, "corr": float(c)} for a, b, c in pairs[:30]],
        "most_correlated": [{"a": a, "b": b, "corr": float(c)} for a, b, c in pairs[-30:]],
    }
    with open(out, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
