#!/usr/bin/env python3
"""
Correlation between existing strategies (ORB, EMA, Momentum) and the
3 new strategy archetypes (MACD, RSI mean reversion, VWAP Bands).

Run each on AMD + GOOGL on full_2yr (2023-2024). Compute daily PnL
correlation matrix. Strategies with low correlation to existing 3 are
real diversifiers.
"""
import os, sys, json, pickle
from pathlib import Path
from collections import defaultdict
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import pandas as pd, pytz, yaml
from src.engine import TradingEngine

_ET = pytz.timezone("America/New_York")
_ROOT = Path(__file__).resolve().parents[1]
CACHE = _ROOT / "logs" / "research" / "bar_cache"

SYMBOLS = ["AMD", "GOOGL"]
STRATEGIES = ["orb", "ema_crossover", "momentum", "macd_crossover", "rsi_reversion", "vwap_bands"]
PERIOD = ("2023-01-01", "2025-01-01")
# Mid-range, not optimized: SL=0.75% RR=3.0
DEFAULT_SL = 0.75
DEFAULT_RR = 3.0


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


def main():
    cfg = yaml.safe_load(open(_ROOT / "config/config.yaml"))
    cfg["mode"] = "backtest"
    cfg["account"] = {"balance": 500.0, "max_position_pct": 0.95}
    engine = TradingEngine(cfg)

    daily_pnl_by_key = {}
    print(f"Running {len(STRATEGIES)} strategies × {len(SYMBOLS)} symbols on full_2yr...\n")

    for sym in SYMBOLS:
        bars = load_cached(sym, *PERIOD)
        if bars is None:
            print(f"  {sym}: no bars, skipping")
            continue
        sma = load_sma(sym)
        for strat in STRATEGIES:
            key = f"{strat}/{sym}"
            print(f"  {key} ...", end="", flush=True)
            override = {
                "active_strategy": strat,
                "stop_loss_pct": DEFAULT_SL,
                "reward_risk":   DEFAULT_RR,
                "volume_mult":   1.2,
                "regime_filter": True,
                "vix_threshold": 25,
                "max_trades_per_day": 100,
                "afternoon_entries": True,
                "entry_window_minutes": 360,
            }
            try:
                res = engine.run_backtest_config(
                    config_override=override, start_date=PERIOD[0], end_date=PERIOD[1],
                    symbol=sym, cached_bars=bars, cached_sma=sma, cached_vix={},
                )
                trades = res.get("trades_list", [])
                day_pnl = defaultdict(float)
                for t in trades:
                    et = pd.to_datetime(t["entry_time"])
                    if et.tz is None: et = et.tz_localize("UTC").tz_convert(_ET)
                    else: et = et.tz_convert(_ET)
                    day_pnl[et.date()] += float(t["pnl"])
                daily_pnl_by_key[key] = day_pnl
                print(f" {len(trades)} trades, ${res['total_pnl']:+.2f}")
            except Exception as e:
                print(f" ERROR {e}")

    # Build DataFrame
    all_days = set()
    for d in daily_pnl_by_key.values():
        all_days.update(d.keys())
    days_sorted = sorted(all_days)
    df = pd.DataFrame({
        k: [d.get(day, 0.0) for day in days_sorted]
        for k, d in daily_pnl_by_key.items()
    }, index=days_sorted)

    corr = df.corr()

    print(f"\n=== Daily PnL correlation matrix ===")
    print(corr.round(3).to_string())

    # Print pairs sorted by absolute correlation
    pairs = []
    cols = list(df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], corr.iloc[i, j]))
    pairs.sort(key=lambda p: abs(p[2]))

    print(f"\n=== 10 LEAST correlated pairs (best diversifiers) ===")
    for a, b, c in pairs[:10]:
        print(f"  {a:<25} vs {b:<25}  corr = {c:+.3f}")

    print(f"\n=== Cross-correlation: NEW strategies vs OLD ===")
    new_strats = ["macd_crossover", "rsi_reversion", "vwap_bands"]
    old_strats = ["orb", "ema_crossover", "momentum"]
    for sym in SYMBOLS:
        print(f"\n  {sym}:")
        for new in new_strats:
            new_key = f"{new}/{sym}"
            if new_key not in df.columns: continue
            for old in old_strats:
                old_key = f"{old}/{sym}"
                if old_key not in df.columns: continue
                c = corr.loc[new_key, old_key]
                marker = "  ⭐ DIVERSIFIER" if abs(c) < 0.2 else "  (similar)" if abs(c) > 0.5 else ""
                print(f"    {new:<14} vs {old:<14}  corr = {c:+.3f}{marker}")

    # Save
    out = _ROOT / "logs/research/studies_2026_04_26/new_strategy_correlation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "strategies": STRATEGIES,
            "symbols": SYMBOLS,
            "period": list(PERIOD),
            "correlation": corr.round(4).to_dict(),
        }, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
