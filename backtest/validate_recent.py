#!/usr/bin/env python3
"""
Validate ALL configs on RECENT data (Oct 2025 - Apr 2026).
Tests: backtest on Jan-Apr 2026, walk-forward with Oct-Dec train / Jan-Apr test.
Uses cached bars — no API calls.
"""
import os, sys, json, pickle, itertools
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import yaml
from src.engine import TradingEngine

_ROOT = Path(__file__).resolve().parents[1]

SYMBOLS = ["AMD", "META", "COIN", "TSLA", "NVDA", "SPY", "QQQ", "AAPL",
           "AMZN", "MSFT", "GOOGL", "JPM", "MSTR", "IWM",
           "XLK", "XLF", "XLE", "XLV", "XLI"]
STRATEGIES = ["orb", "ema_crossover", "momentum"]
STOP_LOSSES = [0.3, 0.5, 0.75, 1.0]
REWARD_RISKS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

cache_dir = _ROOT / "logs" / "research" / "bar_cache"


def load_cached(sym, start, end):
    import pandas as pd
    pq = cache_dir / f"{sym}_{start}_{end}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    p = cache_dir / f"{sym}_{start}_{end}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def load_sma(sym):
    jf = cache_dir / f"{sym}_sma.json"
    if jf.exists():
        data = json.load(open(jf))
        return {date.fromisoformat(k): v for k, v in data.items()}
    p = cache_dir / f"{sym}_sma.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return {}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", default="1/1")
    args = parser.parse_args()

    with open(_ROOT / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["mode"] = "backtest"
    cfg["account"] = {"balance": 500.0, "max_position_pct": 0.95}
    engine = TradingEngine(cfg)

    all_symbols = list(SYMBOLS)
    slice_n, slice_m = map(int, args.slice.split("/"))
    chunk = len(all_symbols) // slice_m
    start_idx = (slice_n - 1) * chunk
    end_idx = start_idx + chunk if slice_n < slice_m else len(all_symbols)
    symbols_slice = all_symbols[start_idx:end_idx]

    all_configs = list(itertools.product(STRATEGIES, symbols_slice, STOP_LOSSES, REWARD_RISKS))
    print(f"Slice {slice_n}/{slice_m}: {len(symbols_slice)} symbols, {len(all_configs)} configs")
    print(f"Symbols: {symbols_slice}")
    print(f"Testing on RECENT data (Jan-Apr 2026)")
    print()

    out_path = _ROOT / "logs" / "research" / f"validation_recent_s{slice_n}.json"
    results = []
    done = 0

    # Group configs by symbol so we load bars once per symbol
    from collections import defaultdict
    by_symbol = defaultdict(list)
    for strat, sym, sl, rr in all_configs:
        by_symbol[sym].append((strat, sl, rr))

    for sym in symbols_slice:
        if sym not in by_symbol:
            continue

        # Load bars once for this symbol
        recent_bars = load_cached(sym, "2026-01-01", "2026-04-14")
        full_bars = load_cached(sym, "2025-10-01", "2026-04-14")
        sma = load_sma(sym)

        if recent_bars is None or recent_bars.empty:
            done += len(by_symbol[sym])
            print(f"  {sym}: no data, skipping {len(by_symbol[sym])} configs")
            continue

        print(f"  {sym}: {len(recent_bars)} bars, testing {len(by_symbol[sym])} configs...", flush=True)

        for strat, sl, rr in by_symbol[sym]:
            done += 1

            base = {
                "active_strategy": strat, "stop_loss_pct": sl, "reward_risk": rr,
                "volume_mult": 1.2, "regime_filter": True, "vix_threshold": 25,
                "max_trades_per_day": 100, "afternoon_entries": True, "entry_window_minutes": 360,
            }

            result = {
                "strategy": strat, "symbol": sym, "sl": sl, "rr": rr,
                "recent_trades": 0, "recent_pf": 0, "recent_wr": 0, "recent_pnl": 0,
                "recent_avg_dur": 0,
                "wf_recent_train_pf": 0, "wf_recent_test_pf": 0,
            }

            m = engine.run_backtest_config(
                config_override=base,
                start_date="2026-01-01", end_date="2026-04-14",
                symbol=sym, cached_bars=recent_bars, cached_sma=sma, cached_vix={},
            )
            result["recent_trades"] = m.get("total_trades", 0)
            result["recent_pf"] = round(m.get("profit_factor", 0), 2)
            result["recent_wr"] = round(m.get("win_rate", 0), 1)
            result["recent_pnl"] = round(m.get("total_pnl", 0), 2)
            result["recent_avg_dur"] = round(m.get("avg_duration_min", 0), 0)

            # 2. Walk-forward: train Oct-Dec 2025, test Jan-Apr 2026
            if full_bars is not None and not full_bars.empty:
                import pytz
                _ET = pytz.timezone("America/New_York")
                ts = full_bars["timestamp"]
                if ts.dt.tz is not None:
                    ts = ts.dt.tz_convert(_ET)

                train_mask = ts < "2026-01-01"
                test_mask = ts >= "2026-01-01"
                train_bars = full_bars[train_mask]
                test_bars = full_bars[test_mask]

                if not train_bars.empty and not test_bars.empty:
                    best_pf = 0
                    best_p = None
                    for p_sl, p_rr in itertools.product(STOP_LOSSES, REWARD_RISKS):
                        try:
                            tm = engine.run_backtest_config(
                                config_override={**base, "stop_loss_pct": p_sl, "reward_risk": p_rr},
                                start_date="2025-10-01", end_date="2025-12-31",
                                symbol=sym, cached_bars=train_bars, cached_sma=sma, cached_vix={},
                            )
                            pf = tm.get("profit_factor", 0)
                            if pf > best_pf and tm.get("total_trades", 0) >= 5:
                                best_pf = pf
                                best_p = (p_sl, p_rr)
                        except:
                            pass

                    result["wf_recent_train_pf"] = round(best_pf, 2)

                    if best_p:
                        try:
                            tm = engine.run_backtest_config(
                                config_override={**base, "stop_loss_pct": best_p[0], "reward_risk": best_p[1]},
                                start_date="2026-01-01", end_date="2026-04-14",
                                symbol=sym, cached_bars=test_bars, cached_sma=sma, cached_vix={},
                            )
                            result["wf_recent_test_pf"] = round(tm.get("profit_factor", 0), 2)
                        except:
                            pass

            if result["recent_trades"] >= 5:
                results.append(result)

            if done % 50 == 0:
                profitable = sum(1 for r in results if r["recent_pf"] > 1.0)
                print(f"    [{done}/{len(all_configs)}] tested={len(results)} profitable={profitable}")
                # Save checkpoint
                with open(out_path, "w") as _f:
                    json.dump(results, _f, indent=2)

        # Free memory after each symbol
        del recent_bars, full_bars, sma
        import gc; gc.collect()

    # Save
    out_path = _ROOT / "logs" / "research" / "validation_recent.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print(f"  RECENT VALIDATION COMPLETE — {len(results)} configs with ≥5 trades")
    print(f"{'='*70}")

    profitable = [r for r in results if r["recent_pf"] > 1.0]
    wf_pass = [r for r in results if r["wf_recent_test_pf"] > 1.0]

    print(f"  Profitable on Jan-Apr 2026: {len(profitable)}/{len(results)}")
    print(f"  Walk-forward pass (recent): {len(wf_pass)}/{len(results)}")

    # Best by symbol
    from collections import defaultdict
    by_sym = defaultdict(list)
    for r in profitable:
        by_sym[r["symbol"]].append(r)

    print(f"\n  BEST PER SYMBOL (Jan-Apr 2026, profitable):")
    print(f"  {'Strategy':<16} {'Symbol':<8} {'SL':<5} {'RR':<5} {'Trades':<7} {'WR%':<6} {'PnL':>8} {'PF':>6} {'Dur':>5} {'WF test':>8}")
    print(f"  {'-'*80}")
    for sym in sorted(by_sym.keys()):
        best = sorted(by_sym[sym], key=lambda x: -x["recent_pf"])[0]
        print(f"  {best['strategy']:<16} {best['symbol']:<8} {best['sl']:<5} {best['rr']:<5} "
              f"{best['recent_trades']:<7} {best['recent_wr']:<6} ${best['recent_pnl']:>+7.2f} "
              f"{best['recent_pf']:>6.2f} {best['recent_avg_dur']:>4.0f}m {best['wf_recent_test_pf']:>8.2f}")

    # Cross-reference with validated configs
    print(f"\n  OUR LIVE CONFIG ON RECENT DATA:")
    live = [
        ("orb", "AMD", 0.5, 5.0),
        ("momentum", "META", 0.75, 4.0),
        ("orb", "COIN", 1.0, 5.0),
        ("ema_crossover", "TSLA", 0.75, 3.0),
    ]
    for strat, sym, sl, rr in live:
        match = [r for r in results if r["strategy"] == strat and r["symbol"] == sym
                 and r["sl"] == sl and r["rr"] == rr]
        if match:
            r = match[0]
            status = "✅" if r["recent_pf"] > 1.0 else "❌"
            print(f"  {status} {strat}/{sym} SL={sl} RR={rr}: {r['recent_trades']} trades, "
                  f"PF={r['recent_pf']}, PnL=${r['recent_pnl']:+.2f}, WR={r['recent_wr']}%")
        else:
            print(f"  ? {strat}/{sym} SL={sl} RR={rr}: no data")


if __name__ == "__main__":
    main()
