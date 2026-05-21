#!/usr/bin/env python3
"""
Study #2 + #7: Extended validate_recent.

Re-run validate_recent with the latest data (Apr 14 → Apr 26). Tests all
1,562 configs on the freshest snapshot — includes the 14 symbols we
dropped to confirm none have flipped to working.
"""
import os, sys, json, itertools, pickle
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import pandas as pd, pytz, yaml
from src.data.fetcher import DataFetcher
from src.engine import TradingEngine

_ET = pytz.timezone("America/New_York")
_ROOT = Path(__file__).resolve().parents[1]

SYMBOLS = ["AMD", "META", "COIN", "TSLA", "NVDA", "SPY", "QQQ", "AAPL",
           "AMZN", "MSFT", "GOOGL", "JPM", "MSTR", "IWM",
           "XLK", "XLF", "XLE", "XLV", "XLI"]
STRATEGIES = ["orb", "ema_crossover", "momentum"]
STOP_LOSSES = [0.3, 0.5, 0.75, 1.0]
REWARD_RISKS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
START = "2026-01-01"
END   = "2026-04-26"


def main():
    fetcher = DataFetcher({})
    cfg = yaml.safe_load(open(_ROOT / "config/config.yaml"))
    cfg["mode"] = "backtest"
    cfg["account"] = {"balance": 500.0, "max_position_pct": 0.95}
    engine = TradingEngine(cfg)

    print(f"=== Extended validate_recent ===")
    print(f"Period: {START} → {END}")
    print(f"Configs: {len(SYMBOLS) * len(STRATEGIES) * len(STOP_LOSSES) * len(REWARD_RISKS)}")
    print()

    out_dir = _ROOT / "logs/research/studies_2026_04_26/validate_recent_extended"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for sym_i, sym in enumerate(SYMBOLS):
        print(f"[{sym_i+1}/{len(SYMBOLS)}] {sym}", end="", flush=True)
        try:
            bars = fetcher.fetch_historical_alpaca(sym, START, END, filter_windows=False)
            print(f" — {len(bars) if bars is not None else 0} bars")
        except Exception as e:
            print(f" — fetch FAILED: {e}")
            continue
        if bars is None or bars.empty:
            print(f"  no bars")
            continue

        for strat in STRATEGIES:
            for sl in STOP_LOSSES:
                for rr in REWARD_RISKS:
                    override = {
                        "active_strategy": strat, "stop_loss_pct": sl, "reward_risk": rr,
                        "volume_mult": 1.2, "regime_filter": True, "vix_threshold": 25,
                        "max_trades_per_day": 100, "afternoon_entries": True,
                        "entry_window_minutes": 360,
                    }
                    try:
                        res = engine.run_backtest_config(
                            config_override=override,
                            start_date=START, end_date=END,
                            symbol=sym, cached_bars=bars, cached_sma={}, cached_vix={},
                        )
                        if res["total_trades"] >= 5:
                            results.append({
                                "strategy": strat, "symbol": sym, "sl": sl, "rr": rr,
                                "trades": res["total_trades"],
                                "pf": round(res["profit_factor"], 3),
                                "wr": round(res["win_rate"], 1),
                                "pnl": round(res["total_pnl"], 2),
                            })
                    except: pass

        # Save incremental
        with open(_ROOT / "logs/research/studies_2026_04_26/validate_recent_extended.json", "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    print(f"\n=== Summary ===")
    profitable = [r for r in results if r["pf"] > 1.0]
    print(f"Total configs with ≥5 trades: {len(results)}")
    print(f"Profitable (PF > 1.0): {len(profitable)}")

    # Top 20 by PnL
    profitable.sort(key=lambda r: -r["pnl"])
    print(f"\nTop 20 (by PnL on extended period):")
    print(f"  {'strategy':<14} {'sym':<6} {'sl':<5} {'rr':<5} {'trades':<7} {'pf':<6} {'wr':<6} {'pnl':<8}")
    for r in profitable[:20]:
        print(f"  {r['strategy']:<14} {r['symbol']:<6} {r['sl']:<5} {r['rr']:<5} "
              f"{r['trades']:<7} {r['pf']:<6.2f} {r['wr']:<6.1f} ${r['pnl']:<+7.2f}")

    # Best per symbol
    print(f"\nBEST per symbol (≥10 trades):")
    by_sym = defaultdict(list)
    for r in profitable:
        if r["trades"] >= 10:
            by_sym[r["symbol"]].append(r)
    for s in sorted(by_sym):
        best = max(by_sym[s], key=lambda r: r["pnl"])
        print(f"  {s:<8} {best['strategy']:<14} SL={best['sl']} RR={best['rr']}  "
              f"trades={best['trades']} pf={best['pf']:.2f} pnl=${best['pnl']:+.2f}")


if __name__ == "__main__":
    main()
