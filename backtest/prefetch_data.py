#!/usr/bin/env python3
"""
Pre-fetch all historical bars for all symbols and cache to disk.
Run once, then validation uses cached data — no API calls during validation.
"""
import os, sys, pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
for p in [_ROOT / ".env", Path.home() / "projects" / "dl_course" / ".env",
          Path.home() / "projects" / "day-trading-bot" / ".env"]:
    if p.exists():
        load_dotenv(p)
        break

from src.data.fetcher import DataFetcher
from src.engine import TradingEngine
import yaml

SYMBOLS = ["AMD", "META", "COIN", "TSLA", "NVDA", "SPY", "QQQ", "AAPL",
           "AMZN", "MSFT", "GOOGL", "JPM", "MSTR", "IWM",
           "XLK", "XLF", "XLE", "XLV", "XLI"]

# All date ranges needed by walk-forward + MC + slippage
DATE_RANGES = [
    ("2022-01-01", "2022-09-30"),   # WF W1-W2 train+test
    ("2022-04-01", "2022-12-31"),   # WF W2-W3
    ("2022-07-01", "2023-03-31"),   # WF W3-W4
    ("2022-10-01", "2023-06-30"),   # WF W4-W5
    ("2023-01-01", "2023-09-30"),   # WF W5-W6
    ("2023-04-01", "2023-12-31"),   # WF W6-W7
    ("2023-07-01", "2024-03-31"),   # WF W7-W8
    ("2023-10-01", "2024-06-30"),   # WF W8-W9
    ("2024-01-01", "2024-09-30"),   # WF W9-W10
    ("2024-04-01", "2024-12-31"),   # WF W10
    ("2023-01-01", "2025-01-01"),   # MC + full_2yr
    ("2022-01-01", "2022-12-31"),   # holdout
    ("2025-06-01", "2026-03-31"),   # bear_2025
]

def main():
    cache_dir = _ROOT / "logs" / "research" / "bar_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    fetcher = DataFetcher()

    with open(_ROOT / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["mode"] = "backtest"
    cfg["account"] = {"balance": 500.0, "max_position_pct": 0.95}
    engine = TradingEngine(cfg)

    total = len(SYMBOLS) * len(DATE_RANGES)
    done = 0
    cached = 0
    errors = 0

    for sym in SYMBOLS:
        # Also cache SMA
        try:
            sma = engine._compute_daily_sma(sym, "2022-01-01", "2025-01-01", 20)
            sma_path = cache_dir / f"{sym}_sma.pkl"
            with open(sma_path, "wb") as f:
                pickle.dump(sma, f)
        except Exception as e:
            print(f"  SMA error for {sym}: {e}")

        for start, end in DATE_RANGES:
            done += 1
            cache_key = f"{sym}_{start}_{end}.pkl"
            cache_path = cache_dir / cache_key

            if cache_path.exists():
                cached += 1
                continue

            try:
                bars = fetcher.fetch_historical(sym, start, end, filter_windows=False)
                if bars is not None and not bars.empty:
                    with open(cache_path, "wb") as f:
                        pickle.dump(bars, f)
                    print(f"  [{done}/{total}] {sym} {start}→{end}: {len(bars)} bars ✓")
                else:
                    # Save empty marker
                    with open(cache_path, "wb") as f:
                        pickle.dump(None, f)
                    print(f"  [{done}/{total}] {sym} {start}→{end}: no data")
            except Exception as e:
                errors += 1
                print(f"  [{done}/{total}] {sym} {start}→{end}: ERROR {e}")

            # Rate limit: sleep between API calls
            import time
            time.sleep(0.5)

    print(f"\nDone: {done} fetched, {cached} already cached, {errors} errors")
    print(f"Cache dir: {cache_dir}")
    print(f"Total files: {len(list(cache_dir.glob('*.pkl')))}")


if __name__ == "__main__":
    main()
