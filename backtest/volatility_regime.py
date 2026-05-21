#!/usr/bin/env python3
"""
Study #3: Volatility regime analysis.

Hypothesis: AMD's 2-week underperformance is because intraday volatility
went up, causing tight stops to clip more often.

Compute daily ATR for AMD over the past 6 months. Compare with V4 trade
outcomes. If high-ATR days correlate with stop-outs, dynamic stop sizing
would help.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import pandas as pd
import pytz
import numpy as np
from src.data.fetcher import DataFetcher

_ET = pytz.timezone("America/New_York")
_ROOT = Path(__file__).resolve().parents[1]


def fetch_daily(sym, start, end):
    fetcher = DataFetcher({})
    bars = fetcher.fetch_historical_alpaca(sym, start, end, filter_windows=False)
    if bars is None or bars.empty:
        return None
    ts = pd.to_datetime(bars["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    bars["ts_et"] = ts.dt.tz_convert(_ET)
    bars["date"]  = bars["ts_et"].dt.date
    # Daily aggregation
    daily = bars.groupby("date").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()
    return daily, bars


def compute_atr(daily, period=14):
    """Compute daily ATR."""
    daily = daily.copy()
    daily["tr"] = pd.concat([
        daily["high"] - daily["low"],
        (daily["high"] - daily["close"].shift()).abs(),
        (daily["low"]  - daily["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    daily["atr"]     = daily["tr"].rolling(period, min_periods=1).mean()
    daily["atr_pct"] = daily["atr"] / daily["close"] * 100
    return daily


def main():
    print("Fetching AMD bars (2025-10-01 → 2026-04-26)...")
    daily, bars = fetch_daily("AMD", "2025-10-01", "2026-04-26")
    daily = compute_atr(daily, period=14)

    print(f"Total trading days: {len(daily)}")
    print()

    # Analyze ATR regime by 30-day windows
    print("=== AMD daily ATR by month ===")
    daily["year_month"] = pd.to_datetime(daily["date"]).dt.to_period("M")
    monthly = daily.groupby("year_month").agg(
        days=("date", "count"),
        avg_atr=("atr", "mean"),
        avg_atr_pct=("atr_pct", "mean"),
        max_atr_pct=("atr_pct", "max"),
        avg_volume=("volume", "mean"),
    )
    print(monthly.round(3).to_string())

    # Specifically: what was ATR during the 2 weeks of underperformance vs the
    # earlier "validated" period?
    print()
    print("=== Comparison: validated period vs last 2 weeks ===")
    val_period = daily[(daily["date"] >= pd.to_datetime("2026-01-01").date()) &
                       (daily["date"] <  pd.to_datetime("2026-04-12").date())]
    recent = daily[daily["date"] >= pd.to_datetime("2026-04-13").date()]

    print(f"  Validated period (Jan-Apr 11):  n={len(val_period)} days")
    print(f"    Avg ATR%: {val_period['atr_pct'].mean():.3f}%")
    print(f"    Med ATR%: {val_period['atr_pct'].median():.3f}%")
    print(f"    Max ATR%: {val_period['atr_pct'].max():.3f}%")
    print(f"  Recent (Apr 13 onwards):        n={len(recent)} days")
    print(f"    Avg ATR%: {recent['atr_pct'].mean():.3f}%")
    print(f"    Med ATR%: {recent['atr_pct'].median():.3f}%")
    print(f"    Max ATR%: {recent['atr_pct'].max():.3f}%")

    diff = recent['atr_pct'].mean() - val_period['atr_pct'].mean()
    pct_change = diff / val_period['atr_pct'].mean() * 100
    print(f"  → ATR change: {diff:+.3f}% absolute ({pct_change:+.1f}% relative)")

    # AMD ORB SL=0.3% — does the recent ATR exceed the stop?
    print()
    print("=== Stop-loss vs ATR check ===")
    print(f"  AMD ORB SL=0.3% means stops trigger at 0.3% adverse move.")
    print(f"  If daily ATR% > 2 × SL%, stops are likely clipped on noise alone.")
    print(f"    Validated period: ATR/SL ratio = {val_period['atr_pct'].mean() / 0.3:.2f}x")
    print(f"    Recent period:    ATR/SL ratio = {recent['atr_pct'].mean() / 0.3:.2f}x")
    print()
    if recent['atr_pct'].mean() > 2 * 0.3:
        print(f"  ⚠️  Recent ATR exceeds 2× the 0.3% stop. Tighter stops are getting clipped on noise.")
    if val_period['atr_pct'].mean() > 2 * 0.3:
        print(f"  ⚠️  Even validated period had high ATR/SL. Why did AMD work earlier?")
    print()

    # Per-day printout for last 2 weeks
    print("=== AMD last 2 weeks ===")
    last_2w = daily.tail(15)
    for _, r in last_2w.iterrows():
        print(f"  {r['date']}: open={r['open']:.2f} high={r['high']:.2f} low={r['low']:.2f} close={r['close']:.2f}  "
              f"day_range%={(r['high']-r['low'])/r['close']*100:.2f}%  ATR={r['atr_pct']:.2f}%")

    # Save daily ATR series
    out = _ROOT / 'logs/research/studies_2026_04_26/amd_volatility.csv'
    daily.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
