#!/usr/bin/env python3
"""
M4: Monte Carlo Permutation Test — is our edge real or just market drift?

For each symbol, runs N backtests where entries are placed at RANDOM times
within the 9:30-10:30 window (instead of strategy-generated signals). If
random entries produce PF > 1.0, our "edge" is just buying in a bull market.
If our strategy PF is in the top 5% of random PFs → edge is statistically
significant (p < 0.05).

Uses a custom RandomEntryStrategy that enters at a random minute each day.
"""
import os, sys, random, time as _time_mod
from pathlib import Path
from datetime import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
import pandas as pd
import numpy as np
from typing import Dict, Optional

_ROOT = Path(__file__).resolve().parents[1]


class RandomEntryStrategy:
    """Enter at a random bar between 9:30 and 10:30 ET — one per day."""
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self._fired = False
        self._target_minute = 0

    @property
    def name(self):
        return "RandomEntry"

    def reset_session(self):
        self._fired = False
        # Pick a random minute between 9:30 (570) and 10:30 (630)
        self._target_minute = self.rng.randint(570, 630)

    def set_prev_close(self, *a): pass
    def set_prev_or_volume(self, *a): pass
    def set_regime_sma(self, *a): pass

    def generate_signal(self, df, idx) -> Optional[Dict]:
        if self._fired or idx < 0 or idx >= len(df):
            return None
        bar = df.iloc[idx]
        import pytz
        _ET = pytz.timezone("America/New_York")
        bt = bar["timestamp"].astimezone(_ET)
        bar_min = bt.hour * 60 + bt.minute
        if bar_min == self._target_minute:
            self._fired = True
            return {
                "signal": "BUY",
                "symbol": None,
                "entry_price": float(bar["close"]),
                "bar_time": bar["timestamp"],
                "reason": "random entry",
            }
        return None


def run_one_backtest(engine, symbol, start, end, sl, rr, seed, cached_bars=None):
    """Run one backtest with random entry."""
    import copy, math
    from src.risk.manager import RiskManager
    from src.broker.simulator import SimulatorBroker

    cfg = copy.deepcopy(engine.config)
    cfg["risk"]["stop_loss_pct"] = sl
    cfg["risk"]["reward_risk_ratio"] = rr

    strategy = RandomEntryStrategy(seed)
    risk_mgr = RiskManager(cfg)
    broker = SimulatorBroker(cfg)

    start_cash = float(cfg.get("account", {}).get("balance", 500.0))
    max_pos_pct = float(cfg.get("account", {}).get("max_position_pct", 0.95))
    import pytz
    _ET = pytz.timezone("America/New_York")

    if cached_bars is not None and not cached_bars.empty:
        full_df = cached_bars
    else:
        full_df = engine.fetcher.fetch_historical(symbol, start, end, filter_windows=False)
    if full_df.empty:
        return {"pf": 0, "trades": 0, "pnl": 0, "wr": 0}

    days = engine._split_by_day(full_df)
    trades_list = []
    peak = start_cash
    max_dd = 0

    for trading_date, day_df in sorted(days.items()):
        broker.reset_daily()
        strategy.reset_session()
        open_pos = None
        for i in range(len(day_df)):
            bar = day_df.iloc[i]
            bar_price = float(bar["close"])
            bar_dt = bar["timestamp"]
            if open_pos is None:
                signal = strategy.generate_signal(day_df, i)
                if signal:
                    signal["symbol"] = symbol
                    bp = math.floor(broker.cash * max_pos_pct * 100) / 100
                    pos = risk_mgr.calculate_position(signal, broker.cash, buying_power=bp)
                    if pos:
                        pos["entry_time"] = bar_dt
                        broker.submit_order(pos)
                        open_pos = pos
            elif open_pos is not None:
                exit_r = risk_mgr.check_exit(open_pos, bar_price, bar_dt)
                if exit_r:
                    open_pos["exit_time"] = bar_dt
                    trade = broker.close_position(open_pos, exit_r["exit_price"], exit_r["reason"])
                    trades_list.append(trade)
                    open_pos = None
        if open_pos is not None:
            last = day_df.iloc[-1]
            open_pos["exit_time"] = last["timestamp"]
            trade = broker.close_position(open_pos, float(last["close"]), "eod_exit")
            trades_list.append(trade)
        bal = broker.cash
        if bal > peak: peak = bal
        if peak > 0:
            dd = (peak - bal) / peak * 100
            if dd > max_dd: max_dd = dd

    if not trades_list:
        return {"pf": 0, "trades": 0, "pnl": 0, "wr": 0}
    pnls = [t["pnl"] for t in trades_list]
    wins = sum(1 for p in pnls if p > 0)
    gross_w = sum(p for p in pnls if p > 0)
    gross_l = abs(sum(p for p in pnls if p <= 0))
    pf = gross_w / gross_l if gross_l > 0 else 999
    return {"pf": min(pf, 999), "trades": len(pnls), "pnl": sum(pnls),
            "wr": wins/len(pnls)*100, "dd": max_dd}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="number of random permutations")
    parser.add_argument("--symbol", default="NVDA")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--sl", type=float, default=0.3)
    parser.add_argument("--rr", type=float, default=2.5)
    args = parser.parse_args()

    with open(_ROOT / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["mode"] = "backtest"

    from src.engine import TradingEngine
    engine = TradingEngine(cfg)

    print(f"\nM4: MONTE CARLO PERMUTATION TEST")
    print(f"  Symbol: {args.symbol}  Period: {args.start} → {args.end}")
    print(f"  Stop: {args.sl}%  R:R: {args.rr}")
    print(f"  Permutations: {args.n}\n")

    # First run the REAL strategy
    print("  Running REAL ORB strategy...")
    real = engine.run_backtest_config(
        config_override={
            "active_strategy": "orb", "stop_loss_pct": args.sl,
            "reward_risk": args.rr, "volume_mult": 1.2,
            "regime_filter": True, "vix_threshold": 30,
            "max_trades_per_day": 1, "afternoon_entries": False,
        },
        start_date=args.start, end_date=args.end, symbol=args.symbol,
    )
    real_pf = min(real["profit_factor"], 999)
    print(f"  REAL ORB:  PF={real_pf:.2f}  WR={real['win_rate']:.1f}%  "
          f"P&L=${real['total_pnl']:+.2f}  trades={real['total_trades']}")

    # Fetch bars once
    bars = engine.fetcher.fetch_historical(args.symbol, args.start, args.end, filter_windows=False)

    # Run N random permutations
    print(f"\n  Running {args.n} random-entry permutations...")
    t0 = _time_mod.time()
    random_pfs = []
    for i in range(args.n):
        r = run_one_backtest(engine, args.symbol, args.start, args.end,
                             args.sl, args.rr, seed=42+i, cached_bars=bars)
        random_pfs.append(r["pf"])
        if (i+1) % 25 == 0:
            print(f"    [{i+1}/{args.n}] avg random PF={np.mean(random_pfs):.2f}")
    elapsed = _time_mod.time() - t0

    random_pfs = np.array(random_pfs)
    # Percentile rank of real PF among random PFs
    pct_rank = (random_pfs < real_pf).sum() / len(random_pfs) * 100
    p_value = 1 - pct_rank / 100

    print(f"\n{'='*70}")
    print(f"  MONTE CARLO RESULTS ({args.n} permutations, {elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"  REAL ORB PF:         {real_pf:.2f}")
    print(f"  Random entry avg PF: {random_pfs.mean():.2f}")
    print(f"  Random entry std PF: {random_pfs.std():.2f}")
    print(f"  Random entry min PF: {random_pfs.min():.2f}")
    print(f"  Random entry max PF: {random_pfs.max():.2f}")
    print(f"  Random PF > 1.0:     {(random_pfs > 1.0).sum()}/{args.n} ({(random_pfs > 1.0).mean()*100:.0f}%)")
    print(f"  Percentile rank:     {pct_rank:.1f}% (real PF beats {pct_rank:.0f}% of random)")
    print(f"  p-value:             {p_value:.3f}")
    print()
    if p_value < 0.05:
        print(f"  ✅ STATISTICALLY SIGNIFICANT (p={p_value:.3f} < 0.05)")
        print(f"     ORB's entry signal provides a REAL edge beyond random chance.")
    elif p_value < 0.10:
        print(f"  🟡 MARGINALLY SIGNIFICANT (p={p_value:.3f})")
        print(f"     Some evidence of edge, but could be noise. More data needed.")
    else:
        print(f"  ❌ NOT SIGNIFICANT (p={p_value:.3f} ≥ 0.10)")
        print(f"     ORB's PF is NOT meaningfully better than random entries.")
        print(f"     The 'edge' may be market drift (buying in a bull market).")

    # Check if random entries are profitable on their own
    if random_pfs.mean() > 1.0:
        print(f"\n  ⚠️  NOTE: random entries average PF={random_pfs.mean():.2f} > 1.0")
        print(f"     This means ANY buy in this period tends to be profitable.")
        print(f"     Our strategy's edge is the DIFFERENCE: {real_pf:.2f} - {random_pfs.mean():.2f} = {real_pf - random_pfs.mean():+.2f}")


if __name__ == "__main__":
    main()
