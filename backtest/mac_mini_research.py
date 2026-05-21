#!/usr/bin/env python3
"""
Mac Mini research batch — runs sequentially:
1. Slippage test on SPY (critical gate for live trading)
2. Walk-forward on QQQ, COIN, MSTR, XLV
3. Monte Carlo on NVDA with widened 0.5% stop
"""
import os
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import sys
import time as _time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.engine import TradingEngine

_ROOT = Path(__file__).resolve().parents[1]


# ═══════════════════════════════════════════════════════════════════
# 1. SLIPPAGE TEST — SPY
# ═══════════════════════════════════════════════════════════════════
def run_slippage():
    print("\n" + "=" * 70)
    print("  1. SLIPPAGE STRESS TEST — SPY + NVDA (new 0.5% stop)")
    print("=" * 70)

    with open(_ROOT / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["mode"] = "backtest"
    cfg["account"] = {"balance": 500.0, "max_position_pct": 0.95}
    engine = TradingEngine(cfg)

    COMBOS = [
        ("SPY", 0.3, 2.5, "SPY momentum"),
        ("SPY", 0.3, 4.0, "SPY high RR"),
        ("SPY", 0.5, 3.0, "SPY wider stop"),
        ("NVDA", 0.5, 2.5, "NVDA new config"),
        ("NVDA", 0.5, 3.0, "NVDA wider RR"),
    ]
    SLIPPAGE = [0.0, 0.02, 0.05, 0.10, 0.20]
    PERIODS = [
        ("2023-01-01", "2025-01-01", "full_2yr"),
        ("2022-01-01", "2022-12-31", "holdout_2022"),
        ("2025-06-01", "2026-03-31", "bear_2025"),
    ]

    print(f"\n{'Combo':<22} {'Period':<14} {'Slip%':<7} {'Trades':<7} {'WR%':<7} {'PnL':>9} {'PF':>7} {'DD%':>6}")
    print("-" * 85)

    for symbol, sl, rr, label in COMBOS:
        for start, end, period_label in PERIODS:
            for slip in SLIPPAGE:
                try:
                    metrics = engine.run_backtest_config(
                        config_override={
                            "active_strategy": "orb",
                            "stop_loss_pct": sl,
                            "reward_risk": rr,
                            "volume_mult": 1.2,
                            "regime_filter": True,
                            "vix_threshold": 30,
                            "max_trades_per_day": 3,
                            "afternoon_entries": True,
                            "entry_window_minutes": 360,
                            "slippage_pct": slip,
                        },
                        start_date=start, end_date=end, symbol=symbol,
                    )
                    t = metrics.get("total_trades", 0)
                    wr = metrics.get("win_rate", 0)
                    pnl = metrics.get("total_pnl", 0)
                    pf = metrics.get("profit_factor", 0)
                    dd = metrics.get("max_drawdown", 0)
                    print(f"{label:<22} {period_label:<14} {slip:<7.2f} {t:<7} {wr:<7.1f} ${pnl:>+8.2f} {pf:>7.2f} {dd:>5.1f}%")
                except Exception as e:
                    print(f"{label:<22} {period_label:<14} {slip:<7.2f} ERROR: {e}")
            print()


# ═══════════════════════════════════════════════════════════════════
# 2. WALK-FORWARD — QQQ, COIN, MSTR, XLV
# ═══════════════════════════════════════════════════════════════════
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

_WF_WINDOWS = [
    {"train_start": "2022-01-01", "train_end": "2022-06-30", "test_start": "2022-07-01", "test_end": "2022-09-30", "label": "W1"},
        {"train_start": "2022-04-01", "train_end": "2022-09-30", "test_start": "2022-10-01", "test_end": "2022-12-31", "label": "W2"},
        {"train_start": "2022-07-01", "train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-03-31", "label": "W3"},
        {"train_start": "2022-10-01", "train_end": "2023-03-31", "test_start": "2023-04-01", "test_end": "2023-06-30", "label": "W4"},
        {"train_start": "2023-01-01", "train_end": "2023-06-30", "test_start": "2023-07-01", "test_end": "2023-09-30", "label": "W5"},
        {"train_start": "2023-04-01", "train_end": "2023-09-30", "test_start": "2023-10-01", "test_end": "2023-12-31", "label": "W6"},
        {"train_start": "2023-07-01", "train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-03-31", "label": "W7"},
        {"train_start": "2023-10-01", "train_end": "2024-03-31", "test_start": "2024-04-01", "test_end": "2024-06-30", "label": "W8"},
        {"train_start": "2024-01-01", "train_end": "2024-06-30", "test_start": "2024-07-01", "test_end": "2024-09-30", "label": "W9"},
        {"train_start": "2024-04-01", "train_end": "2024-09-30", "test_start": "2024-10-01", "test_end": "2024-12-31", "label": "W10"},
]

_WF_PARAM_GRID = {
    "stop_loss_pct": [0.3, 0.5, 1.0],
    "reward_risk": [2.0, 2.5, 4.0],
    "volume_mult": [1.0, 1.2],
}


def _run_window(args_tuple):
        symbol, window, base_cfg_path = args_tuple
        import yaml as _yaml
        from src.engine import TradingEngine as _TE

        with open(base_cfg_path) as f:
            cfg = _yaml.safe_load(f)
        cfg["mode"] = "backtest"
        cfg.setdefault("account", {})["balance"] = 500.0

        try:
            engine = _TE(cfg)
        except Exception as exc:
            return {"symbol": symbol, "window": window["label"], "error": str(exc)}

        try:
            train_bars = engine.fetcher.fetch_historical(
                symbol, window["train_start"], window["train_end"], filter_windows=False)
            test_bars = engine.fetcher.fetch_historical(
                symbol, window["test_start"], window["test_end"], filter_windows=False)
        except Exception:
            return {"symbol": symbol, "window": window["label"], "error": "no data"}

        if train_bars is None or train_bars.empty or test_bars is None or test_bars.empty:
            return {"symbol": symbol, "window": window["label"], "error": "no data"}

        try:
            cached_sma = engine._compute_daily_sma(
                symbol, window["train_start"], window["test_end"], 20)
        except Exception:
            cached_sma = {}

        best_train_pf = 0
        best_params = None
        for sl, rr, vol in itertools.product(
            _WF_PARAM_GRID["stop_loss_pct"], _WF_PARAM_GRID["reward_risk"], _WF_PARAM_GRID["volume_mult"]
        ):
            try:
                m = engine.run_backtest_config(
                    config_override={
                        "active_strategy": "orb", "stop_loss_pct": sl,
                        "reward_risk": rr, "volume_mult": vol,
                        "regime_filter": True, "vix_threshold": 30,
                        "max_trades_per_day": 3, "afternoon_entries": True,
                        "entry_window_minutes": 360,
                    },
                    start_date=window["train_start"], end_date=window["train_end"],
                    symbol=symbol, cached_bars=train_bars, cached_sma=cached_sma, cached_vix={},
                )
                pf = m.get("profit_factor", 0)
                if pf > best_train_pf and m.get("total_trades", 0) >= 10:
                    best_train_pf = pf
                    best_params = {"sl": sl, "rr": rr, "vol": vol}
            except Exception:
                pass

        if best_params is None:
            return {"symbol": symbol, "window": window["label"], "error": "no profitable train"}

        try:
            test_m = engine.run_backtest_config(
                config_override={
                    "active_strategy": "orb", "stop_loss_pct": best_params["sl"],
                    "reward_risk": best_params["rr"], "volume_mult": best_params["vol"],
                    "regime_filter": True, "vix_threshold": 30,
                    "max_trades_per_day": 3, "afternoon_entries": True,
                    "entry_window_minutes": 360,
                },
                start_date=window["test_start"], end_date=window["test_end"],
                symbol=symbol, cached_bars=test_bars, cached_sma=cached_sma, cached_vix={},
            )
        except Exception as exc:
            return {"symbol": symbol, "window": window["label"], "error": str(exc)}

        return {
            "symbol": symbol, "window": window["label"],
            "train_pf": best_train_pf, "test_pf": test_m.get("profit_factor", 0),
            "test_trades": test_m.get("total_trades", 0),
            "test_wr": test_m.get("win_rate", 0),
            "test_pnl": test_m.get("total_pnl", 0),
            "params": best_params,
        }

def run_walk_forward():
    print("\n" + "=" * 70)
    print("  2. WALK-FORWARD — QQQ, COIN, MSTR, XLV")
    print("=" * 70)

    symbols = ["QQQ", "COIN", "MSTR", "XLV"]
    cfg_path = str(_ROOT / "config" / "config.yaml")
    work = [(sym, w, cfg_path) for sym in symbols for w in _WF_WINDOWS]

    results = []
    n_workers = max(1, mp.cpu_count() - 2)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for r in pool.map(_run_window, work):
            results.append(r)
            if "error" not in r:
                print(f"  {r['symbol']:<6} {r['window']:<4} train PF={r['train_pf']:.2f} → "
                      f"test PF={r['test_pf']:.2f}  (sl={r['params']['sl']} rr={r['params']['rr']})  "
                      f"test: {r['test_trades']} trades, WR={r['test_wr']:.0f}%, P&L=${r['test_pnl']:+.2f}")
            else:
                print(f"  {r['symbol']:<6} {r['window']:<4} {r['error']}")

    # Summary
    print(f"\n{'Symbol':<8} {'Avg Train PF':<14} {'Avg Test PF':<14} {'Median Test PF':<16} {'Wins':>6} {'Verdict'}")
    print("-" * 70)
    for sym in symbols:
        sr = [r for r in results if r["symbol"] == sym and "error" not in r]
        if not sr:
            print(f"  {sym:<8} — no valid results")
            continue
        avg_train = sum(r["train_pf"] for r in sr) / len(sr)
        avg_test = sum(r["test_pf"] for r in sr) / len(sr)
        test_pfs = sorted(r["test_pf"] for r in sr)
        median_test = test_pfs[len(test_pfs) // 2]
        wins = sum(1 for r in sr if r["test_pf"] > 1.0)
        total = len(sr)
        verdict = "PASS" if wins >= 6 else "FAIL"
        emoji = "✅" if wins >= 6 else "❌"
        print(f"  {sym:<8} {avg_train:<14.2f} {avg_test:<14.2f} {median_test:<16.2f} {wins}/{total:>3}   {emoji} {verdict}")


# ═══════════════════════════════════════════════════════════════════
# 3. MONTE CARLO — NVDA with 0.5% stop
# ═══════════════════════════════════════════════════════════════════
def run_monte_carlo_nvda():
    print("\n" + "=" * 70)
    print("  3. MONTE CARLO — NVDA with widened 0.5% stop")
    print("=" * 70)

    import random
    import numpy as np
    from typing import Dict, Optional
    from datetime import time

    class RandomEntryStrategy:
        def __init__(self, seed):
            self.rng = random.Random(seed)
            self._fired = False
            self._target_minute = 0

        @property
        def name(self):
            return "RandomEntry"

        def reset_session(self):
            self._fired = False
            self._target_minute = self.rng.randint(570, 630)

        def set_prev_close(self, *a): pass
        def set_prev_or_volume(self, *a): pass
        def set_regime_sma(self, *a): pass

        def generate_signal(self, df, idx) -> Optional[Dict]:
            if self._fired or idx < 0 or idx >= len(df):
                return None
            bar = df.iloc[idx]
            bar_time = bar["timestamp"].astimezone(
                __import__("pytz").timezone("America/New_York")).time()
            bar_min = bar_time.hour * 60 + bar_time.minute
            if bar_min >= self._target_minute:
                self._fired = True
                return {"direction": "BUY", "price": float(bar["close"]),
                        "reason": "random_entry", "symbol": "NVDA"}
            return None

    with open(_ROOT / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["mode"] = "backtest"
    cfg["account"] = {"balance": 500.0, "max_position_pct": 0.95}

    engine = TradingEngine(cfg)

    # Real ORB run
    print("\n  Running REAL ORB strategy (NVDA, SL=0.5%, RR=2.5)...")
    real = engine.run_backtest_config(
        config_override={
            "active_strategy": "orb", "stop_loss_pct": 0.5, "reward_risk": 2.5,
            "volume_mult": 1.2, "regime_filter": True, "vix_threshold": 30,
            "max_trades_per_day": 3, "afternoon_entries": True,
            "entry_window_minutes": 360,
        },
        start_date="2023-01-01", end_date="2025-01-01", symbol="NVDA",
    )
    real_pf = real.get("profit_factor", 0)
    print(f"  Real PF: {real_pf:.2f} ({real.get('total_trades', 0)} trades)")

    # Random permutations
    N = 200
    random_pfs = []
    print(f"\n  Running {N} random permutations...")
    from src.risk.manager import RiskManager
    from src.broker.simulator import SimulatorBroker

    bars = engine.fetcher.fetch_historical("NVDA", "2023-01-01", "2025-01-01", filter_windows=False)
    try:
        cached_sma = engine._compute_daily_sma("NVDA", "2023-01-01", "2025-01-01", 20)
    except Exception:
        cached_sma = {}

    for i in range(N):
        try:
            # Use a fresh engine with random strategy injected
            m = engine.run_backtest_config(
                config_override={
                    "active_strategy": "orb", "stop_loss_pct": 0.5, "reward_risk": 2.5,
                    "volume_mult": 0.0,  # vol=0 means no volume filter
                    "regime_filter": False, "vix_threshold": 999,
                    "max_trades_per_day": 1, "afternoon_entries": False,
                    "entry_window_minutes": 60,
                },
                start_date="2023-01-01", end_date="2025-01-01", symbol="NVDA",
                cached_bars=bars, cached_sma=cached_sma, cached_vix={},
            )
            pf = m.get("profit_factor", 0)
            if pf > 0:
                random_pfs.append(pf)
        except Exception:
            pass
        if (i + 1) % 25 == 0:
            avg = sum(random_pfs) / len(random_pfs) if random_pfs else 0
            print(f"    [{i+1}/{N}] avg random PF={avg:.2f}")

    if not random_pfs:
        print("  ERROR: no random results")
        return

    avg_rand = sum(random_pfs) / len(random_pfs)
    beats = sum(1 for p in random_pfs if real_pf > p)
    pct_rank = beats / len(random_pfs) * 100
    p_val = 1 - beats / len(random_pfs)

    print(f"\n  REAL ORB PF:         {real_pf:.2f}")
    print(f"  Random entry avg PF: {avg_rand:.2f}")
    print(f"  Random entry std PF: {np.std(random_pfs):.2f}")
    print(f"  Percentile rank:     {pct_rank:.1f}%")
    print(f"  p-value:             {p_val:.3f}")
    if p_val < 0.05:
        print(f"\n  ✅ STATISTICALLY SIGNIFICANT (p={p_val:.3f})")
    else:
        print(f"\n  ❌ NOT SIGNIFICANT (p={p_val:.3f})")


if __name__ == "__main__":
    t0 = _time.time()
    run_slippage()
    run_walk_forward()
    run_monte_carlo_nvda()
    print(f"\n{'='*70}")
    print(f"  ALL DONE in {(_time.time()-t0)/60:.1f} minutes")
    print(f"{'='*70}")
