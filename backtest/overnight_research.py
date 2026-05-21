#!/usr/bin/env python3
"""
Overnight Mac Mini research (~12 hours):
  Phase 1: QQQ + COIN full validation (MC + slippage)
  Phase 2: Slippage-aware grid (5 symbols × realistic 0.05% slip)
"""
import os
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import sys
import time as _time
import random
import itertools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
import numpy as np
import pytz

_ROOT = Path(__file__).resolve().parents[1]
_ET = pytz.timezone("America/New_York")


# ═══════════════════════════════════════════════════════════════════
# PHASE 1a: MONTE CARLO — QQQ + COIN
# ═══════════════════════════════════════════════════════════════════

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
        bar_time = bar["timestamp"].astimezone(_ET).time()
        bar_min = bar_time.hour * 60 + bar_time.minute
        if bar_min >= self._target_minute:
            self._fired = True
            return {"direction": "BUY", "price": float(bar["close"]),
                    "reason": "random_entry", "symbol": "MC"}
        return None


def run_monte_carlo(symbol, sl, rr, n_perms=200):
    from src.engine import TradingEngine

    print(f"\n  MONTE CARLO — {symbol} (SL={sl}%, RR={rr}, {n_perms} permutations)")

    with open(_ROOT / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["mode"] = "backtest"
    cfg["account"] = {"balance": 500.0, "max_position_pct": 0.95}
    engine = TradingEngine(cfg)

    real = engine.run_backtest_config(
        config_override={
            "active_strategy": "orb", "stop_loss_pct": sl, "reward_risk": rr,
            "volume_mult": 1.2, "regime_filter": True, "vix_threshold": 30,
            "max_trades_per_day": 3, "afternoon_entries": True,
            "entry_window_minutes": 360,
        },
        start_date="2023-01-01", end_date="2025-01-01", symbol=symbol,
    )
    real_pf = real.get("profit_factor", 0)
    print(f"  Real PF: {real_pf:.2f} ({real.get('total_trades', 0)} trades)")

    bars = engine.fetcher.fetch_historical(symbol, "2023-01-01", "2025-01-01", filter_windows=False)
    try:
        cached_sma = engine._compute_daily_sma(symbol, "2023-01-01", "2025-01-01", 20)
    except Exception:
        cached_sma = {}

    random_pfs = []
    for i in range(n_perms):
        try:
            m = engine.run_backtest_config(
                config_override={
                    "active_strategy": "orb", "stop_loss_pct": sl, "reward_risk": rr,
                    "volume_mult": 0.0, "regime_filter": False, "vix_threshold": 999,
                    "max_trades_per_day": 1, "afternoon_entries": False,
                    "entry_window_minutes": 60,
                },
                start_date="2023-01-01", end_date="2025-01-01", symbol=symbol,
                cached_bars=bars, cached_sma=cached_sma, cached_vix={},
            )
            pf = m.get("profit_factor", 0)
            if pf > 0:
                random_pfs.append(pf)
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            avg = sum(random_pfs) / len(random_pfs) if random_pfs else 0
            print(f"    [{i+1}/{n_perms}] avg random PF={avg:.2f}")

    if not random_pfs:
        print("  ERROR: no random results")
        return

    avg_rand = sum(random_pfs) / len(random_pfs)
    beats = sum(1 for p in random_pfs if real_pf > p)
    p_val = 1 - beats / len(random_pfs)

    print(f"\n  REAL PF:      {real_pf:.2f}")
    print(f"  Random avg:   {avg_rand:.2f} (std={np.std(random_pfs):.2f})")
    print(f"  Percentile:   {beats/len(random_pfs)*100:.1f}%")
    print(f"  p-value:      {p_val:.3f}")
    if p_val < 0.05:
        print(f"  ✅ SIGNIFICANT (p={p_val:.3f})")
    else:
        print(f"  ❌ NOT SIGNIFICANT (p={p_val:.3f})")


# ═══════════════════════════════════════════════════════════════════
# PHASE 1b: SLIPPAGE — QQQ + COIN
# ═══════════════════════════════════════════════════════════════════

def run_slippage_test(combos):
    from src.engine import TradingEngine

    with open(_ROOT / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["mode"] = "backtest"
    cfg["account"] = {"balance": 500.0, "max_position_pct": 0.95}
    engine = TradingEngine(cfg)

    SLIPPAGE = [0.0, 0.02, 0.05, 0.10, 0.20]
    PERIODS = [
        ("2023-01-01", "2025-01-01", "full_2yr"),
        ("2022-01-01", "2022-12-31", "holdout_2022"),
        ("2025-06-01", "2026-03-31", "bear_2025"),
    ]

    print(f"\n{'Combo':<22} {'Period':<14} {'Slip%':<7} {'Trades':<7} {'WR%':<7} {'PnL':>9} {'PF':>7} {'DD%':>6}")
    print("-" * 85)

    for symbol, sl, rr, label in combos:
        for start, end, plabel in PERIODS:
            for slip in SLIPPAGE:
                try:
                    m = engine.run_backtest_config(
                        config_override={
                            "active_strategy": "orb", "stop_loss_pct": sl,
                            "reward_risk": rr, "volume_mult": 1.2,
                            "regime_filter": True, "vix_threshold": 30,
                            "max_trades_per_day": 3, "afternoon_entries": True,
                            "entry_window_minutes": 360, "slippage_pct": slip,
                        },
                        start_date=start, end_date=end, symbol=symbol,
                    )
                    t = m.get("total_trades", 0)
                    wr = m.get("win_rate", 0)
                    pnl = m.get("total_pnl", 0)
                    pf = m.get("profit_factor", 0)
                    dd = m.get("max_drawdown", 0)
                    print(f"{label:<22} {plabel:<14} {slip:<7.2f} {t:<7} {wr:<7.1f} ${pnl:>+8.2f} {pf:>7.2f} {dd:>5.1f}%")
                except Exception as e:
                    print(f"{label:<22} {plabel:<14} {slip:<7.2f} ERROR: {e}")
            print()


# ═══════════════════════════════════════════════════════════════════
# PHASE 2: SLIPPAGE-AWARE GRID (the big one)
# ═══════════════════════════════════════════════════════════════════

SLIP_GRID = {
    "symbols": ["NVDA", "AMD", "SPY", "QQQ", "COIN"],
    "strategies": ["orb", "ema_crossover", "momentum"],
    "periods": [
        {"label": "bull_2023", "start": "2023-01-01", "end": "2023-12-31"},
        {"label": "bull_2024", "start": "2024-01-01", "end": "2024-12-31"},
        {"label": "bear_2025", "start": "2025-06-01", "end": "2026-03-31"},
        {"label": "full_2yr", "start": "2023-01-01", "end": "2025-01-01"},
        {"label": "holdout_2022", "start": "2022-01-01", "end": "2022-12-31"},
    ],
    "stop_loss_pct": [0.3, 0.5, 0.75, 1.0],
    "reward_risk": [2.0, 2.5, 3.0, 4.0, 5.0],
    "volume_mult": [1.0, 1.2],
    "entry_window_minutes": [180, 360],
    "slippage_pct": 0.05,  # realistic IEX slippage
}


def _slip_worker(args):
    """Worker for slippage-aware grid."""
    symbol, strategy, period, sl, rr, vol, ew = args

    import yaml as _yaml
    from src.engine import TradingEngine
    from src.logger.results_store import ResultsStore, make_run_id

    # Build a unique run_id that includes slippage
    params = {
        "strategy": strategy, "symbol": symbol,
        "period_label": period["label"],
        "start_date": period["start"], "end_date": period["end"],
        "stop_loss_pct": sl, "reward_risk": rr,
        "volume_mult": vol, "regime_filter": True,
        "vix_threshold": 30, "max_trades_per_day": 3,
        "afternoon_entries": True,
        "entry_window_minutes": ew,
        "slippage_pct": 0.05,
    }
    run_id = make_run_id(params)

    store = ResultsStore()
    if store.run_exists(run_id):
        return {"status": "skipped"}

    with open(str(_ROOT / "config" / "config.yaml")) as f:
        cfg = _yaml.safe_load(f)
    cfg["mode"] = "backtest"
    cfg.setdefault("account", {})["balance"] = 500.0

    try:
        engine = TradingEngine(cfg)
        metrics = engine.run_backtest_config(
            config_override={
                "active_strategy": strategy,
                "stop_loss_pct": sl,
                "reward_risk": rr,
                "volume_mult": vol,
                "regime_filter": True,
                "vix_threshold": 30,
                "max_trades_per_day": 3,
                "afternoon_entries": True,
                "entry_window_minutes": ew,
                "slippage_pct": 0.05,
            },
            start_date=period["start"],
            end_date=period["end"],
            symbol=symbol,
        )
    except Exception as e:
        return {"status": "error", "error": str(e)}

    # Save to results.db with slippage tag in period_label
    try:
        store.save_run(
            run_id=run_id,
            params={**params, "period_label": f"slip005_{period['label']}"},
            metrics=metrics,
        )
    except Exception:
        pass

    return {
        "status": "done",
        "strategy": strategy, "symbol": symbol,
        "period": period["label"],
        "sl": sl, "rr": rr,
        "trades": metrics.get("total_trades", 0),
        "pf": metrics.get("profit_factor", 0),
        "pnl": metrics.get("total_pnl", 0),
    }


def run_slippage_grid():
    g = SLIP_GRID
    combos = list(itertools.product(
        g["symbols"], g["strategies"], g["periods"],
        g["stop_loss_pct"], g["reward_risk"], g["volume_mult"],
        g["entry_window_minutes"],
    ))
    print(f"\n  Total combos: {len(combos):,}")
    print(f"  Slippage: {g['slippage_pct']}% on every run")
    print(f"  Workers: {max(1, mp.cpu_count() - 2)}")

    done = 0
    skipped = 0
    errors = 0
    best_by_symbol = {}

    n_workers = max(1, mp.cpu_count() - 2)
    t0 = _time.time()

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(_slip_worker, c): c for c in combos}
        for fut in as_completed(futures):
            r = fut.result()
            if r["status"] == "skipped":
                skipped += 1
            elif r["status"] == "error":
                errors += 1
            else:
                done += 1
                sym = r["symbol"]
                if sym not in best_by_symbol or r["pf"] > best_by_symbol[sym]["pf"]:
                    if r["trades"] >= 15 and r["pf"] < 100:
                        best_by_symbol[sym] = r

            total_processed = done + skipped + errors
            if total_processed % 500 == 0:
                elapsed = _time.time() - t0
                rate = total_processed / elapsed if elapsed > 0 else 0
                eta_min = (len(combos) - total_processed) / rate / 60 if rate > 0 else 0
                print(f"  [{total_processed:,}/{len(combos):,}] done={done} skip={skipped} err={errors} "
                      f"rate={rate:.1f}/s ETA={eta_min:.0f}min")

    elapsed = (_time.time() - t0) / 60
    print(f"\n  SLIPPAGE-AWARE GRID COMPLETE")
    print(f"  Done: {done:,} | Skipped: {skipped:,} | Errors: {errors:,}")
    print(f"  Time: {elapsed:.1f} minutes")

    print(f"\n  BEST CONFIG PER SYMBOL (with 0.05% slippage, ≥15 trades):")
    print(f"  {'Symbol':<8} {'Strategy':<16} {'Period':<14} {'SL':<5} {'RR':<5} {'Trades':<7} {'PF':>7} {'PnL':>9}")
    print(f"  {'-'*75}")
    for sym in g["symbols"]:
        if sym in best_by_symbol:
            b = best_by_symbol[sym]
            print(f"  {sym:<8} {b['strategy']:<16} {b['period']:<14} {b['sl']:<5} {b['rr']:<5} {b['trades']:<7} {b['pf']:>7.2f} ${b['pnl']:>+8.2f}")
        else:
            print(f"  {sym:<8} — no profitable config survives 0.05% slippage")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = _time.time()

    # Phase 1a: Monte Carlo
    print("=" * 70)
    print("  PHASE 1a: MONTE CARLO — QQQ + COIN")
    print("=" * 70)
    run_monte_carlo("QQQ", sl=0.3, rr=2.5, n_perms=200)
    run_monte_carlo("COIN", sl=1.0, rr=3.0, n_perms=200)

    # Phase 1b: Slippage test
    print("\n" + "=" * 70)
    print("  PHASE 1b: SLIPPAGE TEST — QQQ + COIN")
    print("=" * 70)
    run_slippage_test([
        ("QQQ", 0.3, 2.5, "QQQ tight"),
        ("QQQ", 0.5, 3.0, "QQQ wider"),
        ("QQQ", 0.5, 4.0, "QQQ high RR"),
        ("COIN", 1.0, 3.0, "COIN live"),
        ("COIN", 0.75, 4.0, "COIN alt"),
    ])

    # Phase 2: Slippage-aware grid
    print("\n" + "=" * 70)
    print("  PHASE 2: SLIPPAGE-AWARE GRID (0.05% slippage on every run)")
    print("=" * 70)
    run_slippage_grid()

    total_min = (_time.time() - t0) / 60
    print(f"\n{'=' * 70}")
    print(f"  ALL DONE in {total_min:.1f} minutes ({total_min/60:.1f} hours)")
    print(f"{'=' * 70}")
