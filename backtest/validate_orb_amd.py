#!/usr/bin/env python3
"""
Focused ORB/AMD holdout 2022 validation — same shape as the
ORB/NVDA holdout 2022 validation we did in Phase 11e.

Saves results to results.db so the dashboard scorecard can use them.
Prints a clean summary table.
"""
import os
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import itertools
import sys
import time as _time_mod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

_ROOT = Path(__file__).resolve().parents[1]


def main():
    with open(_ROOT / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["mode"] = "backtest"

    from src.engine import TradingEngine
    from src.logger.results_store import ResultsStore, make_run_id

    engine = TradingEngine(cfg)
    store = ResultsStore()

    print("Fetching AMD 2022 bars + SMA + VIX...")
    bars = engine.fetcher.fetch_historical(
        "AMD", "2022-01-01", "2022-12-31", filter_windows=False
    )
    if bars is None or bars.empty:
        print("ERROR: No AMD 2022 data from Alpaca")
        return
    print(f"  bars: {len(bars):,}")
    sma_days = cfg.get("filters", {}).get("regime_sma_days", 20)
    sma = engine._compute_daily_sma("AMD", "2022-01-01", "2022-12-31", sma_days)
    vix = engine._fetch_vix_data("2022-01-01", "2022-12-31")
    print(f"  sma days: {len(sma)}  vix days: {len(vix)}\n")

    # Same grid as the ORB/NVDA validation: 80 combos
    combos = list(itertools.product(
        [0.3, 0.5, 0.75],     # sl
        [2.0, 2.5, 3.0, 4.0, 5.0],  # rr
        [True, False],         # regime
        [25, 30],              # vix
        [1.0, 1.2],            # vol_mult
        [1, 2],                # max_t
    ))
    print(f"Running {len(combos)} combos and saving to results.db...\n")

    results = []
    saved = 0
    for sl, rr, regime, vix_t, vol_m, max_t in combos:
        params = {
            "strategy":           "orb",
            "symbol":             "AMD",
            "period_label":       "holdout_2022",
            "start_date":         "2022-01-01",
            "end_date":           "2022-12-31",
            "stop_loss_pct":      sl,
            "reward_risk":        rr,
            "volume_mult":        vol_m,
            "regime_filter":      regime,
            "vix_threshold":      vix_t,
            "max_trades_per_day": max_t,
            "afternoon_entries":  False,
        }
        run_id = make_run_id(params)
        if store.run_exists(run_id):
            continue
        m = engine.run_backtest_config(
            config_override={
                "active_strategy":    "orb",
                "stop_loss_pct":      sl,
                "reward_risk":        rr,
                "volume_mult":        vol_m,
                "regime_filter":      regime,
                "vix_threshold":      vix_t,
                "max_trades_per_day": max_t,
                "afternoon_entries":  False,
            },
            start_date="2022-01-01",
            end_date="2022-12-31",
            symbol="AMD",
            cached_bars=bars,
            cached_sma=sma,
            cached_vix=vix,
        )
        trades = m.pop("trades_list", [])
        run_config = dict(params)
        run_config["run_id"] = run_id
        store.save_run(run_config, m, trades)
        saved += 1
        pf = m["profit_factor"]
        if pf == float("inf"): pf = 999.0
        results.append({
            "sl": sl, "rr": rr, "regime": regime, "vix": vix_t,
            "vol": vol_m, "max_t": max_t,
            "trades": m["total_trades"],
            "wr": m["win_rate"], "pf": pf,
            "pnl": m["total_pnl"], "dd": m["max_drawdown"],
        })

    results.sort(key=lambda r: r["pf"], reverse=True)

    print("=" * 90)
    print(f"ORB/AMD/holdout_2022 — saved {saved} new combos. Top 15 by PF:")
    print("=" * 90)
    print(f"{'sl':>5} {'rr':>5} {'reg':>4} {'vix':>4} {'vol':>4} {'mt':>3} "
          f"{'trades':>7} {'wr':>7} {'pf':>7} {'pnl':>9} {'dd':>7}")
    print("-" * 90)
    for r in results[:15]:
        print(f"{r['sl']:>5.2f} {r['rr']:>5.1f} {str(r['regime'])[:3]:>4} "
              f"{r['vix']:>4} {r['vol']:>4.1f} {r['max_t']:>3} "
              f"{r['trades']:>7} {r['wr']:>6.1f}% {r['pf']:>7.2f} "
              f"{r['pnl']:>+8.2f} {r['dd']:>6.2f}%")

    n_pass = sum(1 for r in results if r["pf"] > 1.0 and r["trades"] >= 15)
    n_total = sum(1 for r in results if r["trades"] >= 15)
    print(f"\nVERDICT: {n_pass}/{n_total} combos pass PF>1.0 with >=15 trades")
    if n_pass > 0:
        best = max((r for r in results if r["trades"] >= 15), key=lambda r: r["pf"])
        print(f"Best: PF={best['pf']:.2f} P&L={best['pnl']:+.2f} WR={best['wr']:.1f}% "
              f"DD={best['dd']:.2f}%")
        print(f"  config: sl={best['sl']} rr={best['rr']} regime={best['regime']} "
              f"vix={best['vix']} vol={best['vol']} max_t={best['max_t']}")


if __name__ == "__main__":
    main()
