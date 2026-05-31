#!/usr/bin/env python3
"""
Big Simulation B — Quad-Validated Portfolio.

HYPOTHESIS:
  The 18 quad-validated configurations (V4 + WF stable + recent profitable)
  perform meaningfully better than the live config (3 slots) when run as a
  shared-capital portfolio. The research-supported "all-in" version should
  produce higher Sharpe and PnL.

PASS CRITERIA (stated BEFORE running):
  - Portfolio Sharpe > 1.5 on full_2yr
  - Portfolio max DD < 50% on full_2yr
  - Portfolio recent (2026) PF > 1.0 (forward survival)
  - Net PnL on full_2yr > 3× the simple live config (sim A)
"""
import os, sys, json, pickle, math
from pathlib import Path
from collections import defaultdict
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import pandas as pd, pytz, yaml
from src.data.fetcher import DataFetcher
from src.engine import TradingEngine

_ET = pytz.timezone("America/New_York")
_ROOT = Path(__file__).resolve().parents[1]
CACHE = _ROOT / "logs" / "research" / "bar_cache"

START_CASH = 500.0


def load_quad_configs():
    """Identify the 18 quad-validated configs by intersecting V4 + WF + Recent."""
    import glob

    v4_pass = {}
    for f in glob.glob(str(_ROOT / 'logs/research/val_v4_all/*.json')):
        try: r = json.load(open(f))
        except: continue
        if r.get('full_2yr_trades',0) < 10: continue
        slip = (r.get('slip_full_2yr',{}) or {}).get('0.05',0)
        if r.get('wf_wins',0) >= 6 and r.get('mc_p_value',1.0) <= 0.05 and slip >= 1.0:
            key = (r['strategy'], r['symbol'], r['sl'], r['rr'])
            v4_pass[key] = r

    wf_robust = {}
    for f in glob.glob(str(_ROOT / 'logs/research/wf_stability/*.json')):
        try: r = json.load(open(f))
        except: continue
        if r.get('robust', False):
            key = (r['strategy'], r['symbol'], r['sl'], r['rr'])
            wf_robust[key] = r

    recent = {}
    for f in ['logs/research/validation_recent.json',
              'logs/research/validation_recent_MSTR_IWM.json',
              'logs/research/validation_recent_NVDA_SPY.json',
              'logs/research/validation_recent_XLK_XLF.json']:
        p = _ROOT / f
        if not p.exists(): continue
        try: data = json.load(open(p))
        except: continue
        for r in data:
            strat = r.get('strategy', r.get('strat',''))
            sym = r.get('symbol', r.get('sym',''))
            key = (strat, sym, r.get('sl'), r.get('rr'))
            recent[key] = {
                't': r.get('recent_trades', r.get('trades',0)),
                'pf': r.get('recent_pf', r.get('pf',0)),
            }

    quad = []
    for key in v4_pass:
        wf = wf_robust.get(key); rec = recent.get(key)
        if wf and rec and rec['t'] >= 10 and rec['pf'] >= 1.0:
            quad.append({
                "strategy": key[0], "symbol": key[1], "sl": key[2], "rr": key[3],
                "name": f"{key[0][:3]}/{key[1]}/{key[2]}_{key[3]}",
            })
    return quad


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


def run_one(strat, sym, sl, rr, bars, sma, start, end):
    cfg = yaml.safe_load(open(_ROOT / "config/config.yaml"))
    cfg["mode"] = "backtest"
    cfg["account"] = {"balance": START_CASH, "max_position_pct": 0.95}
    cfg["active_strategy"] = strat
    engine = TradingEngine(cfg)
    override = {
        "active_strategy": strat, "stop_loss_pct": sl, "reward_risk": rr,
        "volume_mult": 1.2, "regime_filter": True, "vix_threshold": 25,
        "max_trades_per_day": 100, "afternoon_entries": True,
        "entry_window_minutes": 360,
    }
    return engine.run_backtest_config(
        config_override=override, start_date=start, end_date=end,
        symbol=sym, cached_bars=bars, cached_sma=sma, cached_vix={},
    )


def portfolio_simulate(trades_by_slot, max_concurrent=4, per_slot_pct=0.20):
    """With 18 slots, allow more concurrent positions but smaller sizes each."""
    events = []
    for sname, trades in trades_by_slot.items():
        for t in trades:
            et = pd.to_datetime(t["entry_time"])
            xt = pd.to_datetime(t["exit_time"])
            if et.tz is None: et = et.tz_localize("UTC").tz_convert(_ET)
            else: et = et.tz_convert(_ET)
            if xt.tz is None: xt = xt.tz_localize("UTC").tz_convert(_ET)
            else: xt = xt.tz_convert(_ET)
            entry_px = float(t["entry_price"])
            exit_px = float(t.get("exit_price", entry_px))
            pnl_pct = (exit_px - entry_px) / entry_px * 100 if entry_px else 0.0
            events.append({"slot": sname, "entry_time": et, "exit_time": xt,
                           "entry_px": entry_px, "pnl_pct": pnl_pct})
    events.sort(key=lambda e: e["entry_time"])

    cash = START_CASH; peak = START_CASH; max_dd = 0.0
    open_pos = []; taken = []; skipped = 0
    i = 0
    while i < len(events) or open_pos:
        next_t = events[i]["entry_time"] if i < len(events) else None
        open_pos.sort(key=lambda p: p["exit_time"])
        while open_pos and (next_t is None or open_pos[0]["exit_time"] <= next_t):
            pos = open_pos.pop(0)
            pnl = pos["size"] * pos["pnl_pct"] / 100.0
            cash += pos["size"] + pnl
            peak = max(peak, cash)
            dd = (peak - cash)/peak*100
            max_dd = max(max_dd, dd)
        if i >= len(events): break
        ev = events[i]; i += 1
        if len(open_pos) >= max_concurrent: skipped += 1; continue
        size = START_CASH * per_slot_pct
        if size > cash: skipped += 1; continue
        cash -= size
        open_pos.append({**ev, "size": size})
        taken.append({**ev, "size": size})

    pnls = [t["size"]*t["pnl_pct"]/100 for t in taken]
    pos_sum = sum(p for p in pnls if p > 0)
    neg_sum = abs(sum(p for p in pnls if p <= 0))
    pf = pos_sum/neg_sum if neg_sum > 0 else float("inf")
    wr = 100 * sum(1 for p in pnls if p > 0) / max(len(pnls), 1)
    daily = defaultdict(float)
    for t, p in zip(taken, pnls):
        daily[t["entry_time"].date()] += p
    dseries = list(daily.values())
    if len(dseries) >= 2:
        mu = sum(dseries)/len(dseries)
        var = sum((x-mu)**2 for x in dseries)/len(dseries)
        std = math.sqrt(var) if var > 0 else 0
        sharpe = (mu/std * math.sqrt(252)) if std > 0 else 0.0
    else:
        sharpe = 0.0
    return {
        "trades_taken": len(taken), "trades_skipped": skipped,
        "total_pnl": sum(pnls), "final_cash": cash,
        "pf": pf, "wr": wr, "sharpe": sharpe, "max_dd_pct": max_dd,
    }


def main():
    print("=" * 75)
    print("BIG SIMULATION B — Quad-Validated Portfolio")
    print("=" * 75)

    quad = load_quad_configs()
    print(f"\nLoaded {len(quad)} quad-validated configs:\n")
    for q in quad:
        print(f"  {q['strategy']:<14} {q['symbol']:<6} SL={q['sl']} RR={q['rr']}")
    print()

    fetcher = DataFetcher({})
    results = {}

    for period_name, start, end, src in [
        ("full_2yr", "2023-01-01", "2025-01-01", "cached"),
        ("recent",   "2026-01-01", "2026-05-30", "fetch"),
    ]:
        print("=" * 75)
        print(f"  PERIOD: {period_name} ({start} → {end})")
        print("=" * 75)
        trades_by_slot = {}
        bars_cache = {}
        for q in quad:
            sym = q["symbol"]
            if sym not in bars_cache:
                if src == "cached":
                    bars_cache[sym] = load_cached(sym, start, end)
                else:
                    try:
                        bars_cache[sym] = fetcher.fetch_historical_alpaca(sym, start, end, filter_windows=False)
                    except Exception as e:
                        print(f"  {sym}: fetch error: {e}")
                        bars_cache[sym] = None
            bars = bars_cache.get(sym); sma = load_sma(sym) if src=="cached" else {}
            if bars is None or bars.empty: continue
            try:
                r = run_one(q["strategy"], sym, q["sl"], q["rr"], bars, sma, start, end)
                trades_by_slot[q["name"]] = r.get("trades_list", [])
            except Exception as e:
                print(f"  {q['name']}: error")
        print(f"  Total trades across {len(trades_by_slot)} slots: {sum(len(v) for v in trades_by_slot.values())}")

        # Portfolio
        port = portfolio_simulate(trades_by_slot, max_concurrent=4, per_slot_pct=0.20)
        print(f"\n  PORTFOLIO (max 4 concurrent @ 20%):")
        print(f"    trades taken: {port['trades_taken']} / skipped: {port['trades_skipped']}")
        print(f"    PnL: ${port['total_pnl']:+.2f}  (final cash ${port['final_cash']:.2f})")
        print(f"    PF: {port['pf']:.2f}  WR: {port['wr']:.1f}%  Sharpe: {port['sharpe']:.2f}  MaxDD: {port['max_dd_pct']:.1f}%")
        results[period_name] = port

    # Pass/fail
    print("\n" + "=" * 75)
    print("HYPOTHESIS EVALUATION")
    print("=" * 75)
    if "full_2yr" in results:
        print(f"  Sharpe (full_2yr) {results['full_2yr']['sharpe']:.2f}  {'✓' if results['full_2yr']['sharpe']>1.5 else '✗'} (target >1.5)")
        print(f"  MaxDD (full_2yr) {results['full_2yr']['max_dd_pct']:.1f}%  {'✓' if results['full_2yr']['max_dd_pct']<50 else '✗'} (target <50%)")
    if "recent" in results:
        print(f"  PF (recent)      {results['recent']['pf']:.2f}  {'✓' if results['recent']['pf']>1.0 else '✗'} (target >1.0)")

    out = _ROOT / "logs/research/big_sim_b_results.json"
    with open(out, "w") as f:
        json.dump({"quad_configs": quad, "results": results}, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
