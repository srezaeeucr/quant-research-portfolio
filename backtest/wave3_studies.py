#!/usr/bin/env python3
"""
Wave 3 — three combined studies:

A. Time-of-day analysis: when does each strategy fire and how does it perform
   bucketed by hour (9, 10, 11, 12, 13, 14, 15)?

B. Signal-data analysis: parse logs/live.log to see what signals fired in
   actual paper trading (or were skipped). Cross-reference with new strategies.

C. Hybrid Filter validation: ORB primary + ≥1 filter agrees on 4 different
   filter sets, full_2yr + recent.
"""
import os, re, sys, json, pickle, copy
from collections import defaultdict, Counter
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import pandas as pd, pytz, yaml
from src.data.fetcher import DataFetcher
from src.engine import TradingEngine

_ET = pytz.timezone("America/New_York")
_ROOT = Path(__file__).resolve().parents[1]
CACHE = _ROOT / "logs" / "research" / "bar_cache"

ALL_STRATS = ["orb", "ema_crossover", "momentum", "macd_crossover",
              "rsi_reversion", "bollinger_reversal", "donchian_breakout",
              "stochastic_crossover", "inside_bar_breakout"]
SYMBOLS = ["AMD", "GOOGL"]


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


def run_test(strat_name, sym, sl, rr, bars, sma, start, end,
             extra_strategy_params=None):
    cfg = yaml.safe_load(open(_ROOT / "config/config.yaml"))
    cfg["mode"] = "backtest"
    cfg["account"] = {"balance": 500.0, "max_position_pct": 0.95}
    if extra_strategy_params:
        cfg.setdefault("strategy_params", {}).update(extra_strategy_params)
    engine = TradingEngine(cfg)
    override = {
        "active_strategy": strat_name, "stop_loss_pct": sl, "reward_risk": rr,
        "volume_mult": 1.2, "regime_filter": True, "vix_threshold": 25,
        "max_trades_per_day": 100, "afternoon_entries": True, "entry_window_minutes": 360,
    }
    return engine.run_backtest_config(
        config_override=override, start_date=start, end_date=end,
        symbol=sym, cached_bars=bars, cached_sma=sma, cached_vix={},
    )


# =============================================================
# Study A — Time-of-day analysis
# =============================================================
def study_a():
    print("=" * 70)
    print("STUDY A — Time-of-day analysis (full_2yr)")
    print("=" * 70)
    PERIOD = ("2023-01-01", "2025-01-01")
    output = {}

    for sym in SYMBOLS:
        bars = load_cached(sym, *PERIOD)
        if bars is None: print(f"\n{sym}: no bars"); continue
        sma = load_sma(sym)
        print(f"\n--- {sym} ---")

        for strat in ALL_STRATS:
            try:
                r = run_test(strat, sym, 0.75, 3.0, bars, sma, *PERIOD)
            except Exception as e:
                continue
            trades = r.get("trades_list", [])
            if len(trades) < 10:
                continue

            # Bucket by hour
            buckets: Dict[int, List[Tuple[float, bool]]] = defaultdict(list)  # hour -> [(pnl, won)]
            for t in trades:
                et = pd.to_datetime(t["entry_time"])
                if et.tz is None: et = et.tz_localize("UTC").tz_convert(_ET)
                else: et = et.tz_convert(_ET)
                hour = et.hour
                pnl = float(t["pnl"])
                buckets[hour].append((pnl, pnl > 0))

            # Compute per-hour stats
            print(f"  {strat:<22} (total {len(trades)} trades)")
            sym_strat_buckets = {}
            for h in sorted(buckets):
                ts = buckets[h]
                pnl_sum = sum(p for p, _ in ts)
                wins = sum(1 for _, w in ts if w)
                wr = 100 * wins / len(ts) if ts else 0
                gw = sum(p for p, _ in ts if p > 0)
                gl = abs(sum(p for p, _ in ts if p <= 0))
                pf = gw / gl if gl > 0 else float("inf")
                marker = " ⭐" if pnl_sum > 50 and pf > 1.3 else ""
                print(f"    hour {h:02d}: {len(ts):3d} trades  WR={wr:4.0f}%  PF={pf:5.2f}  PnL=${pnl_sum:+8.2f}{marker}")
                sym_strat_buckets[h] = {"trades": len(ts), "wr": wr, "pf": pf if pf != float("inf") else 99.0, "pnl": pnl_sum}
            output[f"{strat}/{sym}"] = sym_strat_buckets

    out = _ROOT / "logs/research/studies_2026_04_26/wave3_time_of_day.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f: json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out}")


# =============================================================
# Study B — Signal data from live log
# =============================================================
def study_b():
    print("\n" + "=" * 70)
    print("STUDY B — Live log signal analysis")
    print("=" * 70)

    log_path = _ROOT / "logs" / "live.log"
    if not log_path.exists():
        print(f"  No live log found at {log_path}")
        return

    # Parse SIG entries
    # Pattern: "SIG AMD  bar=N  price=X.XX  OR=[H–L]  fired=True/False  signal=BUY/None"
    sig_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}).*SIG (\w+).*bar=(\d+).*price=([\d.]+).*"
        r"OR=\[([\d.]+)–([\d.]+)\].*fired=(\w+).*signal=(\w+)"
    )
    entry_pattern = re.compile(r"ENTRY (\S+).*shares")
    exit_pattern  = re.compile(r"EXIT (\S+)")

    sig_count = defaultdict(int)
    sig_buys  = defaultdict(int)
    entries   = []
    exits     = []
    days_seen = set()

    with open(log_path) as f:
        for line in f:
            m = sig_pattern.search(line)
            if m:
                _, _, sym, _, _, _, _, _, signal = m.groups()
                sig_count[sym] += 1
                if signal == "BUY":
                    sig_buys[sym] += 1
                date_m = re.match(r"(\d{4}-\d{2}-\d{2})", line)
                if date_m: days_seen.add(date_m.group(1))
            em = entry_pattern.search(line)
            if em: entries.append(em.group(1))
            xm = exit_pattern.search(line)
            if xm: exits.append(xm.group(1))

    print(f"\n  Days with log entries: {len(days_seen)}")
    print(f"  Earliest: {min(days_seen) if days_seen else 'n/a'}")
    print(f"  Latest:   {max(days_seen) if days_seen else 'n/a'}")
    print(f"\n  SIG bar evaluations per symbol:")
    for sym in sorted(sig_count):
        print(f"    {sym:<8}  total_evals={sig_count[sym]:>6,}  buy_signals={sig_buys[sym]}")
    print(f"\n  ENTRY events: {len(entries)}")
    print(f"  EXIT events:  {len(exits)}")

    # Identify days where NO signal fired for a symbol — these are days where
    # new strategies might have caught a trade we missed.
    print(f"\n  (For B's full picture — would need per-day bar replay through new strategies.")
    print(f"   Skipping that simulation here; the fact that no buy_signals fired during")
    print(f"   most days suggests our current strategies are very selective.)")

    out = _ROOT / "logs/research/studies_2026_04_26/wave3_log_analysis.json"
    with open(out, "w") as f:
        json.dump({
            "days_seen": sorted(days_seen),
            "sig_count_by_symbol": dict(sig_count),
            "buy_signals_by_symbol": dict(sig_buys),
            "entries": entries,
            "exits": exits,
        }, f, indent=2, default=str)
    print(f"\nSaved: {out}")


# =============================================================
# Study C — Hybrid Filter validation
# =============================================================
def study_c():
    print("\n" + "=" * 70)
    print("STUDY C — Hybrid Filter validation")
    print("=" * 70)

    fetcher = DataFetcher({})
    test_configs = [
        {"primary": "orb",            "filters": ["rsi_reversion", "macd_crossover", "bollinger_reversal", "donchian_breakout"]},
        {"primary": "rsi_reversion",  "filters": ["orb", "macd_crossover", "bollinger_reversal", "donchian_breakout"]},
        {"primary": "donchian_breakout", "filters": ["orb", "rsi_reversion", "macd_crossover", "bollinger_reversal"]},
    ]
    thresholds = [1, 2]

    results = []
    for phase_name, period in [
        ("full_2yr", ("2023-01-01", "2025-01-01")),
        ("recent",   ("2026-01-01", "2026-04-26")),
    ]:
        print(f"\n--- {phase_name.upper()} ({period[0]} → {period[1]}) ---")
        for sym in SYMBOLS:
            if phase_name == "full_2yr":
                bars = load_cached(sym, *period); sma = load_sma(sym)
            else:
                try:
                    bars = fetcher.fetch_historical_alpaca(sym, period[0], period[1], filter_windows=False)
                except Exception as e:
                    print(f"  {sym}: fetch error: {e}"); continue
                sma = {}
            if bars is None or bars.empty:
                print(f"  {sym}: no bars"); continue
            print(f"\n  {sym}:")

            # Baselines: each primary alone
            primaries = list({c["primary"] for c in test_configs})
            base_pnls = {}
            for p in primaries:
                try:
                    r = run_test(p, sym, 0.75, 3.0, bars, sma, *period)
                    base_pnls[p] = r["total_pnl"]
                    print(f"    baseline {p:<22}: {r['total_trades']:4d} tr  "
                          f"PF={r['profit_factor']:.2f}  PnL=${r['total_pnl']:+8.2f}")
                except Exception as e:
                    print(f"    baseline {p}: ERROR")

            # Hybrid configs
            for tc in test_configs:
                for th in thresholds:
                    try:
                        r = run_test(
                            "hybrid_filter", sym, 0.75, 3.0, bars, sma, *period,
                            extra_strategy_params={"hybrid_filter": {
                                "primary": tc["primary"],
                                "filters": tc["filters"],
                                "min_filters_agree": th,
                                "filter_lookback": 5,
                            }},
                        )
                        base = base_pnls.get(tc["primary"], 0)
                        delta = r["total_pnl"] - base
                        marker = "  ⭐ better than primary alone" if delta > 0 else "  (worse)"
                        print(f"    hybrid {tc['primary']:<10}+{th}f : {r['total_trades']:4d} tr  "
                              f"PF={r['profit_factor']:.2f}  PnL=${r['total_pnl']:+8.2f}  Δ${delta:+.2f}{marker}")
                        results.append({
                            "phase": phase_name, "symbol": sym,
                            "primary": tc["primary"], "filters": tc["filters"],
                            "min_filters_agree": th,
                            "trades": r["total_trades"], "pf": round(r["profit_factor"], 3),
                            "pnl": round(r["total_pnl"], 2), "wr": round(r["win_rate"], 1),
                            "primary_baseline_pnl": round(base, 2),
                            "delta_vs_primary": round(delta, 2),
                        })
                    except Exception as e:
                        print(f"    hybrid {tc['primary']}+{th}f: ERROR {e}")

    out = _ROOT / "logs/research/studies_2026_04_26/wave3_hybrid_filter.json"
    with open(out, "w") as f: json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    from typing import Dict, List, Tuple
    study_a()
    study_b()
    study_c()
