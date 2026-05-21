#!/usr/bin/env python3
"""
GPU Mega Research — 10 studies, ~12 hours on superpower.
GPU (XGBoost CUDA) + 56 CPU cores in parallel.

Studies:
1. Feature Discovery (GPU, ~1h)
2. Optimal Exit Timing (CPU, ~3h)
3. Ensemble Strategy (CPU+GPU, ~2h)
4. Regime Detection (GPU, ~1h)
5. VIX-Regime Specific Configs (CPU, ~2h)
6. Day-of-Week + Hour-of-Day Analysis (GPU, ~30min)
7. Cross-Asset Correlation (GPU, ~30min)
8. Walk-Forward Meta-Learner (CPU+GPU, ~2h)
9. Slippage Model (GPU, ~1h)
10. Multi-Timeframe Features (GPU, ~1h)
"""
import os
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import sys
import time as _time
import sqlite3
import warnings
import hashlib
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_ROOT = Path(__file__).resolve().parents[1]


def _get_device():
    try:
        import xgboost as xgb
        if xgb.build_info().get("USE_CUDA"):
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _load_db():
    merged = _ROOT / "logs" / "results_merged.db"
    db_path = merged if merged.exists() else _ROOT / "logs" / "results.db"
    c = sqlite3.connect(str(db_path), timeout=30)
    return c


def _header(num, title):
    print(f"\n{'='*70}")
    print(f"  STUDY {num}: {title}")
    print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════════
# STUDY 1: FEATURE DISCOVERY
# ═══════════════════════════════════════════════════════════════════

def study_1_feature_discovery():
    _header(1, "FEATURE DISCOVERY — Which technical indicators predict wins?")
    import xgboost as xgb
    device = _get_device()
    print(f"  Device: {device}")

    # We'll compute features from the backtest metadata
    c = _load_db()
    df = pd.read_sql_query("""
        SELECT strategy, symbol, period_label, stop_loss_pct, reward_risk,
               volume_mult, regime_filter, vix_threshold, max_trades_per_day,
               afternoon_entries, total_trades, win_rate, total_pnl,
               profit_factor, max_drawdown, avg_duration_min, sharpe_ratio
        FROM backtest_runs WHERE total_trades >= 10 AND profit_factor < 100
    """, c)
    c.close()
    print(f"  Loaded {len(df):,} runs")

    # Engineer 50+ features
    le_s = LabelEncoder(); le_sym = LabelEncoder(); le_p = LabelEncoder()
    df["strat_enc"] = le_s.fit_transform(df["strategy"])
    df["sym_enc"] = le_sym.fit_transform(df["symbol"])
    df["per_enc"] = le_p.fit_transform(df["period_label"])

    # Derived features
    df["be_wr"] = 100.0 / (1.0 + df["reward_risk"])
    df["wr_buffer"] = df["win_rate"] - df["be_wr"]
    df["pnl_per_trade"] = df["total_pnl"] / df["total_trades"].clip(1)
    df["risk_per_trade"] = df["stop_loss_pct"] * 500 / 100  # approx $ risk
    df["target_per_trade"] = df["risk_per_trade"] * df["reward_risk"]
    df["sl_rr_product"] = df["stop_loss_pct"] * df["reward_risk"]
    df["sl_rr_ratio"] = df["stop_loss_pct"] / df["reward_risk"].clip(0.1)
    df["wide_stop"] = (df["stop_loss_pct"] >= 0.75).astype(int)
    df["tight_stop"] = (df["stop_loss_pct"] <= 0.3).astype(int)
    df["high_rr"] = (df["reward_risk"] >= 4.0).astype(int)
    df["low_rr"] = (df["reward_risk"] <= 1.5).astype(int)
    df["vol_filter_off"] = (df["volume_mult"] < 0.5).astype(int)
    df["vol_filter_strict"] = (df["volume_mult"] >= 1.5).astype(int)
    df["calm_vix"] = (df["vix_threshold"] <= 20).astype(int)
    df["multi_trade"] = (df["max_trades_per_day"] >= 3).astype(int)
    df["all_day"] = df["afternoon_entries"].astype(int)

    # Period regime features
    regime_map = {"holdout_2022": -0.5, "bull_2023": 1.0, "bull_2024": 1.0,
                  "bear_2025": -1.0, "full_2yr": 0.5}
    df["regime"] = df["period_label"].map(regime_map).fillna(0)

    # Interaction features
    df["sl_x_regime"] = df["stop_loss_pct"] * df["regime"]
    df["rr_x_regime"] = df["reward_risk"] * df["regime"]
    df["vol_x_regime"] = df["volume_mult"] * df["regime"]
    df["sl_x_sym"] = df["stop_loss_pct"] * df["sym_enc"]
    df["rr_x_sym"] = df["reward_risk"] * df["sym_enc"]
    df["strat_x_sym"] = df["strat_enc"] * df["sym_enc"]
    df["sl_x_strat"] = df["stop_loss_pct"] * df["strat_enc"]
    df["rr_x_strat"] = df["reward_risk"] * df["strat_enc"]

    # Strategy-specific flags
    df["is_orb"] = (df["strategy"] == "orb").astype(int)
    df["is_ema"] = (df["strategy"] == "ema_crossover").astype(int)
    df["is_mom"] = (df["strategy"] == "momentum").astype(int)

    # Symbol volatility proxy (from period results)
    sym_vol = df.groupby("symbol")["max_drawdown"].mean().to_dict()
    df["sym_volatility"] = df["symbol"].map(sym_vol)

    feature_cols = [
        "strat_enc", "sym_enc", "per_enc", "stop_loss_pct", "reward_risk",
        "volume_mult", "regime_filter", "vix_threshold", "max_trades_per_day",
        "afternoon_entries", "be_wr", "risk_per_trade", "target_per_trade",
        "sl_rr_product", "sl_rr_ratio", "wide_stop", "tight_stop",
        "high_rr", "low_rr", "vol_filter_off", "vol_filter_strict",
        "calm_vix", "multi_trade", "all_day", "regime",
        "sl_x_regime", "rr_x_regime", "vol_x_regime",
        "sl_x_sym", "rr_x_sym", "strat_x_sym", "sl_x_strat", "rr_x_strat",
        "is_orb", "is_ema", "is_mom", "sym_volatility",
    ]

    df["profitable"] = (df["profit_factor"] > 1.0).astype(int)
    df["good"] = ((df["profit_factor"] > 1.3) & (df["wr_buffer"] > 5)).astype(int)

    X = df[feature_cols].values
    y_prof = df["profitable"].values
    y_good = df["good"].values

    # Walk-forward split
    train = df["period_label"].isin(["holdout_2022", "bull_2023", "bull_2024"])
    test = df["period_label"] == "bear_2025"

    for target_name, y in [("profitable", y_prof), ("good_config", y_good)]:
        X_tr, y_tr = X[train], y[train]
        X_te, y_te = X[test], y[test]

        model = xgb.XGBClassifier(
            max_depth=8, learning_rate=0.03, n_estimators=1500,
            subsample=0.7, colsample_bytree=0.7,
            device=device, eval_metric="auc", early_stopping_rounds=50,
            verbosity=0, random_state=42,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
        print(f"\n  Target: {target_name}  AUC={auc:.4f}")

        imp = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1])
        print(f"  Top 15 features:")
        for name, val in imp[:15]:
            bar = "█" * int(val * 80)
            print(f"    {name:<22} {val:.4f} {bar}")


# ═══════════════════════════════════════════════════════════════════
# STUDY 2: OPTIMAL EXIT TIMING
# ═══════════════════════════════════════════════════════════════════

def study_2_exit_timing():
    _header(2, "OPTIMAL EXIT TIMING — Are fixed stops/targets the best exit?")

    import yaml
    from src.engine import TradingEngine

    with open(_ROOT / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["mode"] = "backtest"
    cfg["account"] = {"balance": 500.0, "max_position_pct": 0.95}
    engine = TradingEngine(cfg)

    symbols = ["NVDA", "AMD", "SPY"]
    periods = [("2023-01-01", "2025-01-01", "full_2yr"), ("2022-01-01", "2022-12-31", "holdout")]

    # Test different R:R ratios (simulates different exit timing)
    # Lower RR = exit sooner (tighter target), Higher RR = hold longer
    rr_values = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]

    print(f"\n  Testing {len(rr_values)} exit distances x {len(symbols)} symbols x {len(periods)} periods")
    print(f"\n  {'Symbol':<8} {'Period':<10} {'RR':<5} {'Trades':<7} {'WR%':<7} {'PnL':>9} {'PF':>7} {'AvgDur':>7}")
    print(f"  {'-'*65}")

    for sym in symbols:
        for start, end, plabel in periods:
            for rr in rr_values:
                try:
                    m = engine.run_backtest_config(
                        config_override={
                            "active_strategy": "orb", "stop_loss_pct": 0.75,
                            "reward_risk": rr, "volume_mult": 1.2,
                            "regime_filter": True, "vix_threshold": 25,
                            "max_trades_per_day": 3, "afternoon_entries": True,
                            "entry_window_minutes": 360,
                        },
                        start_date=start, end_date=end, symbol=sym,
                    )
                    t = m.get("total_trades", 0)
                    if t >= 10:
                        print(f"  {sym:<8} {plabel:<10} {rr:<5} {t:<7} {m['win_rate']:<7.1f} "
                              f"${m['total_pnl']:>+8.2f} {m['profit_factor']:>7.2f} {m['avg_duration_min']:>6.0f}m")
                except Exception:
                    pass
            print()


# ═══════════════════════════════════════════════════════════════════
# STUDY 3: ENSEMBLE STRATEGY
# ═══════════════════════════════════════════════════════════════════

def study_3_ensemble():
    _header(3, "ENSEMBLE — Do multiple strategies agreeing improve results?")
    import xgboost as xgb
    device = _get_device()

    c = _load_db()
    # For each (symbol, period, params), check how many strategies are profitable
    df = pd.read_sql_query("""
        SELECT strategy, symbol, period_label, stop_loss_pct, reward_risk,
               volume_mult, afternoon_entries, profit_factor, total_trades, win_rate
        FROM backtest_runs
        WHERE total_trades >= 15 AND profit_factor < 100
    """, c)
    c.close()

    # Group by (symbol, period, sl, rr) and count profitable strategies
    group_cols = ["symbol", "period_label", "stop_loss_pct", "reward_risk"]
    grouped = df.groupby(group_cols + ["strategy"])["profit_factor"].first().reset_index()

    agreement = grouped.groupby(group_cols).agg(
        n_strategies=("strategy", "count"),
        n_profitable=("profit_factor", lambda x: (x > 1.0).sum()),
        avg_pf=("profit_factor", "mean"),
        max_pf=("profit_factor", "max"),
    ).reset_index()

    print(f"  Configs tested: {len(agreement):,}")
    print(f"\n  AGREEMENT vs PERFORMANCE:")
    print(f"  {'Strategies Agree':<20} {'Count':>8} {'Avg PF':>8} {'% Profitable':>14}")
    print(f"  {'-'*55}")

    for n in sorted(agreement["n_profitable"].unique()):
        subset = agreement[agreement["n_profitable"] == n]
        avg_pf = subset["avg_pf"].mean()
        pct_prof = (subset["avg_pf"] > 1.0).mean() * 100
        print(f"  {n} strategies agree  {len(subset):>8,} {avg_pf:>8.2f} {pct_prof:>13.1f}%")

    # When 3+ strategies agree it's profitable, what's the actual PF?
    strong = agreement[agreement["n_profitable"] >= 3]
    if not strong.empty:
        print(f"\n  STRONG CONSENSUS (3+ strategies profitable):")
        print(f"  {len(strong):,} configs, avg PF={strong['avg_pf'].mean():.2f}, "
              f"max PF={strong['max_pf'].max():.2f}")

        # Best configs with strong consensus
        best_consensus = strong.sort_values("avg_pf", ascending=False).head(10)
        print(f"\n  Top consensus configs:")
        for _, r in best_consensus.iterrows():
            print(f"    {r['symbol']:<8} {r['period_label']:<14} SL={r['stop_loss_pct']} RR={r['reward_risk']} "
                  f"agree={int(r['n_profitable'])}/{int(r['n_strategies'])} avg_pf={r['avg_pf']:.2f}")


# ═══════════════════════════════════════════════════════════════════
# STUDY 4: REGIME DETECTION
# ═══════════════════════════════════════════════════════════════════

def study_4_regime_detection():
    _header(4, "REGIME DETECTION — Can we predict when to sit out?")
    import xgboost as xgb
    device = _get_device()

    c = _load_db()
    df = pd.read_sql_query("""
        SELECT symbol, period_label, stop_loss_pct, reward_risk,
               volume_mult, afternoon_entries, profit_factor, total_trades,
               win_rate, max_drawdown
        FROM backtest_runs
        WHERE total_trades >= 15 AND profit_factor < 100 AND strategy = 'orb'
    """, c)
    c.close()

    # The question: can we predict from the CONFIG whether bear_2025 will lose?
    # If yes, we know when to sit out.
    df["is_bear"] = (df["period_label"] == "bear_2025").astype(int)
    df["loses_in_bear"] = ((df["period_label"] == "bear_2025") & (df["profit_factor"] < 1.0)).astype(int)

    # Which configs survive bear markets?
    bear = df[df["period_label"] == "bear_2025"]
    bull = df[df["period_label"].isin(["bull_2023", "bull_2024"])]

    print(f"  Bear configs: {len(bear):,}, profitable: {(bear['profit_factor']>1).sum():,} ({(bear['profit_factor']>1).mean()*100:.1f}%)")
    print(f"  Bull configs: {len(bull):,}, profitable: {(bull['profit_factor']>1).sum():,} ({(bull['profit_factor']>1).mean()*100:.1f}%)")

    # What's different about configs that survive bears?
    bear_winners = bear[bear["profit_factor"] > 1.0]
    bear_losers = bear[bear["profit_factor"] <= 1.0]

    if not bear_winners.empty and not bear_losers.empty:
        print(f"\n  BEAR SURVIVORS vs LOSERS (ORB):")
        for col in ["stop_loss_pct", "reward_risk", "volume_mult", "afternoon_entries"]:
            w_mean = bear_winners[col].mean()
            l_mean = bear_losers[col].mean()
            diff = w_mean - l_mean
            direction = "↑" if diff > 0 else "↓"
            print(f"    {col:<22} winners={w_mean:.2f}  losers={l_mean:.2f}  diff={diff:+.3f} {direction}")

    # Per-symbol bear survival
    print(f"\n  BEAR SURVIVAL BY SYMBOL:")
    for sym in ["NVDA", "AMD", "SPY", "QQQ", "AAPL", "COIN", "META"]:
        sb = bear[bear["symbol"] == sym]
        if not sb.empty:
            surv = (sb["profit_factor"] > 1.0).mean() * 100
            avg_pf = sb["profit_factor"].mean()
            print(f"    {sym:<8} {len(sb):>5} configs, {surv:>5.1f}% survive, avg PF={avg_pf:.2f}")


# ═══════════════════════════════════════════════════════════════════
# STUDY 5: VIX-REGIME SPECIFIC CONFIGS
# ═══════════════════════════════════════════════════════════════════

def study_5_vix_regimes():
    _header(5, "VIX-REGIME CONFIGS — Best params for each VIX level")

    c = _load_db()
    df = pd.read_sql_query("""
        SELECT strategy, symbol, period_label, stop_loss_pct, reward_risk,
               volume_mult, vix_threshold, profit_factor, total_trades, win_rate
        FROM backtest_runs
        WHERE total_trades >= 15 AND profit_factor < 100
    """, c)
    c.close()

    print(f"  Loaded {len(df):,} runs")

    # Group by VIX threshold and find best configs
    print(f"\n  PERFORMANCE BY VIX THRESHOLD:")
    print(f"  {'VIX':<8} {'Count':>8} {'Avg PF':>8} {'% Prof':>8} {'Best PF':>8}")
    print(f"  {'-'*44}")
    for vix in sorted(df["vix_threshold"].unique()):
        sub = df[df["vix_threshold"] == vix]
        print(f"  ≤{vix:<6} {len(sub):>8,} {sub['profit_factor'].mean():>8.2f} "
              f"{(sub['profit_factor']>1).mean()*100:>7.1f}% {sub['profit_factor'].max():>8.2f}")

    # Best config per VIX level per symbol
    print(f"\n  BEST CONFIG PER VIX LEVEL (full_2yr + holdout):")
    for vix in sorted(df["vix_threshold"].unique()):
        print(f"\n  VIX ≤ {vix}:")
        sub = df[(df["vix_threshold"] == vix) &
                 (df["period_label"].isin(["full_2yr", "holdout_2022"]))]
        for sym in ["NVDA", "AMD", "SPY", "QQQ"]:
            sym_sub = sub[sub["symbol"] == sym].sort_values("profit_factor", ascending=False)
            if not sym_sub.empty:
                best = sym_sub.iloc[0]
                print(f"    {sym:<8} {best['strategy']:<16} SL={best['stop_loss_pct']} RR={best['reward_risk']} "
                      f"PF={best['profit_factor']:.2f} trades={int(best['total_trades'])}")


# ═══════════════════════════════════════════════════════════════════
# STUDY 6: DAY-OF-WEEK + HOUR-OF-DAY
# ═══════════════════════════════════════════════════════════════════

def study_6_temporal():
    _header(6, "TEMPORAL PATTERNS — Day-of-week and hour-of-day effects")
    import xgboost as xgb
    device = _get_device()

    c = _load_db()
    df = pd.read_sql_query("""
        SELECT strategy, symbol, period_label, stop_loss_pct, reward_risk,
               profit_factor, total_trades, win_rate, avg_duration_min
        FROM backtest_runs
        WHERE total_trades >= 20 AND profit_factor < 100
    """, c)
    c.close()

    # Avg duration tells us something about when trades happen
    # Short duration (~15-30 min) = morning trades
    # Long duration (~200+ min) = held to EOD (missed target)

    print(f"  Loaded {len(df):,} runs")

    print(f"\n  TRADE DURATION vs PROFITABILITY:")
    bins = [(0, 30, "Quick (<30m)"), (30, 60, "Medium (30-60m)"),
            (60, 120, "Long (1-2h)"), (120, 240, "Extended (2-4h)"), (240, 999, "EOD (4h+)")]

    for lo, hi, label in bins:
        sub = df[(df["avg_duration_min"] >= lo) & (df["avg_duration_min"] < hi)]
        if not sub.empty:
            pct = (sub["profit_factor"] > 1.0).mean() * 100
            avg = sub["profit_factor"].mean()
            print(f"  {label:<20} {len(sub):>6,} runs  avg PF={avg:.2f}  profitable={pct:.1f}%")

    # Quick exits (target hit) vs long exits (EOD) by strategy
    print(f"\n  QUICK vs SLOW by STRATEGY:")
    for strat in ["orb", "ema_crossover", "momentum"]:
        sub = df[df["strategy"] == strat]
        quick = sub[sub["avg_duration_min"] < 60]
        slow = sub[sub["avg_duration_min"] >= 120]
        if not quick.empty and not slow.empty:
            print(f"  {strat:<16} quick PF={quick['profit_factor'].mean():.2f} ({len(quick):,})  "
                  f"slow PF={slow['profit_factor'].mean():.2f} ({len(slow):,})")


# ═══════════════════════════════════════════════════════════════════
# STUDY 7: CROSS-ASSET CORRELATION
# ═══════════════════════════════════════════════════════════════════

def study_7_correlation():
    _header(7, "CROSS-ASSET CORRELATION — Symbol diversification analysis")

    c = _load_db()
    # Get PF for each (symbol, period, config) to compute correlations
    df = pd.read_sql_query("""
        SELECT symbol, period_label, stop_loss_pct, reward_risk,
               profit_factor
        FROM backtest_runs
        WHERE total_trades >= 15 AND profit_factor < 100 AND strategy = 'orb'
    """, c)
    c.close()

    # Pivot: rows = (period, sl, rr), columns = symbols, values = PF
    df["config"] = df["period_label"] + "_" + df["stop_loss_pct"].astype(str) + "_" + df["reward_risk"].astype(str)
    pivot = df.pivot_table(index="config", columns="symbol", values="profit_factor", aggfunc="mean")

    symbols = ["NVDA", "AMD", "SPY", "QQQ", "AAPL", "COIN", "META"]
    pivot = pivot[[s for s in symbols if s in pivot.columns]].dropna()

    if pivot.empty:
        print("  Not enough overlapping data")
        return

    corr = pivot.corr()
    print(f"  Correlation matrix ({len(pivot)} shared configs):")
    print(f"\n  {'':8}", end="")
    for s in corr.columns:
        print(f"{s:>8}", end="")
    print()
    for s1 in corr.index:
        print(f"  {s1:<8}", end="")
        for s2 in corr.columns:
            v = corr.loc[s1, s2]
            marker = "●" if abs(v) > 0.5 else "○"
            print(f"{v:>7.2f}{marker}", end="")
        print()

    # Best diversification pairs (lowest correlation)
    pairs = []
    syms = list(corr.columns)
    for i in range(len(syms)):
        for j in range(i+1, len(syms)):
            pairs.append((syms[i], syms[j], corr.loc[syms[i], syms[j]]))
    pairs.sort(key=lambda x: x[2])

    print(f"\n  BEST DIVERSIFICATION PAIRS (lowest correlation):")
    for s1, s2, c_val in pairs[:5]:
        print(f"    {s1} + {s2}: correlation = {c_val:.3f}")
    print(f"\n  WORST (most correlated):")
    for s1, s2, c_val in pairs[-3:]:
        print(f"    {s1} + {s2}: correlation = {c_val:.3f}")


# ═══════════════════════════════════════════════════════════════════
# STUDY 8: WALK-FORWARD META-LEARNER
# ═══════════════════════════════════════════════════════════════════

def study_8_meta_learner():
    _header(8, "WALK-FORWARD META-LEARNER — ML-guided config selection")
    import xgboost as xgb
    device = _get_device()

    c = _load_db()
    df = pd.read_sql_query("""
        SELECT strategy, symbol, period_label, stop_loss_pct, reward_risk,
               volume_mult, vix_threshold, max_trades_per_day, afternoon_entries,
               profit_factor, total_trades, win_rate
        FROM backtest_runs
        WHERE total_trades >= 15 AND profit_factor < 100
    """, c)
    c.close()

    le_s = LabelEncoder(); le_sym = LabelEncoder()
    df["strat_enc"] = le_s.fit_transform(df["strategy"])
    df["sym_enc"] = le_sym.fit_transform(df["symbol"])
    df["profitable"] = (df["profit_factor"] > 1.0).astype(int)

    features = ["strat_enc", "sym_enc", "stop_loss_pct", "reward_risk",
                "volume_mult", "vix_threshold", "max_trades_per_day", "afternoon_entries"]

    # Walk-forward: train on period N, pick top configs, test on period N+1
    period_order = ["holdout_2022", "bull_2023", "bull_2024", "bear_2025"]

    print(f"  Testing ML-guided config selection across 3 transitions")
    print(f"\n  {'Train Period':<16} {'Test Period':<16} {'ML Top-10 PF':>13} {'Random 10 PF':>13} {'Best 10 PF':>12}")
    print(f"  {'-'*72}")

    for i in range(len(period_order) - 1):
        train_p = period_order[i]
        test_p = period_order[i + 1]

        train_df = df[df["period_label"] == train_p]
        test_df = df[df["period_label"] == test_p]

        if len(train_df) < 100 or len(test_df) < 100:
            continue

        X_tr = train_df[features].values
        y_tr = train_df["profitable"].values

        model = xgb.XGBClassifier(
            max_depth=6, learning_rate=0.05, n_estimators=500,
            device=device, verbosity=0, random_state=42,
        )
        model.fit(X_tr, y_tr)

        # Score test configs
        X_te = test_df[features].values
        test_df = test_df.copy()
        test_df["pred_prob"] = model.predict_proba(X_te)[:, 1]

        # Pick top 10 by model
        top10_ml = test_df.nlargest(10, "pred_prob")
        random10 = test_df.sample(10, random_state=42)
        top10_actual = test_df.nlargest(10, "profit_factor")

        ml_pf = top10_ml["profit_factor"].mean()
        rand_pf = random10["profit_factor"].mean()
        best_pf = top10_actual["profit_factor"].mean()

        print(f"  {train_p:<16} {test_p:<16} {ml_pf:>13.2f} {rand_pf:>13.2f} {best_pf:>12.2f}")


# ═══════════════════════════════════════════════════════════════════
# STUDY 9: SLIPPAGE MODEL
# ═══════════════════════════════════════════════════════════════════

def study_9_slippage_model():
    _header(9, "SLIPPAGE MODEL — Predict which configs are slippage-resistant")
    import xgboost as xgb
    device = _get_device()

    c = _load_db()
    # Check if we have slippage results (from overnight research)
    df = pd.read_sql_query("""
        SELECT strategy, symbol, period_label, stop_loss_pct, reward_risk,
               volume_mult, profit_factor, total_trades, win_rate, max_drawdown
        FROM backtest_runs
        WHERE total_trades >= 15 AND profit_factor < 100
              AND period_label LIKE 'slip005_%'
    """, c)

    if df.empty:
        print("  No slippage-aware results in DB (slip005_ prefix)")
        # Fall back: analyze relationship between PF and stop width
        df = pd.read_sql_query("""
            SELECT strategy, symbol, stop_loss_pct, reward_risk,
                   volume_mult, profit_factor, total_trades
            FROM backtest_runs
            WHERE total_trades >= 15 AND profit_factor < 100
                  AND period_label IN ('full_2yr', 'holdout_2022')
        """, c)
        c.close()

        print(f"  Analyzing slippage resistance from {len(df):,} runs")
        print(f"  Proxy: configs with wider stops and lower R:R are more slippage-resistant")

        # Slippage resistance proxy: PF * (1 - 0.05/stop_loss_pct)
        # Wider stop = less PF lost to slippage
        df["slip_factor"] = 0.05 / df["stop_loss_pct"].clip(0.1)
        df["adjusted_pf"] = df["profit_factor"] * (1 - df["slip_factor"])

        print(f"\n  SLIPPAGE-ADJUSTED PF by STOP WIDTH:")
        for sl in sorted(df["stop_loss_pct"].unique()):
            sub = df[df["stop_loss_pct"] == sl]
            raw = sub["profit_factor"].mean()
            adj = sub["adjusted_pf"].mean()
            print(f"    SL={sl}%: raw PF={raw:.3f} → adjusted PF={adj:.3f} (lost {(raw-adj)/raw*100:.1f}%)")
    else:
        c.close()
        print(f"  Found {len(df):,} slippage-aware results")
        print(f"  Analyzing slippage-aware PF by config...")

        for col in ["stop_loss_pct", "reward_risk", "volume_mult"]:
            print(f"\n  {col}:")
            for val in sorted(df[col].unique()):
                sub = df[df[col] == val]
                if not sub.empty:
                    avg = sub["profit_factor"].mean()
                    pct = (sub["profit_factor"] > 1.0).mean() * 100
                    print(f"    {val}: avg PF={avg:.3f}, profitable={pct:.1f}%")


# ═══════════════════════════════════════════════════════════════════
# STUDY 10: MULTI-TIMEFRAME
# ═══════════════════════════════════════════════════════════════════

def study_10_multi_timeframe():
    _header(10, "MULTI-TIMEFRAME — Does trade duration predict success?")
    import xgboost as xgb
    device = _get_device()

    c = _load_db()
    df = pd.read_sql_query("""
        SELECT strategy, symbol, period_label, stop_loss_pct, reward_risk,
               volume_mult, afternoon_entries, profit_factor, total_trades,
               win_rate, avg_duration_min, sharpe_ratio, max_drawdown
        FROM backtest_runs
        WHERE total_trades >= 15 AND profit_factor < 100 AND avg_duration_min > 0
    """, c)
    c.close()

    le_s = LabelEncoder(); le_sym = LabelEncoder(); le_p = LabelEncoder()
    df["strat_enc"] = le_s.fit_transform(df["strategy"])
    df["sym_enc"] = le_sym.fit_transform(df["symbol"])
    df["per_enc"] = le_p.fit_transform(df["period_label"])

    # Duration-based features
    df["quick_exit"] = (df["avg_duration_min"] < 30).astype(int)
    df["medium_exit"] = ((df["avg_duration_min"] >= 30) & (df["avg_duration_min"] < 120)).astype(int)
    df["eod_exit"] = (df["avg_duration_min"] >= 240).astype(int)
    df["duration_bucket"] = pd.cut(df["avg_duration_min"], bins=[0, 30, 60, 120, 240, 999], labels=[0,1,2,3,4]).astype(int)

    df["profitable"] = (df["profit_factor"] > 1.0).astype(int)

    features = ["strat_enc", "sym_enc", "per_enc", "stop_loss_pct", "reward_risk",
                "volume_mult", "afternoon_entries", "avg_duration_min",
                "quick_exit", "medium_exit", "eod_exit", "duration_bucket"]

    X = df[features].values
    y = df["profitable"].values

    train = df["period_label"].isin(["holdout_2022", "bull_2023", "bull_2024"])
    test = df["period_label"] == "bear_2025"

    model = xgb.XGBClassifier(
        max_depth=6, learning_rate=0.05, n_estimators=500,
        device=device, eval_metric="auc", early_stopping_rounds=50,
        verbosity=0, random_state=42,
    )
    model.fit(X[train], y[train], eval_set=[(X[test], y[test])], verbose=False)
    auc = roc_auc_score(y[test], model.predict_proba(X[test])[:, 1])
    print(f"\n  AUC with duration features: {auc:.4f}")

    imp = sorted(zip(features, model.feature_importances_), key=lambda x: -x[1])
    print(f"  Feature importance:")
    for name, val in imp:
        bar = "█" * int(val * 60)
        print(f"    {name:<22} {val:.4f} {bar}")

    # Key finding: do quick exits predict profitability?
    print(f"\n  DURATION vs PROFITABILITY (bear_2025 only):")
    bear = df[df["period_label"] == "bear_2025"]
    for label, mask in [("Quick <30m", bear["avg_duration_min"] < 30),
                        ("Medium 30-120m", (bear["avg_duration_min"] >= 30) & (bear["avg_duration_min"] < 120)),
                        ("Long 2-4h", (bear["avg_duration_min"] >= 120) & (bear["avg_duration_min"] < 240)),
                        ("EOD 4h+", bear["avg_duration_min"] >= 240)]:
        sub = bear[mask]
        if not sub.empty:
            pct = (sub["profit_factor"] > 1.0).mean() * 100
            avg = sub["profit_factor"].mean()
            print(f"    {label:<16} {len(sub):>5} runs  avg PF={avg:.2f}  profitable={pct:.1f}%")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = _time.time()

    study_1_feature_discovery()
    study_2_exit_timing()
    study_3_ensemble()
    study_4_regime_detection()
    study_5_vix_regimes()
    study_6_temporal()
    study_7_correlation()
    study_8_meta_learner()
    study_9_slippage_model()
    study_10_multi_timeframe()

    total = (_time.time() - t0) / 60
    print(f"\n{'='*70}")
    print(f"  ALL 10 STUDIES COMPLETE in {total:.1f} minutes ({total/60:.1f} hours)")
    print(f"{'='*70}")
