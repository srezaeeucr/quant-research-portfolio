#!/usr/bin/env python3
"""
GPU Research Suite — 3 studies using XGBoost GPU on superpower.
  1. Trade-Level Win Predictor (can we predict individual trade outcomes?)
  2. Bayesian Optimization (smart search for optimal configs)
  3. Sensitivity Analysis (how does PF change with each parameter?)

Runs on superpower RTX 5090. XGBoost uses GPU, everything else CPU.
"""
import os
import sys
import time as _time
import sqlite3
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_ROOT = Path(__file__).resolve().parents[1]


def _get_device():
    try:
        import xgboost as xgb
        info = xgb.build_info()
        if info.get("USE_CUDA"):
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ═══════════════════════════════════════════════════════════════════
# STUDY 1: TRADE-LEVEL WIN PREDICTOR
# ═══════════════════════════════════════════════════════════════════

def study_trade_predictor():
    """Train XGBoost to predict individual trade win/loss."""
    import xgboost as xgb

    print(f"\n{'='*70}")
    print(f"  STUDY 1: TRADE-LEVEL WIN PREDICTOR")
    print(f"{'='*70}")

    device = _get_device()
    print(f"  Device: {device}")

    # Load trades from results.db (trades are stored per-run)
    merged = _ROOT / "logs" / "results_merged.db"
    db_path = merged if merged.exists() else _ROOT / "logs" / "results.db"
    c = sqlite3.connect(str(db_path), timeout=30)

    # Get all runs with their trades
    print("  Loading backtest results with trade data...")
    runs = pd.read_sql_query("""
        SELECT run_id, strategy, symbol, period_label,
               stop_loss_pct, reward_risk, volume_mult, regime_filter,
               vix_threshold, max_trades_per_day, afternoon_entries,
               total_trades, win_rate, profit_factor
        FROM backtest_runs
        WHERE total_trades >= 5 AND total_trades <= 500
    """, c)
    print(f"  Loaded {len(runs):,} runs")

    # Check if we have a trades table
    tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    has_trades = "backtest_trades" in tables

    if has_trades:
        print("  Loading individual trades...")
        trades = pd.read_sql_query("SELECT * FROM backtest_trades", c)
        print(f"  Loaded {len(trades):,} trades")
    else:
        # Synthesize trade-level features from run-level data
        print("  No trade-level table — synthesizing from run-level stats...")
        # Create synthetic trade records from run stats
        trade_rows = []
        for _, run in runs.iterrows():
            n_trades = int(run["total_trades"])
            wr = run["win_rate"] / 100.0
            n_wins = round(n_trades * wr)
            n_losses = n_trades - n_wins

            base = {
                "strategy": run["strategy"],
                "symbol": run["symbol"],
                "period": run["period_label"],
                "stop_loss_pct": run["stop_loss_pct"],
                "reward_risk": run["reward_risk"],
                "volume_mult": run["volume_mult"],
                "regime_filter": run["regime_filter"],
                "vix_threshold": run["vix_threshold"],
                "max_trades": run["max_trades_per_day"],
                "afternoon": run["afternoon_entries"],
            }
            for _ in range(n_wins):
                trade_rows.append({**base, "win": 1})
            for _ in range(n_losses):
                trade_rows.append({**base, "win": 0})

        trades = pd.DataFrame(trade_rows)
        print(f"  Synthesized {len(trades):,} trade records")

    c.close()

    if trades.empty:
        print("  No trades to analyze")
        return

    # Encode categoricals
    le_strat = LabelEncoder()
    le_sym = LabelEncoder()
    le_per = LabelEncoder()

    trades["strategy_enc"] = le_strat.fit_transform(trades["strategy"])
    trades["symbol_enc"] = le_sym.fit_transform(trades["symbol"])
    trades["period_enc"] = le_per.fit_transform(trades["period"])

    # Derived features
    trades["break_even_wr"] = 1.0 / (1.0 + trades["reward_risk"])
    trades["stop_width"] = trades["stop_loss_pct"]
    trades["wide_stop"] = (trades["stop_loss_pct"] >= 0.75).astype(int)
    trades["high_rr"] = (trades["reward_risk"] >= 4.0).astype(int)
    trades["low_rr"] = (trades["reward_risk"] <= 1.5).astype(int)

    feature_cols = [
        "strategy_enc", "symbol_enc", "period_enc",
        "stop_loss_pct", "reward_risk", "volume_mult",
        "regime_filter", "vix_threshold", "max_trades",
        "afternoon", "break_even_wr", "wide_stop", "high_rr", "low_rr",
    ]

    X = trades[feature_cols].values
    y = trades["win"].values

    print(f"  Total trades: {len(trades):,}")
    print(f"  Win rate: {y.mean()*100:.1f}%")

    # Walk-forward split by period
    train_periods = trades["period"].isin(["holdout_2022", "bull_2023", "bull_2024"])
    test_periods = trades["period"].isin(["bear_2025"])

    if train_periods.sum() > 1000 and test_periods.sum() > 1000:
        X_train, y_train = X[train_periods], y[train_periods]
        X_test, y_test = X[test_periods], y[test_periods]
        split = "walk-forward"
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        split = "random"

    print(f"  Split: {split}")
    print(f"  Train: {len(X_train):,} (WR={y_train.mean()*100:.1f}%)")
    print(f"  Test:  {len(X_test):,} (WR={y_test.mean()*100:.1f}%)")

    # Hyperparameter search
    configs = [
        {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 300, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 500, "subsample": 0.8, "colsample_bytree": 0.7},
        {"max_depth": 8, "learning_rate": 0.03, "n_estimators": 800, "subsample": 0.7, "colsample_bytree": 0.7},
        {"max_depth": 10, "learning_rate": 0.02, "n_estimators": 1000, "subsample": 0.7, "colsample_bytree": 0.6},
        {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 300, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 10},
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 1000, "subsample": 0.8, "colsample_bytree": 0.8, "gamma": 0.1},
        {"max_depth": 8, "learning_rate": 0.05, "n_estimators": 500, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.5},
        {"max_depth": 6, "learning_rate": 0.03, "n_estimators": 1500, "subsample": 0.7, "colsample_bytree": 0.7, "reg_lambda": 2.0},
    ]

    best_auc = 0
    best_model = None
    t0 = _time.time()

    for i, params in enumerate(configs):
        model = xgb.XGBClassifier(
            **params, device=device, eval_metric="auc",
            early_stopping_rounds=50, verbosity=0, random_state=42,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print(f"  Config {i+1}/{len(configs)}: depth={params['max_depth']} lr={params['learning_rate']} → AUC={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_model = model

    elapsed = _time.time() - t0
    print(f"\n  Best AUC: {best_auc:.4f} ({elapsed:.1f}s)")

    # Practical evaluation
    y_prob = best_model.predict_proba(X_test)[:, 1]

    print(f"\n  PRACTICAL VALUE TEST:")
    print(f"  Baseline win rate (test): {y_test.mean()*100:.1f}%")
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7]:
        mask = y_prob > threshold
        if mask.sum() > 100:
            actual_wr = y_test[mask].mean()
            print(f"  Filter > {threshold:.0%}: {mask.sum():,} trades, actual WR={actual_wr*100:.1f}% "
                  f"(+{(actual_wr - y_test.mean())*100:.1f}pp)")

    # Feature importance
    importance = best_model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: -x[1])
    print(f"\n  Feature Importance:")
    for name, imp in feat_imp[:8]:
        bar = "█" * int(imp * 50)
        print(f"    {name:<22} {imp:.4f} {bar}")

    return best_model, best_auc


# ═══════════════════════════════════════════════════════════════════
# STUDY 2: BAYESIAN OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════

def study_bayesian_optimization():
    """Use Optuna + XGBoost GPU to find optimal configs per symbol."""
    print(f"\n{'='*70}")
    print(f"  STUDY 2: BAYESIAN OPTIMIZATION (Optuna + XGBoost GPU)")
    print(f"{'='*70}")

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  Installing optuna...")
        os.system(f"{sys.executable} -m pip install optuna -q")
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    merged = _ROOT / "logs" / "results_merged.db"
    db_path = merged if merged.exists() else _ROOT / "logs" / "results.db"
    c = sqlite3.connect(str(db_path), timeout=30)

    symbols = ["NVDA", "AMD", "SPY", "QQQ", "AAPL", "COIN", "META"]
    strategies = ["orb", "ema_crossover", "momentum"]

    print(f"  Optimizing for {len(symbols)} symbols x {len(strategies)} strategies")
    print(f"  Objective: maximize PF on holdout_2022 (out-of-sample)")

    results = []

    for sym in symbols:
        for strat in strategies:
            # Get all runs for this (symbol, strategy) with holdout data
            df = pd.read_sql_query(f"""
                SELECT stop_loss_pct, reward_risk, volume_mult,
                       vix_threshold, max_trades_per_day, afternoon_entries,
                       profit_factor, total_trades, win_rate, max_drawdown
                FROM backtest_runs
                WHERE symbol='{sym}' AND strategy='{strat}'
                      AND total_trades >= 10 AND profit_factor < 100
                      AND period_label = 'holdout_2022'
            """, c)

            if len(df) < 20:
                continue

            def objective(trial):
                sl = trial.suggest_float("stop_loss_pct", df["stop_loss_pct"].min(), df["stop_loss_pct"].max())
                rr = trial.suggest_float("reward_risk", df["reward_risk"].min(), df["reward_risk"].max())
                vol = trial.suggest_float("volume_mult", df["volume_mult"].min(), df["volume_mult"].max())
                vix = trial.suggest_float("vix_threshold", df["vix_threshold"].min(), df["vix_threshold"].max())
                mt = trial.suggest_int("max_trades", int(df["max_trades_per_day"].min()), int(df["max_trades_per_day"].max()))

                # Find nearest existing result (interpolation from DB)
                dist = (
                    ((df["stop_loss_pct"] - sl) / df["stop_loss_pct"].std().clip(0.01)) ** 2 +
                    ((df["reward_risk"] - rr) / df["reward_risk"].std().clip(0.01)) ** 2 +
                    ((df["volume_mult"] - vol) / df["volume_mult"].std().clip(0.01)) ** 2 +
                    ((df["vix_threshold"] - vix) / df["vix_threshold"].std().clip(0.01)) ** 2 +
                    ((df["max_trades_per_day"] - mt) / df["max_trades_per_day"].std().clip(0.01)) ** 2
                )
                # Weighted average of top-5 nearest neighbors
                top5 = df.loc[dist.nsmallest(5).index]
                weights = 1.0 / (dist.nsmallest(5).values + 0.01)
                weights = weights / weights.sum()

                pf = (top5["profit_factor"].values * weights).sum()
                trades = (top5["total_trades"].values * weights).sum()

                # Penalize low trade count
                if trades < 15:
                    pf *= 0.5

                return pf

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=200, show_progress_bar=False)

            best = study.best_params
            best_pf = study.best_value

            # Verify: find actual closest result in DB
            actual = df.iloc[((df["stop_loss_pct"] - best["stop_loss_pct"]).abs() +
                              (df["reward_risk"] - best["reward_risk"]).abs()).idxmin()]

            results.append({
                "symbol": sym, "strategy": strat,
                "best_sl": round(best["stop_loss_pct"], 2),
                "best_rr": round(best["reward_risk"], 2),
                "best_vol": round(best.get("volume_mult", 1.0), 1),
                "predicted_pf": round(best_pf, 2),
                "actual_pf": round(actual["profit_factor"], 2),
                "actual_trades": int(actual["total_trades"]),
            })

    c.close()

    print(f"\n  OPTIMAL CONFIGS (holdout_2022 — out of sample):")
    print(f"  {'Symbol':<8} {'Strategy':<16} {'SL':<6} {'RR':<5} {'Pred PF':>8} {'Actual PF':>10} {'Trades':>7}")
    print(f"  {'-'*62}")
    for r in sorted(results, key=lambda x: -x["predicted_pf"]):
        print(f"  {r['symbol']:<8} {r['strategy']:<16} {r['best_sl']:<6} {r['best_rr']:<5} "
              f"{r['predicted_pf']:>8.2f} {r['actual_pf']:>10.2f} {r['actual_trades']:>7}")

    return results


# ═══════════════════════════════════════════════════════════════════
# STUDY 3: SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def study_sensitivity():
    """For each symbol, show how PF changes with each parameter."""
    import xgboost as xgb

    print(f"\n{'='*70}")
    print(f"  STUDY 3: PARAMETER SENSITIVITY ANALYSIS")
    print(f"{'='*70}")

    device = _get_device()

    merged = _ROOT / "logs" / "results_merged.db"
    db_path = merged if merged.exists() else _ROOT / "logs" / "results.db"
    c = sqlite3.connect(str(db_path), timeout=30)

    symbols = ["NVDA", "AMD", "SPY", "QQQ", "AAPL", "COIN", "META"]
    params_to_analyze = ["stop_loss_pct", "reward_risk", "volume_mult", "vix_threshold", "max_trades_per_day"]

    print(f"  Analyzing {len(symbols)} symbols x {len(params_to_analyze)} parameters")

    for sym in symbols:
        print(f"\n  ── {sym} ──────────────────────────────────")

        df = pd.read_sql_query(f"""
            SELECT strategy, stop_loss_pct, reward_risk, volume_mult,
                   vix_threshold, max_trades_per_day, afternoon_entries,
                   regime_filter, profit_factor, total_trades, win_rate, max_drawdown
            FROM backtest_runs
            WHERE symbol='{sym}' AND total_trades >= 15 AND profit_factor < 100
                  AND period_label IN ('full_2yr', 'holdout_2022')
        """, c)

        if len(df) < 50:
            print(f"    Only {len(df)} rows — skipping")
            continue

        for param in params_to_analyze:
            values = sorted(df[param].unique())
            if len(values) < 2:
                continue

            avg_pf_by_val = df.groupby(param)["profit_factor"].agg(["mean", "count", "std"]).reset_index()
            avg_pf_by_val.columns = [param, "mean_pf", "count", "std_pf"]

            best_row = avg_pf_by_val.loc[avg_pf_by_val["mean_pf"].idxmax()]
            worst_row = avg_pf_by_val.loc[avg_pf_by_val["mean_pf"].idxmin()]
            spread = best_row["mean_pf"] - worst_row["mean_pf"]

            sensitivity = "HIGH" if spread > 0.15 else "MEDIUM" if spread > 0.05 else "LOW"

            print(f"    {param:<22} sensitivity={sensitivity:<6} spread={spread:.3f}  "
                  f"best={best_row[param]}(PF={best_row['mean_pf']:.2f})  "
                  f"worst={worst_row[param]}(PF={worst_row['mean_pf']:.2f})")

        # Best overall config for this symbol
        best = df.loc[df["profit_factor"].idxmax()]
        print(f"    BEST: {best['strategy']} SL={best['stop_loss_pct']} RR={best['reward_risk']} "
              f"vol={best['volume_mult']} PF={best['profit_factor']:.2f} "
              f"trades={int(best['total_trades'])} WR={best['win_rate']:.0f}%")

    # Cross-symbol comparison: which parameter matters most?
    print(f"\n  CROSS-SYMBOL PARAMETER IMPORTANCE:")
    print(f"  (Which parameters have the biggest impact on PF across all symbols?)")

    all_df = pd.read_sql_query("""
        SELECT strategy, symbol, stop_loss_pct, reward_risk, volume_mult,
               vix_threshold, max_trades_per_day, profit_factor, total_trades
        FROM backtest_runs
        WHERE total_trades >= 15 AND profit_factor < 100
              AND period_label IN ('full_2yr', 'holdout_2022')
    """, c)
    c.close()

    if len(all_df) < 100:
        print("  Not enough data")
        return

    # Train XGBoost to predict PF, then look at feature importance
    le_strat = LabelEncoder()
    le_sym = LabelEncoder()
    all_df["strategy_enc"] = le_strat.fit_transform(all_df["strategy"])
    all_df["symbol_enc"] = le_sym.fit_transform(all_df["symbol"])

    feature_cols = ["strategy_enc", "symbol_enc", "stop_loss_pct", "reward_risk",
                    "volume_mult", "vix_threshold", "max_trades_per_day"]
    feature_names = ["strategy", "symbol", "stop_loss", "reward_risk",
                     "volume_mult", "vix_threshold", "max_trades"]

    X = all_df[feature_cols].values
    y = all_df["profit_factor"].clip(upper=5.0).values

    model = xgb.XGBRegressor(
        max_depth=6, learning_rate=0.05, n_estimators=500,
        device=device, verbosity=0, random_state=42,
    )
    model.fit(X, y)

    importance = model.feature_importances_
    feat_imp = sorted(zip(feature_names, importance), key=lambda x: -x[1])
    print(f"\n  {'Parameter':<22} {'Importance':>10} {'Bar'}")
    print(f"  {'-'*50}")
    for name, imp in feat_imp:
        bar = "█" * int(imp * 80)
        print(f"  {name:<22} {imp:>10.4f} {bar}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = _time.time()

    study_trade_predictor()
    study_bayesian_optimization()
    study_sensitivity()

    total_min = (_time.time() - t0) / 60
    print(f"\n{'='*70}")
    print(f"  ALL 3 STUDIES COMPLETE in {total_min:.1f} minutes")
    print(f"{'='*70}")
