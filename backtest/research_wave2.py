#!/usr/bin/env python3
"""
Research Wave 2 — Trade Feature Analysis (logistic + XGBoost variants)

Uses the existing run_trades table (~17M trades) to fit two simple
classifiers that predict win/loss from features available at entry time.
The goal is NOT a deployable filter — it's to find which features have
real predictive power so we can optionally add them as additional gates
to the live strategy.

Features (all known at entry time, no lookahead):
  - hour_of_day, minute_of_hour, day_of_week, month
  - reward_risk, stop_loss_pct, volume_mult, regime_filter, vix_threshold
  - max_trades_per_day, afternoon_entries
  - period_label (one-hot: bull / bear / holdout / full)
  - duration_minutes (NOTE: only for analysis post-hoc, NOT a runtime feature)
  - shares (size proxy)

Outputs:
  logs/research/wave2_features.txt    Top features by coefficient + AUC report
  logs/research/wave2_predictions.csv  Per-trade predicted_proba on a held-out set

This is the simpler, more interpretable cousin of XGBoost in Prompt 3 —
linear logistic regression on a huge dataset is hard to overfit and gives
honest signal.  We also fit XGBoost as a comparison.
"""
import os
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import argparse
import sqlite3
import sys
import time as _time_mod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]


def load_trades_with_run_params(min_trades_per_run: int = 5):
    """Load run_trades joined with the parent run params from backtest_runs."""
    print("Loading trades + run params from results.db...")
    conn = sqlite3.connect(str(_ROOT / "logs" / "results.db"), timeout=60)
    sql = """
        SELECT
            t.run_id, t.symbol, t.shares, t.entry_price, t.exit_price,
            t.pnl, t.pnl_pct, t.reason, t.entry_time, t.exit_time,
            t.duration_minutes,
            r.strategy, r.period_label, r.start_date, r.end_date,
            r.stop_loss_pct, r.reward_risk, r.volume_mult,
            r.regime_filter, r.vix_threshold, r.max_trades_per_day,
            r.afternoon_entries
        FROM run_trades t
        JOIN backtest_runs r ON t.run_id = r.run_id
    """
    df = pd.read_sql_query(sql, conn)
    conn.close()
    print(f"  loaded {len(df):,} trades")
    return df


def build_features(df: pd.DataFrame) -> tuple:
    """Extract features + labels from the trades DataFrame."""
    print("Building features...")
    df = df.copy()
    df["entry_dt"] = pd.to_datetime(df["entry_time"], errors="coerce", utc=True)
    df = df.dropna(subset=["entry_dt"])

    # Convert to ET to get realistic hour/minute
    et = df["entry_dt"].dt.tz_convert("America/New_York")
    df["hour"]        = et.dt.hour
    df["minute"]      = et.dt.minute
    df["day_of_week"] = et.dt.dayofweek
    df["month"]       = et.dt.month

    # Numeric features
    feats = pd.DataFrame({
        "hour":           df["hour"],
        "minute":         df["minute"],
        "day_of_week":    df["day_of_week"],
        "month":          df["month"],
        "reward_risk":    df["reward_risk"],
        "stop_loss_pct":  df["stop_loss_pct"],
        "volume_mult":    df["volume_mult"],
        "regime_filter":  df["regime_filter"].astype(int),
        "vix_threshold":  df["vix_threshold"],
        "max_trades":     df["max_trades_per_day"],
        "afternoon":      df["afternoon_entries"].astype(int),
        "shares":         df["shares"],
        "entry_price":    df["entry_price"],
        # Derived
        "shares_x_price": df["shares"] * df["entry_price"],
    })
    # One-hot strategy
    for s in ["orb", "vwap_reversion", "gap_fill", "ema_crossover", "momentum"]:
        feats[f"is_{s}"] = (df["strategy"] == s).astype(int)
    # One-hot period (high-level)
    feats["period_bull"] = df["period_label"].str.contains("bull",    na=False).astype(int)
    feats["period_bear"] = df["period_label"].str.contains("bear",    na=False).astype(int)
    feats["period_full"] = df["period_label"].str.contains("full",    na=False).astype(int)
    feats["period_hold"] = df["period_label"].str.contains("holdout", na=False).astype(int)

    # Label: 1 if win, 0 if loss/break-even
    label = (df["pnl"] > 0).astype(int)

    # Drop NaNs
    mask = feats.notna().all(axis=1) & label.notna()
    feats = feats[mask].reset_index(drop=True)
    label = label[mask].reset_index(drop=True)

    print(f"  features shape: {feats.shape}")
    print(f"  win rate: {label.mean():.3f}")
    return feats, label


def train_logistic(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, classification_report

    print("\n=== LOGISTIC REGRESSION ===")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    t0 = _time_mod.time()
    model = LogisticRegression(max_iter=1000, n_jobs=-1, C=1.0)
    model.fit(X_train_s, y_train)
    print(f"  trained in {_time_mod.time()-t0:.1f}s")

    proba = model.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"  Test AUC: {auc:.4f}")
    print(classification_report(y_test, model.predict(X_test_s),
                                 target_names=["loss", "win"]))

    # Top features by absolute coefficient (on STANDARDIZED features so
    # the magnitude is comparable)
    coefs = pd.DataFrame({
        "feature": X_train.columns,
        "coef":    model.coef_[0],
        "abs":     abs(model.coef_[0]),
    }).sort_values("abs", ascending=False)

    print("\nTop 15 features by |coefficient| (logistic):")
    for _, row in coefs.head(15).iterrows():
        sign = "+" if row["coef"] > 0 else "-"
        print(f"  {sign} {row['feature']:<25} coef={row['coef']:+.4f}")
    return auc, coefs, proba


def train_xgboost(X_train, y_train, X_test, y_test):
    try:
        import xgboost as xgb
    except ImportError:
        print("\n=== XGBOOST: skipping (xgboost not installed) ===")
        return None, None, None
    from sklearn.metrics import roc_auc_score, classification_report

    print("\n=== XGBOOST ===")
    t0 = _time_mod.time()
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric="auc",
    )
    model.fit(X_train, y_train)
    print(f"  trained in {_time_mod.time()-t0:.1f}s")

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"  Test AUC: {auc:.4f}")
    print(classification_report(y_test, model.predict(X_test),
                                 target_names=["loss", "win"]))

    importances = pd.DataFrame({
        "feature":    X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nTop 15 features by importance (XGBoost):")
    for _, row in importances.head(15).iterrows():
        print(f"  {row['feature']:<25} {row['importance']:.4f}")
    return auc, importances, proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0,
                        help="If >0, use only N random trades (faster)")
    parser.add_argument("--no-xgboost", action="store_true")
    args = parser.parse_args()

    print("=== Wave 2 :: TRADE FEATURE ANALYSIS ===\n")
    df = load_trades_with_run_params()

    if args.sample > 0:
        df = df.sample(n=min(args.sample, len(df)), random_state=42).reset_index(drop=True)
        print(f"  sampled to {len(df):,} trades")

    feats, label = build_features(df)

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        feats, label, test_size=0.2, random_state=42, stratify=label
    )
    print(f"\nTrain: {len(X_train):,}  Test: {len(X_test):,}")

    log_auc, log_coefs, log_proba = train_logistic(X_train, y_train, X_test, y_test)

    if args.no_xgboost:
        xgb_auc = None
    else:
        xgb_auc, xgb_imp, xgb_proba = train_xgboost(X_train, y_train, X_test, y_test)

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  Logistic AUC:  {log_auc:.4f}")
    if xgb_auc is not None:
        print(f"  XGBoost AUC:   {xgb_auc:.4f}")
        delta = xgb_auc - log_auc
        print(f"  Δ (XGB - Log): {delta:+.4f}  "
              f"({'meaningful nonlinear signal' if delta > 0.02 else 'mostly linear'})")
    if max(log_auc, xgb_auc or 0) < 0.55:
        print("  → AUC < 0.55: NO meaningful predictive edge in these features.")
        print("    The strategies' filters are already capturing what's predictable.")
    elif max(log_auc, xgb_auc or 0) < 0.60:
        print("  → AUC 0.55-0.60: Weak signal. Top features may be worth adding")
        print("    as soft filters, but expect single-digit % improvement at best.")
    else:
        print("  → AUC ≥ 0.60: Real predictive signal. Top features are worth")
        print("    investigating as additional strategy gates.")

    # Save outputs
    out_dir = _ROOT / "logs" / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "wave2_features.txt", "w") as f:
        f.write("=== Wave 2 Trade Feature Analysis ===\n\n")
        f.write(f"Trades: {len(df):,}\n")
        f.write(f"Train/Test: {len(X_train):,} / {len(X_test):,}\n")
        f.write(f"Win rate: {label.mean():.3f}\n\n")
        f.write(f"Logistic AUC: {log_auc:.4f}\n")
        if xgb_auc is not None:
            f.write(f"XGBoost AUC:  {xgb_auc:.4f}\n")
        f.write("\nTop features by |coef| (logistic):\n")
        for _, row in log_coefs.head(20).iterrows():
            f.write(f"  {row['feature']:<25}  coef={row['coef']:+.5f}\n")
        if xgb_auc is not None:
            f.write("\nTop features by importance (XGBoost):\n")
            for _, row in xgb_imp.head(20).iterrows():
                f.write(f"  {row['feature']:<25}  imp={row['importance']:.5f}\n")
    print(f"\n  → wrote logs/research/wave2_features.txt")


if __name__ == "__main__":
    main()
