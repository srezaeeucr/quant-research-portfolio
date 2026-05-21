#!/usr/bin/env python3
"""
GPU Config Scorer — Predicts which (strategy, symbol, params) combos will be
profitable under given market conditions.

Approach:
1. Feature engineering from 320k backtest results + market regime features
2. XGBoost (GPU) baseline — tabular data king
3. PyTorch neural net with attention — captures cross-feature interactions
4. Walk-forward validation — train on earlier periods, predict later ones
5. Practical output: ranked configs for "tomorrow" given current market state

Designed for superpower (RTX 5090, 32GB VRAM).
"""
import os
import sys
import time as _time
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report, mean_squared_error,
    precision_score, recall_score, f1_score,
)

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_ROOT = Path(__file__).resolve().parents[1]


# ═══════════════════════════════════════════════════════════════════
# 1. DATA LOADING & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════

def load_data(db_path):
    """Load backtest results and engineer features."""
    print("\n  Loading data from results.db...")
    c = sqlite3.connect(str(db_path), timeout=30)
    df = pd.read_sql_query("""
        SELECT strategy, symbol, period_label, start_date, end_date,
               stop_loss_pct, reward_risk, volume_mult, regime_filter,
               vix_threshold, max_trades_per_day, afternoon_entries,
               or_duration_min, entry_window_start, entry_window_end,
               total_trades, win_rate, total_pnl, max_drawdown,
               profit_factor, avg_duration_min, sharpe_ratio, final_balance
        FROM backtest_runs
        WHERE total_trades >= 10
    """, c)
    c.close()
    print(f"  Loaded {len(df):,} rows (filtered: trades >= 10)")

    # ── Encode categoricals ──────────────────────────────────────
    le_strategy = LabelEncoder()
    le_symbol = LabelEncoder()
    le_period = LabelEncoder()

    df["strategy_enc"] = le_strategy.fit_transform(df["strategy"])
    df["symbol_enc"] = le_symbol.fit_transform(df["symbol"])
    df["period_enc"] = le_period.fit_transform(df["period_label"])

    # ── Derived features ─────────────────────────────────────────
    # Break-even win rate for given R:R
    df["break_even_wr"] = 100.0 / (1.0 + df["reward_risk"])
    df["wr_above_be"] = df["win_rate"] - df["break_even_wr"]

    # Risk-adjusted metrics
    df["pnl_per_trade"] = df["total_pnl"] / df["total_trades"].clip(lower=1)
    df["risk_reward_ratio"] = df["reward_risk"]
    df["stop_width"] = df["stop_loss_pct"]

    # Period features (encode market regime)
    period_regime = {
        "bull_2023": 1.0,
        "bull_2024": 1.0,
        "bear_2025": -1.0,
        "full_2yr": 0.5,
        "holdout_2022": -0.5,
    }
    df["regime_score"] = df["period_label"].map(period_regime).fillna(0)

    # Is this a "wide stop" config?
    df["wide_stop"] = (df["stop_loss_pct"] >= 0.75).astype(int)
    # Is this an "all-day" config?
    df["all_day"] = df["afternoon_entries"].astype(int)
    # High R:R flag
    df["high_rr"] = (df["reward_risk"] >= 4.0).astype(int)

    # ── Targets ──────────────────────────────────────────────────
    # Binary: will this config be profitable?
    df["profitable"] = (df["profit_factor"] > 1.0).astype(int)
    # Binary: will this be a GOOD config? (PF > 1.3 AND WR > break-even + 5%)
    df["good_config"] = ((df["profit_factor"] > 1.3) & (df["wr_above_be"] > 5)).astype(int)

    print(f"  Profitable: {df['profitable'].sum():,} ({df['profitable'].mean()*100:.1f}%)")
    print(f"  Good config: {df['good_config'].sum():,} ({df['good_config'].mean()*100:.1f}%)")

    encoders = {
        "strategy": le_strategy,
        "symbol": le_symbol,
        "period": le_period,
    }

    return df, encoders


def get_features(df):
    """Select feature columns for modeling."""
    feature_cols = [
        "strategy_enc", "symbol_enc", "period_enc",
        "stop_loss_pct", "reward_risk", "volume_mult",
        "regime_filter", "vix_threshold", "max_trades_per_day",
        "afternoon_entries", "or_duration_min",
        "regime_score", "wide_stop", "all_day", "high_rr",
        "break_even_wr",
    ]
    return feature_cols


# ═══════════════════════════════════════════════════════════════════
# 2. XGBOOST (GPU) — BASELINE
# ═══════════════════════════════════════════════════════════════════

def train_xgboost(df, feature_cols, target="profitable"):
    """Train XGBoost classifier on GPU."""
    import xgboost as xgb

    print(f"\n{'='*70}")
    print(f"  XGBOOST (GPU) — predicting '{target}'")
    print(f"{'='*70}")

    X = df[feature_cols].values
    y = df[target].values

    # Walk-forward split: train on earlier periods, test on later
    # Period order: holdout_2022 < bull_2023 < bull_2024 < bear_2025 < full_2yr
    train_mask = df["period_label"].isin(["holdout_2022", "bull_2023", "bull_2024"])
    test_mask = df["period_label"].isin(["bear_2025"])

    if train_mask.sum() < 1000 or test_mask.sum() < 1000:
        # Fallback to random split
        print("  Using random 80/20 split (not enough period data for walk-forward)")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        split_method = "random"
    else:
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        split_method = "walk-forward (train: 2022-2024, test: bear_2025)"
        print(f"  Split: {split_method}")

    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  Train positive rate: {y_train.mean()*100:.1f}% | Test: {y_test.mean()*100:.1f}%")

    # Check if GPU is available
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        gpu_available = False

    device = "cuda" if gpu_available else "cpu"
    print(f"  Device: {device}")

    # Hyperparameter search
    best_auc = 0
    best_params = None
    best_model = None

    param_grid = [
        {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 500, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 1000, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 8, "learning_rate": 0.03, "n_estimators": 1500, "subsample": 0.7, "colsample_bytree": 0.7},
        {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 500, "subsample": 0.9, "colsample_bytree": 0.9},
        {"max_depth": 10, "learning_rate": 0.01, "n_estimators": 2000, "subsample": 0.8, "colsample_bytree": 0.6},
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 1500, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 5},
        {"max_depth": 6, "learning_rate": 0.02, "n_estimators": 2000, "subsample": 0.8, "colsample_bytree": 0.7, "gamma": 0.1},
        {"max_depth": 8, "learning_rate": 0.05, "n_estimators": 1000, "subsample": 0.7, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0},
    ]

    t0 = _time.time()
    for i, params in enumerate(param_grid):
        model = xgb.XGBClassifier(
            **params,
            device=device,
            eval_metric="auc",
            early_stopping_rounds=50,
            verbosity=0,
            random_state=42,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print(f"  Config {i+1}/{len(param_grid)}: depth={params['max_depth']} lr={params['learning_rate']} "
              f"n={params['n_estimators']} → AUC={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_params = params
            best_model = model

    elapsed = _time.time() - t0
    print(f"\n  Best AUC: {best_auc:.4f} (trained in {elapsed:.1f}s)")
    print(f"  Best params: {best_params}")

    # Detailed evaluation
    y_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    print(f"\n  Classification Report (threshold=0.5):")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"  F1:        {f1_score(y_test, y_pred):.4f}")
    print(f"  AUC:       {best_auc:.4f}")

    # High-confidence predictions
    high_conf = y_prob > 0.7
    if high_conf.sum() > 0:
        actual_pf_high = df.loc[test_mask, "profit_factor"].values[high_conf]
        print(f"\n  High-confidence predictions (prob > 0.7): {high_conf.sum()}")
        print(f"  Actual profitable rate: {(actual_pf_high > 1.0).mean()*100:.1f}%")
        print(f"  Actual mean PF: {actual_pf_high.mean():.2f}")
        print(f"  vs baseline (all test): {df.loc[test_mask, 'profit_factor'].mean():.2f}")

    # Feature importance
    importance = best_model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: -x[1])
    print(f"\n  Feature Importance:")
    for name, imp in feat_imp:
        bar = "█" * int(imp * 100)
        print(f"    {name:<22} {imp:.4f} {bar}")

    return best_model, best_auc, split_method


def train_xgboost_regression(df, feature_cols):
    """Train XGBoost regressor to predict PF directly."""
    import xgboost as xgb

    print(f"\n{'='*70}")
    print(f"  XGBOOST REGRESSION (GPU) — predicting profit_factor")
    print(f"{'='*70}")

    # Cap PF at 5 to reduce outlier impact
    df_reg = df.copy()
    df_reg["pf_capped"] = df_reg["profit_factor"].clip(upper=5.0)

    X = df_reg[feature_cols].values
    y = df_reg["pf_capped"].values

    train_mask = df_reg["period_label"].isin(["holdout_2022", "bull_2023", "bull_2024"])
    test_mask = df_reg["period_label"].isin(["bear_2025"])

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    param_grid = [
        {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 1000, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 8, "learning_rate": 0.03, "n_estimators": 1500, "subsample": 0.7, "colsample_bytree": 0.7},
        {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 500, "subsample": 0.9, "colsample_bytree": 0.9},
        {"max_depth": 10, "learning_rate": 0.01, "n_estimators": 2000, "subsample": 0.8, "colsample_bytree": 0.6},
    ]

    best_rmse = float("inf")
    best_model = None

    t0 = _time.time()
    for i, params in enumerate(param_grid):
        model = xgb.XGBRegressor(
            **params,
            device=device,
            eval_metric="rmse",
            early_stopping_rounds=50,
            verbosity=0,
            random_state=42,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        print(f"  Config {i+1}/{len(param_grid)}: depth={params['max_depth']} → RMSE={rmse:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

    elapsed = _time.time() - t0
    print(f"\n  Best RMSE: {best_rmse:.4f} ({elapsed:.1f}s)")

    # Evaluate: when model predicts PF > 1.3, how often is it actually > 1.0?
    y_pred = best_model.predict(X_test)
    actual_pf = df_reg.loc[test_mask, "profit_factor"].values

    for threshold in [1.0, 1.2, 1.3, 1.5]:
        predicted_good = y_pred > threshold
        if predicted_good.sum() > 0:
            actual_good_rate = (actual_pf[predicted_good] > 1.0).mean()
            actual_mean_pf = actual_pf[predicted_good].mean()
            print(f"  Predicted PF>{threshold}: {predicted_good.sum()} configs, "
                  f"actual profitable={actual_good_rate*100:.1f}%, actual mean PF={actual_mean_pf:.2f}")

    return best_model, best_rmse


# ═══════════════════════════════════════════════════════════════════
# 3. PYTORCH NEURAL NET (GPU)
# ═══════════════════════════════════════════════════════════════════

def train_neural_net(df, feature_cols, target="profitable"):
    """Train PyTorch neural net with attention on GPU."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("\n  PyTorch not available — skipping neural net")
        return None, 0

    print(f"\n{'='*70}")
    print(f"  PYTORCH NEURAL NET (GPU) — predicting '{target}'")
    print(f"{'='*70}")

    # RTX 5090 (sm_120/Blackwell) not supported by PyTorch stable — force CPU
    device = torch.device("cpu")
    print(f"  Device: {device} (PyTorch doesn't support RTX 5090 yet — XGBoost uses GPU)")

    X = df[feature_cols].values.astype(np.float32)
    y = df[target].values.astype(np.float32)

    # Standardize
    scaler = StandardScaler()

    train_mask = df["period_label"].isin(["holdout_2022", "bull_2023", "bull_2024"]).values
    test_mask = df["period_label"].isin(["bear_2025"]).values

    X_train = scaler.fit_transform(X[train_mask])
    X_test = scaler.transform(X[test_mask])
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )
    train_dl = DataLoader(train_ds, batch_size=2048, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=4096)

    n_features = X_train.shape[1]

    class ConfigScorer(nn.Module):
        def __init__(self, n_in, hidden_sizes, dropout=0.3):
            super().__init__()
            layers = []
            prev = n_in
            for h in hidden_sizes:
                layers.extend([
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
                prev = h
            # Attention layer
            self.attention = nn.Sequential(
                nn.Linear(prev, prev),
                nn.Tanh(),
                nn.Linear(prev, 1),
            )
            self.backbone = nn.Sequential(*layers)
            self.head = nn.Linear(prev, 1)

        def forward(self, x):
            h = self.backbone(x)
            attn_weights = torch.softmax(self.attention(h), dim=0)
            h = h * attn_weights
            return self.head(h).squeeze(-1)

    # Hyperparameter search
    configs = [
        {"hidden": [128, 64, 32], "dropout": 0.3, "lr": 1e-3, "epochs": 100},
        {"hidden": [256, 128, 64], "dropout": 0.4, "lr": 5e-4, "epochs": 150},
        {"hidden": [512, 256, 128, 64], "dropout": 0.3, "lr": 3e-4, "epochs": 200},
        {"hidden": [128, 128, 64, 32], "dropout": 0.2, "lr": 1e-3, "epochs": 120},
        {"hidden": [256, 128, 64, 32], "dropout": 0.35, "lr": 5e-4, "epochs": 180},
    ]

    best_auc = 0
    best_model = None
    best_cfg = None

    t0 = _time.time()
    for ci, cfg in enumerate(configs):
        model = ConfigScorer(n_features, cfg["hidden"], cfg["dropout"]).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(cfg["epochs"]):
            model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Evaluate
        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in test_dl:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(yb.numpy())

        y_prob = np.concatenate(all_probs)
        y_true = np.concatenate(all_labels)
        auc = roc_auc_score(y_true, y_prob)
        print(f"  Config {ci+1}/{len(configs)}: {cfg['hidden']} dropout={cfg['dropout']} "
              f"lr={cfg['lr']} epochs={cfg['epochs']} → AUC={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_cfg = cfg

    elapsed = _time.time() - t0
    print(f"\n  Best AUC: {best_auc:.4f} ({elapsed:.1f}s)")
    print(f"  Best config: {best_cfg}")

    return best_model, best_auc


# ═══════════════════════════════════════════════════════════════════
# 4. PRACTICAL OUTPUT — RANK CONFIGS FOR TOMORROW
# ═══════════════════════════════════════════════════════════════════

def rank_configs(model, df, feature_cols, encoders):
    """Use the trained model to rank all configs for bear_2025 (most recent)."""
    import xgboost as xgb

    print(f"\n{'='*70}")
    print(f"  CONFIG RANKINGS — Best configs for current market")
    print(f"{'='*70}")

    # Get bear_2025 rows (most similar to current market)
    recent = df[df["period_label"] == "bear_2025"].copy()
    if recent.empty:
        print("  No bear_2025 data")
        return

    X = recent[feature_cols].values
    probs = model.predict_proba(X)[:, 1]
    recent["pred_prob"] = probs

    # Rank by predicted probability of profitability
    ranked = recent.sort_values("pred_prob", ascending=False)

    # Show top 30 unique (strategy, symbol, config) combos
    print(f"\n  {'Rank':<5} {'Strategy':<16} {'Symbol':<8} {'SL':<5} {'RR':<5} {'PM':<4} "
          f"{'Pred%':>6} {'Actual PF':>9} {'Actual PnL':>10} {'Trades':>7}")
    print(f"  {'-'*85}")

    seen = set()
    rank = 0
    for _, row in ranked.iterrows():
        key = (row["strategy"], row["symbol"], row["stop_loss_pct"], row["reward_risk"], row["afternoon_entries"])
        if key in seen:
            continue
        seen.add(key)
        rank += 1
        if rank > 30:
            break
        actual_pf = row["profit_factor"]
        pf_color = "+" if actual_pf > 1.0 else "-"
        print(f"  {rank:<5} {row['strategy']:<16} {row['symbol']:<8} {row['stop_loss_pct']:<5} "
              f"{row['reward_risk']:<5} {int(row['afternoon_entries']):<4} "
              f"{row['pred_prob']*100:>5.1f}% {actual_pf:>9.2f} ${row['total_pnl']:>+9.2f} {row['total_trades']:>7.0f}")

    # Compare: model's top 10 vs random 10 vs actual best 10
    top10_pred = ranked.drop_duplicates(subset=["strategy", "symbol", "stop_loss_pct", "reward_risk"]).head(10)
    random10 = recent.sample(10, random_state=42)
    best10_actual = recent.sort_values("profit_factor", ascending=False).drop_duplicates(
        subset=["strategy", "symbol", "stop_loss_pct", "reward_risk"]).head(10)

    print(f"\n  COMPARISON (bear_2025):")
    print(f"  Model top-10 mean PF:  {top10_pred['profit_factor'].mean():.2f}  "
          f"(profitable: {(top10_pred['profit_factor']>1).sum()}/10)")
    print(f"  Random 10 mean PF:     {random10['profit_factor'].mean():.2f}  "
          f"(profitable: {(random10['profit_factor']>1).sum()}/10)")
    print(f"  Actual best-10 PF:     {best10_actual['profit_factor'].mean():.2f}  "
          f"(profitable: {(best10_actual['profit_factor']>1).sum()}/10)")


# ═══════════════════════════════════════════════════════════════════
# 5. CROSS-PERIOD GENERALIZATION TEST
# ═══════════════════════════════════════════════════════════════════

def cross_period_test(df, feature_cols):
    """Train on each period, test on every other — full generalization matrix."""
    import xgboost as xgb

    print(f"\n{'='*70}")
    print(f"  CROSS-PERIOD GENERALIZATION MATRIX")
    print(f"{'='*70}")

    periods = ["holdout_2022", "bull_2023", "bull_2024", "bear_2025"]

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    results = {}
    for train_p in periods:
        for test_p in periods:
            if train_p == test_p:
                continue

            train_mask = df["period_label"] == train_p
            test_mask = df["period_label"] == test_p

            if train_mask.sum() < 100 or test_mask.sum() < 100:
                continue

            X_train = df.loc[train_mask, feature_cols].values
            y_train = df.loc[train_mask, "profitable"].values
            X_test = df.loc[test_mask, feature_cols].values
            y_test = df.loc[test_mask, "profitable"].values

            model = xgb.XGBClassifier(
                max_depth=6, learning_rate=0.05, n_estimators=500,
                subsample=0.8, colsample_bytree=0.8,
                device=device, verbosity=0, random_state=42,
                eval_metric="auc", early_stopping_rounds=30,
            )
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            results[(train_p, test_p)] = auc

    # Print matrix
    print(f"\n  {'Train →':<16}", end="")
    for p in periods:
        print(f"  {p[:8]:>10}", end="")
    print()
    print(f"  {'Test ↓':<16}", end="")
    for _ in periods:
        print(f"  {'--------':>10}", end="")
    print()

    for test_p in periods:
        print(f"  {test_p:<16}", end="")
        for train_p in periods:
            if train_p == test_p:
                print(f"  {'—':>10}", end="")
            elif (train_p, test_p) in results:
                auc = results[(train_p, test_p)]
                marker = "✓" if auc > 0.6 else "✗"
                print(f"  {auc:>8.3f} {marker}", end="")
            else:
                print(f"  {'N/A':>10}", end="")
        print()

    avg_auc = np.mean(list(results.values()))
    print(f"\n  Average cross-period AUC: {avg_auc:.4f}")
    if avg_auc > 0.65:
        print("  ✅ Model generalizes well across market regimes")
    elif avg_auc > 0.55:
        print("  ⚠️  Model partially generalizes — use with caution")
    else:
        print("  ❌ Model does NOT generalize — predictions are period-specific")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = _time.time()

    # Use results_merged.db if available (has all 320k rows from Mac Mini),
    # otherwise fall back to local results.db
    merged = _ROOT / "logs" / "results_merged.db"
    db_path = merged if merged.exists() else _ROOT / "logs" / "results.db"
    df, encoders = load_data(db_path)
    feature_cols = get_features(df)

    # 2a. XGBoost classification (profitable yes/no)
    xgb_model, xgb_auc, split = train_xgboost(df, feature_cols, target="profitable")

    # 2b. XGBoost classification (good config)
    xgb_model_good, xgb_auc_good, _ = train_xgboost(df, feature_cols, target="good_config")

    # 2c. XGBoost regression (predict PF)
    xgb_reg_model, xgb_rmse = train_xgboost_regression(df, feature_cols)

    # 3. Neural net
    nn_model, nn_auc = train_neural_net(df, feature_cols, target="profitable")

    # 4. Rank configs using best model
    rank_configs(xgb_model, df, feature_cols, encoders)

    # 5. Cross-period generalization
    cross_period_test(df, feature_cols)

    # Summary
    total_min = (_time.time() - t0) / 60
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  XGBoost (profitable):  AUC = {xgb_auc:.4f}")
    print(f"  XGBoost (good config): AUC = {xgb_auc_good:.4f}")
    print(f"  XGBoost (PF regr):     RMSE = {xgb_rmse:.4f}")
    if nn_model:
        print(f"  Neural Net:            AUC = {nn_auc:.4f}")
    print(f"  Total time: {total_min:.1f} minutes")
    print(f"\n  INTERPRETATION:")
    print(f"  AUC > 0.70 = useful signal (can improve config selection)")
    print(f"  AUC 0.60-0.70 = weak signal (marginal value)")
    print(f"  AUC < 0.60 = noise (config selection can't be predicted from features alone)")
