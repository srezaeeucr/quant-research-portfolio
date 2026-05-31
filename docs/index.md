---
title: Quant Research Portfolio
---

# Quant Research Portfolio

A statistical research framework for designing, backtesting, and validating intraday equity trading strategies.

The repository documents **methodology over results** — emphasizing out-of-sample validation, multiple-testing correction, and the explicit recognition of overfitting across a sequence of independent studies.

[**View source on GitHub →**](https://github.com/srezaeeucr/quant-research-portfolio)

---

## What's in scope

- **14 strategy archetypes** — breakout, trend, momentum, mean reversion, volume, pattern, and composite families, all implementing a common `BaseStrategy` interface.
- **Four-stage validation chain** (V4) — walk-forward + Monte Carlo + slippage stress + out-of-sample recent.
- **Multiple-testing correction** — Benjamini-Hochberg FDR and Bonferroni, applied to 1,596 simultaneous tests.
- **Anti-overfitting discipline** — documented rejection of period-optimal configurations in favor of regime-stable defaults.
- **171-test test suite** for strategies, engine, risk, and broker layers.

---

## Methodology highlights

### The V4 validation chain

A configuration is V4-validated only if all three pass:

| Test | Threshold |
|---|---|
| **Walk-forward** (10 rolling windows, 6mo train / 3mo test) | ≥ 6 of 10 windows profitable |
| **Monte Carlo** (200 random-entry permutations) | p-value ≤ 0.05 |
| **Slippage stress** at realistic 0.05% | profit factor ≥ 1.0 |

Of **1,596 candidate configurations**, only **100 passed**.

### The quad-validation extension

Recognizing that V4 used 2022-2024 data exclusively, an independent **2026 out-of-sample test** was added:

- **V4-validated**: 100 configurations
- **+ profitable on 2026 data** (triple): 30 configurations
- **+ stable across 12 quarterly windows** (quad): **18 configurations**

The numbers shrank by ~5× because most historical edges did not survive forward-time evidence.

### Multiple-testing correction

With 1,596 simultaneous tests at α=0.05, roughly **80 false positives are expected by chance alone**.

- **Bonferroni** correction (α/N = 3.1 × 10⁻⁵) — all 100 V4 passers survived
- **Benjamini-Hochberg FDR** (q=0.05) — all 100 survived

The unadjusted result was statistically robust.

---

## Case studies

Four deep-dive write-ups of the methodology in practice:

### **[Anti-overfit discipline →](case-studies/anti-overfit.html)**

How four different "best AMD configurations" emerged in 13 days as new validation tests were added — and the disciplined decision to reject all four in favor of a mid-range, period-agnostic default.

> When the "best" answer changes with every test, the test framework itself is being tuned, not the strategy.

### **[Regime shift detection →](case-studies/regime-shift.html)**

META and TSLA — both with 21+ quarterly-stable configurations through 2024 — saw **every single configuration become unprofitable** on 2026 data. Direct evidence that historical robustness does not guarantee forward performance.

> Of 60 META+TSLA configurations validated in-sample, zero survived out-of-sample.

### **[Multiple testing correction →](case-studies/multiple-testing.html)**

Application of Benjamini-Hochberg FDR and Bonferroni to 1,596 simultaneous tests. The unadjusted result was rock-solid (all 100 passers survived Bonferroni) — and yet 70% of those passers failed out-of-sample. Statistical significance is necessary but not sufficient.

### **[Three big simulations →](case-studies/big-simulation.html)**

Three end-to-end backtests comparing (A) a small mid-range 3-slot configuration, (B) an 18-config "all-in" portfolio of the quad-validated set, and (C) a 12-strategy ranking sweep. The simpler Sim A beats the more ambitious Sim B on out-of-sample data; Sim C's #1 historical strategy (Donchian Breakout) fails on out-of-sample AMD — the same regime-shift pattern that disqualified META and TSLA earlier.

> The most ambitious configuration is worse on the most relevant period. More configurations is not free.

---

## Documented negative results

| Study | Verdict |
|---|---|
| ATR dynamic stops | Rejected — lost 5/7 combos vs fixed stops |
| Trailing stops | Rejected — lost on holdout 2022 |
| Partial exits | Rejected — lost on holdout 2022 |
| Time-based exits (COIN) | Rejected — hold-to-EOD outperformed all alternatives by 50–90% on 918 trades |
| Symmetric ensemble (N-of-N agreement) | Rejected — does not replicate the GPU mega claim at bar level |
| ML config-selection (XGBoost) | Rejected — AUC 0.55 (barely above random) for predicting config profitability |

Each rejection is documented in `backtest/` and contributes to confidence in the configurations that did survive.

---

## Repository structure

```
src/         engine, 14 strategies, risk, brokers, data, logger
backtest/    20+ research scripts (one per study)
tests/       180-test test suite
docs/        methodology + case studies (this site)
```

[**Full repo on GitHub**](https://github.com/srezaeeucr/quant-research-portfolio) — including reproducible backtest scripts, complete test suite, and detailed documentation.

---

## Limitations

- **Symbol universe** — 19 names chosen for liquidity and prior interest; broader universe would reduce selection bias
- **Monte Carlo permutation count** — 200 per configuration; smallest measurable p-value is therefore 0.005
- **Overnight risk** — not modeled; framework assumes end-of-day position closure
- **Live execution effects** — fill latency and adverse selection not fully reflected in backtests
- **Transaction costs** — slippage modeled (up to 0.20%); commissions not (broker is commission-free)
