---
title: Methodology
---

[← Back to overview](index.html)

# Methodology

A walk-through of how the validation chain works, what each component
controls for, and what it does *not* control for.

---

## The problem with naive backtesting

A naive backtest is: pick a strategy, run it on historical data, see the
profit factor. If it's > 1, claim a positive edge.

This approach is wrong in several ways that are well-documented in the
quantitative trading literature:

1. **Lookahead bias** — using data the algorithm wouldn't have seen at decision time.
2. **Survivorship bias** — selecting symbols that exist today, ignoring delisted ones.
3. **Optimization overfitting** — picking parameters that fit the specific data window.
4. **Regime overfitting** — finding patterns that worked in the training period but not generally.
5. **Multiple testing** — running enough variations that some succeed by chance.
6. **Slippage neglect** — assuming you fill at the bar's close price.
7. **No out-of-sample evidence** — no test on data the model didn't see.

The V4 validation chain combines several independent controls against
these issues.

---

## V4 chain — four stages

### Stage 1: Quick filter

Each configuration is first run as a single backtest on `full_2yr`
(Jan 2023 – Jan 2025). Configurations producing fewer than 10 trades
are excluded — they don't have enough data to make a meaningful claim.

### Stage 2: Walk-forward optimization

For configurations that pass Stage 1, walk-forward optimization is run:

- **Window structure**: 6-month train, 3-month test, rolled by 3 months → 10 windows total
- **Inner optimization**: within each train window, a parameter sweep finds the best
  (SL, RR, volume) — that "best" is then evaluated forward on the test window
- **Pass criterion**: profitable in at least 6 of 10 test windows

This controls against the assumption that any fixed parameter set was correct
across the entire period. If the strategy needs the same parameters for every
quarter of every year to be profitable, it's likely overfit. If different
parameter sets work in different quarters, the strategy has a real edge that
can be re-tuned forward.

### Stage 3: Monte Carlo permutation test

This is the most important statistical test in the chain.

For each configuration, the engine is run with the actual entry rule, then
run again with random entries (same number of trades, same days, same hold
duration). The "real" profit factor is compared against the distribution
of 200 random-entry profit factors.

- **p-value**: fraction of random-entry trials that beat the real result
- **Pass criterion**: p ≤ 0.05

This controls against the possibility that the strategy is "successful"
only because the market trended favorably during the test period. If a
random-entry baseline beats the real strategy as often as not, the strategy
is no better than buying at random in a bull market.

### Stage 4: Slippage stress

Real fills are never at the bar's close price. The slippage stress test
runs each configuration at 5 different slippage levels (0%, 0.02%, 0.05%,
0.10%, 0.20%) across 3 periods.

- **Pass criterion**: slippage-adjusted profit factor at 0.05% ≥ 1.0

This controls against the possibility that the edge exists but is too thin
to survive realistic execution costs.

---

## The combined chain

A configuration is **V4-validated** only if it passes all three of WF, MC,
and slippage. Of 1,596 configurations tested, **100 passed**.

This is a strong control set. But it has known weaknesses — see the
case studies for details.

---

## Multiple testing correction

With 1,596 simultaneous tests at α=0.05, by pure chance ~80 false
positives are expected. Two corrections are applied:

### Bonferroni (strict)

Threshold = α / N = 0.05 / 1596 = **3.13 × 10⁻⁵**

A configuration is Bonferroni-significant if its individual p-value is
below this threshold. All 100 V4 passers survived this test.

### Benjamini-Hochberg FDR (less strict)

Sort p-values in ascending order. Find the largest k such that
`p(k) ≤ (k/N) × q`. Declare all i ≤ k as significant.

With q = 0.05, this allows up to ~5% of declared passers to be false
discoveries. Again, all 100 V4 passers survived.

---

## Out-of-sample (recent) validation

V4 used 2022-2024 data exclusively. To probe forward-time robustness,
each of the 100 V4 passers was tested on **January-April 2026** data —
data that had not been used in the V4 chain.

Configurations that remained profitable on recent data are called
**triple-validated** (V4 + recent). Of the 100 V4 passers, **30 were
triple-validated**.

The 70 that failed were not "wrong" in the in-sample sense — they were
statistically significant at α << 0.05. They simply failed to predict
forward.

This is the most important test in the entire framework, because it is
the only one that produces evidence about forward performance.

---

## Quarterly stability check

A further test: each configuration's PnL is broken down by quarter
(12 quarters in 2022-2024). A configuration is **quarterly-stable**
if it is profitable in at least 8 of 12 quarters and the worst
quarterly PF is at least 0.7.

Of the 30 triple-validated configurations, **18 were also quarterly-stable**.
This is the **quad-validation** set.

---

## What V4 does NOT control for

The chain is explicit about its limitations:

- **Regime shift** — even quarterly-stable configurations can become
  unprofitable when the underlying regime changes. See the
  [regime-shift case study](case-studies/regime-shift.html) for direct
  evidence (META and TSLA).
- **Adverse selection at fill time** — real fills experience selection
  bias (your order arrives in the second a faster trader is already
  fading your move). Not modeled.
- **Network latency** — backtest assumes instant execution; real
  paper trading shows 1-2 second delays.
- **Multi-strategy interaction** — backtests test each configuration in
  isolation. A portfolio of multiple configurations interacts through
  shared capital. See `backtest/portfolio_backtest.py`.
- **Look-ahead bias in indicator construction** — controlled-for in
  the engine but each strategy implementation should be reviewed
  independently.

---

## Reproducibility

Each stage of the V4 chain is implemented in `backtest/full_validation_v4.py`.
The multiple-testing correction is in `backtest/fdr_correction.py`. Recent
validation is in `backtest/validate_recent.py`. Quarterly stability is in
`backtest/wf_window_stability.py`.

All are self-contained Python scripts. Run any of them with:

```bash
python backtest/<script>.py
```

with `ALPACA_API_KEY` set in environment (or rely on yfinance fallback).

---

[← Back to overview](index.html)
