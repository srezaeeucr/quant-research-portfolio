---
title: Case Study — Three Big Simulations
---

[← Back to overview](../index.html)

# Case Study — Three Big Simulations

> Three independent backtests stress-testing what the framework's research
> claims actually mean, in dollars.

This case study runs three end-to-end simulations that test specific
empirical claims of the framework: (A) that a small, simple configuration
holds up across regimes; (B) that "all-in" portfolio scaling of the
quad-validated set actually helps; and (C) that the strategy rankings
implied by the framework are consistent with historical aggregate PnL.

The three simulations correspond to `backtest/big_sim_{a,b,c}_*.py` in
the repository.

---

## Setup

All three simulations use:

- **In-sample period**: January 2023 – January 2025 (the V4 training period)
- **Out-of-sample period**: January 2026 – May 2026 (data not seen by V4)
- **Initial capital**: $500
- **Trading window**: 9:30 ET – 15:50 ET, all-day entries
- **Risk filters**: regime SMA on, VIX < 25
- **Slippage**: not modeled in this comparison (separately validated to 0.05% elsewhere)

---

## Simulation A — Simple three-slot configuration

**Question.** Does the small, mid-range, anti-overfit-driven configuration
work in both the in-sample and out-of-sample periods?

**Configuration tested:**

| Slot | Strategy | Symbol | SL | RR |
|---|---|---|---|---|
| 1 | ORB | AMD | 1.0% | 1.5 |
| 2 | RSI Reversion | AMD | 0.5% | 2.0 |
| 3 | ORB | GOOGL | 0.75% | 2.0 |

Slots are aggregated into a shared-capital portfolio with **max 2 concurrent
positions at 50% size** (the allocation policy that won in
`backtest/portfolio_backtest.py`).

**Pass criteria** (stated before running):

- Combined PF ≥ 1.2 on in-sample
- Combined PF ≥ 1.0 on out-of-sample
- At least one slot profitable in each period

**Results:**

| Metric | In-sample (2023-24) | Out-of-sample (2026 Jan-May) |
|---|---|---|
| Trades | 612 | 183 |
| PnL | **+$125.80** (25% on $500) | **+$55.03** (11% on $500) |
| PF | 1.25 | 1.37 |
| Sharpe | 1.90 | 2.73 |
| Max DD | 50.8% | 50.9% |
| Slots positive | 2/3 | **3/3** |

**Pass criteria met**: PF in both periods, all slots profitable on
out-of-sample. The high max-DD is a structural property of small-account
compounding (50% size on losses), not a strategy failure — Sharpe and PF
are the meaningful signals.

**Verdict**: The simple configuration is genuinely working. The out-of-sample
result is the stronger evidence (smaller sample but truly forward-time).

---

## Simulation B — Quad-validated portfolio (the "all-in" version)

**Question.** Does scaling up to all 18 quadruple-validated configurations
beat the simple 3-slot version?

**Hypothesis**: Yes, the larger, more diversified portfolio should produce
better risk-adjusted returns (higher Sharpe) and at least matched
out-of-sample profitability.

**Configurations tested**: All 18 from the quadruple-validation
intersection (V4 + WF stability + recent profitability), as documented in
`docs/research-results.md`. The set is dominated by AMD ORB variants
(14 of 18) plus 4 GOOGL Momentum configurations.

**Portfolio policy**: max 4 concurrent positions at 20% size (smaller
per-position sizing to accommodate more slots).

**Pass criteria**:

- Sharpe > 1.5 on in-sample
- PF > 1.0 on out-of-sample
- PnL on out-of-sample ≥ Sim A's PnL on out-of-sample

**Results:**

| Metric | In-sample | Out-of-sample |
|---|---|---|
| Slots running | 18 | 18 |
| Total signals across slots | 3,262 | 925 |
| Trades taken / skipped (concurrency) | 1,125 / 2,137 | 279 / 646 |
| PnL | +$137 | **+$26** |
| PF | 1.46 | 1.32 |
| Sharpe | **2.87** | 2.06 |
| Max DD | 59.8% | 61.2% |

**Pass result**: in-sample Sharpe target met. Out-of-sample PnL ($26)
is **below** Sim A's out-of-sample PnL ($55) — by $29.

**Verdict**: The more ambitious portfolio does NOT beat the simpler version
on out-of-sample data, despite having strictly more information. Diagnosis:

- 14 of 18 quad-validated configurations are AMD ORB variants with
  similar entry rules. The "diversification" of 18 slots is illusory
  because they fire on the same bars.
- The 4 GOOGL configurations are all Momentum-strategy — but on recent
  data, **GOOGL ORB outperforms GOOGL Momentum** (this was the rationale
  for the simple Sim A configuration).
- Concurrency-capping skips 70% of signals, further concentrating
  exposure to a handful of bars.

This is direct empirical support for the anti-overfit philosophy
documented separately. **More configurations is not better when the
configurations are correlated**.

---

## Simulation C — Strategy-class ranking sweep

**Question.** At neutral, non-period-optimized parameters (SL=0.75%,
RR=3.0), which strategies in the framework produce profitable aggregate
PnL on AMD + GOOGL combined?

**Pass criteria**:

- ORB ranks in the top 5 by aggregate PnL (its standalone-edge claim)
- ≥ half of (strategy × symbol) pairs are profitable

**Result (in-sample full_2yr aggregate, AMD + GOOGL combined PnL)**:

| Rank | Strategy | Trades | Aggregate PnL |
|---|---|---|---|
| 1 | **Donchian Breakout** | 669 | **+$206.73** |
| 2 | **ORB** | 436 | +$181.04 |
| 3 | **RSI Reversion** | 432 | +$150.13 |
| 4 | MACD Crossover | 518 | +$79.84 |
| 5 | Momentum | 181 | +$44.78 |
| 6 | VWAP Bands | 602 | +$33.03 |
| 7 | Gap Fill | 351 | −$1.76 |
| 8 | Inside Bar Breakout | 582 | −$17.79 |
| 9 | Bollinger Reversal | 670 | −$27.54 |
| 10 | Stochastic Crossover | 652 | −$35.53 |
| 11 | EMA Crossover | 642 | **−$123.90** |
| 12 | VWAP Reversion | 1,477 | **−$161.71** |

**Pass result**: ORB at #2 (in top 5). 13 of 24 (strategy × symbol) pairs
profitable, meeting threshold.

**Out-of-sample (Jan-May 2026) check on the top 5**:

| Strategy | AMD recent | GOOGL recent | Forward verdict |
|---|---|---|---|
| **Donchian Breakout** | **−$41.09** | +$2.58 | **❌ Fails on AMD** |
| ORB | −$22.64 | +$51.77 | ⚠️ Mixed |
| RSI Reversion | +$19.86 | −$2.30 | ✓ AMD survives |
| MACD | +$13.28 | +$6.57 | ✓ Both marginal |
| Momentum | +$19.52 | +$16.45 | ✓ Both survive |

**Critical finding**: **Donchian Breakout — the #1 historical performer —
becomes a clear loser on out-of-sample AMD data.** This is the same
pattern documented in the regime-shift case study: in-sample
optimization does not predict forward performance.

**Verdict**: The framework's strategy rankings are consistent with the
underlying claims (ORB ranks high, EMA-alone is a loser, VWAP Reversion
over-trades). But ranking by in-sample PnL produces a misleading top
choice (Donchian) that the out-of-sample test correctly disqualifies.

**Implication**: do not deploy Donchian to live trading based on the
in-sample ranking, despite it being #1.

---

## Combined synthesis

Across the three simulations:

1. **The simple Sim A configuration is the strongest empirical proposition** —
   profitable in both periods, with all slots positive on out-of-sample data.

2. **The larger Sim B "all-in" portfolio is worse on the more relevant period.**
   This is empirically the case for THIS dataset; the conceptual point —
   more configurations is not free — generalizes.

3. **The Sim C aggregate ranking surfaces a tempting overfit candidate
   (Donchian Breakout)** that the forward test correctly rejects. This
   demonstrates the necessity of out-of-sample validation even after
   in-sample statistical significance has been established.

**Recommended action based on results**: NO change to the simple Sim A
configuration. The framework's anti-overfit discipline (don't trade
the in-sample ranking; require out-of-sample survival) is itself
validated by Sim B and Sim C.

---

## What this case study demonstrates as research methodology

- **Pre-stated hypotheses and pass criteria.** All three simulations
  declare what would falsify them BEFORE running. This prevents the
  motivated-reasoning failure mode where researchers gradually re-interpret
  ambiguous results in their favor.

- **Honest reporting of refuted hypotheses.** Sim B's hypothesis
  (more configs = better) is refuted by the data. This is reported as
  a finding, not as something to re-frame.

- **Resistance to in-sample optima.** Sim C produces a clear in-sample
  #1 (Donchian) that an undisciplined researcher would deploy to live.
  The case study explicitly rejects this decision based on out-of-sample
  evidence.

- **Quantitative ties between simulations.** Sim A's $55 out-of-sample
  becomes the benchmark Sim B must beat — it does not, and the gap
  is explained mechanically (correlated configurations).

---

## See also

- [Anti-overfit discipline](anti-overfit.html) — the methodological
  rule applied here
- [Regime shift detection](regime-shift.html) — the same pattern
  (Donchian failing out-of-sample on AMD) generalizing META/TSLA's
  earlier failure
- `backtest/big_sim_a_simple_config.py`
- `backtest/big_sim_b_quad_portfolio.py`
- `backtest/big_sim_c_strategy_comparison.py`
