---
title: Case Study — Regime Shift Detection
---

[← Back to overview](../index.html)

# Case Study — Regime Shift Detection

> Historical robustness is not a guarantee of forward performance.

This case study documents a sharp regime change observed when V4-validated
configurations were tested on out-of-sample 2026 data: configurations that
had been profitable in **every quarter of 2022-2024** were observed to be
unprofitable on every quarter of early 2026, for specific symbols. The
finding has direct methodological implications for how to weight historical
backtests against current-regime evidence.

---

## Setting

The V4 validation chain (walk-forward + Monte Carlo + slippage stress)
produced **100 statistically robust configurations** on 2022-2024 data
across 5 symbols:

| Symbol | V4 passers | Quarterly stable (8+/12 quarters) |
|---|---|---|
| TSLA | 38 | 22 |
| AMD | 24 | 21 |
| META | 22 | 21 |
| GOOGL | 15 | 4 |
| MSFT | 1 | 0 |

A separate, fully independent study was then run: each of the 100 passers
was tested on January–April 2026 data — data that had not been used in
the V4 chain.

---

## Finding

The result for each symbol:

| Symbol | V4 passers | Still profitable on 2026 | Survival rate |
|---|---|---|---|
| AMD | 24 | 15 | 62.5% |
| GOOGL | 15 | 15 | 100% |
| META | 22 | **0** | **0%** |
| TSLA | 38 | **0** | **0%** |
| MSFT | 1 | 0 | — |

**META and TSLA — both historically robust, with 21 and 22 quarterly-stable
configurations through 2024 — had every single passing configuration
become unprofitable on 2026 data.**

This is not a small statistical anomaly. With 22 independent META
configurations all profitable in 2022-2024 and all losing on 2026 data,
the probability of this happening by chance alone (assuming the underlying
edge persists) is approximately 2⁻²² ≈ 1 in 4 million. The hypothesis that
META's historical edge persists into 2026 can be confidently rejected.

For TSLA the same logic applies with 38 configurations.

---

## Diagnosis

This is a textbook example of **regime change** — the underlying statistical
relationship that an algorithm exploited has broken down, even though the
historical record gave no warning. Possible mechanisms (the analysis does
not distinguish between them):

1. **Market-microstructure change** — execution costs, liquidity, or
   participant composition shifted in a way that disadvantages a particular
   entry signal.
2. **Volatility regime change** — META and TSLA spent 2025-2026 in different
   volatility states (parabolic moves with frequent reversals) than during
   the training period.
3. **Crowding** — if other systematic traders have discovered and exploited
   the same edge, the edge decays.
4. **Idiosyncratic stock behavior** — fundamental shifts in earnings dynamics,
   issuance, or technical patterns specific to these two symbols.

For AMD and GOOGL, the edges appear to persist — at least at the recent
sample of 12-13 weeks of 2026 data. They may also decay later. The point
is that **no historical backtest, however rigorous, can demonstrate
future robustness.**

---

## Methodological consequence

This finding motivated the **quad-validation** framework, in which a
configuration is considered live-ready only if it passes:

1. V4 (walk-forward + Monte Carlo + slippage stress) on 2022-2024
2. Quarterly stability across 12 quarters
3. Recent out-of-sample (Jan-Apr 2026)
4. Slippage stress at 0.05% (realistic IEX feed cost)

Configurations that pass tests 1–2 only — what we call "historically
robust" — are explicitly insufficient for live deployment. META and TSLA
are the empirical demonstration of why.

The triple-validated and quad-validated configuration counts:

- V4 only: 100 configurations
- V4 + recent profitable: **30 configurations**
- V4 + recent + quarterly stable: **18 configurations**

The cost of the additional gating is that **the number of "deployable"
configurations shrinks by a factor of 5–6**. The benefit is that the
remaining configurations have not, at least, been disqualified by the
single most informative out-of-sample test available.

---

## Why this matters for a researcher

There is a strong temptation in research to weight historical evidence
heavily — there's more of it, the statistical tests are well-developed,
and the conclusions are quantifiable. But the actual value of a quant
research program is in **predicting forward performance**, not in
maximizing fit to history.

When the data tells you that 60 configurations (META + TSLA combined)
that were robust in-sample have died out-of-sample, the right response
is not to investigate why or attempt to fix it — the right response is
to **lower one's confidence in similar in-sample-only validation across
all studies**, and to insist on out-of-sample evidence as a precondition
for any production decision.

---

## See also

- `backtest/full_validation_v4.py` — the V4 chain
- `backtest/validate_recent.py` — the recent-data validation
- `backtest/wf_window_stability.py` — quarterly robustness check
- `backtest/correlation_study_full.py` — full strategy correlation matrix
  (relevant: cross-strategy correlations on the failed symbols)
