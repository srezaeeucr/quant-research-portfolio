---
title: Case Study — Anti-Overfit Discipline
---

[← Back to overview](../index.html)

# Case Study — Anti-Overfit Discipline

> When the "best" answer changes with every test, the test framework itself
> is being tuned, not the strategy.

This case study documents how, over a 13-day window, **four distinct "best AMD
configurations"** emerged from successive validation studies — and the
disciplined decision to reject all of them in favor of a mid-range,
period-agnostic configuration.

It is the single most important methodological commitment of this research
project. The point is not to demonstrate clever optimization; the point is to
demonstrate **the discipline of rejecting one's own optimal results when
they fail a generalization test.**

---

## Setting

The validation framework had produced a strong baseline result: 100 of 1,596
configurations passed the V4 validation chain (walk-forward + Monte Carlo +
slippage stress, all on 2022–2024 data). Among these, AMD ORB at
`SL=0.75%, RR=5.0` was a strong candidate for the symbol AMD.

Over the following two weeks, four additional studies were performed.
Each one — using sound methodology in isolation — produced a different
"best AMD configuration."

---

## The flip-flop

| Study | Date | Selected AMD config | PnL on the source data | Justification at the time |
|---|---|---|---|---|
| DEFINITIVE chain | Apr 13 | **ORB** / SL=0.75% / RR=5.0 | +$X (full_2yr) | Best of 419 holdout-screened configs through WF+MC+Slip |
| Recent-data switch | Apr 15 | **Momentum** / SL=0.3% / RR=1.5 | +$33.81 (Jan–Apr 2026) | Best PF (2.63, 72% WR) on Jan–Apr 2026 |
| Quad-pass | Apr 21 | **ORB** / SL=0.3% / RR=5.0 | +$25.52 (Jan–Apr 2026) | Best slip-PF in V4-passers also profitable on recent |
| Extended-recent | Apr 26 | **Momentum** / SL=1.0% / RR=5.0 | +$33.81 (Jan–Apr 26 extended) | Best PF across more recent data |

Each pick was the "best" by the study's own metric. None of them was the same
as the previous pick. **The strategy choice flipped between ORB and Momentum
twice. The stop-loss varied between 0.3%, 0.75%, and 1.0%. The reward-risk
ratio varied between 1.5 and 5.0.**

In a 13-day window. With no new data, only new tests.

---

## What's happening?

The "best" configuration depends on the metric used to evaluate it. Each
study introduced its own metric (cumulative PnL over a period, walk-forward
stability, multi-test agreement). When the metrics are noisy and partially
correlated, the maximum of each can fall on a different configuration.

In other words: **with 1,596 candidates and a dozen different scoring rules,
the empirically optimal configuration becomes a function of which rule
you ask, not of which strategy is genuinely best.**

This is "the garden of forking paths" — a well-known phenomenon in
statistics. The more analyses you run, the more likely you are to find
a spuriously strong result by chance.

---

## The discipline

After noticing the flip-flop, the research project adopted the following
rule: **the live configuration uses ONLY parameters and symbols that are
consistent across multiple independent studies.**

The resulting configuration was deliberately mid-range:

```yaml
symbols: [AMD, GOOGL]
both: ORB / SL=0.75% / RR=3.0
```

- **ORB** because it was the only strategy that survived as a "passer"
  across all validation rounds (also the only strategy that works *standalone*
  per the ensemble analysis — others required multi-strategy agreement to
  be profitable).
- **SL=0.75%, RR=3.0** as a mid-range default, not optimized to any specific
  data window.
- **AMD and GOOGL** as the only two symbols appearing in essentially every
  passer list across studies.
- **Explicitly accepting lower expected backtest performance** in exchange
  for parameter stability across regimes.

The configuration was committed with this commit message:

> ```
> Simplify config to AMD+GOOGL ORB SL=0.75/RR=3.0 (anti-overfit revert)
>
> After 7 fresh studies the "best (symbol, strategy, SL, RR)" picks kept
> flip-flopping between studies — 4 different "best AMD" configs in 13 days.
> Strong overfitting signal.
>
> Reverted to a mid-range, period-agnostic config using only claims that
> hold consistently across cumulative studies:
> - ORB strategy works alone (MC p=0.000, ensemble study confirms it doesn't
>   need other strategies to be profitable, unlike EMA)
> - AMD and GOOGL appear in essentially every passer list
> - Mid-range stops (0.75%) survive most regimes
>
> Trade-off accepted: this config will likely under-perform period-optimized
> configs on recent backtests but is more likely to keep working across
> future regime shifts.
> ```

---

## What we learned

### 1. Many "best" answers in a noisy domain ≈ no genuine best

If small changes in the test rule produce large changes in the optimum,
the data is not strongly informative about the true underlying ranking.
The right response is **not** to choose any particular optimum but to
notice that the data does not support strong claims about ranking — and
to fall back to robust defaults.

### 2. Methodological self-skepticism is more valuable than methodological complexity

The validation framework is already sophisticated: walk-forward, Monte
Carlo, slippage stress, multiple-testing correction. None of that
prevents overfitting if the framework itself becomes a candidate to
tune the strategy against.

The fix is not more validation tests. The fix is **fewer, with the
honest recognition that the test set is now polluted as a forward
indicator**.

### 3. Stop changing the answer

Across the four "best" candidates, three of them — when run forward
on live paper data — failed within 5 trades. Only the mid-range
fallback consistently behaved like its backtest. This is consistent
with the hypothesis that the period-optimal picks were each overfit
to their respective windows.

### 4. The right number of studies is bounded

This project ran 42 documented studies. By study 30 or so, the marginal
information about strategy quality from each new study was likely
negative — each new study added another scoring rule and another chance
of an unrepresentative optimum.

A more disciplined research program would have stopped earlier and
committed to one robust default with deliberately broad parameters.

---

## Why this matters for a researcher

A quantitative researcher's value is not in finding the highest backtest
PnL on a given dataset. Any optimization library can do that. The value
is in **knowing what the backtest does NOT tell you, and refusing to
draw conclusions the data does not actually support.**

This case study exists because the discipline of admitting "I don't have
the right answer, but here's why my four previous attempts were not
better" is harder than producing one more "best" configuration. It is
also the difference between research that survives in live deployment
and research that does not.

---

## See also

- `backtest/full_validation_v4.py` — the V4 chain that produced the original
  100 passers
- `backtest/fdr_correction.py` — the multiple-testing correction analysis
- `backtest/wf_window_stability.py` — quarterly robustness check that produced
  the third "best" answer
- `backtest/validate_recent.py` — the recent-data study that produced the
  second "best" answer
