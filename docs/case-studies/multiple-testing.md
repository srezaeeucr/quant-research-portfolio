---
title: Case Study — Multiple Testing Correction
---

[← Back to overview](../index.html)

# Case Study — Multiple Testing Correction

> With 1,596 simultaneous tests at α=0.05, roughly 80 false-positive
> "passers" are expected by pure chance.

This case study documents the application of Benjamini-Hochberg FDR
and Bonferroni corrections to the V4 validation chain results, and
the finding that the unadjusted result was — in this specific case —
robust to multiple comparisons.

---

## Setting

The V4 chain tested **1,596 configurations** (19 symbols × 3 strategies
× 4 stop-loss levels × 7 reward-risk ratios), each through Monte Carlo
permutation testing yielding a p-value. The unadjusted "passer" criterion
was MC p ≤ 0.05.

The basic statistical issue: at α=0.05 with 1,596 simultaneous tests,
the **expected number of false positives** under the null hypothesis is
1,596 × 0.05 = **~80 false positives**.

If 100 configurations pass the unadjusted test, somewhere between 0
and 80 are likely false positives. The unadjusted result therefore
does not directly support the claim that any specific configuration
has a real entry-signal edge — only that on average, the population has
an edge.

This is the same statistical issue famously associated with garden-of-forking-paths
analysis, with the 2015 Reproducibility Crisis in psychology, and with
the "Five Sigma" requirement in particle physics.

---

## Two correction procedures

Two standard approaches were applied:

### Bonferroni correction (strict)

The Bonferroni method ensures the family-wise error rate (FWER) — the
probability of ANY false positive — is at most α. The corrected per-test
threshold is α/N:

```
α_adj = 0.05 / 1596 = 3.13 × 10⁻⁵
```

A configuration passes Bonferroni-adjusted significance only if its
individual p-value is below 3.13 × 10⁻⁵.

This is **very conservative** — it controls for the worst case, treating
each test as if it could be the lone false positive.

### Benjamini-Hochberg FDR correction (less strict)

The BH-FDR method controls the **false discovery rate** — the expected
proportion of false positives among the declared passers. For a chosen
FDR level q:

1. Sort p-values in ascending order: p(1) ≤ p(2) ≤ ... ≤ p(N).
2. Find the largest k such that p(k) ≤ (k/N) × q.
3. Declare all i ≤ k as significant.

With q=0.05, this allows ~5% of the passers to be false discoveries
in expectation — a much more practical threshold for research.

---

## Implementation

```python
def benjamini_hochberg(p_values, alpha=0.05):
    """Returns boolean array; True = significant under BH-FDR."""
    n = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    passes = sorted_p <= thresholds
    if not passes.any():
        return np.zeros(n, dtype=bool)
    last_pass = np.where(passes)[0].max()
    result = np.zeros(n, dtype=bool)
    result[order[:last_pass + 1]] = True
    return result
```

Implementation lives in `backtest/fdr_correction.py`.

---

## Result on the V4 data

Of the 100 unadjusted "passers":

| Correction | Threshold | Passers |
|---|---|---|
| None | p ≤ 0.05 | 100 |
| Bonferroni | p ≤ 3.13 × 10⁻⁵ | **100** |
| BH-FDR (q=0.05) | per-rank | **100** |

**All 100 unadjusted passers survived Bonferroni.**

This is a strong result. The Monte Carlo permutation tests in this dataset
have p-values clustered tightly at zero (with 200 permutations, the
practical lower bound is ~0.005, but most passers had observed
significance well below this). The bulk of "p=0.000" results held up
under the strictest correction.

Practically: the unadjusted V4 chain was — for this particular dataset
— statistically valid in the multiple-testing sense.

---

## Why this still matters

The Bonferroni-survival result does NOT mean the strategy edges are
real in real-world trading. It only means **the strategies are doing
something genuinely different from random entry**, not that what they
are doing will be profitable after costs.

Of the 100 Bonferroni-corrected passers, only 30 also passed
out-of-sample validation on 2026 data (see [regime-shift case study](regime-shift.md)).
This is a useful comparison: even when in-sample statistical
significance is rock-solid, **70% of in-sample passers fail
out-of-sample**. The Bonferroni correction protects against being
fooled by RANDOMNESS in the in-sample data, but provides no protection
against being fooled by REGIME CHANGE.

The combined picture:

- Unadjusted V4 → 100 passers
- + Bonferroni correction → still 100
- + 2026 recent out-of-sample → **30 passers**
- + quarterly stability check → **18 passers**

Multiple-testing correction is necessary but not sufficient. The much
larger constraint is **out-of-sample robustness across regimes**.

---

## Why this matters for a researcher

Two important habits illustrated by this analysis:

### 1. Always run the correction

It is faster than the validation chain itself (it is just post-processing
of p-values) and provides a clean answer to the "is this just statistical
noise?" question. Skipping it leaves the entire research vulnerable to a
trivially-fixable critique.

### 2. Know what each test does and does not protect against

| Test | Protects against |
|---|---|
| Multiple testing correction | Spurious in-sample significance |
| Out-of-sample validation | In-sample overfit to specific data |
| Walk-forward optimization | Stationary-period assumption |
| Slippage stress | Costs that backtest ignores |
| Quarterly stability | Lucky training-period regime |

No single test is sufficient. The V4 chain combines several but it's
the **combination** that provides protection — not any individual test.

---

## See also

- `backtest/full_validation_v4.py` — the V4 chain that produced the
  1,596 p-values
- `backtest/fdr_correction.py` — BH and Bonferroni implementations
- [regime-shift case study](regime-shift.md) — for why correction is
  necessary but not sufficient
